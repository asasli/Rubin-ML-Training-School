"""Like/dislike metric head (notebook 06 of asasli/SimilaritySearch) on top of the frozen SupCon encoder.

Liked    : the 69 DP2 transients with a coherent change (from the reliability >= 0.3 set).
Disliked : the flat or incoherent objects of the same set.
Test     : DP2 objects the head never saw (the no-reliability-cut set minus the training set), scored against the
           same rule computed directly on their light curves. The rule made the labels, so this measures how well a few
           likes transfer to new objects, not whether the objects are real transients.
"""
import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tap import DATA_DIR
from features6 import features6, nightly, quality6, coherent

warnings.filterwarnings("ignore"); torch.manual_seed(0)
HERE = Path(__file__).resolve().parent; M = HERE / "model"
sc = json.load(open(M / "supcon_scaler.json")); cols = sc["cols"]; med = pd.Series(sc["median"], index=cols); iqr = pd.Series(sc["iqr"], index=cols)
FEATURES = json.load(open(M / "features.json"))["features"]
def prep(X):
    v = ((X[cols] - med) / iqr).clip(-6, 6)
    return torch.tensor(np.hstack([v.fillna(0.0).values, X[cols].isna().values.astype(float)]).astype(np.float32))
class Encoder(nn.Module):
    def __init__(self, d_in, d_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
enc = Encoder(2 * len(cols)); enc.load_state_dict(torch.load(M / "supcon_encoder.pt")); enc.eval()
def embed(X):
    with torch.no_grad():
        return enc(prep(X))

class UserSpecificMetricHead(nn.Module):
    def __init__(self, d_in=64, d_out=16):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_in), nn.ReLU(), nn.Linear(d_in, d_out))
    def forward(self, z):
        return F.normalize(self.proj(z), dim=-1)
def finetune_user_head(z_liked, z_disliked, n_epochs=300):
    torch.manual_seed(0); head = UserSpecificMetricHead()
    opt = torch.optim.Adam(head.parameters(), lr=5e-4, weight_decay=1e-3)
    centroid = z_liked.mean(0, keepdim=True)
    for _ in range(n_epochs):
        z_l, z_d = head(z_liked), head(z_disliked); c = F.normalize(head.proj(centroid), dim=-1)
        loss = (1 - (z_l * c).sum(-1)).mean() + F.relu(0.5 + (z_d * c).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    def score(z):
        with torch.no_grad():
            return (head(z) * F.normalize(head.proj(centroid), dim=-1)).sum(-1).numpy()
    return score

# ---------------------------------------------------------------- training set: the reliability >= 0.3 selection
feat = pd.read_parquet(DATA_DIR / "dp2_transients_table_loose03.parquet")
clean_ids = set(pd.read_parquet(DATA_DIR / "dp2_transients_table_loose03_clean.parquet").diaObjectId)
Xa = pd.read_parquet(DATA_DIR / "dp2_nightly_features_loose03.parquet")
za = embed(Xa[FEATURES]); ya = Xa.diaObjectId.isin(clean_ids).values.astype(int)
print(f"training set: {ya.sum()} liked, {(1 - ya).sum()} disliked", flush=True)
cv = np.zeros(len(ya)); cv_lr = np.zeros(len(ya)); cv_cos = np.zeros(len(ya))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(za, ya):
    zl, zd = za[a][ya[a] == 1], za[a][ya[a] == 0]
    cv[b] = finetune_user_head(zl, zd)(za[b])
    cv_cos[b] = (za[b] @ F.normalize(zl.mean(0), dim=0)).numpy()
    cv_lr[b] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(za[a].numpy(), ya[a]).predict_proba(za[b].numpy())[:, 1]
for nm, s in (("metric head", cv), ("cosine to liked centroid, no head", cv_cos), ("logistic regression on the embedding", cv_lr)):
    print(f"   5-fold within the training set, {nm:38s} AUC {roc_auc_score(ya, s):.3f}  average precision {average_precision_score(ya, s):.3f} (base rate {ya.mean():.3f})", flush=True)
score = finetune_user_head(za[ya == 1], za[ya == 0])
lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(za.numpy(), ya)

# ---------------------------------------------------------------- unseen objects: no reliability cut
cand = pd.read_parquet(DATA_DIR / "stageB_loose00_candidates.parquet")
lo = pd.read_parquet(DATA_DIR / "loose_objects.parquet")[["diaObjectId", "loose_0.3"]]
cand = cand.merge(lo, on="diaObjectId", how="left")
print(f"\nno-reliability-cut set: {len(cand):,} candidates | quiet baseline and no dip: {int(cand.transient.sum()):,} | with events: {int((cand.transient & cand.has_events).sum()):,}")
new = cand[cand.transient & cand.has_events & ~cand["loose_0.3"].fillna(False)].reset_index(drop=True)
saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 2)
rows, ok, rule = [], [], []
with np.load(DATA_DIR / "stageB_loose00_events.npz") as z:
    for i in new.diaObjectId:
        ev = nightly(z[str(i)]); good = quality6(ev); ok.append(good)
        rows.append(features6(ev) if good else {}); rule.append(coherent(ev) if good else False)
os.dup2(saved, 2)
ok, rule = np.array(ok), np.array(rule)
new = new[ok].reset_index(drop=True); Xn = pd.DataFrame(rows).reindex(columns=FEATURES)[ok].reset_index(drop=True); yn = rule[ok].astype(int)
zn = embed(Xn); sn = score(zn); sl = lr.predict_proba(zn.numpy())[:, 1]; scos = (zn @ F.normalize(za[ya == 1].mean(0), dim=0)).numpy()
print(f"unseen objects (mean reliability below 0.3) with six nightly epochs in two bands: {len(new):,}; the rule finds a coherent change in {yn.sum()} ({yn.mean():.3f})")
print(f"   mean reliability quantiles of the unseen objects: {new.rel_ok.quantile([.1, .5, .9]).round(3).to_dict()} | of the rule-positive ones: {new.rel_ok[yn == 1].quantile([.1, .5, .9]).round(3).to_dict()}")
for nm, s in (("metric head", sn), ("cosine to liked centroid, no head", scos), ("logistic regression on the embedding", sl)):
    o = np.argsort(-s); top = lambda k: yn[o[:k]].mean()
    print(f"   unseen, {nm:38s} AUC {roc_auc_score(yn, s):.3f}  average precision {average_precision_score(yn, s):.3f} | "
          f"rule-positive in the top 50: {top(50):.2f}, top 100: {top(100):.2f}, top 200: {top(200):.2f} | recall in the top 10%: {yn[o[:len(o)//10]].sum() / max(yn.sum(), 1):.2f}", flush=True)
new = new.assign(rule=yn, head=sn, lr=sl)
new.to_parquet(DATA_DIR / "userhead_unseen_scores.parquet")
Xn.assign(diaObjectId=new.diaObjectId.values).to_parquet(DATA_DIR / "dp2_nightly_features_loose00_unseen.parquet")
json.dump({"n_liked": int(ya.sum()), "n_disliked": int((1 - ya).sum()), "cv_auc_head": float(roc_auc_score(ya, cv)), "n_unseen": int(len(new)),
           "unseen_rule_positive": int(yn.sum()), "unseen_auc_head": float(roc_auc_score(yn, sn)), "unseen_ap_head": float(average_precision_score(yn, sn)),
           "unseen_top100_precision_head": float(yn[np.argsort(-sn)[:100]].mean())}, open(M / "userhead_metrics.json", "w"), indent=1)
