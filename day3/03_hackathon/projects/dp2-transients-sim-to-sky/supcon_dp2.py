"""Metric learning (supervised contrastive loss, as in notebook 03 of asasli/SimilaritySearch) on the SASSAFRAS
features, then projection of the real DP2 transients onto the learned embedding.

SIM : training on the train split, model selection on validation, kNN accuracy on the test split.
REAL: the 588 selected DP2 transients (nightly averaged), 69 of them with a coherent change. No labels.
"""
import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, lightgbm as lgb, umap
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tap import DATA_DIR
from features6 import features6, nightly, CLASSES

warnings.filterwarnings("ignore"); torch.manual_seed(0); np.random.seed(0)
HERE = Path(__file__).resolve().parent; M = HERE / "model"; PREP = HERE.parent / "sassafras" / "prepared"
OUTFIG = HERE.parent / "talks" / "school" / "run" / "dp2_lightning"
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)

FEATURES = json.load(open(M / "features.json"))["features"]
split = pd.read_csv(M / "sim_split.csv", dtype={"SNID": str})
Xall = pd.read_parquet(PREP / "sixband_six_h100.parquet").loc[split.SNID].reset_index(drop=True)[FEATURES]
y = split.y.values; tr, va, te = ((split.split == s).values for s in ("train", "val", "test"))
is_count = lambda f: f in ("n_obs_total", "n_bands", "span_total") or f.endswith("_n") or f.split("_", 1)[-1] in ("n_obs", "span")
cols = [f for f in FEATURES if not is_count(f)]
med = Xall[tr][cols].median(); iqr = (Xall[tr][cols].quantile(.75) - Xall[tr][cols].quantile(.25)).replace(0, 1.0)
def prep(X):                                   # robust scale, clip, median imputation, plus a missing-value mask
    v = ((X[cols] - med) / iqr).clip(-6, 6)
    return np.hstack([v.fillna(0.0).values, X[cols].isna().values.astype(float)]).astype(np.float32)
json.dump({"cols": cols, "median": med.tolist(), "iqr": iqr.tolist()}, open(M / "supcon_scaler.json", "w"))

class Encoder(nn.Module):
    def __init__(self, d_in, d_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def supervised_contrastive_loss(z, labels, temperature=0.1):
    B = z.shape[0]; sim = z @ z.T / temperature
    labels = labels.unsqueeze(1); eye = torch.eye(B, device=z.device)
    pos = (labels == labels.T).float() - eye
    log_denom = torch.log((torch.exp(sim) * (1 - eye)).sum(1, keepdim=True) + 1e-8)
    return (-((sim - log_denom) * pos).sum(1) / (pos.sum(1) + 1e-8)).mean()

Xtr, Xva, Xte = (torch.tensor(prep(Xall[m])) for m in (tr, va, te))
ytr = torch.tensor(y[tr]); nfeat = len(cols)
enc = Encoder(Xtr.shape[1]).to(dev)
opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS = 80; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
w = 1.0 / np.bincount(y[tr])[y[tr]]; w = torch.tensor(w / w.sum())
def embed(X):
    enc.eval()
    with torch.no_grad():
        return enc(X.to(dev)).cpu().numpy()
def knn_bacc(Za, ya, Zb, yb, k=10):
    return balanced_accuracy_score(yb, KNeighborsClassifier(k, weights="distance").fit(Za, ya).predict(Zb))
def augment(x):                                 # noise on the values, and random features set to missing
    v, m = x[:, :nfeat].clone(), x[:, nfeat:].clone()
    v = v + 0.1 * torch.randn_like(v); drop = torch.rand_like(v) < 0.1
    v[drop] = 0.0; m[drop] = 1.0
    return torch.cat([v, m], 1)
best, best_state = -1, None
for ep in range(EPOCHS):
    enc.train(); tot = 0
    for _ in range(len(Xtr) // 512):
        i = torch.multinomial(w, 512, replacement=True)
        xb, yb = Xtr[i].to(dev), ytr[i].to(dev)
        loss = supervised_contrastive_loss(enc(torch.cat([augment(xb), augment(xb)])), torch.cat([yb, yb]))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        b = knn_bacc(embed(Xtr), y[tr], embed(Xva), y[va])
        print(f"epoch {ep+1:3d} loss {tot / (len(Xtr) // 512):.3f} validation kNN balanced accuracy {b:.3f}", flush=True)
        if b > best:
            best, best_state = b, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
enc.load_state_dict(best_state); torch.save(best_state, M / "supcon_encoder.pt")
Ztr, Zte = embed(Xtr), embed(Xte)

# reference: the same kNN in the plain feature space (PCA 20)
plain = lambda X: ((X[cols].fillna(med) - med) / iqr).clip(-6, 6).values
pca = PCA(20, random_state=0).fit(plain(Xall[tr])); Ptr, Pte = pca.transform(plain(Xall[tr])), pca.transform(plain(Xall[te]))
metrics = {"sim_test_knn_bacc_supcon": knn_bacc(Ztr, y[tr], Zte, y[te]), "sim_test_knn_bacc_plain_features": knn_bacc(Ptr, y[tr], Pte, y[te]),
           "sim_test_bacc_lightgbm": json.load(open(M / "metrics_sim.json"))["test_balanced_accuracy"], "val_knn_bacc_supcon": best}
print("SIM test, balanced accuracy:", {k: round(v, 3) for k, v in metrics.items()}, flush=True)

# ---------------------------------------------------------------- real DP2 transients on the learned embedding
feat = pd.read_parquet(DATA_DIR / "dp2_transients_table_loose03.parquet")
clean_ids = set(pd.read_parquet(DATA_DIR / "dp2_transients_table_loose03_clean.parquet").diaObjectId)
is_clean = feat.diaObjectId.isin(clean_ids).values
saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 2)
with np.load(DATA_DIR / "stageB_loose03_events.npz") as z:
    Xb = pd.DataFrame([features6(nightly(z[str(i)])) for i in feat.diaObjectId]).reindex(columns=FEATURES)
os.dup2(saved, 2)
Xb.assign(diaObjectId=feat.diaObjectId.values).to_parquet(DATA_DIR / "dp2_nightly_features_loose03.parquet")
Zre = embed(torch.tensor(prep(Xb))); Pre = pca.transform(plain(Xb))
ysim = y[te]
def report(S, R, name):
    P = np.vstack([S, R]); lab = np.r_[np.zeros(len(S)), np.ones(len(R))]
    mx = lab[NearestNeighbors(n_neighbors=11).fit(P).kneighbors(R)[1][:, 1:]].mean(1)
    nb = ysim[NearestNeighbors(n_neighbors=10).fit(S).kneighbors(R)[1]]
    vote = np.array(CLASSES)[[np.bincount(r, minlength=8).argmax() for r in nb]]
    purity = np.array([np.bincount(r, minlength=8).max() / 10 for r in nb])
    agree = vote == feat.pred.values
    print(f"{name:14s} real neighbours of a real object: {mx.mean()*10:.1f} of 10 (coherent {mx[is_clean].mean()*10:.1f}); chance {len(R)/(len(S)+len(R))*10:.1f} | "
          f"neighbour vote equals LightGBM top class: all {agree.mean():.2f}, coherent {agree[is_clean].mean():.2f} | mean vote purity {purity.mean():.2f}", flush=True)
    print(f"{'':14s} vote fractions, all: {pd.Series(vote).value_counts(normalize=True).round(2).to_dict()}")
    print(f"{'':14s} vote fractions, coherent: {pd.Series(vote[is_clean]).value_counts(normalize=True).round(2).to_dict()}", flush=True)
    return vote, purity
report(Pte, Pre, "plain features")
vote, purity = report(Zte, Zre, "SupCon")
pos = feat.position.values
ct = pd.crosstab(vote[is_clean], pos[is_clean]); print("\nSupCon neighbour vote against position, 69 coherent transients:"); print(ct.to_string())
# purity of the vote on SIM test itself, for reference
nbs = y[tr][NearestNeighbors(n_neighbors=10).fit(Ztr).kneighbors(Zte)[1]]
print(f"mean vote purity: SIM test {np.mean([np.bincount(r, minlength=8).max() / 10 for r in nbs]):.2f} | REAL all {purity.mean():.2f} | REAL coherent {purity[is_clean].mean():.2f}")
metrics.update(real_vote_purity=float(purity.mean()), real_vote_agree=float((vote == feat.pred.values).mean()))
json.dump(metrics, open(M / "supcon_metrics.json", "w"), indent=1)
np.savez(DATA_DIR / "supcon_loose03.npz", Zre=Zre, vote=vote, purity=purity, diaObjectId=feat.diaObjectId.values, is_clean=is_clean)

# ---------------------------------------------------------------- three maps
um = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0, metric="cosine").fit(Zte)
old = np.load(DATA_DIR / "embeddings_loose03.npz")
coords = {"Light-curve features": (old["features_sim"], old["features_real"]), "Classifier scores": (old["classifier_sim"], old["classifier_real"]),
          "Contrastive embedding (SupCon)": (um.embedding_, um.transform(Zre))}
BG, FG, MUTED = "#141A2E", "#E8E9F2", "#B5BAD0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "font.size": 12, "axes.titlesize": 13})
CC = plt.cm.magma(np.linspace(0.18, 0.95, 8))
fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.0))
for a, (name, (S, R)) in zip(ax, coords.items()):
    a.scatter(S[:, 0], S[:, 1], s=3, c=CC[ysim], alpha=0.55, lw=0, rasterized=True)
    a.scatter(R[~is_clean, 0], R[~is_clean, 1], s=10, facecolors="none", edgecolors="white", lw=0.5, alpha=0.75)
    a.scatter(R[is_clean, 0], R[is_clean, 1], s=36, marker="*", color="white", edgecolors="k", lw=0.3, zorder=5)
    a.set_title(name); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
h = [Line2D([], [], marker="o", ls="", color=CC[i], ms=7, label=c) for i, c in enumerate(CLASSES)]
h += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="white", ms=7, label=f"DP2, all selected ({len(feat)})"),
      Line2D([], [], marker="*", ls="", color="white", ms=11, label=f"DP2, coherent change ({int(is_clean.sum())})")]
fig.legend(handles=h, loc="lower center", ncol=10, frameon=False, fontsize=10, handletextpad=0.2, columnspacing=1.0)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(OUTFIG / "embedding3.png", dpi=170); plt.close()
print("figure:", OUTFIG / "embedding3.png")
