"""Reproduces the numbers of the lightning talk from the cached private files (no TAP queries).
Upstream, once: train_sassafras6.py, loose_dp2.py, stageB_dp2.py (SET=loose03, SET=loose00), apply_dp2.py (SET=loose03, then SAMPLE=clean),
supcon_dp2.py, userhead_dp2.py, build_comb.py. Run: sbatch -J repro run.sbatch reproduce.py
SIM = SASSAFRAS test split (labels). REAL = DP2 deep fields (no labels). Aggregate numbers only are printed."""
import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tap import DATA_DIR as D
from features6 import features6, quality6, nightly, coherent, prediction_sets, CLASSES

warnings.filterwarnings("ignore"); torch.manual_seed(0)
M = Path(__file__).resolve().parent / "model"; PREP = Path(__file__).resolve().parent.parent / "sassafras" / "prepared"
NT = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
model = lgb.Booster(model_file=str(M / "sassafras6_lgbm.txt"))
FEATURES = json.load(open(M / "features.json"))["features"]
qhat = np.array([json.load(open(M / "conformal.json"))["qhat"][c] for c in CLASSES])
split = pd.read_csv(M / "sim_split.csv", dtype={"SNID": str})
Xall = pd.read_parquet(PREP / "sixband_six_h100.parquet").loc[split.SNID].reset_index(drop=True)[FEATURES]
tr, te = (split.split == "train").values, (split.split == "test").values
Xsim, ysim, P_sim = Xall[te].reset_index(drop=True), split.y.values[te], np.load(M / "sim_P_test.npy")

# ---- 1. selection funnel
lo = pd.read_parquet(D / "loose_objects.parquet"); c3 = pd.read_parquet(D / "stageB_loose03_candidates.parquet")
cand = pd.read_parquet(D / "stageB_comb_candidates.parquet"); cand = cand[cand.transient & cand.has_events].reset_index(drop=True)
saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 2)
with np.load(D / "stageB_comb_events.npz") as z:
    ev = {i: nightly(z[str(i)]) for i in cand.diaObjectId}
keep = np.array([quality6(ev[i]) and coherent(ev[i]) for i in cand.diaObjectId])
ok3 = np.array([quality6(ev[i]) for i in cand.diaObjectId]) & ~cand.below_reliability_cut.values
cand = cand[keep].reset_index(drop=True)
X = pd.DataFrame([features6(ev[i]) for i in cand.diaObjectId]).reindex(columns=FEATURES)
os.dup2(saved, 2)
n = len(cand); new = int(cand.below_reliability_cut.sum())
print(f"FUNNEL  reliability >= 0.3: {int(lo['loose_0.3'].sum()):,} -> quiet baseline, no star: {int((c3.transient & c3.has_events).sum())} -> six nightly epochs in two bands: {int(ok3.sum())}"
      f" -> coherent change: {n - new} -> plus recovered below the cut: {n}")
print(f"        stellar counterparts removed: {int((c3.position == 'stellar counterpart').sum())}")

# ---- 2. the gap
P = model.predict(X); pred = np.array(CLASSES)[P.argmax(1)]; s_sim, s_real = prediction_sets(P_sim, qhat).sum(1), prediction_sets(P, qhat).sum(1)
print(f"GAP     singletons SIM {np.mean(s_sim == 1):.3f} -> REAL {np.mean(s_real == 1):.3f} ({int((s_real == 1).sum())} of {n}) | mean set {s_sim.mean():.2f} -> {s_real.mean():.2f}"
      f" | p > 0.9: {np.mean(P_sim.max(1) > .9):.2f} -> {np.mean(P.max(1) > .9):.2f}")
def domain_auc(Xs, Xr):
    Z = pd.concat([Xs, Xr], ignore_index=True); y = np.r_[np.zeros(len(Xs)), np.ones(len(Xr))]; p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        p[b] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                  class_weight="balanced", verbosity=-1, n_jobs=NT, random_state=0).fit(Z.iloc[a], y[a]).predict_proba(Z.iloc[b])[:, 1]
    return roc_auc_score(y, p)
perm = np.random.default_rng(0).permutation(len(Xsim)); h = len(Xsim) // 2
print(f"        domain AUC SIM against REAL {domain_auc(Xsim, X):.3f} | control SIM against SIM {domain_auc(Xsim.iloc[perm[:h]], Xsim.iloc[perm[h:h + 250]]):.3f}")
sc = json.load(open(M / "supcon_scaler.json")); cols = sc["cols"]; med = pd.Series(sc["median"], index=cols); iqr = pd.Series(sc["iqr"], index=cols)
plain = lambda A: ((A[cols].fillna(med) - med) / iqr).clip(-6, 6).values
pca = PCA(20, random_state=0).fit(plain(Xall[tr])); Ztr, Zte, Zre = (pca.transform(plain(A)) for A in (Xall[tr], Xsim, X))
iso = IsolationForest(n_estimators=300, random_state=0, n_jobs=NT).fit(Ztr); nn_ = NearestNeighbors(n_neighbors=10).fit(Ztr)
sco = lambda Z: (-iso.score_samples(Z), nn_.kneighbors(Z)[0].mean(1)); nb = lambda v: np.digitize(v, [8, 12, 20, 35])
(a_te, k_te), (a_re, k_re) = sco(Zte), sco(Zre); b_te, b_re = nb(Xsim.n_obs_total.values), nb(X.n_obs_total.values)
cal = lambda ref, s: np.array([(ref[b_te == bb] < ss).mean() for ss, bb in zip(s, b_re)])
rank = (cal(a_te, a_re) + cal(k_te, k_re)) / 2
print(f"        anomaly rank against SIM at matched epoch count: median {np.median(rank):.2f} (0.50 without a gap), above 0.9: {np.mean(rank > .9):.2f}")

# ---- 3. host position, never given to the model
nuc = (cand.position == "nuclear").values
print("HOST    " + " | ".join(f"{c}: {int((pred == c).sum())} calls, {int(((pred == c) & nuc).sum())} nuclear" for c in ("TDE", "SN II", "SN Ia", "SN Ibc")))

# ---- 4. outliers and families within the real sample
Zr = Zre[:, :10]; Zr = (Zr - Zr.mean(0)) / Zr.std(0); t = pd.DataFrame({"n": X.n_obs_total.values})
t["a"] = -IsolationForest(n_estimators=300, random_state=0).fit(Zr).score_samples(Zr)
dist, idx = NearestNeighbors(n_neighbors=6).fit(Zr).kneighbors(Zr); t["k"] = dist[:, 1:].mean(1); t["bin"] = pd.qcut(t.n, 4, duplicates="drop").astype(str)
t["r"] = (t.groupby("bin").a.rank(pct=True) + t.groupby("bin").k.rank(pct=True)) / 2
oi = np.flatnonzero(t.r >= t.r.quantile(0.85)); parent = {i: i for i in oi}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in oi:
    for j in idx[i, 1:]:
        if j in parent: parent[find(i)] = find(j)
size = pd.Series([find(i) for i in oi]).value_counts()
print(f"OUTLIER {len(oi)} outliers: {int(size[size >= 3].sum())} in {int((size >= 3).sum())} families, {int((size == 1).sum())} singletons")

# ---- 5. learned metric (SupCon encoder trained on SIM) and Gaussian mixture
class Encoder(nn.Module):
    def __init__(self, d_in, d_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, d_out))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)
enc = Encoder(2 * len(cols)); enc.load_state_dict(torch.load(M / "supcon_encoder.pt", map_location="cpu")); enc.eval()
def embed(A):
    v = ((A[cols] - med) / iqr).clip(-6, 6)
    with torch.no_grad(): return enc(torch.tensor(np.hstack([v.fillna(0.0).values, A[cols].isna().values.astype(float)]).astype(np.float32))).numpy()
Es, Er = embed(Xsim), embed(X)
vote = np.array(CLASSES)[[np.bincount(r, minlength=8).argmax() for r in ysim[NearestNeighbors(n_neighbors=10).fit(Es).kneighbors(Er)[1]]]]
print(f"METRIC  vote of 10 simulated neighbours equals the classifier: {np.mean(vote == pred):.2f} | TDE votes: {int((vote == 'TDE').sum())}, nuclear {int(((vote == 'TDE') & nuc).sum())}")
G = PCA(10, random_state=0).fit(embed(Xall[tr])).transform(Er)
fits = [GaussianMixture(k, covariance_type="diag", n_init=3, reg_covar=1e-3, random_state=0).fit(G) for k in range(1, 9)]
g = min(fits, key=lambda m: m.bic(G)); lab = g.predict(G)
print(f"GMM     BIC picks {g.n_components} components, sizes {sorted(np.bincount(lab).tolist(), reverse=True)}, ARI with the classifier {adjusted_rand_score(pred, lab):.2f}")
print(pd.crosstab(lab, pred).to_string()); print(pd.crosstab(lab, cand.position.values).to_string())
