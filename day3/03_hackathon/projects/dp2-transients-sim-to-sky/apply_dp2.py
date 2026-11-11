"""Steps 4 and 5: apply the SASSAFRAS six-band model to the DP2 transients and measure the gap.

REAL: the DP2 transients selected by select_dp2.py and stageB_dp2.py (deep fields only, no labels).
SIM : the SASSAFRAS test split, which the model never saw in training.
Row-level outputs and figures go to the private DATA_DIR; the log holds aggregate numbers only.
"""
import json, os, contextlib, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import ks_2samp, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tap import DATA_DIR
from features6 import features6, quality6, prediction_sets, CLASSES, BANDS, LOG_CONST

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; M = HERE / "model"; PREP = HERE.parent / "sassafras" / "prepared"
FIG = DATA_DIR / "figs"; FIG.mkdir(exist_ok=True)
NT = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C_SIM, C_REAL, C_BIN = plt.cm.magma(0.25), plt.cm.magma(0.62), plt.cm.magma(0.82)

model = lgb.Booster(model_file=str(M / "sassafras6_lgbm.txt"))
FEATURES = json.load(open(M / "features.json"))["features"]
qhat = np.array([json.load(open(M / "conformal.json"))["qhat"][c] for c in CLASSES])
split = pd.read_csv(M / "sim_split.csv", dtype={"SNID": str})
Xsim_all = pd.read_parquet(PREP / "sixband_six_h100.parquet").loc[split.SNID].reset_index(drop=True)[FEATURES]
te, tr = (split.split == "test").values, (split.split == "train").values
Xsim, ysim = Xsim_all[te].reset_index(drop=True), split.y.values[te]
P_sim = np.load(M / "sim_P_test.npy")

SET = os.environ.get("SET", "strict"); T = "" if SET == "strict" else "_" + SET
cand = pd.read_parquet(DATA_DIR / f"stageB{T}_candidates.parquet")
cand = cand[cand.transient & cand.has_events].reset_index(drop=True)
with np.load(DATA_DIR / f"stageB{T}_events.npz") as z:
    events = {int(k): z[k] for k in z.files}
Xreal = pd.read_parquet(DATA_DIR / f"stageB{T}_features6.parquet").loc[cand.diaObjectId].reset_index(drop=True)[FEATURES]

# ---------------------------------------------------------------- nightly binning (cadence harmonisation)
def nightly(ev, gap=0.5):
    out = []
    for b in np.unique(ev[:, 1]):
        e = ev[ev[:, 1] == b]; e = e[np.argsort(e[:, 0])]
        grp = np.cumsum(np.r_[0, np.diff(e[:, 0]) > gap])
        for g in np.unique(grp):
            x = e[grp == g]; fl = 10.0 ** x[:, 2].astype(float); er = x[:, 3] * fl / LOG_CONST; w = 1 / er ** 2
            f = (fl * w).sum() / w.sum(); s = 1 / np.sqrt(w.sum())
            out.append([(x[:, 0] * w).sum() / w.sum(), b, np.log10(f), s / f * LOG_CONST])
    out = np.array(out); out = out[np.argsort(out[:, 0], kind="stable")]; out[:, 0] -= out[0, 0]
    return out.astype(np.float32)

@contextlib.contextmanager
def quiet_c_stderr():
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2); yield
    finally:
        os.dup2(saved, 2); os.close(devnull); os.close(saved)

def per_night_multiplicity(ev):
    n = len(ev); return n / max(len(nightly(ev)), 1)

with np.load(PREP / "sixband_events.npz") as z:
    sim_mult = np.array([per_night_multiplicity(z[s]) for s in split.SNID[te].values[:1500]])
real_mult = np.array([per_night_multiplicity(events[i]) for i in cand.diaObjectId])
print(f"epochs per band-night: SIM median {np.median(sim_mult):.2f} (90% {np.quantile(sim_mult,.9):.2f}) | REAL median {np.median(real_mult):.2f} (90% {np.quantile(real_mult,.9):.2f})")

binned = {i: nightly(events[i]) for i in cand.diaObjectId}

# ---------------------------------------------------------------- cleaned sample (SAMPLE=clean)
# A transient must brighten or fade coherently: more than 0.5 mag at 5 sigma in one band (two brightest against two
# faintest nightly points), supported by more than 0.3 mag at 3 sigma in a second band with the same direction in time.
def band_changes(ev):
    out = []
    for b in np.unique(ev[:, 1]):
        e = ev[ev[:, 1] == b]
        if len(e) < 4: continue
        fl = 10.0 ** e[:, 2].astype(float); er = e[:, 3] * fl / LOG_CONST; o = np.argsort(fl); lo, hi = o[:2], o[-2:]
        wm = lambda k: ((fl[k] / er[k] ** 2).sum() / (1 / er[k] ** 2).sum(), 1 / np.sqrt((1 / er[k] ** 2).sum()))
        (fh, eh), (fl_, el) = wm(hi), wm(lo)
        out.append((fh / fl_, (fh - fl_) / np.hypot(eh, el), np.sign(e[hi, 0].mean() - e[lo, 0].mean())))
    return out
def coherent(ev):
    ch = band_changes(ev)
    for a, (ra, sa, da) in enumerate(ch):
        if ra > 1.585 and sa > 5 and any(rb > 1.3 and sb > 3 and db == da for b, (rb, sb, db) in enumerate(ch) if b != a):
            return True
    return False
SAMPLE = os.environ.get("SAMPLE", "all"); TAG = T + ("" if SAMPLE == "all" else "_" + SAMPLE)
is_clean = np.array([coherent(binned[i]) for i in cand.diaObjectId])
print(f"coherent change in two bands: {is_clean.sum()} of {len(cand)} | by position: {cand.position[is_clean].value_counts().to_dict()} | "
      f"with a quiet baseline before the first detection: {int((is_clean & (cand.n_pre >= 3)).sum())}")
print("rejected as flat or incoherent, by position:", cand.position[~is_clean].value_counts().to_dict())
if SAMPLE == "clean":
    Xreal = Xreal[is_clean].reset_index(drop=True); cand = cand[is_clean].reset_index(drop=True)
FIG = DATA_DIR / f"figs{TAG}"; FIG.mkdir(exist_ok=True)
print("SET =", SET, "| SAMPLE =", SAMPLE, "| objects:", len(cand), flush=True)
okb = np.array([quality6(binned[i]) for i in cand.diaObjectId])
with quiet_c_stderr():
    Xbin = pd.DataFrame([features6(binned[i]) if ok else {} for i, ok in zip(cand.diaObjectId, okb)]).reindex(columns=FEATURES)
print(f"REAL transients: {len(cand)}; after nightly binning {okb.sum()} still have 6 epochs in 2 bands")

# ---------------------------------------------------------------- step 4: classes and conformal sets
def summarise(P, name):
    sets = prediction_sets(P, qhat); size = sets.sum(1)
    return {"sample": name, "n": len(P), "median max prob": np.median(P.max(1)), "max prob > 0.9": (P.max(1) > 0.9).mean(),
            "mean set size": size.mean(), "singletons": (size == 1).mean(), "empty sets": (size == 0).mean(), "size >= 4": (size >= 4).mean()}

P_real = model.predict(Xreal); P_bin = model.predict(Xbin[okb])
sn = ysim <= 2                                             # control for the class mix: simulated supernovae only
tab = pd.DataFrame([summarise(P_sim, "SIM test"), summarise(P_sim[sn], "SIM test, SN Ia/Ibc/II only"), summarise(P_real, "REAL per visit"), summarise(P_bin, "REAL nightly")]).set_index("sample")
print("\nConfidence and 90% conformal sets (calibrated on SIM validation):"); print(tab.round(3).to_string())
# matched number of epochs: reweight SIM test to the REAL n_obs distribution
nb = np.array([0, 8, 12, 20, 35, 60, 1e9])
def matched(P, n_sim, n_real):
    hs, hr = np.histogram(n_sim, nb)[0], np.histogram(n_real, nb)[0]
    w = (hr / hr.sum() / np.clip(hs / hs.sum(), 1e-9, None))[np.digitize(n_sim, nb) - 1]
    size = prediction_sets(P, qhat).sum(1)
    return {"mean set size": np.average(size, weights=w), "singletons": np.average(size == 1, weights=w), "empty sets": np.average(size == 0, weights=w),
            "median max prob (unweighted bins)": np.nan}
print("SIM test reweighted to the REAL epoch counts: per visit", {k: round(v, 3) for k, v in matched(P_sim, Xsim.n_obs_total.values, Xreal.n_obs_total.values).items() if np.isfinite(v)},
      "| nightly", {k: round(v, 3) for k, v in matched(P_sim, Xsim.n_obs_total.values, Xbin[okb].n_obs_total.values).items() if np.isfinite(v)})

pred_real, pred_bin = P_real.argmax(1), P_bin.argmax(1)

# purity check: objects whose brightness changes by more than 0.5 mag at 5 sigma in at least one band
def evolving(ev):
    for b in np.unique(ev[:, 1]):
        e = ev[ev[:, 1] == b]
        if len(e) < 2: continue
        fl = 10.0 ** e[:, 2].astype(float); er = e[:, 3] * fl / LOG_CONST; i, j = fl.argmax(), fl.argmin()
        if fl[i] / fl[j] > 1.585 and (fl[i] - fl[j]) / np.hypot(er[i], er[j]) > 5: return True
    return False
evol = np.array([evolving(binned[i]) for i in cand.diaObjectId[okb]])
print(f"\nREAL nightly objects that change by more than 0.5 mag at 5 sigma in one band: {evol.sum()} of {okb.sum()}")
print(pd.DataFrame([summarise(P_bin[evol], "REAL nightly, evolving"), summarise(P_bin[~evol], "REAL nightly, flat")]).set_index("sample").round(3).to_string())
print("predicted class, evolving:", pd.Series(np.array(CLASSES)[pred_bin[evol]]).value_counts().to_dict(), "| flat:", pd.Series(np.array(CLASSES)[pred_bin[~evol]]).value_counts().to_dict())
print("position, evolving:", cand.position[okb][evol].value_counts().to_dict(), "| flat:", cand.position[okb][~evol].value_counts().to_dict())
cls = pd.DataFrame({"SIM test (true mix is balanced by model)": pd.Series(P_sim.argmax(1)).value_counts(normalize=True),
                    "REAL per visit": pd.Series(pred_real).value_counts(normalize=True),
                    "REAL nightly": pd.Series(pred_bin).value_counts(normalize=True)}).reindex(range(len(CLASSES))).fillna(0)
cls.index = CLASSES
print("\nPredicted class fractions:"); print(cls.round(3).to_string())
print(f"per visit against nightly: same top class for {(pred_real[okb] == pred_bin).mean():.3f} of {okb.sum()} objects")
ct = pd.crosstab(np.array(CLASSES)[pred_bin], cand.position[okb].values)
print("\nPredicted class (nightly) by position in dp2.Object:"); print(ct.to_string())
conf_bin = P_bin.max(1) > 0.9
print("confident (> 0.9) nightly predictions by class:", pd.Series(np.array(CLASSES)[pred_bin][conf_bin]).value_counts().to_dict())

# ---------------------------------------------------------------- the gap in feature space
def domain_auc(Xs, Xr, cols):
    X = pd.concat([Xs[cols], Xr[cols]], ignore_index=True); y = np.r_[np.zeros(len(Xs)), np.ones(len(Xr))]
    p = np.zeros(len(y)); gain = np.zeros(len(cols))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, class_weight="balanced", verbosity=-1, n_jobs=NT, random_state=0).fit(X.iloc[a], y[a])
        p[b] = m.predict_proba(X.iloc[b])[:, 1]; gain += m.booster_.feature_importance("gain")
    return roc_auc_score(y, p), pd.Series(gain / gain.sum(), index=cols).sort_values(ascending=False)

is_count = lambda f: f in ("n_obs_total", "n_bands", "span_total") or f.endswith("_n") or "_n_" in f or f.split("_", 1)[-1] in (
    "n", "n_obs", "span", "dt_median", "dt_max", "dt_min", "dt_mean", "cadence", "duration", "n_det")
shape_cols = [f for f in FEATURES if not is_count(f)]
flux_free = [f for f in shape_cols if not any(k in f for k in ("logflux_peak", "logflux_mean", "logflux_median", "logflux_min", "logflux_max", "flux_peak", "amplitude"))
             or "ratio" in f]
print(f"\nDomain classifier SIM test against REAL (5-fold AUC; 0.5 means indistinguishable):")
gap = {}
for nm, Xr in (("per visit", Xreal), ("nightly", Xbin[okb].reset_index(drop=True))):
    for cn, cols in (("all 329 features", FEATURES), (f"without counts and cadence ({len(shape_cols)})", shape_cols)):
        auc, g = domain_auc(Xsim, Xr, cols); gap[(nm, cn)] = auc
        print(f"   REAL {nm:9s} {cn:38s} AUC {auc:.3f}   top features: {', '.join(f'{k} ({v:.2f})' for k, v in g.head(6).items())}", flush=True)
# control: SIM against SIM
h = len(Xsim) // 2; perm = np.random.default_rng(0).permutation(len(Xsim))
auc0, _ = domain_auc(Xsim.iloc[perm[:h]].reset_index(drop=True), Xsim.iloc[perm[h:h + 250]].reset_index(drop=True), FEATURES)
print(f"   control, SIM test against 250 other SIM test objects: AUC {auc0:.3f}")

Xb = Xbin[okb].reset_index(drop=True)
ks = []
for f in FEATURES:
    a, b = Xsim[f].dropna().values, Xb[f].dropna().values
    if len(a) > 50 and len(b) > 30:
        ks.append((f, ks_2samp(a, b).statistic, np.median(a), np.median(b), Xsim[f].isna().mean(), Xb[f].isna().mean()))
ks = pd.DataFrame(ks, columns=["feature", "KS", "SIM median", "REAL median", "SIM missing", "REAL missing"]).sort_values("KS", ascending=False)
print("\nLargest one-dimensional shifts (nightly REAL against SIM test):"); print(ks.head(18).round(3).to_string(index=False))
print("KS quantiles over features:", ks.KS.quantile([.25, .5, .75, .9]).round(2).to_dict())
band_cov = pd.DataFrame({"SIM": [(Xsim[f"{b}_logflux_peak"].notna()).mean() for b in BANDS], "REAL": [(Xb[f"{b}_logflux_peak"].notna()).mean() for b in BANDS]}, index=BANDS)
print("fraction of objects with data in each band:"); print(band_cov.round(2).T.to_string())
peakmag = lambda X: 23.9 - 2.5 * X["r_logflux_peak"].fillna(X["i_logflux_peak"]).fillna(X["g_logflux_peak"])
print("peak magnitude (r, else i, else g) quantiles: SIM", peakmag(Xsim).quantile([.1, .5, .9]).round(2).to_dict(), " REAL", peakmag(Xb).quantile([.1, .5, .9]).round(2).to_dict())
print("epochs per object: SIM", Xsim.n_obs_total.quantile([.1, .5, .9]).to_dict(), " REAL per visit", Xreal.n_obs_total.quantile([.1, .5, .9]).to_dict(),
      " REAL nightly", Xb.n_obs_total.quantile([.1, .5, .9]).to_dict())
print("span (d): SIM", Xsim.span_total.quantile([.1, .5, .9]).round(0).to_dict(), " REAL", Xb.span_total.quantile([.1, .5, .9]).round(0).to_dict())

# ---------------------------------------------------------------- step 5: anomaly rank and similarity
# (a) against the simulation: is a real transient unusual for SASSAFRAS, among simulated objects with a similar number of epochs?
med = Xsim_all[tr].median(); lo, hi = Xsim_all[tr].quantile(0.25), Xsim_all[tr].quantile(0.75)
scale = (hi - lo).replace(0, 1.0)
def Z(X, cols=shape_cols):
    return ((X[cols].fillna(med[cols]) - med[cols]) / scale[cols]).clip(-6, 6).values
pca = PCA(20, random_state=0).fit(Z(Xsim_all[tr]))
Ztr, Zte, Zre = (pca.transform(Z(x)) for x in (Xsim_all[tr], Xsim, Xb))
iso = IsolationForest(n_estimators=300, random_state=0, n_jobs=NT).fit(Ztr)
nn_sim = NearestNeighbors(n_neighbors=10).fit(Ztr)
def scores(Zx):
    return -iso.score_samples(Zx), nn_sim.kneighbors(Zx)[0].mean(1)
(a_te, k_te), (a_re, k_re) = scores(Zte), scores(Zre)
nbin = lambda n: np.digitize(n, [8, 12, 20, 35])
b_te, b_re = nbin(Xsim.n_obs_total.values), nbin(Xb.n_obs_total.values)
def calibrated(s_ref, b_ref, s, b):                      # fraction of reference objects in the same n_obs bin with a lower score
    return np.array([(s_ref[b_ref == bb] < ss).mean() if (b_ref == bb).sum() >= 20 else np.nan for ss, bb in zip(s, b)])
r_if, r_nn = calibrated(a_te, b_te, a_re, b_re), calibrated(k_te, b_te, k_re, b_re)
rank_sim = (r_if + r_nn) / 2
print("\nAnomaly rank of REAL transients against SIM objects with a similar number of epochs (uniform between 0 and 1 if there were no gap):")
print("   isolation forest: median", np.nanmedian(r_if).round(3), " fraction above 0.9:", np.nanmean(r_if > 0.9).round(3), " above 0.99:", np.nanmean(r_if > 0.99).round(3))
print("   neighbour distance: median", np.nanmedian(r_nn).round(3), " fraction above 0.9:", np.nanmean(r_nn > 0.9).round(3), " above 0.99:", np.nanmean(r_nn > 0.99).round(3))
print("   n_obs bins of REAL:", np.bincount(b_re, minlength=5).tolist(), " SIM reference:", np.bincount(b_te, minlength=5).tolist())
# does the gap follow the model's confidence?
print("   Spearman, rank against SIM with the model's max probability:", round(spearmanr(rank_sim, P_bin.max(1), nan_policy="omit")[0], 2),
      "| with conformal set size:", round(spearmanr(rank_sim, prediction_sets(P_bin, qhat).sum(1), nan_policy="omit")[0], 2))

# (b) within the real sample: outliers, families and singletons, as in the starter notebook
feat = cand[okb].reset_index(drop=True)[["diaObjectId", "field", "position", "ra", "dec", "t_first", "duration", "host_sep", "host_r"]].copy()
feat["n_obs"] = Xb.n_obs_total.values; feat["pred"] = np.array(CLASSES)[pred_bin]; feat["max_prob"] = P_bin.max(1)
feat["evolving"] = evol; feat["set_size"] = prediction_sets(P_bin, qhat).sum(1); feat["rank_vs_sim"] = rank_sim
Zr = Zre[:, :10]; Zr = (Zr - Zr.mean(0)) / Zr.std(0)
feat["anomaly"] = -IsolationForest(n_estimators=300, random_state=0).fit(Zr).score_samples(Zr)
K = 5
nn = NearestNeighbors(n_neighbors=K + 1).fit(Zr); dist, idx = nn.kneighbors(Zr)
feat["knn_dist"] = dist[:, 1:].mean(1)
feat["n_bin"] = pd.qcut(feat.n_obs, 4, duplicates="drop")
feat["anomaly_rank"] = feat.groupby("n_bin", observed=True).anomaly.rank(pct=True)
feat["knn_rank"] = feat.groupby("n_bin", observed=True).knn_dist.rank(pct=True)
feat["outlier_rank"] = (feat.anomaly_rank + feat.knn_rank) / 2
print(f"\nWithin the REAL sample: isolation forest against neighbour distance, Spearman {spearmanr(feat.anomaly_rank, feat.knn_rank)[0]:.2f};"
      f" raw anomaly score against n_obs {spearmanr(feat.anomaly, feat.n_obs)[0]:.2f}, calibrated rank against n_obs {spearmanr(feat.outlier_rank, feat.n_obs)[0]:.2f}")
OUTLIER_FRACTION = 0.15
is_out = (feat.outlier_rank >= feat.outlier_rank.quantile(1 - OUTLIER_FRACTION)).values
oi = np.flatnonzero(is_out); parent = {i: i for i in oi}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i in oi:
    for j in idx[i, 1:]:
        if j in parent:
            parent[find(i)] = find(j)
root = np.array([find(i) for i in oi]); size = pd.Series(root).map(pd.Series(root).value_counts()).values
feat["family"] = -1; feat["family_size"] = 0
feat.loc[oi, "family"] = root; feat.loc[oi, "family_size"] = size
out = feat[is_out]
print(f"{is_out.sum()} outliers (top {OUTLIER_FRACTION:.0%}): {int((out.family_size >= 3).sum())} in {out[out.family_size >= 3].family.nunique()} families of three or more, "
      f"{int((out.family_size == 2).sum())} in pairs, {int((out.family_size == 1).sum())} singletons")
print("outliers by predicted class:", out.pred.value_counts().to_dict(), "| by position:", out.position.value_counts().to_dict())
print("outliers: mean conformal set size", round(out.set_size.mean(), 2), "against", round(feat[~is_out].set_size.mean(), 2), "for the rest;",
      "median rank against SIM", round(out.rank_vs_sim.median(), 3), "against", round(feat[~is_out].rank_vs_sim.median(), 3))

def more_like_this(oid, k=5):
    """The k most similar real transients to oid in feature space, with distance, outlier rank and the model's reading."""
    i = feat.index[feat.diaObjectId == oid][0]
    return feat.iloc[idx[i, 1:k + 1]][["diaObjectId", "pred", "max_prob", "set_size", "outlier_rank", "family_size", "position"]].assign(distance=dist[i, 1:k + 1])

feat.assign(n_bin=feat.n_bin.astype(str)).to_parquet(DATA_DIR / f"dp2_transients_table{TAG}.parquet")
np.save(DATA_DIR / f"dp2_P_nightly{TAG}.npy", P_bin); np.save(DATA_DIR / f"dp2_neighbours{TAG}.npy", idx)

# ---------------------------------------------------------------- figures (private folder)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
bins = np.linspace(0.12, 1, 23)
ax[0].hist(P_sim.max(1), bins, density=True, histtype="stepfilled", alpha=.55, color=C_SIM, label="simulation, test split")
ax[0].hist(P_bin.max(1), bins, density=True, histtype="step", lw=2, color=C_REAL, label=f"DP2 transients (n = {len(P_bin)})")
ax[0].set(xlabel="highest class probability", ylabel="density", title="Confidence"); ax[0].legend(fontsize=7, frameon=False)
s_sim, s_real = prediction_sets(P_sim, qhat).sum(1), prediction_sets(P_bin, qhat).sum(1)
x = np.arange(0, 9)
ax[1].bar(x - .2, np.bincount(s_sim, minlength=9)[:9] / len(s_sim), .4, color=C_SIM, label="simulation")
ax[1].bar(x + .2, np.bincount(s_real, minlength=9)[:9] / len(s_real), .4, color=C_REAL, label="DP2")
ax[1].set(xlabel="classes in the 90% conformal set", ylabel="fraction", title="Prediction set size"); ax[1].legend(fontsize=7, frameon=False)
ax[2].bar(np.arange(8) - .2, cls.iloc[:, 0], .4, color=C_SIM, label="simulation"); ax[2].bar(np.arange(8) + .2, cls["REAL nightly"], .4, color=C_REAL, label="DP2")
ax[2].set_xticks(range(8)); ax[2].set_xticklabels(CLASSES, rotation=40, ha="right"); ax[2].set(ylabel="fraction", title="Predicted class"); ax[2].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(FIG / "gap_confidence_sets_classes.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[0].hist(rank_sim[np.isfinite(rank_sim)], np.linspace(0, 1, 21), color=C_REAL); ax[0].axhline(np.isfinite(rank_sim).sum() / 20, color="k", ls=":", lw=1)
ax[0].set(xlabel="anomaly rank among simulated objects\nwith a similar number of epochs", ylabel="DP2 transients", title="Dotted line: no gap")
labels = ["all features", "without counts\nand cadence"]
for k, (nm, c) in enumerate((("per visit", C_SIM), ("nightly", C_REAL))):
    vals = [v for (a, b), v in gap.items() if a == nm]
    ax[1].bar(np.arange(2) + (k - .5) * .38, vals, .38, color=c, label=f"DP2 {nm}")
ax[1].axhline(0.5, color="k", ls=":", lw=1); ax[1].axhline(auc0, color="0.5", ls="--", lw=1)
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(labels); ax[1].set(ylim=(0.4, 1.02), ylabel="AUC, simulation against DP2", title="Domain classifier"); ax[1].legend(fontsize=7, frameon=False)
pm_s, pm_r = peakmag(Xsim), peakmag(Xb)
ax[2].hist(pm_s.dropna(), np.linspace(17, 26, 37), density=True, histtype="stepfilled", alpha=.55, color=C_SIM, label="simulation")
ax[2].hist(pm_r.dropna(), np.linspace(17, 26, 37), density=True, histtype="step", lw=2, color=C_REAL, label="DP2")
ax[2].set(xlabel="peak magnitude", ylabel="density", title="Brightness"); ax[2].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(FIG / "gap_feature_space.png"); plt.close()

# light-curve galleries (row-level, private)
BC = dict(zip(range(6), plt.cm.magma(np.linspace(0.1, 0.88, 6))))
def plot_lc(ax, oid, title):
    ev = binned[oid]
    for b in np.unique(ev[:, 1]).astype(int):
        m = ev[:, 1] == b; mag = 23.9 - 2.5 * ev[m, 2]
        ax.errorbar(ev[m, 0], mag, 2.5 * ev[m, 3], fmt="o", ms=3, lw=.7, color=BC[b], label=BANDS[b])
    ax.invert_yaxis(); ax.set_title(title, fontsize=7)
def gallery(rows, titles, path, suptitle):
    n = len(rows); nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(13, 2.7 * nr), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, oid, t in zip(axes.ravel(), rows, titles):
        plot_lc(ax, oid, t)
    axes[0, 0].legend(fontsize=6, ncol=6, frameon=False); fig.suptitle(suptitle, y=1.0); plt.tight_layout(); plt.savefig(path); plt.close()
t = lambda r: f"{r.pred} p={r.max_prob:.2f} set={r.set_size} | {r.position}\nrank {r.outlier_rank:.2f}, vs sim {r.rank_vs_sim:.2f}, family {r.family_size}"
top = feat.sort_values("max_prob", ascending=False).head(12); gallery(top.diaObjectId, [t(r) for r in top.itertuples()], FIG / "gallery_most_confident.png", "DP2 transients the simulation-trained model is most sure about")
top = feat.sort_values("outlier_rank", ascending=False).head(12); gallery(top.diaObjectId, [t(r) for r in top.itertuples()], FIG / "gallery_outliers.png", "DP2 transients with the highest calibrated outlier rank")
top = feat[feat.evolving].sort_values("max_prob", ascending=False).head(12); gallery(top.diaObjectId, [t(r) for r in top.itertuples()], FIG / "gallery_evolving_most_confident.png", "Evolving DP2 transients, most confident predictions")
top = feat.sort_values("rank_vs_sim", ascending=False).head(12); gallery(top.diaObjectId, [t(r) for r in top.itertuples()], FIG / "gallery_unlike_simulation.png", "DP2 transients least like the simulation")
fam = out[out.family_size >= 3]
if len(fam):
    f0 = fam.groupby("family").size().idxmax(); q = fam[fam.family == f0].outlier_rank.idxmax(); sim_ = more_like_this(feat.diaObjectId[q], 7)
    rows = [feat.diaObjectId[q]] + sim_.diaObjectId.tolist()
    gallery(rows, ["QUERY: " + t(feat.loc[q])] + [f"distance {d:.2f} | " + t(feat.loc[i]) for i, d in zip(sim_.index, sim_.distance)],
            FIG / "gallery_family.png", "more_like_this: the most unusual member of the largest outlier family and its neighbours")
print("\nfigures written to the private folder", FIG)
