"""Embeddings: where do the real DP2 transients fall on maps built from the simulation?
Two spaces: (a) light-curve features (robust-scaled, PCA 20), (b) the classifier's eight raw class scores.
UMAP is fitted on the SASSAFRAS test split only; real objects are projected onto it.
Coordinates are saved in the private DATA_DIR; the figure has one dot per object and no identifiers."""
import json, os, contextlib, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tap import DATA_DIR
from features6 import features6, CLASSES, LOG_CONST

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; M = HERE / "model"; PREP = HERE.parent / "sassafras" / "prepared"
OUTFIG = HERE.parent / "talks" / "school" / "run" / "dp2_lightning"; OUTFIG.mkdir(parents=True, exist_ok=True)
SET = os.environ.get("SET", "loose03")

model = lgb.Booster(model_file=str(M / "sassafras6_lgbm.txt"))
FEATURES = json.load(open(M / "features.json"))["features"]
split = pd.read_csv(M / "sim_split.csv", dtype={"SNID": str})
Xall = pd.read_parquet(PREP / "sixband_six_h100.parquet").loc[split.SNID].reset_index(drop=True)[FEATURES]
te, tr = (split.split == "test").values, (split.split == "train").values
Xsim, ysim = Xall[te].reset_index(drop=True), split.y.values[te]

feat = pd.read_parquet(DATA_DIR / f"dp2_transients_table_{SET}.parquet")
clean_ids = set(pd.read_parquet(DATA_DIR / f"dp2_transients_table_{SET}_clean.parquet").diaObjectId)
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
saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 2)
with np.load(DATA_DIR / f"stageB_{SET}_events.npz") as z:
    Xb = pd.DataFrame([features6(nightly(z[str(i)])) for i in feat.diaObjectId]).reindex(columns=FEATURES)
os.dup2(saved, 2)
is_clean = feat.diaObjectId.isin(clean_ids).values
print(f"SIM test {len(Xsim):,} | REAL {len(Xb)} ({is_clean.sum()} coherent)", flush=True)

is_count = lambda f: f in ("n_obs_total", "n_bands", "span_total") or f.endswith("_n") or f.split("_", 1)[-1] in ("n_obs", "span")
cols = [f for f in FEATURES if not is_count(f)]
med = Xall[tr][cols].median(); iqr = (Xall[tr][cols].quantile(.75) - Xall[tr][cols].quantile(.25)).replace(0, 1.0)
Zf = lambda X: ((X[cols].fillna(med) - med) / iqr).clip(-6, 6).values
pca = PCA(20, random_state=0).fit(Zf(Xall[tr]))
spaces = {"features": (pca.transform(Zf(Xsim)), pca.transform(Zf(Xb))),
          "classifier": (model.predict(Xsim, raw_score=True), model.predict(Xb, raw_score=True))}

def mixing(S, R, k=10):
    """For each real object: fraction of its k nearest neighbours (pooled sample) that are real. Chance = share of real objects."""
    P = np.vstack([S, R]); lab = np.r_[np.zeros(len(S)), np.ones(len(R))]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(R)[1][:, 1:]
    return lab[idx].mean(1)
coords = {}
for name, (S, R) in spaces.items():
    mu, sd = S.mean(0), S.std(0) + 1e-9
    S, R = (S - mu) / sd, (R - mu) / sd
    um = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit(S)
    coords[name] = (um.embedding_, um.transform(R))
    mx = mixing(S, R); chance = len(R) / (len(S) + len(R))
    # nearest simulated class of each real object, against the classifier's own top class
    nn = NearestNeighbors(n_neighbors=10).fit(S); nb = ysim[nn.kneighbors(R)[1]]
    vote = np.array([np.bincount(r, minlength=8).argmax() for r in nb])
    print(f"{name:10s}: real neighbours of a real object, mean {mx.mean():.2f} (chance {chance:.2f}); all, coherent only {mx[is_clean].mean():.2f} | "
          f"objects with >= 8 of 10 real neighbours {np.mean(mx >= 0.8):.2f} | neighbour-vote class equals the model's top class {np.mean(np.array(CLASSES)[vote] == feat.pred.values):.2f}", flush=True)
    print(f"            neighbour-vote class fractions: {pd.Series(np.array(CLASSES)[vote]).value_counts(normalize=True).round(2).to_dict()}", flush=True)
np.savez(DATA_DIR / f"embeddings_{SET}.npz", **{f"{k}_{w}": v[i] for k, v in coords.items() for i, w in enumerate(("sim", "real"))}, ysim=ysim, is_clean=is_clean)

BG, FG, MUTED = "#141A2E", "#E8E9F2", "#B5BAD0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG,
                     "axes.edgecolor": MUTED, "xtick.color": FG, "ytick.color": FG, "font.size": 12, "axes.titlesize": 13})
CC = plt.cm.magma(np.linspace(0.18, 0.95, 8))
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
titles = {"features": "Light-curve feature space", "classifier": "Classifier space (eight class scores)"}
for a, name in zip(ax, coords):
    S, R = coords[name]
    a.scatter(S[:, 0], S[:, 1], s=4, c=CC[ysim], alpha=0.55, lw=0, rasterized=True)
    a.scatter(R[~is_clean, 0], R[~is_clean, 1], s=14, facecolors="none", edgecolors="white", lw=0.6, alpha=0.8)
    a.scatter(R[is_clean, 0], R[is_clean, 1], s=42, marker="*", color="white", edgecolors="k", lw=0.3, zorder=5)
    a.set_title(titles[name]); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_visible(False)
h = [Line2D([], [], marker="o", ls="", color=CC[i], ms=7, label=c) for i, c in enumerate(CLASSES)]
h += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="white", ms=7, label=f"DP2, all selected ({len(R)})"),
      Line2D([], [], marker="*", ls="", color="white", ms=11, label=f"DP2, coherent change ({int(is_clean.sum())})")]
fig.legend(handles=h, loc="lower center", ncol=10, frameon=False, fontsize=10, handletextpad=0.2, columnspacing=1.0)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(OUTFIG / "embedding.png", dpi=170); plt.close()
print("figure:", OUTFIG / "embedding.png")
