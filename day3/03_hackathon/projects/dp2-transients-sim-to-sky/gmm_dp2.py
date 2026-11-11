"""Unsupervised grouping with Gaussian mixtures: how many components does BIC ask for, and what do they contain?

Spaces : (a) light-curve features, PCA-10 fitted on the SIM train split; (b) SupCon embedding, PCA-10 fitted on SIM train.
Samples: REAL DP2, all selected (588) and the clean coherent set (98); SIM test subsamples of the same sizes as a
         reference, because the number of components BIC supports grows with the sample size.
No labels on the real objects: components are compared with the LightGBM top class, the position in the host and the
coherent-change flag. Row-level output only in the private DATA_DIR.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tap import DATA_DIR
from features6 import CLASSES

warnings.filterwarnings("ignore"); torch.manual_seed(0); rng = np.random.default_rng(0)
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parent; M = HERE / "model"; PREP = HERE.parent / "sassafras" / "prepared"
FEATURES = json.load(open(M / "features.json"))["features"]
sc = json.load(open(M / "supcon_scaler.json")); cols = sc["cols"]; med = pd.Series(sc["median"], index=cols); iqr = pd.Series(sc["iqr"], index=cols)
plain = lambda X: ((X[cols].fillna(med) - med) / iqr).clip(-6, 6).values
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
enc = Encoder(2 * len(cols)); enc.load_state_dict(torch.load(M / "supcon_encoder.pt", map_location="cpu")); enc.eval()
def embed(X):
    with torch.no_grad():
        return enc(prep(X)).numpy()

# ---------------------------------------------------------------- data
split = pd.read_csv(M / "sim_split.csv", dtype={"SNID": str})
Xs = pd.read_parquet(PREP / "sixband_six_h100.parquet").loc[split.SNID].reset_index(drop=True)[FEATURES]
y = split.y.values; tr, te = (split.split == "train").values, (split.split == "test").values
t588 = pd.read_parquet(DATA_DIR / "dp2_transients_table_loose03.parquet")
t98 = pd.read_parquet(DATA_DIR / "dp2_transients_table_comb_clean.parquet")
Xa = pd.read_parquet(DATA_DIR / "dp2_nightly_features_loose03.parquet")
Xu = pd.read_parquet(DATA_DIR / "dp2_nightly_features_loose00_unseen.parquet")
Xr = pd.concat([Xa, Xu], ignore_index=True).drop_duplicates("diaObjectId").set_index("diaObjectId")
X588, X98 = Xr.loc[t588.diaObjectId, FEATURES], Xr.loc[t98.diaObjectId, FEATURES]
coh588 = t588.diaObjectId.isin(set(t98.diaObjectId)).values
print(f"SIM train {tr.sum():,}, SIM test {te.sum():,} | REAL all selected {len(X588)}, clean {len(X98)}", flush=True)

pf = PCA(10, random_state=0).fit(plain(Xs[tr])); pz = PCA(10, random_state=0).fit(embed(Xs[tr]))
SPACES = {"features PCA-10": lambda X: pf.transform(plain(X)), "SupCon PCA-10": lambda X: pz.transform(embed(X))}

def fit_bic(Z, kmax, cov="diag"):
    best = None
    for k in range(1, kmax + 1):
        g = GaussianMixture(k, covariance_type=cov, n_init=3, reg_covar=1e-3, random_state=0).fit(Z); b = g.bic(Z)
        if best is None or b < best[0]:
            best = (b, k, g)
    return best[1], best[2]

def describe(Z, g, tab, name, extra=None):
    lab = g.predict(Z); post = g.predict_proba(Z).max(1)
    sil = silhouette_score(Z, lab) if len(set(lab)) > 1 else np.nan
    print(f"   {name}: {g.n_components} components, sizes {sorted(np.bincount(lab).tolist(), reverse=True)} | mean top posterior {post.mean():.2f}, "
          f"objects with posterior > 0.9: {(post > 0.9).mean():.2f} | silhouette {sil:.2f} | ARI with LightGBM class {adjusted_rand_score(tab.pred.values, lab):.2f}")
    print("   component against LightGBM top class:"); print(pd.crosstab(lab, tab.pred.values).to_string())
    print("   component against position:"); print(pd.crosstab(lab, tab.position.values).to_string())
    if extra is not None:
        print("   component against coherent change:"); print(pd.crosstab(lab, extra).to_string())
    return lab

out = {}
for sname, f in SPACES.items():
    Zte = f(Xs[te]); yte = y[te]
    print(f"\n================ {sname}", flush=True)
    k, g = fit_bic(Zte, 20); lab = g.predict(Zte)
    print(f"SIM test, all {len(Zte):,}: BIC picks {k} components (8 true classes); ARI with the true class {adjusted_rand_score(yte, lab):.2f}", flush=True)
    for n, kmax, Z, tab, extra in ((588, 12, f(X588), t588, coh588), (98, 8, f(X98), t98, None)):
        ks, aris = [], []
        for r in range(10):                                           # SIM reference at the same sample size
            i = rng.choice(len(Zte), n, replace=False); kk, gg = fit_bic(Zte[i], kmax)
            ks.append(kk); aris.append(adjusted_rand_score(yte[i], gg.predict(Zte[i])))
        print(f"\nSIM test, 10 random subsamples of {n}: components {sorted(ks)}, ARI with the true class {np.mean(aris):.2f}", flush=True)
        k, g = fit_bic(Z, kmax)
        kb = [fit_bic(Z[rng.choice(len(Z), int(0.8 * len(Z)), replace=False)], kmax)[0] for _ in range(20)]
        print(f"REAL, n = {n}: BIC picks {k} components (diagonal covariance); in 20 subsamples of 80%: {pd.Series(kb).value_counts().sort_index().to_dict()}", flush=True)
        lab = describe(Z, g, tab, f"REAL n = {n}", extra)
        out[f"{sname}|{n}"] = pd.DataFrame({"diaObjectId": tab.diaObjectId.values, "component": lab})
        if n == 588:
            kf, gf = fit_bic(Z, kmax, "full"); print(f"   same with full covariance: {kf} components, sizes {sorted(np.bincount(gf.predict(Z)).tolist(), reverse=True)}")
        # do the real objects fall into components of a mixture fitted on SIM plus REAL together?
        J = np.vstack([Zte[rng.choice(len(Zte), 5 * n, replace=False)], Z]); isreal = np.r_[np.zeros(5 * n, bool), np.ones(n, bool)]
        kj, gj = fit_bic(J, 20); lj = gj.predict(J); fr = pd.Series(isreal).groupby(lj).agg(["mean", "sum", "size"])
        dom = fr[fr["mean"] > 0.5]
        print(f"   mixture on SIM (5 x {n}) + REAL together: {kj} components; components where real objects are the majority: {len(dom)}, "
              f"holding {int(dom['sum'].sum())} of {n} real objects (real fraction expected by chance {1/6:.2f})", flush=True)
pd.concat(out, names=["space_n"]).reset_index(level=0).to_parquet(DATA_DIR / "gmm_components.parquet")
