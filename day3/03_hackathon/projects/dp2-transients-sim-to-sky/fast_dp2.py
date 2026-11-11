"""Fast transients in the DP2 deep fields: objects that appear, fade by one magnitude within days and stay gone.

Selection from the per-object DiaSource aggregates (loose_objects.parquet, all deep-field DiaObjects), with three or more
clean detections instead of six and a detection span of at most 14 d. Then forced photometry, nightly averages, a quiet
window before and after, the time to fade by one magnitude, and the position check in dp2.Object.
REAL data, no labels. The SASSAFRAS test split gives the detection span and decline rate per simulated class for reference.
Row-level outputs only in the private DATA_DIR.
"""
import json, os, contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tap import service, query, data_dir
from features6 import features6, quality6, nightly, prediction_sets, BANDS, BAND_ID, CLASSES, LOG_CONST

OUT = data_dir(); HERE = Path(__file__).resolve().parent; M = HERE / "model"; PREP = HERE.parent / "sassafras" / "prepared"
MAX_SPAN, MAX_OBJ, T_FADE = 14.0, 3000, 6.0

# ---------------------------------------------------------------- SIM reference: how fast are the simulated classes?
split = pd.read_csv(M / "sim_split.csv", dtype={"SNID": str}); te = split[split.split == "test"]
def det_stats(ev):                               # detections only: span, and steepest decline from the peak of a band (mag per day)
    rate = np.nan
    for b in np.unique(ev[:, 1]):
        e = ev[ev[:, 1] == b]
        if len(e) < 2: continue
        k = e[:, 2].argmax(); after = e[k + 1:]; after = after[after[:, 0] - e[k, 0] >= 0.7]
        if len(after):
            rate = np.nanmax([rate, (2.5 * (e[k, 2] - after[:, 2]) / (after[:, 0] - e[k, 0])).max()])
    return ev[:, 0].max(), rate
with np.load(PREP / "sixband_events.npz") as z:
    st = pd.DataFrame([det_stats(nightly(z[s])) for s in te.SNID], columns=["span", "rate"]); st["cls"] = np.array(CLASSES)[te.y.values]
ref = st.groupby("cls").agg(n=("span", "size"), span_median=("span", "median"), span_le_14=("span", lambda v: (v <= MAX_SPAN).mean()),
                            rate_median=("rate", "median"), rate_gt_03=("rate", lambda v: (v > 0.3).mean()))
print("SIM test (100 d horizon), detection span in days and steepest decline in mag per day, by class:"); print(ref.round(2).to_string(), flush=True)

# ---------------------------------------------------------------- selection from the DiaSource aggregates
d = pd.read_parquet(OUT / "loose_objects.parquet")
vis = pd.read_parquet(OUT / "stageA_visits.parquet").groupby("field").expMidptMJD.agg(f_start="min", f_end="max").reset_index()
d = d.drop(columns=[x for x in ("f_start", "f_end") if x in d.columns]).merge(vis, on="field"); span = d.t1_ok - d.t0_ok
keep = np.ones(len(d), bool); print(f"\nREAL, deep fields: {len(d):,} DiaObjects")
steps = [("3 to 80 clean DiaSources", d.n_ok.between(3, 80)), ("no negative DiaSource", d.n_neg == 0), (f"detections span 0.5 to {MAX_SPAN:.0f} d", span.between(0.5, MAX_SPAN)),
         ("at least 2 bands", d.b0_ok != d.b1_ok), ("peak SNR at least 7", d.snr_max_ok >= 7),
         ("field observed at least 10 d before and 10 d after", ((d.t0_ok - d.f_start) >= 10) & ((d.f_end - d.t1_ok) >= 10))]
for label, m in steps:
    keep &= np.asarray(m); print(f"   {label:55s} {keep.sum():9,}", flush=True)
for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
    k = keep & (d.rel_ok >= thr).values; print(f"   mean reliability >= {thr:<4}                               {k.sum():9,}")
    if k.sum() <= MAX_OBJ: break
print(f"reliability threshold used: {thr} (the loosest that leaves at most {MAX_OBJ:,} objects for the forced-photometry download)")
cand = d[k].rename(columns={"t0_ok": "t_first", "t1_ok": "t_last"})[["diaObjectId", "ra", "dec", "field", "t_first", "t_last", "n_ok", "rel_ok", "snr_max_ok"]].reset_index(drop=True)
print("per field:", cand.field.value_counts().to_dict(), "| clean sources per object:", cand.n_ok.quantile([.1, .5, .9]).to_dict(),
      "| already in the six-detection set:", int(d.loc[k, "loose_0.0"].sum()), flush=True)

# ---------------------------------------------------------------- forced photometry
FCOLS = ("f.diaObjectId, f.visit, f.band, f.psfDiffFlux, f.psfDiffFluxErr, f.psfDiffFlux_flag, f.psfFlux, f.psfFluxErr, f.psfFlux_flag, "
         "f.pixelFlags_bad, f.pixelFlags_crCenter, f.pixelFlags_edge, f.pixelFlags_interpolatedCenter, f.pixelFlags_saturatedCenter, "
         "f.pixelFlags_suspectCenter, f.diff_PixelFlags_nodataCenter, f.invalidPsfFlag, v.expMidptMJD")
ids = cand.diaObjectId.astype("int64").tolist(); fp_path = OUT / "fast_forced.parquet"
if fp_path.exists():
    fp = pd.read_parquet(fp_path)
else:
    def forced(j):
        return query(f"SELECT {FCOLS} FROM dp2.ForcedSourceOnDiaObject AS f JOIN dp2.Visit AS v ON f.visit = v.visit "
                     f"WHERE f.diaObjectId IN ({','.join(map(str, ids[j:j + 60]))})", service(), f"forced {j}")
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        with ThreadPoolExecutor(4) as ex:
            fp = pd.concat(list(ex.map(forced, range(0, len(ids), 60))), ignore_index=True)
    fp.to_parquet(fp_path)
flags = [c for c in fp.columns if "lag" in c]
for c in flags: fp[c] = fp[c].fillna(False).astype(bool)
fp = fp[~fp[[c for c in flags if c != "psfFlux_flag"]].any(axis=1) & np.isfinite(fp.psfDiffFlux) & (fp.psfDiffFluxErr > 0)].rename(columns={"expMidptMJD": "mjd"})
fp = fp.merge(cand[["diaObjectId", "t_first", "t_last"]], on="diaObjectId")
print(f"forced photometry: {len(fp):,} clean epochs, median {fp.groupby('diaObjectId').size().median():.0f} per object", flush=True)
fp["far"] = (fp.mjd < fp.t_first - 10) | (fp.mjd > fp.t_last + 30)
off = fp[fp.far].groupby(["diaObjectId", "band"]).psfDiffFlux.agg(["median", "size"]); off = off[off["size"] >= 5]["median"].rename("off")
fp = fp.join(off, on=["diaObjectId", "band"]); fp["flux"] = fp.psfDiffFlux - fp.off.fillna(0.0)
sci = fp[fp.far & ~fp.psfFlux_flag & (fp.psfFluxErr > 0)]
sci = sci.assign(z=sci.psfFlux / sci.psfFluxErr).groupby("diaObjectId").agg(base_sci=("psfFlux", "median"), base_sci_z=("z", "median"), n_sci=("z", "size"))
fp["night"] = np.floor(fp.mjd - 0.5); fp["w"] = 1 / fp.psfDiffFluxErr ** 2; fp["fw"] = fp.flux * fp.w; fp["tw"] = fp.mjd * fp.w
nt = fp.groupby(["diaObjectId", "band", "night"]).agg(w=("w", "sum"), fw=("fw", "sum"), tw=("tw", "sum")).reset_index()
nt["flux"], nt["err"], nt["t"] = nt.fw / nt.w, 1 / np.sqrt(nt.w), nt.tw / nt.w; nt["z"] = nt.flux / nt.err

# ---------------------------------------------------------------- light-curve tests on nightly averages
rows = []; groups = dict(tuple(nt.groupby("diaObjectId")))
for r in cand.itertuples():
    g = groups.get(r.diaObjectId)
    if g is None: continue
    det = g[(g.z >= 5) & (g.t > r.t_first - 2) & (g.t < r.t_last + 2)]
    q = {"diaObjectId": r.diaObjectId, "n_det_nights": det.night.nunique(), "min_z": g.z.min(), "peak_diff": g.flux.max()}
    if len(det):
        ta, tb = det.t.min(), det.t.max(); win = g[(g.t >= ta - 0.7) & (g.t <= tb + 3)]
        pre, post = g[(g.t < ta - 0.7) & (g.t > ta - 40)], g[(g.t > tb + 3) & (g.t < tb + 60)]; rest = g[(g.t < ta - 0.7) | (g.t > tb + 3)]
        q.update(t_a=ta, span5=tb - ta, n_bands=win[win.z >= 3].band.nunique(), n_pre=pre.night.nunique(), n_post=post.night.nunique(),
                 max_pre=pre.z.max() if len(pre) else np.nan, max_post=post.z.max() if len(post) else np.nan,
                 frac5_rest=(rest.z.abs() > 5).mean() if len(rest) else np.nan, gap_pre=ta - pre.t.max() if len(pre) else np.nan)
        fade, rate = np.nan, np.nan
        for b, e in g.groupby("band"):
            e = e.sort_values("t"); p = e[(e.t >= ta - 0.7) & (e.t <= tb + 0.7) & (e.z >= 5)]
            if not len(p): continue
            pk = p.loc[p.flux.idxmax()]; later = e[e.t > pk.t + 0.3]
            gone = later[later.flux + later.err < 0.4 * pk.flux]
            if len(gone): fade = np.nanmin([fade, gone.t.iloc[0] - pk.t])
            dl = later[(later.z >= 3) & (later.t - pk.t >= 0.7) & (later.t <= tb + 0.7)]
            if len(dl): rate = np.nanmax([rate, (2.5 * np.log10(pk.flux / dl.flux) / (dl.t - pk.t)).max()])
        q.update(t_1mag=fade, rate=rate)
    rows.append(q)
c = cand.merge(pd.DataFrame(rows), on="diaObjectId").join(sci, on="diaObjectId")
keep = np.ones(len(c), bool); print("\nLight-curve tests on nightly forced photometry (REAL):")
tests = [("downloaded", keep.copy()), ("5 sigma on at least 2 nights", c.n_det_nights >= 2), ("3 sigma in at least 2 bands", c.n_bands >= 2),
         (f"5 sigma nights span at most {MAX_SPAN:.0f} d", c.span5 <= MAX_SPAN), ("no forced night below -5 sigma", c.min_z > -5),
         ("at least 2 nights within 40 d before, all below 3 sigma", (c.n_pre >= 2) & (c.max_pre < 3)),
         ("at least 2 nights within 60 d after, all below 3 sigma", (c.n_post >= 2) & (c.max_post < 3)),
         ("at most 5% of all other nights beyond 5 sigma", c.frac5_rest.fillna(0) <= 0.05),
         (f"fades by 1 mag within {T_FADE:.0f} d of the peak in one band", c.t_1mag <= T_FADE)]
for label, m in tests:
    keep &= np.asarray(m); print(f"   {label:60s} {keep.sum():7,}", flush=True)
appear = keep & (c.gap_pre <= 5).values
print(f"   of these, last non-detection at most 5 d before the first detection: {appear.sum()}")
f = c[keep].reset_index(drop=True)
print("fast candidates: time to fade by 1 mag (d)", f.t_1mag.quantile([.1, .5, .9]).round(1).to_dict(), "| 5 sigma span (d)", f.span5.quantile([.1, .5, .9]).round(1).to_dict(),
      "| decline between detections (mag/d)", f.rate.quantile([.1, .5, .9]).round(2).to_dict(), "| mean reliability", f.rel_ok.quantile([.1, .5, .9]).round(2).to_dict())
print("per field:", f.field.value_counts().to_dict(), "| clean DiaSources:", f.n_ok.quantile([.1, .5, .9]).to_dict())

# ---------------------------------------------------------------- position check in dp2.Object
OB = "objectId, coord_ra, coord_dec, refExtendedness, r_cModelFlux, i_cModelFlux"
def near(r):
    try:
        o = query(f"SELECT {OB} FROM dp2.Object WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {r.ra}, {r.dec}, {15/3600})) = 1", service(), "obj")
    except Exception:
        return {"diaObjectId": r.diaObjectId, "position": "query failed"}
    o["sep"] = 3600 * np.hypot((o.coord_ra - r.ra) * np.cos(np.deg2rad(r.dec)), o.coord_dec - r.dec); o = o.sort_values("sep")
    q = {"diaObjectId": r.diaObjectId, "position": "hostless", "host_sep": np.nan, "host_r": np.nan}
    if len(o):
        n = o.iloc[0]; ext = o[o.refExtendedness == 1]
        persistent = np.isfinite(r.base_sci_z) and r.base_sci_z > 5 and r.base_sci > 0.2 * r.peak_diff
        if n.sep < 0.7 and n.refExtendedness == 0 and persistent: q["position"] = "stellar counterpart"
        elif n.sep < 0.7 and n.refExtendedness == 0: q["position"] = "faint point source"
        elif len(ext) and ext.iloc[0].sep < 0.4: q["position"] = "nuclear"
        elif len(ext) and ext.iloc[0].sep < 10: q["position"] = "offset in a galaxy"
        if len(ext):
            h = ext.iloc[0]; q.update(host_sep=h.sep, host_r=31.4 - 2.5 * np.log10(h.r_cModelFlux) if h.r_cModelFlux > 0 else np.nan)
    return q
with contextlib.redirect_stdout(open(os.devnull, "w")):
    with ThreadPoolExecutor(6) as ex:
        pos = pd.DataFrame(list(ex.map(near, f.itertuples())))
f = f.merge(pos, on="diaObjectId")
print("\nposition of the fast candidates in dp2.Object:", f.position.value_counts().to_dict())
print("   with a non-detection at most 5 d before:", f[f.gap_pre <= 5].position.value_counts().to_dict())
gal = f.position.isin(["offset in a galaxy", "nuclear", "hostless"])
print("   host r magnitude of those in or near a galaxy:", f.host_r[gal].quantile([.1, .5, .9]).round(1).to_dict(), "| separation (arcsec):", f.host_sep[gal].quantile([.1, .5, .9]).round(1).to_dict())

# ---------------------------------------------------------------- what the simulation-trained classifier says (when there are enough epochs)
model = lgb.Booster(model_file=str(M / "sassafras6_lgbm.txt")); FEATURES = json.load(open(M / "features.json"))["features"]
qhat = np.array([json.load(open(M / "conformal.json"))["qhat"][k] for k in CLASSES])
f["pred"] = "too few epochs"; f["set_size"] = np.nan; f["p_kn"] = np.nan; feats = {}
saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 2)
for i, r in f.iterrows():
    e = fp[(fp.diaObjectId == r.diaObjectId) & (fp.flux / fp.psfDiffFluxErr >= 3) & (fp.mjd >= r.t_a - 0.7) & (fp.mjd <= r.t_a + 100)].sort_values("mjd")
    if len(e) < 2: continue
    ev = nightly(np.column_stack([e.mjd - e.mjd.iloc[0], e.band.map(BAND_ID), np.log10(e.flux / 1000.0), e.psfDiffFluxErr / e.flux * LOG_CONST]).astype(np.float32))
    if quality6(ev): feats[i] = features6(ev)
os.dup2(saved, 2)
if feats:
    X = pd.DataFrame.from_dict(feats, orient="index").reindex(columns=FEATURES); P = model.predict(X)
    f.loc[X.index, "pred"] = np.array(CLASSES)[P.argmax(1)]; f.loc[X.index, "set_size"] = prediction_sets(P, qhat).sum(1); f.loc[X.index, "p_kn"] = P[:, CLASSES.index("KN")]
    in_set = prediction_sets(P, qhat)[:, CLASSES.index("KN")]
    print(f"\nclassifier (trained on SIM) on the {len(X)} fast candidates with six nightly epochs in two bands: {pd.Series(f.loc[X.index, 'pred']).value_counts().to_dict()}"
          f" | KN inside the 90% conformal set: {int(in_set.sum())} | mean set size {f.set_size.mean():.2f}")
    print("   top class against position:"); print(pd.crosstab(f.loc[X.index, "pred"], f.loc[X.index, "position"]).to_string())
print(f"fast candidates with too few epochs for the classifier: {int((f.pred == 'too few epochs').sum())} of {len(f)}")
f.to_parquet(OUT / "fast_candidates.parquet")

# ---------------------------------------------------------------- private gallery for the visual check
FIG = OUT / "figs_fast"; FIG.mkdir(exist_ok=True); BC = dict(zip(BANDS, plt.cm.magma(np.linspace(0.1, 0.88, 6))))
order = f.assign(k=~f.position.isin(["stellar counterpart", "faint point source"])).sort_values(["k", "t_1mag"], ascending=[False, True]).reset_index(drop=True)
for page in range(0, min(len(order), 96), 24):
    sub = order.iloc[page:page + 24]; fig, axes = plt.subplots(6, 4, figsize=(14, 16), squeeze=False)
    for ax in axes.ravel()[len(sub):]: ax.axis("off")
    for ax, r in zip(axes.ravel(), sub.itertuples()):
        g = groups[r.diaObjectId]; g = g[(g.t > r.t_a - 30) & (g.t < r.t_a + r.span5 + 40)]
        for b, e in g.groupby("band"):
            ax.errorbar(e.t - r.t_a, e.flux, e.err, fmt="o", ms=3, lw=.7, color=BC[b], label=b)
        ax.axhline(0, color="0.6", lw=.5); ax.set_title(f"{r.position} | 1 mag in {r.t_1mag:.1f} d | {r.pred}\nrel {r.rel_ok:.2f}, n {int(r.n_ok)}, {r.field}", fontsize=7)
    axes[0, 0].legend(fontsize=6, ncol=6, frameon=False); plt.tight_layout(); plt.savefig(FIG / f"fast_{page // 24 + 1}.png", dpi=110); plt.close()
print("gallery (nightly forced flux in nJy against days since the first 5 sigma night) in the private folder", FIG)
