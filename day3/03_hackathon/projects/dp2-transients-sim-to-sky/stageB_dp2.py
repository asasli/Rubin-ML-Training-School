"""Stage B: forced photometry, quiet-baseline test, position check against dp2.Object, six-band events
and features for the stage A candidates. Row-level outputs only in the private DATA_DIR."""
import sys, os, contextlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from tap import service, query, data_dir
from features6 import features6, quality6, BAND_ID, LOG_CONST

OUT = data_dir(); svc = service()
SET = os.environ.get("SET", "strict"); T = "" if SET == "strict" else "_" + SET      # SET=loose03: loose cuts, mean reliability >= 0.3
if SET == "strict":
    summ = pd.read_parquet(OUT / "stageA_summary.parquet")
    cand = summ[summ.keep].reset_index(drop=True)
else:
    lo = pd.read_parquet(OUT / "loose_objects.parquet")
    cand = lo[lo[{"loose03": "loose_0.3", "loose00": "loose_0.0"}[SET]]].rename(columns={"t0_ok": "t_first", "t1_ok": "t_last"}).reset_index(drop=True)
    cand["duration"] = cand.t_last - cand.t_first
    cand = cand[["diaObjectId", "ra", "dec", "field", "t_first", "t_last", "duration", "n_ok", "rel_ok", "snr_max_ok"]]
print("SET =", SET, flush=True)
ids = cand.diaObjectId.astype("int64").tolist()
print(f"{len(ids)} stage A candidates", flush=True)

# ---------------------------------------------------------------- forced photometry
FCOLS = ("f.diaObjectId, f.visit, f.band, f.psfDiffFlux, f.psfDiffFluxErr, f.psfDiffFlux_flag, f.psfFlux, f.psfFluxErr, f.psfFlux_flag, "
         "f.pixelFlags_bad, f.pixelFlags_crCenter, f.pixelFlags_edge, f.pixelFlags_interpolatedCenter, f.pixelFlags_saturatedCenter, "
         "f.pixelFlags_suspectCenter, f.diff_PixelFlags_nodataCenter, f.invalidPsfFlag, v.expMidptMJD")
fp_path = OUT / ("stageB" + T + "_forced.parquet")
if fp_path.exists():
    fp = pd.read_parquet(fp_path)
else:
    def forced(k):
        chunk = ",".join(map(str, ids[k:k + 60]))
        return query(f"SELECT {FCOLS} FROM dp2.ForcedSourceOnDiaObject AS f JOIN dp2.Visit AS v ON f.visit = v.visit "
                     f"WHERE f.diaObjectId IN ({chunk})", service(), f"forced {k}")
    with ThreadPoolExecutor(4) as ex:
        parts = list(ex.map(forced, range(0, len(ids), 60)))
    fp = pd.concat(parts, ignore_index=True); fp.to_parquet(fp_path)
flag_cols = [c for c in fp.columns if "lag" in c]
for c in flag_cols:
    fp[c] = fp[c].fillna(False).astype(bool)
fp["clean"] = ~fp[[c for c in flag_cols if c != "psfFlux_flag"]].any(axis=1) & np.isfinite(fp.psfDiffFlux) & (fp.psfDiffFluxErr > 0)
fp = fp.rename(columns={"expMidptMJD": "mjd"}).sort_values(["diaObjectId", "mjd"]).reset_index(drop=True)
fp["snr"] = fp.psfDiffFlux / fp.psfDiffFluxErr
print(f"forced photometry: {len(fp):,} epochs, {fp.clean.mean():.3f} clean, median {fp.groupby('diaObjectId').size().median():.0f} per object", flush=True)

# ---------------------------------------------------------------- quiet baseline
# Baseline epochs: more than 20 d before the first detection or more than 80 d after the last one.
# The per-band median of the baseline difference flux is subtracted (the template can contain transient light).
GAP_PRE, GAP_POST = 20, 80
fp["flux"] = fp.psfDiffFlux.astype(float); fp["is_base"] = False
rows = []
for r in cand.itertuples():
    m = (fp.diaObjectId == r.diaObjectId).values
    f = fp[m & fp.clean.values]
    isb = (f.mjd < r.t_first - GAP_PRE) | (f.mjd > r.t_last + GAP_POST)
    off = f[isb].groupby("band").psfDiffFlux.agg(["median", "size"])
    off = off[off["size"] >= 5]["median"]
    corr = f.band.map(off).fillna(0.0)
    fp.loc[f.index, "flux"] = f.psfDiffFlux - corr; fp.loc[f.index, "is_base"] = isb
    f = f.assign(flux=f.psfDiffFlux - corr, z=(f.psfDiffFlux - corr) / f.psfDiffFluxErr, night=np.floor(f.mjd - 0.5))
    d = {"diaObjectId": r.diaObjectId, "min_snr": f.snr.min() if len(f) else np.nan, "peak_diff": f.flux.max() if len(f) else np.nan,
         "offset_over_peak": (off / f.groupby("band").flux.max()).median() if len(off) else np.nan,
         "offset_snr": (f[isb].snr.median() if isb.sum() >= 5 else np.nan)}
    for nm, s in (("pre", f[f.mjd < r.t_first - GAP_PRE]), ("post", f[f.mjd > r.t_last + GAP_POST])):
        w = 1 / s.psfDiffFluxErr ** 2
        nz = ((s.flux * w).groupby(s.night).sum() / np.sqrt(w.groupby(s.night).sum())) if len(s) else pd.Series(dtype=float)
        d[f"n_{nm}"] = len(nz); d[f"frac5_{nm}"] = (nz.abs() > 5).mean() if len(nz) else np.nan
        d[f"mad_{nm}"] = 1.4826 * (s.z - s.z.median()).abs().median() if len(s) else np.nan
        d[f"rawmed_{nm}"] = s.snr.median() if len(s) else np.nan
    sci = f[isb & ~f.psfFlux_flag & (f.psfFluxErr > 0)]
    d["base_sci"] = sci.psfFlux.median() if len(sci) >= 3 else np.nan
    d["base_sci_z"] = (sci.psfFlux / sci.psfFluxErr).median() if len(sci) >= 3 else np.nan
    rows.append(d)
base = pd.DataFrame(rows)
has_pre, has_post = base.n_pre >= 3, base.n_post >= 3
ok_pre = ~has_pre | ((base.rawmed_pre.abs() < 1) & (base.mad_pre < 1.5) & (base.frac5_pre <= 0.1))
ok_post = ~has_post | ((base.rawmed_post.abs() < 1) & (base.mad_post < 1.5) & (base.frac5_post <= 0.1))
base["quiet"] = (has_pre | has_post) & ok_pre & ok_post
base["no_dip"] = base.min_snr > -5
cand = cand.merge(base, on="diaObjectId")
fp["snr"] = fp.flux / fp.psfDiffFluxErr
print("baseline offset (median forced SNR of baseline epochs, before correction): quantiles",
      base.offset_snr.quantile([.1, .5, .9]).round(2).to_dict(), "| offset over peak flux", base.offset_over_peak.quantile([.1, .5, .9]).round(3).to_dict())

# ---------------------------------------------------------------- position check in dp2.Object
OB = ("objectId, coord_ra, coord_dec, refExtendedness, refBand, g_psfFlux, r_psfFlux, i_psfFlux, z_psfFlux, "
      "g_cModelFlux, r_cModelFlux, i_cModelFlux, z_cModelFlux, r_kronRad, i_kronRad, sersic_reff_major, sersic_index, ebv")
ob_path = OUT / ("stageB" + T + "_objects.parquet")
if os.environ.get("SKIP_OBJECTS"):          # no position check: features only
    ob = pd.DataFrame(columns=["diaObjectId", "sep", "refExtendedness"])
elif ob_path.exists():
    ob = pd.read_parquet(ob_path)
else:
    def near(r):
        s = service()
        try:
            d = query(f"SELECT {OB} FROM dp2.Object WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {r.ra}, {r.dec}, {15/3600})) = 1", s, "obj")
        except Exception:
            return None
        d["diaObjectId"] = r.diaObjectId
        d["sep"] = 3600 * np.hypot((d.coord_ra - r.ra) * np.cos(np.deg2rad(r.dec)), d.coord_dec - r.dec)
        return d
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(near, cand.itertuples()))
    print("object cone queries failed:", sum(x is None for x in res), flush=True)
    ob = pd.concat([x for x in res if x is not None], ignore_index=True); ob.to_parquet(ob_path)

pos = []
for r in cand.itertuples():
    o = ob[ob.diaObjectId == r.diaObjectId].sort_values("sep")
    d = {"diaObjectId": r.diaObjectId, "n_obj15": len(o), "position": "hostless", "sep_near": np.nan, "near_ext": np.nan,
         "host_sep": np.nan, "host_r": np.nan, "host_i": np.nan, "host_gr": np.nan, "host_ri": np.nan, "host_iz": np.nan}
    if len(o):
        n = o.iloc[0]; d["sep_near"], d["near_ext"] = n.sep, n.refExtendedness
        persistent = np.isfinite(r.base_sci_z) and r.base_sci_z > 5 and r.base_sci > 0.2 * r.peak_diff
        ext = o[o.refExtendedness == 1]
        if n.sep < 0.7 and n.refExtendedness == 0 and persistent:
            d["position"] = "stellar counterpart"
        elif len(ext) and ext.iloc[0].sep < 0.4:
            d["position"] = "nuclear"
        elif len(ext) and ext.iloc[0].sep < 10:
            d["position"] = "offset in a galaxy"
        if len(ext):
            h = ext.iloc[0]
            mag = lambda f: 31.4 - 2.5 * np.log10(f) if np.isfinite(f) and f > 0 else np.nan
            g, rr, i, z = (mag(h[f"{b}_cModelFlux"]) for b in "griz")
            d.update(host_sep=h.sep, host_r=rr, host_i=i, host_gr=g - rr, host_ri=rr - i, host_iz=i - z)
    pos.append(d)
cand = cand.merge(pd.DataFrame(pos), on="diaObjectId")

print("\nStage B selection (DP2 real data):")
keep = np.ones(len(cand), bool)
for label, m in [("stage A candidates", keep.copy()), ("no forced point below -5 sigma", cand.no_dip),
                 ("quiet baseline on every side that has one", cand.quiet),
                 ("no persistent point source at the position", cand.position != "stellar counterpart")]:
    keep &= np.asarray(m); print(f"   {label:55s} {keep.sum():6,}", flush=True)
print("   baseline available: before only", int((has_pre & ~has_post).sum()), " after only", int((~has_pre & has_post).sum()),
      " both", int((has_pre & has_post).sum()), " neither", int((~has_pre & ~has_post).sum()))
cand["transient"] = keep
print("position of the kept objects:", cand[keep].position.value_counts().to_dict())
print("position of the rejected objects:", cand[~keep].position.value_counts().to_dict())

# ---------------------------------------------------------------- six-band events, same convention as the simulation
@contextlib.contextmanager
def quiet_c_stderr():
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2); yield
    finally:
        os.dup2(saved, 2); os.close(devnull); os.close(saved)

events, feats = {}, {}
with quiet_c_stderr():
    for r in cand.itertuples():
        f = fp[(fp.diaObjectId == r.diaObjectId) & fp.clean & (fp.snr >= 3) & (fp.mjd >= r.t_first - 1e-4) & (fp.mjd <= r.t_first + 100)]
        if len(f) < 2:
            continue
        flux = f.flux.values / 1000.0                                          # nJy -> microJansky
        ev = np.column_stack([f.mjd.values - f.mjd.values[0], f.band.map(BAND_ID).values, np.log10(flux),
                              (f.psfDiffFluxErr.values / f.flux.values) * LOG_CONST]).astype(np.float32)
        if quality6(ev):
            events[str(r.diaObjectId)] = ev; feats[r.diaObjectId] = features6(ev)
cand["has_events"] = cand.diaObjectId.isin(feats)
X = pd.DataFrame.from_dict(feats, orient="index"); X.index.name = "diaObjectId"
np.savez_compressed(OUT / ("stageB" + T + "_events.npz"), **events)
X.to_parquet(OUT / ("stageB" + T + "_features6.parquet")); cand.to_parquet(OUT / ("stageB" + T + "_candidates.parquet"))
k = cand[cand.transient & cand.has_events]
print(f"\nwith >= 6 epochs at 3 sigma in >= 2 bands within 100 d of first detection: {cand.has_events.sum()} of {len(cand)} candidates, "
      f"{len(k)} of {int(cand.transient.sum())} transients")
ne = pd.Series({i: len(events[str(i)]) for i in k.diaObjectId}); nb = pd.Series({i: len(np.unique(events[str(i)][:, 1])) for i in k.diaObjectId})
print("transients: epochs per object", ne.quantile([.1, .5, .9]).to_dict(), "| bands", nb.value_counts().sort_index().to_dict(),
      "| per field", k.field.value_counts().to_dict())
