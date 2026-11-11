"""Stage A of the DP2 transient selection: in the deep fields with a long time baseline, download the
DiaObjects that pass the catalogue cuts and their DiaSources, then keep the objects whose detections are
positive, reliable and confined to a window that leaves the field observed before or after it.

Row-level outputs go only to the private DATA_DIR. The log holds aggregate counts only.
"""
import sys
import numpy as np, pandas as pd
from tap import service, query, data_dir

OUT = data_dir(); svc = service()
B = "ugrizy"
FIELDS = {"ELAIS-S1": (9.45, -44.02), "XMM-LSS": (35.57, -4.82), "ECDFS": (52.98, -28.12),
          "EDFS-a": (58.90, -49.32), "EDFS-b": (63.60, -47.60), "COSMOS": (150.11, 2.23)}
RADIUS = 2.0
MIN_SRC = 6

never_negative = " AND ".join(f"(o.{b}_psfFluxMin IS NULL OR o.{b}_psfFluxMin > 0)" for b in B)
bright = "(" + " OR ".join(f"o.{b}_psfFluxMax > 10*o.{b}_psfFluxErrMean" for b in B) + ")"
CUTS = f"o.nDiaSources BETWEEN {MIN_SRC} AND 300 AND {never_negative} AND {bright}"
OCOLS = "o.diaObjectId, o.ra, o.dec, o.nDiaSources, " + ", ".join(
    f"o.{b}_psfFluxMax, o.{b}_psfFluxErrMean, o.{b}_psfFluxNdata, o.{b}_scienceFluxMean" for b in B)
SCOLS = ("s.diaObjectId, s.diaSourceId, s.visit, s.midpointMjdTai, s.band, s.psfFlux, s.psfFluxErr, s.snr, s.reliability, "
         "s.extendedness, s.isDipole, s.isNegative, s.scienceFlux, s.templateFlux, s.glint_trail, s.pixelFlags_streakCenter, "
         "s.pixelFlags_saturatedCenter, s.pixelFlags_crCenter, s.pixelFlags_edge")

objs, srcs, vis = [], [], []
for name, (ra, dec) in FIELDS.items():
    circle = f"CIRCLE('ICRS', {ra}, {dec}, {RADIUS})"
    v = query(f"SELECT visit, band, ra, dec, expMidptMJD FROM dp2.Visit WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {RADIUS + 1.8})) = 1", svc, f"{name} visits")
    v["field"] = name; vis.append(v)
    o = query(f"SELECT {OCOLS} FROM dp2.DiaObject AS o WHERE CONTAINS(POINT('ICRS', o.ra, o.dec), {circle}) = 1 AND {CUTS}", svc, f"{name} DiaObjects")
    o["field"] = name; objs.append(o)
    try:
        s = query(f"SELECT {SCOLS} FROM dp2.DiaSource AS s JOIN dp2.DiaObject AS o ON s.diaObjectId = o.diaObjectId "
                  f"WHERE CONTAINS(POINT('ICRS', o.ra, o.dec), {circle}) = 1 AND {CUTS}", svc, f"{name} DiaSources (join)", retries=1)
    except Exception:
        s = query(f"SELECT {SCOLS} FROM dp2.DiaSource AS s WHERE CONTAINS(POINT('ICRS', s.ra, s.dec), {circle}) = 1", svc, f"{name} DiaSources (cone)")
        s = s[s.diaObjectId.isin(o.diaObjectId)]
    s["field"] = name; srcs.append(s)
    print(f"{name:9s} visits {len(v):5d} nights {v.expMidptMJD.astype(int).nunique():3d} span {np.ptp(v.expMidptMJD):5.0f} d | "
          f"DiaObjects after catalogue cuts {len(o):7,} | DiaSources {len(s):9,}", flush=True)

obj = pd.concat(objs, ignore_index=True).drop_duplicates("diaObjectId")
src = pd.concat(srcs, ignore_index=True).drop_duplicates("diaSourceId")
vis = pd.concat(vis, ignore_index=True)
obj.to_parquet(OUT / "stageA_diaobject.parquet"); src.to_parquet(OUT / "stageA_diasource.parquet"); vis.to_parquet(OUT / "stageA_visits.parquet")

# ---------------------------------------------------------------- detection-level selection
for c in ("isDipole", "isNegative", "glint_trail", "pixelFlags_streakCenter", "pixelFlags_saturatedCenter", "pixelFlags_crCenter", "pixelFlags_edge"):
    src[c] = src[c].fillna(False).astype(bool)
src["night"] = np.floor(src.midpointMjdTai - 0.5).astype(int)
src["bad"] = src.isDipole | src.isNegative | src.glint_trail | src.pixelFlags_streakCenter | src.pixelFlags_saturatedCenter | src.pixelFlags_crCenter | src.pixelFlags_edge
g = src.groupby("diaObjectId")
summ = pd.DataFrame({
    "n_src": g.size(), "n_bad": g.bad.sum(), "n_neg": g.isNegative.sum(), "n_dipole": g.isDipole.sum(),
    "t_first": g.midpointMjdTai.min(), "t_last": g.midpointMjdTai.max(), "n_nights": g.night.nunique(), "n_bands": g.band.nunique(),
    "rel_med": g.reliability.median(), "snr_max": g.snr.max(), "ext_med": g.extendedness.median(),
    "tmpl_over_peak": g.templateFlux.median() / g.psfFlux.max(),
}).reset_index()
summ = summ.merge(obj[["diaObjectId", "ra", "dec", "field"]], on="diaObjectId")
span = vis.groupby("field").expMidptMJD.agg(f_start="min", f_end="max").reset_index()
summ = summ.merge(span, on="field")
summ["duration"] = summ.t_last - summ.t_first
summ["pre"] = summ.t_first - summ.f_start; summ["post"] = summ.f_end - summ.t_last

steps = [
    ("all after catalogue cuts", np.ones(len(summ), bool)),
    ("no negative, dipole, streak, glint, saturated, cosmic-ray or edge detection", summ.n_bad == 0),
    ("median reliability >= 0.6", summ.rel_med >= 0.6),
    ("detected on >= 3 nights in >= 2 bands", (summ.n_nights >= 3) & (summ.n_bands >= 2)),
    ("detections span 3 to 150 days", summ.duration.between(3, 150)),
    ("field observed >= 20 d before first or after last detection", (summ.pre >= 20) | (summ.post >= 20)),
]
keep = np.ones(len(summ), bool)
print("\nDetection-level selection (DP2 real data, deep fields only):")
for label, m in steps:
    keep &= np.asarray(m)
    print(f"   {label:78s} {keep.sum():8,}", flush=True)
summ["keep"] = keep
summ.to_parquet(OUT / "stageA_summary.parquet")
k = summ[keep]
print("\nkept per field:", k.field.value_counts().to_dict())
print("kept: detections per object quantiles", k.n_src.quantile([.1, .5, .9]).round(0).to_dict(),
      "| nights", k.n_nights.quantile([.1, .5, .9]).to_dict(), "| bands", k.n_bands.value_counts().sort_index().to_dict(),
      "| duration", k.duration.quantile([.1, .5, .9]).round(0).to_dict())
