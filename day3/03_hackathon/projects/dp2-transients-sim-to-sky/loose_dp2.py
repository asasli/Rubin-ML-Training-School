"""What remains in the DP2 deep fields under loose, broker-like cuts. Per-object aggregates are computed
server side (one row per DiaObject), saved in the private DATA_DIR; the log holds counts only."""
import numpy as np, pandas as pd
from tap import service, query, data_dir

OUT = data_dir(); svc = service()
FIELDS = {"ELAIS-S1": (9.45, -44.02), "ECDFS": (52.98, -28.12), "EDFS-a": (58.90, -49.32), "EDFS-b": (63.60, -47.60), "COSMOS": (150.11, 2.23)}
RADIUS = 2.0
AGG = ("diaObjectId, COUNT(*) AS n, AVG(reliability) AS rel, MIN(midpointMjdTai) AS t0, MAX(midpointMjdTai) AS t1, "
       "MIN(band) AS b0, MAX(band) AS b1, MAX(snr) AS snr_max, AVG(ra) AS ra, AVG(dec) AS dec")
parts = []
for name, (ra, dec) in FIELDS.items():
    cone = f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {RADIUS})) = 1"
    a = query(f"SELECT {AGG} FROM dp2.DiaSource WHERE {cone} GROUP BY diaObjectId", svc, f"{name} all sources")
    try:
        b = query(f"SELECT {AGG} FROM dp2.DiaSource WHERE {cone} AND pixelFlags = 0 AND (ssObjectId IS NULL OR ssObjectId = 0) GROUP BY diaObjectId", svc, f"{name} broker-like sources")
    except Exception as e:
        print("broker-like query failed", str(e)[:200]); b = a.iloc[:0]
    ng = query(f"SELECT diaObjectId, COUNT(*) AS n_neg FROM dp2.DiaSource WHERE {cone} AND isNegative = 1 GROUP BY diaObjectId", svc, f"{name} negative sources")
    d = a.merge(b.add_suffix("_ok").rename(columns={"diaObjectId_ok": "diaObjectId"}), on="diaObjectId", how="left").merge(ng, on="diaObjectId", how="left")
    d["field"] = name; parts.append(d)
    print(f"{name}: {len(a):,} DiaObjects with any source, {int(a.n.sum()):,} DiaSources", flush=True)
d = pd.concat(parts, ignore_index=True).drop_duplicates("diaObjectId")
d[["n_ok", "n_neg"]] = d[["n_ok", "n_neg"]].fillna(0)
d.to_parquet(OUT / "loose_objects.parquet")
vis = pd.read_parquet(OUT / "stageA_visits.parquet").groupby("field").expMidptMJD.agg(f_start="min", f_end="max").reset_index()
d = d.merge(vis, on="field")
stageA = pd.read_parquet(OUT / "stageA_summary.parquet"); strict = set(stageA.diaObjectId[stageA.keep])

def flow(title, steps):
    keep = np.ones(len(d), bool); print(f"\n{title}")
    for label, m in steps:
        keep &= np.asarray(m); print(f"   {label:72s} {keep.sum():9,}   (of the 273 strict candidates: {d.diaObjectId[keep].isin(strict).sum()})", flush=True)
    return keep

span_ok, multi_band, multi_night = (d.t1_ok - d.t0_ok).between(1, 150), d.b0_ok != d.b1_ok, (d.t1_ok - d.t0_ok) >= 1
seen_edge = ((d.t0_ok - d.f_start) >= 20) | ((d.f_end - d.t1_ok) >= 20)
print(f"\nDeep fields, {len(d):,} DiaObjects in total")
k_boom = flow("A. Broker-like (what BOOM passes on today): no pixel flag, not a solar system object", [
    ("any DiaSource", d.n >= 1), ("at least 1 clean DiaSource", d.n_ok >= 1), ("at least 2 clean DiaSources", d.n_ok >= 2), ("at least 6 clean DiaSources", d.n_ok >= 6)])
for thr in (0.0, 0.3, 0.6, 0.8):
    k = flow(f"B. Loose transient cuts, mean reliability >= {thr}", [
        ("at least 6 clean DiaSources", d.n_ok >= 6), (f"mean reliability >= {thr}", d.rel_ok >= thr),
        ("no negative DiaSource", d.n_neg == 0), ("detections on more than one night, within 150 d", span_ok),
        ("at least 2 bands", multi_band), ("field observed >= 20 d before or after the detections", seen_edge)])
    d[f"loose_{thr}"] = k
    print("      per field:", d.field[k].value_counts().to_dict(), "| clean sources per object, quantiles:", d.n_ok[k].quantile([.1, .5, .9]).to_dict(),
          "| max SNR quantiles:", d.snr_max_ok[k].quantile([.1, .5, .9]).round(0).to_dict())
k3 = flow("C. Same as B with reliability >= 0.6, but only 3 clean DiaSources", [
    ("at least 3 clean DiaSources", d.n_ok >= 3), ("mean reliability >= 0.6", d.rel_ok >= 0.6), ("no negative DiaSource", d.n_neg == 0),
    ("detections on more than one night, within 150 d", span_ok), ("at least 2 bands", multi_band), ("field observed >= 20 d before or after", seen_edge)])
d["loose3_0.6"] = k3
d.to_parquet(OUT / "loose_objects.parquet")
print("\nreliability of clean sources, per-object mean, quantiles over objects with >= 6:", d.rel_ok[d.n_ok >= 6].quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
