"""Combined candidate set 'comb': the reliability >= 0.3 selection plus the objects below that cut in which the
coherent-change rule fires (found by userhead_dp2.py). The new objects get the same position check in dp2.Object
as the others. Row-level outputs only in the private DATA_DIR."""
import numpy as np, pandas as pd
from tap import service, query, DATA_DIR as OUT

new = pd.read_parquet(OUT / "userhead_unseen_scores.parquet"); new = new[new.rule == 1].reset_index(drop=True)
print(f"{len(new)} rule-positive objects below the reliability cut", flush=True)
OB = ("objectId, coord_ra, coord_dec, refExtendedness, refBand, g_psfFlux, r_psfFlux, i_psfFlux, z_psfFlux, "
      "g_cModelFlux, r_cModelFlux, i_cModelFlux, z_cModelFlux, r_kronRad, i_kronRad, sersic_reff_major, sersic_index, ebv")
svc = service(); pos = []
for r in new.itertuples():
    o = query(f"SELECT {OB} FROM dp2.Object WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {r.ra}, {r.dec}, {15/3600})) = 1", svc, "obj")
    o["sep"] = 3600 * np.hypot((o.coord_ra - r.ra) * np.cos(np.deg2rad(r.dec)), o.coord_dec - r.dec); o = o.sort_values("sep")
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
pos = pd.DataFrame(pos)
print("position of the new objects:", pos.position.value_counts().to_dict(), flush=True)

c0 = pd.read_parquet(OUT / "stageB_loose00_candidates.parquet")
c0 = c0[c0.diaObjectId.isin(new.diaObjectId)].drop(columns=[c for c in pos.columns if c != "diaObjectId"]).merge(pos, on="diaObjectId")
c0["transient"] = c0.transient & (c0.position != "stellar counterpart"); c0["below_reliability_cut"] = True
c3 = pd.read_parquet(OUT / "stageB_loose03_candidates.parquet"); c3["below_reliability_cut"] = False
comb = pd.concat([c3, c0], ignore_index=True)
comb.to_parquet(OUT / "stageB_comb_candidates.parquet")
ev = {}
with np.load(OUT / "stageB_loose03_events.npz") as z: ev.update({k: z[k] for k in z.files})
with np.load(OUT / "stageB_loose00_events.npz") as z: ev.update({str(i): z[str(i)] for i in c0.diaObjectId})
np.savez_compressed(OUT / "stageB_comb_events.npz", **ev)
f3, f0 = pd.read_parquet(OUT / "stageB_loose03_features6.parquet"), pd.read_parquet(OUT / "stageB_loose00_features6.parquet")
pd.concat([f3, f0.loc[c0.diaObjectId]]).to_parquet(OUT / "stageB_comb_features6.parquet")
print(f"combined set: {len(comb)} candidates, {int((comb.transient & comb.has_events).sum())} transients with events; new objects kept: {int(c0.transient.sum())}")
