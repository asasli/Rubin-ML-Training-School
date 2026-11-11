"""Aggregate-only scoping of DP2 for the transient selection. Prints counts, saves nothing row-level."""
import pandas as pd
from tap import service, query

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
svc = service()
B = "ugrizy"

v = query("SELECT band, COUNT(*) AS n, MIN(expMidptMJD) AS t0, MAX(expMidptMJD) AS t1, MIN(ra) AS ra0, MAX(ra) AS ra1, "
          "MIN(dec) AS dec0, MAX(dec) AS dec1 FROM dp2.Visit GROUP BY band", svc, "visits by band")
print(v.round(2).to_string(), flush=True)

# visits per night-ish: how the survey span is populated
vv = query("SELECT visit, band, ra, dec, expMidptMJD FROM dp2.Visit", svc, "all visits")
vv["month"] = ((vv.expMidptMJD - vv.expMidptMJD.min()) // 30).astype(int)
print("visits per 30-day bin:", vv.groupby("month").size().to_dict())
print("nights with data:", vv.expMidptMJD.astype(int).nunique(), " span (d):", round(vv.expMidptMJD.max() - vv.expMidptMJD.min(), 1))
# coarse pointing map, 2-degree cells
vv["cell"] = (vv.ra // 2 * 2).astype(int).astype(str) + "," + (vv.dec // 2 * 2).astype(int).astype(str)
top = vv.groupby("cell").agg(n=("visit", "size"), nights=("expMidptMJD", lambda s: s.astype(int).nunique()), bands=("band", "nunique"),
                             span=("expMidptMJD", lambda s: s.max() - s.min())).sort_values("n", ascending=False)
print(f"{len(top)} 2-degree cells with visits; the 25 most visited:"); print(top.head(25).round(1).to_string(), flush=True)

never_negative = " AND ".join(f"({b}_psfFluxMin IS NULL OR {b}_psfFluxMin > 0)" for b in B)
bright = "(" + " OR ".join(f"{b}_psfFluxMax > 10*{b}_psfFluxErrMean" for b in B) + ")"
for label, where in [
    ("nDiaSources >= 6", "nDiaSources >= 6"),
    ("6..200", "nDiaSources BETWEEN 6 AND 200"),
    ("6..200, never negative", f"nDiaSources BETWEEN 6 AND 200 AND {never_negative}"),
    ("6..200, never negative, 10 sigma peak", f"nDiaSources BETWEEN 6 AND 200 AND {never_negative} AND {bright}"),
    ("10..200, never negative, 10 sigma peak", f"nDiaSources BETWEEN 10 AND 200 AND {never_negative} AND {bright}"),
]:
    try:
        c = query(f"SELECT COUNT(*) AS n FROM dp2.DiaObject WHERE {where}", svc, label)
        print(f"   {label:45s} {int(c.n[0]):>12,}", flush=True)
    except Exception as e:
        print("   failed:", label, str(e)[:200], flush=True)
