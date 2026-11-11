"""Six-band FLARE feature extractor, copied unchanged from later/sassafras/build_sixband_notebook.py
(section 2) so that simulated and real light curves pass through the same code.

Event table convention: (n, 4) float array with columns
    dt        days since the first detection
    band_id   u g r i z y = 0..5
    logflux   log10 of the flux in microJansky
    logerr    error on logflux
sorted in time, 3-sigma positive points only, 0 <= dt <= 100.
"""
import numpy as np
from flare import data as D
from flare.features import _band_features

BANDS = ["u", "g", "r", "i", "z", "y"]
BAND_ID = {b: i for i, b in enumerate(BANDS)}
CLASSES = ["SN Ia", "SN Ibc", "SN II", "SLSN", "TDE", "KN", "CART", "ILOT"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
LOG_CONST = 1.0 / np.log(10)
PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("z", "y")]
COLOUR_SUFFIXES = ["mean", "wmean", "std", "slope", "early", "late", "late_minus_early"]
_DUMMY = D.BandLC(t=np.array([0., 1, 2, 3]), flux=np.ones(4), flux_err=np.ones(4) * .1,
                  logflux=np.zeros(4), logflux_err=np.ones(4) * .04)


def bands_of(ev):
    out = {}
    for b in BANDS:
        m = ev[:, 1] == BAND_ID[b]
        if not m.any():
            continue
        t, lf, le = (ev[m, k].astype(float) for k in (0, 2, 3))
        u = np.concatenate([[True], np.diff(t) > 0])                 # strictly increasing times
        t, lf, le = t[u], lf[u], le[u]
        fl = 10.0 ** lf
        out[b] = D.BandLC(t=t, flux=fl, flux_err=le * fl / D.LOG_CONST, logflux=lf, logflux_err=le)
    return out


def colour_series(bl, b1, b2, window=1.5):
    if b1 not in bl or b2 not in bl:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    a, b = bl[b1], bl[b2]; ts, vs, es = [], [], []
    for i, t in enumerate(a.t):
        j = int(np.argmin(np.abs(b.t - t)))
        if abs(b.t[j] - t) <= window:
            ts.append(t); vs.append(-2.5 * (a.logflux[i] - b.logflux[j])); es.append(2.5 * np.hypot(a.logflux_err[i], b.logflux_err[j]))
    return np.array(ts), np.array(vs), np.array(es)


def colour_stats(name, t, v, e):
    out = {f"{name}_n": float(len(v))}
    if len(v) == 0:
        out.update({f"{name}_{s}": np.nan for s in COLOUR_SUFFIXES}); return out
    n, w = len(v), 1.0 / np.clip(e, 1e-6, None) ** 2
    out[f"{name}_mean"], out[f"{name}_wmean"] = float(v.mean()), float(np.average(v, weights=w))
    out[f"{name}_std"] = float(v.std()) if n > 1 else np.nan
    out[f"{name}_slope"] = float(np.polyfit(t, v, 1)[0]) if n > 2 and np.ptp(t) > 0 else np.nan
    half = t.min() + np.ptp(t) / 2 if n > 1 else t[0]
    early, late = v[t <= half], v[t > half]
    out[f"{name}_early"] = float(early.mean()) if len(early) else np.nan
    out[f"{name}_late"] = float(late.mean()) if len(late) else np.nan
    out[f"{name}_late_minus_early"] = out[f"{name}_late"] - out[f"{name}_early"] if np.isfinite(out[f"{name}_late"]) and np.isfinite(out[f"{name}_early"]) else np.nan
    return out


def features6(ev):
    bl = bands_of(ev)
    out = {"n_obs_total": float(len(ev)), "n_bands": float(len(bl)), "span_total": float(ev[:, 0].max())}
    for b in BANDS:
        out.update(_band_features(bl[b], b) if b in bl else {k: np.nan for k in _band_features(_DUMMY, b)})
    for b1, b2 in PAIRS:
        out.update(colour_stats(f"{b1}_{b2}", *colour_series(bl, b1, b2)))
        both = b1 in bl and b2 in bl
        out[f"{b1}{b2}_peak_dt"] = out[f"{b1}_t_peak"] - out[f"{b2}_t_peak"] if both else np.nan
        out[f"{b1}{b2}_peak_logflux_ratio"] = out[f"{b1}_logflux_peak"] - out[f"{b2}_logflux_peak"] if both else np.nan
    return out


def quality6(ev, min_total=6, min_bands=2):
    return ev is not None and len(ev) >= min_total and len(np.unique(ev[:, 1])) >= min_bands


def conformal_calibrate(P_cal, y_cal, alpha=0.10):
    """Mondrian (class-conditional) split conformal: per-class quantile of 1 - p_true."""
    qhat = np.ones(P_cal.shape[1])
    for c in range(P_cal.shape[1]):
        s = 1.0 - P_cal[y_cal == c, c]; n = len(s)
        if n:
            k = int(np.ceil((n + 1) * (1 - alpha))); qhat[c] = 1.0 if k > n else np.sort(s)[k - 1]
    return qhat


def prediction_sets(P, qhat):
    return P >= (1.0 - np.asarray(qhat))[None, :]


def nightly(ev, gap=0.5):
    """Average the epochs of one band that lie within `gap` days of each other (inverse-variance weights)."""
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


def band_changes(ev):
    out = []
    for b in np.unique(ev[:, 1]):
        e = ev[ev[:, 1] == b]
        if len(e) < 4:
            continue
        fl = 10.0 ** e[:, 2].astype(float); er = e[:, 3] * fl / LOG_CONST; o = np.argsort(fl); lo, hi = o[:2], o[-2:]
        wm = lambda k: ((fl[k] / er[k] ** 2).sum() / (1 / er[k] ** 2).sum(), 1 / np.sqrt((1 / er[k] ** 2).sum()))
        (fh, eh), (fl_, el) = wm(hi), wm(lo)
        out.append((fh / fl_, (fh - fl_) / np.hypot(eh, el), np.sign(e[hi, 0].mean() - e[lo, 0].mean())))
    return out


def coherent(ev):
    """Brightens or fades by more than 0.5 mag at 5 sigma in one band, supported by 0.3 mag at 3 sigma in a second band
    with the same direction in time (nightly averaged events)."""
    ch = band_changes(ev)
    for a, (ra, sa, da) in enumerate(ch):
        if ra > 1.585 and sa > 5 and any(rb > 1.3 and sb > 3 and db == da for b, (rb, sb, db) in enumerate(ch) if b != a):
            return True
    return False
