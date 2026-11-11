"""Retrain the six-band SASSAFRAS model (configuration B of rubin_flare_sixband.ipynb: 329 features,
no host block) with the notebook's split and recipe, and save everything needed to apply it to DP2.

Everything here is SIMULATION: training, validation, test and conformal calibration are SASSAFRAS objects.
"""
import json, os, sys, time, contextlib
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from features6 import features6, CLASSES, CLASS_ID, conformal_calibrate, prediction_sets

PREP = HERE.parent / "sassafras" / "prepared"
OUT = HERE / "model"; OUT.mkdir(exist_ok=True)
SEED, NT = 42, int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

meta = pd.read_csv(PREP / "sixband_meta.csv", dtype={"SNID": str})
meta["y"] = meta.cls.map(CLASS_ID)
X = pd.read_parquet(PREP / "sixband_six_h100.parquet").loc[meta.SNID].reset_index(drop=True)
FEATURES = list(X.columns)
y = meta.y.values
print(f"{len(meta):,} simulated objects, {len(FEATURES)} features", flush=True)

# 1. the standalone extractor must reproduce the cached feature table
@contextlib.contextmanager
def quiet_c_stderr():
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2); yield
    finally:
        os.dup2(saved, 2); os.close(devnull); os.close(saved)

rng = np.random.default_rng(0)
check = rng.choice(len(meta), 300, replace=False)
with np.load(PREP / "sixband_events.npz") as z, quiet_c_stderr():
    rows = [features6(z[meta.SNID[i]]) for i in check]
R = pd.DataFrame(rows)[FEATURES].values.astype(float); C = X.iloc[check].values.astype(float)
same = np.isclose(R, C, rtol=1e-4, atol=1e-6, equal_nan=True)
print(f"extractor check on 300 objects: {same.mean():.5f} of cells identical, "
      f"{(~same).any(0).sum()} features with any difference", flush=True)
extractor_match = float(same.mean())

# 2. the notebook's split and recipe
idx = np.arange(len(meta))
idx_trainval, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, stratify=y[idx_trainval], random_state=SEED)
PARAMS = {"objective": "multiclass", "metric": "multi_logloss", "learning_rate": 0.05, "num_leaves": 31,
          "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "verbosity": -1, "seed": SEED, "num_threads": NT, "num_class": len(CLASSES)}
counts = np.bincount(y[idx_train], minlength=len(CLASSES)).astype(float)
w = (len(idx_train) / (len(CLASSES) * counts))[y[idx_train]]
t0 = time.time()
dtrain = lgb.Dataset(X.iloc[idx_train], label=y[idx_train], weight=w)
dval = lgb.Dataset(X.iloc[idx_val], label=y[idx_val], reference=dtrain)
model = lgb.train(PARAMS, dtrain, num_boost_round=3000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"trained {model.best_iteration} rounds in {time.time()-t0:.0f} s", flush=True)

P_val = model.predict(X.iloc[idx_val], num_iteration=model.best_iteration)
P_test = model.predict(X.iloc[idx_test], num_iteration=model.best_iteration)
pred, yt = P_test.argmax(1), y[idx_test]
qhat = conformal_calibrate(P_val, y[idx_val], alpha=0.10)
sets = prediction_sets(P_test, qhat)
metrics = {
    "data": "SASSAFRAS simulation only",
    "n_train": int(len(idx_train)), "n_val": int(len(idx_val)), "n_test": int(len(idx_test)),
    "trees": int(model.best_iteration), "extractor_match_fraction": extractor_match,
    "test_accuracy": float((pred == yt).mean()), "test_balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
    "test_macro_f1": float(f1_score(yt, pred, average="macro")),
    "test_recall": {c: float((pred[yt == CLASS_ID[c]] == CLASS_ID[c]).mean()) for c in CLASSES},
    "conformal_alpha": 0.10, "conformal_marginal_coverage": float(sets[np.arange(len(yt)), yt].mean()),
    "conformal_mean_set_size": float(sets.sum(1).mean()), "conformal_singletons": float((sets.sum(1) == 1).mean()),
    "conformal_empty": float((sets.sum(1) == 0).mean()),
    "test_max_prob_quantiles": {q: float(np.quantile(P_test.max(1), float(q))) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
}
print(json.dumps(metrics, indent=1), flush=True)

model.save_model(str(OUT / "sassafras6_lgbm.txt"), num_iteration=model.best_iteration)
json.dump({"features": FEATURES, "classes": CLASSES}, open(OUT / "features.json", "w"), indent=1)
json.dump({"alpha": 0.10, "qhat": dict(zip(CLASSES, map(float, qhat))), "calibration": "SASSAFRAS validation split"},
          open(OUT / "conformal.json", "w"), indent=1)
json.dump(metrics, open(OUT / "metrics_sim.json", "w"), indent=1)
# simulated reference for the sim-to-real comparison: split labels and test probabilities
split = np.full(len(meta), "train", dtype=object); split[idx_val] = "val"; split[idx_test] = "test"
pd.DataFrame({"SNID": meta.SNID, "cls": meta.cls, "y": y, "split": split}).to_csv(OUT / "sim_split.csv", index=False)
np.save(OUT / "sim_P_test.npy", P_test.astype(np.float32)); np.save(OUT / "sim_P_val.npy", P_val.astype(np.float32))
print("saved to", OUT)
