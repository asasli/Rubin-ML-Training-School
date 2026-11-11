# Active learning for classification and follow-up on ZTF

**Wednesday, November 11, 2026 · 9:30–10:45 · Room 401-20 · Dimitrios Kourtesis**

This lecture and hands-on session introduces **pool-based active learning**: choosing which unlabeled objects to label next under a limited follow-up budget. We distinguish candidate selection from acquiring a trustworthy label, then compare uncertainty-based strategies and the learning curves obtained as labels accumulate.

Two self-contained, **CPU-only notebooks** are included here. Both can be rerun from a fresh clone of this school repository after installing the separately maintained [active-learning Python package](https://github.com/DimitrisKour/astro-transient-active-learning).

## Hands-on notebooks

| Notebook | Activity | What actually runs |
| --- | --- | --- |
| [01 — Active learning with handwritten digits](01_active_learning_digits.ipynb) | Compare random, predictive-entropy and margin acquisition; explore budgets, validation, and five guided exercises | **Trains a small scikit-learn classifier** through active-learning cycles using the bundled digits dataset |
| [02 — Active learning for astronomical transients](02_applecider_astronomy_active_learning.ipynb) | Analyze recorded AstroMiNN macro-F1 trajectories, TDE/CV class metrics, acquisition overlap, and seed variation | **Reloads and replots archived validation measurements** included in [`teaching_data/`](teaching_data/README.md); does **not** retrain AstroMiNN |

The figures already embedded in these notebooks are for convenient GitHub viewing. They are not a substitute for execution: the code cells, small accompanying astronomy CSVs, and an independent re-execution script are all supplied.

**Scientific interpretation:** Notebook 02 compares some methods across independently executed experiment suites. Matching initialization scores and label budgets alone do not establish fully controlled comparisons. No raw astronomical images, candidate identities, hidden labels, model weights, private HPC paths, credentials or Rubin DP2 data are required.

## Installation (school environment)

From the **root** of a local [Rubin-ML-Training-School](https://github.com/asasli/Rubin-ML-Training-School) checkout, use the existing `rubin-ml-school` environment:

```bash
conda env create -f environment.yml   # skip if already created
conda activate rubin-ml-school
python -m pip install --upgrade pip
python -m pip install "al-pipeline[teaching] @ git+https://github.com/DimitrisKour/astro-transient-active-learning.git@0585b42d2346da4c729279c7374f5c2a1c1944dd"
python -m ipykernel install --user --name rubin-ml-school --display-name "Rubin ML School"
```

The package installation retrieves original active-learning *software*, not datasets, model weights, or historical experiment artifacts. Installation of the school environment may include heavier dependencies used on other days; these notebooks themselves need neither PyTorch nor a GPU.

If you are only doing this activity, a minimal isolated environment (Python 3.11+) is sufficient:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "al-pipeline[teaching] @ git+https://github.com/DimitrisKour/astro-transient-active-learning.git@0585b42d2346da4c729279c7374f5c2a1c1944dd"
```

## Reproduce both notebooks from a clean checkout

From the **school repository root**:

```bash
python day3/01_active_learning/run_notebooks.py --kernel rubin-ml-school
```

For the minimal virtual environment, you can use the default `python3` kernel instead:

```bash
python day3/01_active_learning/run_notebooks.py
```

The runner executes each notebook again, checks for expected output markers and figures, and saves newly executed copies under the Git-ignored `results/active_learning/` directory. It **does not overwrite the distributed student notebooks**. The digits notebook runs with the unsolved exercises disabled; students can enable them after completing the TODOs.

Alternatively, open either `.ipynb` in JupyterLab or VS Code with the environment above and choose **Restart Kernel and Run All**. Notebook 02 discovers its bundled data from either the school repository root or the `day3/01_active_learning/` directory.

Successful execution verifies the teaching workflow, not numerical retraining of the original AstroMiNN model. The published records in Notebook 02 are fixed historical validation results rather than generated astronomical observations.

## Learning goals

1. Distinguish labeled, unlabeled and pending pools, and separate acquisition policies from the labeling oracle.
2. Compare random, entropy and margin acquisition at matched budgets, using validation macro-F1.
3. Understand class imbalance, seed variability, and why initial labeled sets and training protocols matter.
4. Inspect real transient-classification summaries without needing private astronomical data or HPC compute.
5. Discuss how a similar acquisition workflow could guide spectroscopic follow-up or be transferred to Rubin DP2, subject to labeled-data availability and survey shift.

## Sources, license, and attribution

The teaching notebooks and accompanying material are contributed by Dimitris Kourtesis (copyright © 2026) under the [training school's GPLv3 license](../../LICENSE). The [original active-learning repository](https://github.com/DimitrisKour/astro-transient-active-learning) remains independently MIT-licensed.

The archived validation tables have accompanying provenance files, and are **not** raw ZTF or Rubin datasets. This repository makes no claim over third-party survey data, model checkpoints or upstream AstroMiNN/AppleCiDEr implementations. See [`teaching_data/README.md`](teaching_data/README.md) and the [source project's attribution](https://github.com/DimitrisKour/astro-transient-active-learning/blob/main/THIRD_PARTY.md).
