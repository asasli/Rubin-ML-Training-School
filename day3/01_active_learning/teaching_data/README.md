# Included astronomical teaching data

This folder accompanies [Notebook 02](../02_applecider_astronomy_active_learning.ipynb). All tables are **small, recorded validation summaries** from completed AstroMiNN pool-based active-learning studies. The notebook reads the tables locally; it does not download observations or retrain astronomical models.

| File | Contents |
| --- | --- |
| `astro_recent_validation.csv` | 1,020 validation metric measurements (macro-F1 and balanced accuracy) over 51 nine-acquisition trajectories, each with a cycle-zero baseline |
| `astro_recent_per_class.csv` | 2,550 validation precision, recall, F1 and support records for AGN, TDE, SN I, SN II and CV |
| `astro_cycle2_overlap_by_seed.csv` | Policy-pair cumulative acquisition overlap for three matched seeds |
| `astro_cycle2_overlap_mean.csv` | Mean cumulative acquisition overlap across those seeds |
| `*_provenance.json` | Source suites, extraction scope, data definitions and limitations |

The cumulative Jaccard table covers the objects selected in the first two acquisition cycles, excluding the common initial labeled set. It contains overlap statistics, **not individual candidate IDs**.

**Interpret carefully.** The notebook compares recorded policies from some independently executed research suites. The same labeling budget and initial scores alone do not demonstrate that all other training, evaluation and checkpointing details are matched. The alternate initial labeled-set study is a separate analysis. Three seeds give descriptive variation, not a formal significance claim.

**No restricted inputs are supplied.** These files contain no raw ZTF/Rubin observations or images, candidate identities, hidden labels, private model checkpoints, trained model weights, or test-set predictions. Source information is recorded in the provenance JSON documents and in the [upstream data guide](https://github.com/DimitrisKour/astro-transient-active-learning/blob/main/examples/data/README.md).

For complete reproductions of the **historical AstroMiNN model-training runs**, separate datasets, checkpoint files, model implementations and computational resources would be necessary. This school exercise intentionally reproduces the **analysis and its plots**, not those underlying historical fits.

The teaching materials are contributed by Dimitris Kourtesis (copyright © 2026) under the [school's GPLv3 license](../../../LICENSE). The original software repository remains MIT-licensed. No ownership of third-party data or models is claimed.
