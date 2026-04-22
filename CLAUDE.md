# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is an **early-stage research implementation project**. At the time of writing, the repository contains only:
- `README.md` — Polish-language notes describing the methodology being implemented (see below).
- `s40537-025-01154-1.pdf` — the source paper the implementation is based on. Read it when methodology questions come up.
- `creditcard.csv` — the Kaggle Credit Card Fraud dataset (~290k rows, 30 features), already downloaded locally. It is gitignored implicitly via size/content; do not commit it.
- `main.py` — currently an empty placeholder (staged as a new file but with no content). Implementation work lands here unless structure is deliberately added.

There is no existing source code, test suite, lint config, or build system yet. When adding structure (splitting modules, adding tests, adding a `requirements.txt` or `pyproject.toml`), prefer to match the minimal scope the task needs rather than scaffolding broadly.

## Environment & commands

A Python 3.12 virtualenv exists at `.venv/`. Use it directly:

```bash
.venv/bin/python main.py          # run the entry script
.venv/bin/pip install <pkg>       # add a dependency
.venv/bin/pip list                # check what's installed
```

Currently installed packages of note: `pandas`, `numpy`, `kagglehub`. The methodology (see below) additionally requires `scikit-learn` (Isolation Forest, classifiers, cross-validation), `shap`, and `keras`/`tensorflow` (autoencoder) — these are **not yet installed**. Install them with pip when the relevant stage is implemented.

No test framework is configured yet. If adding tests, pytest is the conventional choice for this stack.

## Project methodology (what the code is meant to do)

The project reproduces a label-refinement pipeline for the unbalanced, unlabeled credit card fraud dataset. Ground-truth labels exist in the CSV but are **deliberately not used during training** — they are only used at the end to measure label quality.

The pipeline has three conceptual stages that must run in order:

1. **Isolation Forest + SHAP feature selection.** Train an Isolation Forest on all 29 features (`Amount`, `V1`–`V28`). Run SHAP on the trained model, take `mean(abs(shap_value))` per feature to rank importance. Produce three reduced feature sets: top-29 (all), top-15, and top-10.

2. **Autoencoder-based label generation.** For each of the 3 feature sets, train an autoencoder to reconstruct inputs, then use reconstruction MSE as an anomaly score. Flag the top-`P` highest-error instances as fraud, with `P ∈ {500, 1000, 1500}`. This yields **9 labeled datasets** total (3 feature sets × 3 values of P). Architecture and hyperparameters (Adam, lr=1e-4, batch=256, MSE loss, 250 epochs, early stopping patience=25, 20% val split, encoder `input→100(ReLU)→50(ReLU)`, decoder `50→100(Tanh)→output(ReLU)`) come from the paper — match them when implementing.

3. **Label-quality evaluation.** For each of the 9 datasets, train 4 supervised classifiers (Decision Tree, Random Forest, Logistic Regression, MLP) with 5-fold CV repeated 10 times (→ 50 folds). Compare against Isolation Forest as the unsupervised baseline. Primary metric is **AUPRC**. No hyperparameter tuning — the goal is to assess label quality, not model quality. Supervised models train on the generated labels (80%) and are evaluated against the original ground-truth labels (20%).

The target output is 2250 runs (9 datasets × 5 models × 50 folds) summarized by AUPRC. Paper reports top-15 features with `P=1500` as a plateau / sweet spot.

When working on a specific stage, keep the stage boundary clean: feature selection outputs should be reusable by the autoencoder stage without re-running SHAP, and generated labels should be reusable by the evaluation stage without re-running the autoencoder. The dataset is large (~150 MB), so avoid re-reading the CSV unnecessarily across stages.

## Language note

The README and any inline documentation in this repo are written in **Polish**. Preserve the existing language when editing docs unless asked otherwise.
