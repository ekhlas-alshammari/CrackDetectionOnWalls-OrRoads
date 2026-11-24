# Surface Crack Classification — ResNet-18 Pipeline

## Overview
An end-to-end reproducible PyTorch pipeline for binary classification of surface cracks (`Negative` vs. `Positive`) using **ResNet-18** pretrained on ImageNet (`224x224` resolution with standard ImageNet normalization). The workflow evaluates the model **before and after fine-tuning** on a fixed, untouched test split using consistent evaluation logic and reproducible random seeds (`SEED = 42`).

## Dataset & Sampling
* **Source:** `arunrk7/surface-crack-detection` (Kaggle), loaded via `kagglehub` (full dataset size: `40,000` images).
* **Balanced Subset:** Randomly sampled `3,000` images (`1,500` per class: `0` = `Negative`, `1` = `Positive`).
* **Duplicate Filtering:** Uses SHA-256 content hashing during selection to skip exact file duplicates (`10` exact duplicates skipped in this run), with a cross-split exact duplicate check and a note that visually similar images may still appear across splits.
* **Data Splits:** Stratified split into:
  * **Train:** `2,100` images (`70%` — `1,050` per class)
  * **Validation:** `450` images (`15%` — `225` per class)
  * **Test:** `450` images (`15%` — `225` per class)

## Key Contributions & Methodology
* **Reproducibility:** Fixes Python, NumPy, and PyTorch (`CPU`/`CUDA` `T4` GPU) seeds along with deterministic cuDNN flags.
* **Balanced & Deduplicated Selection:** Ensures equal class representation and prevents exact content duplication.
* **Stratified Splitting & Integrity Checks:** Preserves class ratios across train, validation, and test sets while verifying split consistency.
* **Pre/Post Fine-Tuning Comparison:** Evaluates a pretrained ResNet-18 backbone with a newly initialized classification head before and after fine-tuning (`BATCH_SIZE = 32`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`), tracking learning curves, test-set metrics (accuracy, precision, recall, F1-score), and confusion matrices.

## Environment & Tools
* **Runtime:** Python 3, Google Colab (`cuda` / NVIDIA `T4` GPU detected).
* **Libraries:** `torch`, `torchvision`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `kagglehub`.
