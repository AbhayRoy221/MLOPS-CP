# Data Split Governance

This document outlines the authoritative partitioning strategy for the **Equitable Student Dropout Prediction** modeling dataset and defines strict governance policies for future model development to prevent data leakage and guarantee unbiased evaluation.

The canonical Day-28 modeling dataset is split into four strict partitions located in `data/processed/splits/`:

## 1. development_train.parquet
- **Purpose**: Future model fitting, calibration fitting where appropriate, and hyperparameter optimization loops (e.g., cross-validation folds).
- **Status**: Available for all training procedures.

## 2. validation.parquet
- **Purpose**: Model-development comparison, evaluation of fairness mitigations, calibration/threshold decisions, and early stopping.
- **Status**: Available for evaluation and model selection only. Must NOT be used for direct model parameter fitting (e.g., no `fit()` calls).

## 3. test.parquet (Locked Test)
- **Purpose**: Historical held-out fairness-mitigation evaluation (already completed).
- **Status**: **LOCKED**.
- **Rule**: The locked test set must not be reused for future model-development decisions.

## 4. final_holdout.parquet
- **Purpose**: Final untouched evaluation after all future model and calibration decisions are completely frozen.
- **Status**: **UNTOUCHED**.
- **Rule**: The final holdout must remain untouched until the final-system evaluation. It must not be used for hyperparameter tuning, calibration, threshold policy development, or model selection.
