# Step 2.14: Automated ML Pipeline

This document describes the automated ML pipeline for the Equitable Student Dropout Prediction project.

## Overview

The automated pipeline connects existing components into a reproducible, end-to-end workflow that trains an XGBoost model on the approved development data, evaluates it using the approved validation workflow, and logs everything to MLflow.

## Pipeline Stages

```
Stage 1: Configuration & Input Validation
    ↓
Stage 2: Data Quality Validation
    ↓
Stage 3: Model Training (XGBoost on development_train)
    ↓
Stage 4: Evaluation (validation split only)
    ↓
Stage 5: MLflow Tracking
    ↓
Stage 6: Output & Summary Report
```

### Stage 1: Configuration & Input Validation

- Loads `configs/pipeline_config.yaml`
- Verifies all required input files exist (`development_train.parquet`, `validation.parquet`, existing model artifact, `model_config.yaml`, `fairness_config.yaml`)
- Confirms locked datasets (`test.parquet`, `final_holdout.parquet`) are NOT listed as pipeline inputs

### Stage 2: Data Quality Validation

- Loads `development_train.parquet` and `validation.parquet`
- Validates all 21 predictive features are present in both splits
- Verifies target column (`is_withdrawn`) exists and is binary (0 or 1)
- Runs leakage audit: confirms no forbidden columns (`final_result`, `date_unregistration`, `id_student`, sensitive attributes, diagnostic flags) appear in `PREDICTIVE_FEATURES`
- Validates feature roles using `src.data.split_dataset.validate_feature_roles()`
- Confirms fairness attributes are present for auditing

### Stage 3: Model Training

- Builds feature matrix from exactly the 21 approved `PREDICTIVE_FEATURES`
- Coerces numeric features using the same logic as `train_baselines.py`
- Uses `get_models()` from `train_baselines.py` with `configs/model_config.yaml`
- Trains XGBoost pipeline on `development_train`
- Saves model artifact to `models/pipeline_runs/xgb_pipeline_<timestamp>.pkl`
- **Never overwrites** the existing `models/baseline/xgb_pipeline.pkl`

### Stage 4: Evaluation

- Evaluates on the validation split using `get_metrics()` from `evaluate_baselines.py`
- Metrics computed: ROC-AUC, PR-AUC, accuracy, precision, recall, F1, log_loss, specificity
- Computes fairness disparity metrics (demographic parity difference/ratio, equalized odds difference) using `fairness_metrics.py` on all 5 sensitive attributes
- **Does NOT compute ECE or other calibration metrics** — the pipeline probabilities are uncalibrated
- **Does NOT select new fairness thresholds or fit a new fairness mitigation strategy**

### Stage 5: MLflow Tracking

- Logs to `sqlite:///mlflow.db` under experiment `student_dropout_automated_pipeline`
- Records: model hyperparameters, dataset sizes, positive rates, all validation metrics, fairness metrics, training time, feature list
- Uses the existing `mlflow_tracking.py` wrapper functions
- **Does NOT register a production model**

### Stage 6: Output & Summary

- Saves JSON execution report to `reports/pipeline/pipeline_run_<timestamp>.json`
- Prints a human-readable summary to stdout

## How to Run

### Full Pipeline

From the repository root:

```bash
python -m src.pipeline.automated_pipeline
```

Or equivalently:

```bash
python src/pipeline/automated_pipeline.py
```

### Dry-Run Mode

To validate configuration and inputs without training:

1. Edit `configs/pipeline_config.yaml` and set `dry_run: true`
2. Run the pipeline as above

Dry-run will:
- Validate all configs and input files exist
- Load and quality-check training and validation data
- Verify feature matrices are correctly constructed
- Confirm locked datasets are not accessed

Dry-run will **NOT**:
- Train a model
- Modify any artifacts
- Write to MLflow
- Read `test.parquet` or `final_holdout.parquet`

## Configuration

All pipeline configuration is in [`configs/pipeline_config.yaml`](../configs/pipeline_config.yaml).

| Key | Description |
|---|---|
| `training_data` | Path to training split (default: `development_train.parquet`) |
| `validation_data` | Path to validation split (default: `validation.parquet`) |
| `existing_model_path` | Path to existing validated model (read-only reference) |
| `output_model_dir` | Directory for newly trained model artifacts |
| `output_reports_dir` | Directory for execution reports |
| `forbidden_feature_columns` | Columns that must never enter model features |
| `locked_datasets` | Datasets the pipeline must never read |
| `mlflow.tracking_uri` | `sqlite:///mlflow.db` |
| `mlflow.experiment_name` | `student_dropout_automated_pipeline` |
| `dry_run` | Set to `true` for validation-only mode |

The pipeline also reads:
- `configs/model_config.yaml` — XGBoost hyperparameters and random seed
- `configs/fairness_config.yaml` — sensitive attributes for disparity metrics

## Output Locations

| Output | Location |
|---|---|
| Trained model artifact | `models/pipeline_runs/xgb_pipeline_<timestamp>.pkl` |
| Execution report | `reports/pipeline/pipeline_run_<timestamp>.json` |
| MLflow experiment | `student_dropout_automated_pipeline` in `mlflow.db` |

## MLflow Tracking Details

Each pipeline run creates one MLflow run with:

- **Tags**: `project`, `pipeline_step`, `task`, `cutoff`, `data_split`, `evaluation_split`, `predictive_features`
- **Parameters**: model type, feature count, sample counts, positive rates, random seed, XGBoost hyperparameters, training time
- **Metrics**: All validation metrics (`val_roc_auc`, `val_pr_auc`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, `val_log_loss`, `val_specificity`), confusion matrix components, fairness metrics

## How to Run Tests

```bash
# Run only the pipeline tests
python -m pytest tests/test_automated_pipeline.py -v

# Run the full test suite (excluding tests that access locked data)
python -m pytest tests/ -v
```

## Dataset Access Policy

| Dataset | Pipeline Access |
|---|---|
| `development_train.parquet` | ✅ Training |
| `validation.parquet` | ✅ Evaluation |
| `test.parquet` | ❌ **LOCKED** — never read |
| `final_holdout.parquet` | ❌ **UNTOUCHED** — never read |
| `train.parquet` | Not used (legacy; equals dev_train + final_holdout) |

> [!IMPORTANT]
> The `final_holdout.parquet` is reserved for **one-time final evaluation** after the evaluation plan is fully frozen and all development/tuning decisions are finalized. It must not be used for model selection, hyperparameter tuning, threshold selection, or pipeline experiments.

## Existing Components Reused

The pipeline does **not** reimplement any existing functionality. It reuses:

| Component | Source |
|---|---|
| 21 predictive features | `src.models.train_baselines.PREDICTIVE_FEATURES` |
| Model construction & preprocessing | `src.models.train_baselines.get_models()` |
| Evaluation metrics | `src.models.evaluate_baselines.get_metrics()` |
| Feature role validation | `src.data.split_dataset.validate_feature_roles()` |
| Fairness metrics | `src.fairness.fairness_metrics.calculate_disparity_metrics()` |
| MLflow helpers | `src.models.mlflow_tracking.*` |

## Limitations

- The pipeline trains and evaluates a fresh XGBoost model; it does not apply calibration or fairness postprocessing (those are handled by the integrated pipeline in Step 2.8)
- Probabilities from the trained model are **uncalibrated** — they are not described as calibrated
- No model registration or deployment occurs
- The pipeline is designed for local development/demonstration, not production orchestration
