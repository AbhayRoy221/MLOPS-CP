"""
Step 2.14: Automated ML Pipeline

A reproducible, automated pipeline connecting existing components for:
  1. Configuration and input validation
  2. Data quality and schema validation
  3. Model training (XGBoost on development_train)
  4. Evaluation on validation split
  5. MLflow experiment tracking
  6. Output summary and report generation

Data governance:
  - Uses development_train for training, validation for evaluation
  - NEVER reads test.parquet or final_holdout.parquet
  - Does NOT overwrite existing validated model artifacts
  - Does NOT register or deploy a production model
"""

import os
import sys
import json
import yaml
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Reuse existing project components
from src.models.train_baselines import PREDICTIVE_FEATURES, get_models
from src.models.evaluate_baselines import get_metrics
from src.models.mlflow_tracking import initialize_mlflow, start_run, log_parameters, log_metrics, add_tags
from src.data.split_dataset import validate_feature_roles
from src.fairness.fairness_metrics import calculate_group_metrics, calculate_disparity_metrics

# ---------------------------------------------------------------------------
# Constants (derived from existing project conventions)
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ['highest_education', 'code_module', 'code_presentation']
NUMERIC_FEATURES = [f for f in PREDICTIVE_FEATURES if f not in CATEGORICAL_FEATURES]
TARGET_COLUMN = 'is_withdrawn'

# Forbidden columns that must never appear in the model feature matrix
FORBIDDEN_COLUMNS = frozenset({
    'is_withdrawn', 'final_result', 'date_unregistration', 'id_student',
    'gender', 'disability', 'age_band', 'region', 'imd_band',
    'missing_registration', 'missing_socioeconomic_information',
    'assessment_submission_before_registration', 'no_vle_activity_by_day28',
})

SENSITIVE_ATTRIBUTES = ['gender', 'disability', 'age_band', 'region', 'imd_band']

logger = logging.getLogger(__name__)


# ===========================================================================
# Stage 1: Configuration & Input Validation
# ===========================================================================
def load_pipeline_config(config_path='configs/pipeline_config.yaml'):
    """Load and validate the pipeline configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if 'pipeline' not in raw_config:
        raise ValueError("Pipeline config missing required 'pipeline' key.")

    config = raw_config['pipeline']

    # Validate required keys
    required_keys = [
        'training_data', 'validation_data', 'target_column',
        'forbidden_feature_columns', 'locked_datasets', 'mlflow',
        'output_model_dir', 'output_reports_dir',
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Pipeline config missing required key: '{key}'")

    # Validate MLflow sub-keys
    for key in ['tracking_uri', 'experiment_name']:
        if key not in config['mlflow']:
            raise ValueError(f"Pipeline config mlflow section missing '{key}'")

    return config


def validate_inputs(config):
    """Verify required input files exist and locked datasets are not accessed."""
    # Training and validation data must exist
    for path_key in ['training_data', 'validation_data']:
        path = config[path_key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required input file missing: {path} (config key: '{path_key}')"
            )

    # Verify existing model artifact exists (we load it for reference, never overwrite)
    existing_model = config.get('existing_model_path', 'models/baseline/xgb_pipeline.pkl')
    if not os.path.exists(existing_model):
        raise FileNotFoundError(
            f"Existing model artifact not found: {existing_model}"
        )

    # Verify locked datasets are NOT accidentally listed as inputs
    locked = set(config.get('locked_datasets', []))
    for path_key in ['training_data', 'validation_data']:
        if config[path_key] in locked:
            raise ValueError(
                f"GOVERNANCE VIOLATION: '{config[path_key]}' is locked but listed as '{path_key}'"
            )

    # Verify model config exists (needed for training)
    if not os.path.exists('configs/model_config.yaml'):
        raise FileNotFoundError("Model config not found: configs/model_config.yaml")

    # Verify fairness config exists
    if not os.path.exists('configs/fairness_config.yaml'):
        raise FileNotFoundError("Fairness config not found: configs/fairness_config.yaml")

    logger.info("Stage 1 PASSED: All inputs validated.")
    return True


# ===========================================================================
# Stage 2: Data Quality Validation
# ===========================================================================
def validate_data_quality(train_df, val_df, config):
    """
    Validate data quality, schema, and governance constraints.
    Does NOT modify the dataframes.
    """
    issues = []

    # 2a. Verify all 21 predictive features present in both splits
    for name, df in [('training', train_df), ('validation', val_df)]:
        missing_features = set(PREDICTIVE_FEATURES) - set(df.columns)
        if missing_features:
            issues.append(
                f"{name} split missing predictive features: {missing_features}"
            )

    # 2b. Verify target column exists and is binary
    for name, df in [('training', train_df), ('validation', val_df)]:
        if TARGET_COLUMN not in df.columns:
            issues.append(f"{name} split missing target column '{TARGET_COLUMN}'")
        else:
            unique_targets = set(df[TARGET_COLUMN].dropna().unique())
            if not unique_targets.issubset({0, 1}):
                issues.append(
                    f"{name} split has non-binary target values: {unique_targets}"
                )
            if df[TARGET_COLUMN].isna().any():
                issues.append(f"{name} split has NaN values in target column")

    # 2c. Validate feature roles using existing function
    try:
        validate_feature_roles(PREDICTIVE_FEATURES)
    except ValueError as e:
        issues.append(f"Feature role validation failed: {e}")

    # 2d. Leakage audit: verify forbidden columns are NOT in PREDICTIVE_FEATURES
    for col in FORBIDDEN_COLUMNS:
        if col in PREDICTIVE_FEATURES:
            issues.append(f"LEAKAGE: Forbidden column '{col}' found in PREDICTIVE_FEATURES")

    # 2e. Verify fairness attributes present for auditing (not in features)
    for name, df in [('training', train_df), ('validation', val_df)]:
        for attr in SENSITIVE_ATTRIBUTES:
            if attr not in df.columns:
                issues.append(
                    f"{name} split missing fairness attribute '{attr}' for auditing"
                )

    # 2f. Verify feature count is exactly 21
    if len(PREDICTIVE_FEATURES) != 21:
        issues.append(
            f"Expected exactly 21 predictive features, got {len(PREDICTIVE_FEATURES)}"
        )

    if issues:
        raise ValueError(
            "Data quality validation FAILED:\n" + "\n".join(f"  - {i}" for i in issues)
        )

    logger.info("Stage 2 PASSED: Data quality validated.")
    return {
        'train_rows': len(train_df),
        'train_positive_rate': float(train_df[TARGET_COLUMN].mean()),
        'val_rows': len(val_df),
        'val_positive_rate': float(val_df[TARGET_COLUMN].mean()),
        'feature_count': len(PREDICTIVE_FEATURES),
    }


# ===========================================================================
# Stage 3: Model Training
# ===========================================================================
def prepare_features(df):
    """
    Build model feature matrix from a dataframe using exclusively the
    21 approved PREDICTIVE_FEATURES. Coerce numeric columns.

    This mirrors the existing prepare_X/load_data patterns in
    train_baselines.py and integrated_pipeline.py.
    """
    X = df[PREDICTIVE_FEATURES].copy()

    # Coerce numeric columns (same logic as existing train_baselines.load_data)
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # Verify no forbidden columns leaked in
    actual_columns = set(X.columns)
    leaked = actual_columns.intersection(FORBIDDEN_COLUMNS)
    if leaked:
        raise ValueError(f"LEAKAGE: Forbidden columns in feature matrix: {leaked}")

    if len(X.columns) != 21:
        raise ValueError(
            f"Feature matrix has {len(X.columns)} columns, expected exactly 21"
        )

    return X


def train_model(train_df, config):
    """
    Train XGBoost pipeline on development_train using the existing
    model configuration and preprocessing.

    Returns the trained pipeline and training metadata.
    """
    # Load existing model config
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)['models']

    # Build feature matrix and target
    X_train = prepare_features(train_df)
    y_train = train_df[TARGET_COLUMN]

    # Use existing get_models to build the pipeline with correct preprocessing
    models = get_models(model_config)
    if 'xgb' not in models:
        raise RuntimeError("XGBoost model not available — check xgboost installation")

    xgb_pipeline = models['xgb']

    logger.info(f"Training XGBoost on {len(X_train)} samples, {len(PREDICTIVE_FEATURES)} features...")
    start_time = time.time()
    xgb_pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")

    # Save to pipeline-specific output directory (never overwrite existing artifacts)
    output_dir = config['output_model_dir']
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"xgb_pipeline_{timestamp}.pkl"
    model_path = os.path.join(output_dir, model_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(xgb_pipeline, f)

    logger.info(f"Model saved to {model_path}")

    return xgb_pipeline, {
        'model_type': 'xgboost',
        'training_samples': len(X_train),
        'training_features': len(PREDICTIVE_FEATURES),
        'training_time_seconds': round(training_time, 2),
        'model_artifact_path': model_path,
        'model_config': {
            'n_estimators': model_config['xgboost']['n_estimators'],
            'max_depth': model_config['xgboost']['max_depth'],
            'learning_rate': model_config['xgboost']['learning_rate'],
            'random_state': model_config['random_state'],
        },
    }


# ===========================================================================
# Stage 4: Evaluation (Development Workflow)
# ===========================================================================
def evaluate_model(model, val_df, config):
    """
    Evaluate trained model on the validation split using the existing
    get_metrics function from evaluate_baselines.py.

    Also computes fairness disparity metrics using the existing
    fairness_metrics.py functions.
    """
    X_val = prepare_features(val_df)
    y_val = val_df[TARGET_COLUMN]

    # Use existing evaluation function
    metrics = get_metrics(model, X_val, y_val)

    # Extract scalar metrics (exclude 'cm' tuple and 'probs' array)
    val_metrics = {}
    for k, v in metrics.items():
        if k not in ('cm', 'probs'):
            val_metrics[f'val_{k}'] = float(v)

    # Confusion matrix components
    tn, fp, fn, tp = metrics['cm']
    val_metrics['val_true_negatives'] = int(tn)
    val_metrics['val_false_positives'] = int(fp)
    val_metrics['val_false_negatives'] = int(fn)
    val_metrics['val_true_positives'] = int(tp)

    # Fairness disparity metrics using existing implementation
    with open('configs/fairness_config.yaml', 'r') as f:
        fairness_config = yaml.safe_load(f)['fairness']

    sens_attrs = fairness_config['sensitive_attributes']
    S_val = val_df[sens_attrs].copy()
    y_pred = model.predict(X_val)

    disparity_metrics = calculate_disparity_metrics(y_val, y_pred, S_val)

    # Log primary fairness metric (gender, as used in existing integrated pipeline)
    primary_attr = 'gender'
    if primary_attr in disparity_metrics:
        for metric_name, metric_val in disparity_metrics[primary_attr].items():
            if metric_name not in ('n_groups', 'min_group_count'):
                val_metrics[f'val_fairness_{primary_attr}_{metric_name}'] = (
                    float(metric_val) if not np.isnan(metric_val) else 0.0
                )

    logger.info("Stage 4 PASSED: Evaluation completed.")
    return val_metrics, disparity_metrics


# ===========================================================================
# Stage 5: MLflow Tracking
# ===========================================================================
def track_in_mlflow(config, data_quality_info, training_info, val_metrics, pipeline_metadata):
    """
    Log pipeline run to MLflow using the existing tracking helpers.
    """
    mlflow_cfg = config['mlflow']
    initialize_mlflow(mlflow_cfg['tracking_uri'], mlflow_cfg['experiment_name'])

    with start_run(run_name=f"pipeline_run_{pipeline_metadata['timestamp']}") as run:
        run_id = run.info.run_id

        # Tags
        add_tags({
            'project': 'equitable_student_dropout',
            'pipeline_step': '2.14_automated_pipeline',
            'task': 'student_withdrawal_prediction',
            'cutoff': 'day28',
            'data_split': 'development_train',
            'evaluation_split': 'validation',
        })

        # Parameters
        params = {
            'model_type': training_info['model_type'],
            'n_predictive_features': training_info['training_features'],
            'training_samples': training_info['training_samples'],
            'validation_samples': data_quality_info['val_rows'],
            'train_positive_rate': data_quality_info['train_positive_rate'],
            'val_positive_rate': data_quality_info['val_positive_rate'],
            'random_state': config.get('random_state', 42),
            'training_time_seconds': training_info['training_time_seconds'],
        }
        # Add model hyperparameters
        for k, v in training_info['model_config'].items():
            params[f'xgb_{k}'] = v

        log_parameters(params)

        # Metrics (only validated, existing metrics)
        log_metrics(val_metrics)

        # Log feature list as a tag (JSON, truncated to fit MLflow limit)
        feature_list_str = json.dumps(PREDICTIVE_FEATURES)
        add_tags({'predictive_features': feature_list_str[:250]})

        logger.info(f"Stage 5 PASSED: MLflow run logged (run_id={run_id})")
        return run_id


# ===========================================================================
# Stage 6: Output & Summary
# ===========================================================================
def generate_summary(config, data_quality_info, training_info, val_metrics,
                     run_id, pipeline_metadata):
    """
    Save execution summary report and print results.
    """
    output_dir = config['output_reports_dir']
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'pipeline_step': '2.14_automated_pipeline',
        'execution_timestamp': pipeline_metadata['timestamp'],
        'execution_time_seconds': pipeline_metadata.get('total_time', 0),
        'dry_run': False,
        'data_quality': data_quality_info,
        'training': {
            'model_type': training_info['model_type'],
            'training_samples': training_info['training_samples'],
            'training_features': training_info['training_features'],
            'training_time_seconds': training_info['training_time_seconds'],
            'model_artifact_path': training_info['model_artifact_path'],
        },
        'validation_metrics': val_metrics,
        'mlflow_run_id': run_id,
        'data_governance': {
            'training_data': config['training_data'],
            'validation_data': config['validation_data'],
            'locked_datasets_accessed': False,
            'existing_model_overwritten': False,
        },
    }

    report_path = os.path.join(output_dir, f"pipeline_run_{pipeline_metadata['timestamp']}.json")
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("AUTOMATED ML PIPELINE -- EXECUTION SUMMARY (Step 2.14)")
    print("=" * 70)
    print(f"Timestamp:        {pipeline_metadata['timestamp']}")
    print(f"Training data:    {config['training_data']} ({data_quality_info['train_rows']} rows)")
    print(f"Validation data:  {config['validation_data']} ({data_quality_info['val_rows']} rows)")
    print(f"Features:         {data_quality_info['feature_count']} predictive features")
    print(f"Model:            XGBoost")
    print(f"Training time:    {training_info['training_time_seconds']:.1f}s")
    print(f"Model saved to:   {training_info['model_artifact_path']}")
    print(f"\n--- Validation Metrics ---")

    for k in ['val_roc_auc', 'val_pr_auc', 'val_accuracy', 'val_precision',
              'val_recall', 'val_f1', 'val_log_loss']:
        if k in val_metrics:
            print(f"  {k}: {val_metrics[k]:.4f}")

    print(f"\n--- Data Governance ---")
    print(f"  Locked datasets accessed: NO")
    print(f"  Existing model overwritten: NO")
    print(f"  MLflow run ID: {run_id}")
    print(f"  Report saved: {report_path}")
    print("=" * 70)

    logger.info(f"Stage 6 PASSED: Summary saved to {report_path}")
    return report_path


# ===========================================================================
# Dry Run
# ===========================================================================
def run_dry_run(config):
    """
    Validate configuration and inputs without training, evaluating,
    or modifying any artifacts. Does NOT read locked datasets.
    """
    print("\n" + "=" * 70)
    print("AUTOMATED ML PIPELINE -- DRY RUN (Step 2.14)")
    print("=" * 70)

    # Stage 1: Config & input validation
    print("\n[Stage 1] Configuration & Input Validation...")
    validate_inputs(config)
    print("  [OK] All required input files exist")
    print("  [OK] Locked datasets are not listed as inputs")
    print("  [OK] Model config and fairness config found")

    # Verify locked datasets are NOT read
    for locked_path in config.get('locked_datasets', []):
        print(f"  [OK] Locked dataset NOT accessed: {locked_path}")

    # Stage 2: Data quality validation (load only training and validation)
    print("\n[Stage 2] Data Quality Validation...")
    train_df = pd.read_parquet(config['training_data'])
    val_df = pd.read_parquet(config['validation_data'])
    data_quality_info = validate_data_quality(train_df, val_df, config)
    print(f"  [OK] Training data: {data_quality_info['train_rows']} rows, "
          f"positive rate: {data_quality_info['train_positive_rate']:.4f}")
    print(f"  [OK] Validation data: {data_quality_info['val_rows']} rows, "
          f"positive rate: {data_quality_info['val_positive_rate']:.4f}")
    print(f"  [OK] Feature count: {data_quality_info['feature_count']}")
    print(f"  [OK] All 21 features present, target is binary, no leakage")

    # Feature matrix audit
    X_train = prepare_features(train_df)
    actual_cols = set(X_train.columns)
    leaked = actual_cols.intersection(FORBIDDEN_COLUMNS)
    print(f"  [OK] Feature matrix columns: {len(X_train.columns)} "
          f"(forbidden leaked: {len(leaked)})")

    # Model config check
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)['models']
    print(f"  [OK] Model config: random_state={model_config['random_state']}, "
          f"xgb_n_estimators={model_config['xgboost']['n_estimators']}")

    print("\n[Stage 3-6] SKIPPED in dry-run mode (no training, evaluation, or MLflow logging)")
    print("\n  [OK] Existing model artifact preserved (not overwritten)")
    print(f"  [OK] Existing model path: {config.get('existing_model_path')}")

    print("\n" + "=" * 70)
    print("DRY RUN PASSED -- Pipeline is ready for execution.")
    print("Run with dry_run: false in configs/pipeline_config.yaml")
    print("=" * 70)

    return data_quality_info


# ===========================================================================
# Main Entry Point
# ===========================================================================
def main(config_path='configs/pipeline_config.yaml'):
    """
    Main pipeline entry point. Runs all 6 stages or dry-run mode.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline_start = time.time()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Stage 1: Load and validate config
    logger.info("=" * 60)
    logger.info("AUTOMATED ML PIPELINE -- Step 2.14")
    logger.info("=" * 60)

    config = load_pipeline_config(config_path)

    # Dry run mode
    if config.get('dry_run', False):
        validate_inputs(config)
        return run_dry_run(config)

    # Full pipeline
    validate_inputs(config)

    # Stage 2: Load data and validate quality
    logger.info("Stage 2: Loading data and validating quality...")
    train_df = pd.read_parquet(config['training_data'])
    val_df = pd.read_parquet(config['validation_data'])
    data_quality_info = validate_data_quality(train_df, val_df, config)

    # Stage 3: Train model
    logger.info("Stage 3: Training model...")
    model, training_info = train_model(train_df, config)

    # Stage 4: Evaluate
    logger.info("Stage 4: Evaluating on validation split...")
    val_metrics, disparity_metrics = evaluate_model(model, val_df, config)

    # Stage 5: MLflow tracking
    logger.info("Stage 5: Logging to MLflow...")
    pipeline_metadata = {
        'timestamp': timestamp,
        'total_time': round(time.time() - pipeline_start, 2),
    }
    run_id = track_in_mlflow(config, data_quality_info, training_info,
                             val_metrics, pipeline_metadata)

    # Stage 6: Summary
    logger.info("Stage 6: Generating summary...")
    pipeline_metadata['total_time'] = round(time.time() - pipeline_start, 2)
    report_path = generate_summary(config, data_quality_info, training_info,
                                   val_metrics, run_id, pipeline_metadata)

    return {
        'report_path': report_path,
        'mlflow_run_id': run_id,
        'val_metrics': val_metrics,
    }


if __name__ == '__main__':
    main()
