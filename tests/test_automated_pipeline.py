"""
Tests for Step 2.14: Automated ML Pipeline

Tests cover:
  - Pipeline configuration and required-input validation
  - Correct training/validation input selection
  - Prevention of forbidden target/leakage columns entering model features
  - Clear failure behavior when required inputs are missing
  - MLflow tracking configuration (mocked)
  - Dry-run safety: no training, no artifact overwrites, no locked data access
  - Feature matrix construction correctness
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train_baselines import PREDICTIVE_FEATURES
from src.pipeline.automated_pipeline import (
    load_pipeline_config,
    validate_inputs,
    validate_data_quality,
    prepare_features,
    FORBIDDEN_COLUMNS,
    TARGET_COLUMN,
    SENSITIVE_ATTRIBUTES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def _make_dummy_df(n_rows=50, include_forbidden=True):
    """Build a synthetic dataframe resembling a split, with all expected columns."""
    np.random.seed(42)
    data = {}
    for f in PREDICTIVE_FEATURES:
        if f in CATEGORICAL_FEATURES:
            data[f] = np.random.choice(['A', 'B', 'C'], n_rows)
        else:
            data[f] = np.random.randn(n_rows)

    # Target
    data['is_withdrawn'] = np.random.randint(0, 2, n_rows)

    if include_forbidden:
        # Columns that exist in real splits but must not enter model features
        data['final_result'] = np.random.choice(['Pass', 'Fail', 'Withdrawn'], n_rows)
        data['id_student'] = np.arange(n_rows)
        data['gender'] = np.random.choice(['M', 'F'], n_rows)
        data['disability'] = np.random.choice(['Y', 'N'], n_rows)
        data['age_band'] = np.random.choice(['0-35', '35-55'], n_rows)
        data['region'] = np.random.choice(['East', 'West'], n_rows)
        data['imd_band'] = np.random.choice(['0-10%', '10-20%'], n_rows)
        data['missing_registration'] = np.zeros(n_rows, dtype=int)
        data['missing_socioeconomic_information'] = np.zeros(n_rows, dtype=int)
        data['assessment_submission_before_registration'] = np.zeros(n_rows, dtype=int)
        data['no_vle_activity_by_day28'] = np.zeros(n_rows, dtype=int)

    return pd.DataFrame(data)


class TestPipelineConfiguration(unittest.TestCase):
    """Tests for Stage 1: Configuration and input validation."""

    def test_load_valid_config(self):
        """Pipeline config loads successfully with all required keys."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        self.assertIn('training_data', config)
        self.assertIn('validation_data', config)
        self.assertIn('target_column', config)
        self.assertIn('forbidden_feature_columns', config)
        self.assertIn('locked_datasets', config)
        self.assertIn('mlflow', config)

    def test_load_missing_config_fails(self):
        """Missing config file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_pipeline_config('nonexistent/config.yaml')

    def test_load_invalid_config_fails(self):
        """Config without 'pipeline' key raises ValueError."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'wrong_key': {}}, f)
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                load_pipeline_config(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_config_has_correct_mlflow_uri(self):
        """MLflow URI matches the authoritative sqlite:///mlflow.db."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        self.assertEqual(config['mlflow']['tracking_uri'], 'sqlite:///mlflow.db')

    def test_config_locked_datasets_listed(self):
        """Locked datasets include test.parquet and final_holdout.parquet."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        locked = config['locked_datasets']
        self.assertTrue(
            any('test.parquet' in p for p in locked),
            "test.parquet not in locked datasets"
        )
        self.assertTrue(
            any('final_holdout.parquet' in p for p in locked),
            "final_holdout.parquet not in locked datasets"
        )

    def test_config_training_data_is_development_train(self):
        """Training data path points to development_train, not train.parquet."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        self.assertIn('development_train', config['training_data'])

    def test_config_validation_data_is_validation(self):
        """Validation data path points to validation.parquet."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        self.assertIn('validation', config['validation_data'])


class TestInputValidation(unittest.TestCase):
    """Tests for input file existence and governance checks."""

    @patch('os.path.exists', return_value=True)
    def test_validate_inputs_passes_with_real_files(self, mock_exists):
        """Validate inputs succeeds when all required files exist."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        # This should not raise
        self.assertTrue(validate_inputs(config))

    @patch('os.path.exists')
    def test_validate_inputs_fails_missing_training_data(self, mock_exists):
        """Missing training data raises FileNotFoundError."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        config['training_data'] = 'nonexistent/train.parquet'
        
        def exists_side_effect(path):
            if path == 'nonexistent/train.parquet': return False
            return True
        mock_exists.side_effect = exists_side_effect
        
        with self.assertRaises(FileNotFoundError):
            validate_inputs(config)

    @patch('os.path.exists')
    def test_validate_inputs_fails_missing_validation_data(self, mock_exists):
        """Missing validation data raises FileNotFoundError."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        config['validation_data'] = 'nonexistent/val.parquet'
        
        def exists_side_effect(path):
            if path == 'nonexistent/val.parquet': return False
            return True
        mock_exists.side_effect = exists_side_effect
        
        with self.assertRaises(FileNotFoundError):
            validate_inputs(config)

    @patch('os.path.exists', return_value=True)
    def test_validate_inputs_rejects_locked_as_training(self, mock_exists):
        """Using a locked dataset as training_data raises ValueError."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        config['training_data'] = config['locked_datasets'][0]
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_inputs(config)

    @patch('os.path.exists', return_value=True)
    def test_validate_inputs_rejects_locked_as_validation(self, mock_exists):
        """Using a locked dataset as validation_data raises ValueError."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        # Point validation to the locked test set
        config['validation_data'] = 'data/processed/splits/test.parquet'
        with self.assertRaises(ValueError):
            validate_inputs(config)


class TestFeatureExclusion(unittest.TestCase):
    """Tests proving only the 21 approved features reach the model."""

    def test_predictive_features_exactly_21(self):
        """PREDICTIVE_FEATURES list contains exactly 21 items."""
        self.assertEqual(len(PREDICTIVE_FEATURES), 21)

    def test_target_excluded_from_features(self):
        """is_withdrawn is NOT in PREDICTIVE_FEATURES."""
        self.assertNotIn('is_withdrawn', PREDICTIVE_FEATURES)

    def test_final_result_excluded(self):
        """final_result is NOT in PREDICTIVE_FEATURES."""
        self.assertNotIn('final_result', PREDICTIVE_FEATURES)

    def test_id_student_excluded(self):
        """id_student is NOT in PREDICTIVE_FEATURES."""
        self.assertNotIn('id_student', PREDICTIVE_FEATURES)

    def test_date_unregistration_excluded(self):
        """date_unregistration is NOT in PREDICTIVE_FEATURES."""
        self.assertNotIn('date_unregistration', PREDICTIVE_FEATURES)

    def test_sensitive_attributes_excluded(self):
        """All 5 sensitive attributes are NOT in PREDICTIVE_FEATURES."""
        for attr in SENSITIVE_ATTRIBUTES:
            self.assertNotIn(attr, PREDICTIVE_FEATURES,
                             f"Sensitive attribute '{attr}' found in PREDICTIVE_FEATURES")

    def test_diagnostic_flags_excluded(self):
        """Diagnostic flags are NOT in PREDICTIVE_FEATURES."""
        diagnostics = [
            'missing_registration', 'missing_socioeconomic_information',
            'assessment_submission_before_registration', 'no_vle_activity_by_day28'
        ]
        for flag in diagnostics:
            self.assertNotIn(flag, PREDICTIVE_FEATURES)

    def test_all_forbidden_columns_excluded(self):
        """Every column in FORBIDDEN_COLUMNS is absent from PREDICTIVE_FEATURES."""
        for col in FORBIDDEN_COLUMNS:
            self.assertNotIn(col, PREDICTIVE_FEATURES)

    def test_prepare_features_returns_only_21_columns(self):
        """prepare_features returns exactly 21 columns, none forbidden."""
        df = _make_dummy_df(n_rows=10, include_forbidden=True)
        X = prepare_features(df)
        self.assertEqual(len(X.columns), 21)
        for col in X.columns:
            self.assertIn(col, PREDICTIVE_FEATURES)
            self.assertNotIn(col, FORBIDDEN_COLUMNS)

    def test_prepare_features_excludes_forbidden_even_if_present(self):
        """
        Even though forbidden columns exist in the input df,
        they do NOT appear in the feature matrix.
        """
        df = _make_dummy_df(n_rows=10, include_forbidden=True)
        # Verify forbidden columns exist in the source
        self.assertIn('is_withdrawn', df.columns)
        self.assertIn('final_result', df.columns)
        self.assertIn('gender', df.columns)
        self.assertIn('id_student', df.columns)

        X = prepare_features(df)
        for col in FORBIDDEN_COLUMNS:
            self.assertNotIn(col, X.columns,
                             f"Forbidden column '{col}' leaked into feature matrix")

    def test_target_kept_separate(self):
        """Target column is NOT in prepare_features output."""
        df = _make_dummy_df(n_rows=10)
        X = prepare_features(df)
        self.assertNotIn(TARGET_COLUMN, X.columns)

    def test_numeric_features_coerced(self):
        """Numeric features are coerced to numeric dtype."""
        df = _make_dummy_df(n_rows=10)
        # Introduce a string in a numeric column
        df.loc[0, 'vle_total_clicks'] = 'bad_value'
        X = prepare_features(df)
        # Should be NaN after coercion, not crash
        self.assertTrue(pd.isna(X.loc[0, 'vle_total_clicks']))


class TestDataQualityValidation(unittest.TestCase):
    """Tests for Stage 2: Data quality checks."""

    def test_validation_passes_with_good_data(self):
        """Data quality validation passes with correctly formed data."""
        df = _make_dummy_df(n_rows=20)
        info = validate_data_quality(df, df, load_pipeline_config()['pipeline']
                                     if False else
                                     load_pipeline_config('configs/pipeline_config.yaml'))
        self.assertEqual(info['feature_count'], 21)

    def test_validation_fails_missing_feature(self):
        """Missing predictive feature raises ValueError."""
        df = _make_dummy_df(n_rows=10)
        df = df.drop(columns=['vle_total_clicks'])
        config = load_pipeline_config('configs/pipeline_config.yaml')
        with self.assertRaises(ValueError) as ctx:
            validate_data_quality(df, df, config)
        self.assertIn('vle_total_clicks', str(ctx.exception))

    def test_validation_fails_missing_target(self):
        """Missing target column raises ValueError."""
        df = _make_dummy_df(n_rows=10)
        df = df.drop(columns=['is_withdrawn'])
        config = load_pipeline_config('configs/pipeline_config.yaml')
        with self.assertRaises(ValueError) as ctx:
            validate_data_quality(df, df, config)
        self.assertIn('is_withdrawn', str(ctx.exception))

    def test_validation_fails_non_binary_target(self):
        """Non-binary target values raise ValueError."""
        df = _make_dummy_df(n_rows=10)
        df.loc[0, 'is_withdrawn'] = 2  # Invalid
        config = load_pipeline_config('configs/pipeline_config.yaml')
        with self.assertRaises(ValueError) as ctx:
            validate_data_quality(df, df, config)
        self.assertIn('non-binary', str(ctx.exception))


class TestMLflowConfiguration(unittest.TestCase):
    """Tests for MLflow tracking configuration (mocked)."""

    @patch('src.pipeline.automated_pipeline.initialize_mlflow')
    @patch('src.pipeline.automated_pipeline.start_run')
    @patch('src.pipeline.automated_pipeline.log_parameters')
    @patch('src.pipeline.automated_pipeline.log_metrics')
    @patch('src.pipeline.automated_pipeline.add_tags')
    def test_mlflow_called_with_correct_uri(self, mock_tags, mock_metrics,
                                             mock_params, mock_start, mock_init):
        """MLflow is initialized with the correct tracking URI."""
        from src.pipeline.automated_pipeline import track_in_mlflow

        # Set up mock run context
        mock_run = MagicMock()
        mock_run.info.run_id = 'test-run-id'
        mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start.return_value.__exit__ = MagicMock(return_value=False)

        config = load_pipeline_config('configs/pipeline_config.yaml')
        data_info = {'train_rows': 100, 'val_rows': 50,
                     'train_positive_rate': 0.3, 'val_positive_rate': 0.3}
        training_info = {
            'model_type': 'xgboost', 'training_features': 21,
            'training_samples': 100, 'training_time_seconds': 1.0,
            'model_config': {'n_estimators': 300, 'max_depth': 4,
                             'learning_rate': 0.05, 'random_state': 42}
        }
        val_metrics = {'val_roc_auc': 0.72}
        pipeline_metadata = {'timestamp': '20260101_120000'}

        run_id = track_in_mlflow(config, data_info, training_info,
                                 val_metrics, pipeline_metadata)

        mock_init.assert_called_once_with('sqlite:///mlflow.db',
                                           'student_dropout_automated_pipeline')
        self.assertEqual(run_id, 'test-run-id')

    @patch('src.pipeline.automated_pipeline.initialize_mlflow')
    @patch('src.pipeline.automated_pipeline.start_run')
    @patch('src.pipeline.automated_pipeline.log_parameters')
    @patch('src.pipeline.automated_pipeline.log_metrics')
    @patch('src.pipeline.automated_pipeline.add_tags')
    def test_mlflow_logs_correct_tags(self, mock_tags, mock_metrics,
                                       mock_params, mock_start, mock_init):
        """MLflow tags include pipeline step and project name."""
        from src.pipeline.automated_pipeline import track_in_mlflow

        mock_run = MagicMock()
        mock_run.info.run_id = 'test-run-id'
        mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_start.return_value.__exit__ = MagicMock(return_value=False)

        config = load_pipeline_config('configs/pipeline_config.yaml')
        data_info = {'train_rows': 100, 'val_rows': 50,
                     'train_positive_rate': 0.3, 'val_positive_rate': 0.3}
        training_info = {
            'model_type': 'xgboost', 'training_features': 21,
            'training_samples': 100, 'training_time_seconds': 1.0,
            'model_config': {'n_estimators': 300, 'max_depth': 4,
                             'learning_rate': 0.05, 'random_state': 42}
        }
        val_metrics = {'val_roc_auc': 0.72}
        pipeline_metadata = {'timestamp': '20260101_120000'}

        track_in_mlflow(config, data_info, training_info,
                        val_metrics, pipeline_metadata)

        # First call should be the tags dict
        tags_call_args = mock_tags.call_args_list[0][0][0]
        self.assertEqual(tags_call_args['project'], 'equitable_student_dropout')
        self.assertEqual(tags_call_args['pipeline_step'], '2.14_automated_pipeline')


class TestDryRunSafety(unittest.TestCase):
    """Tests proving dry-run does not train, overwrite, or access locked data."""

    @patch('src.pipeline.automated_pipeline.pd.read_parquet')
    @patch('os.path.exists', return_value=True)
    def test_dry_run_does_not_train(self, mock_exists, mock_read_parquet):
        """Dry-run completes without calling train_model."""
        from src.pipeline.automated_pipeline import run_dry_run
        config = load_pipeline_config('configs/pipeline_config.yaml')
        
        # Synthetic data to pass quality validation
        df = _make_dummy_df(n_rows=20)
        mock_read_parquet.return_value = df

        with patch('src.pipeline.automated_pipeline.train_model') as mock_train:
            result = run_dry_run(config)
            mock_train.assert_not_called()

    @patch('src.pipeline.automated_pipeline.pd.read_parquet')
    @patch('os.path.exists', return_value=True)
    def test_dry_run_does_not_overwrite_existing_model(self, mock_exists, mock_read_parquet):
        """Dry-run does not write to the existing model artifact path."""
        from src.pipeline.automated_pipeline import run_dry_run
        config = load_pipeline_config('configs/pipeline_config.yaml')
        
        df = _make_dummy_df(n_rows=20)
        mock_read_parquet.return_value = df
        
        existing_model = config.get('existing_model_path',
                                     'models/baseline/xgb_pipeline.pkl')

        # Record modification time before dry run
        if os.path.exists(existing_model):
            mtime_before = os.path.getmtime(existing_model)

        run_dry_run(config)

        if os.path.exists(existing_model):
            mtime_after = os.path.getmtime(existing_model)
            self.assertEqual(mtime_before, mtime_after,
                             "Dry-run modified the existing model artifact!")

    @patch('os.path.exists', return_value=True)
    def test_dry_run_does_not_read_locked_datasets(self, mock_exists):
        """Dry-run does not attempt to read test.parquet or final_holdout.parquet."""
        from src.pipeline.automated_pipeline import run_dry_run
        config = load_pipeline_config('configs/pipeline_config.yaml')

        original_read_parquet = pd.read_parquet
        locked_paths = config.get('locked_datasets', [])
        accessed_locked = []

        def tracked_read_parquet(path, *args, **kwargs):
            path_str = str(path)
            for locked_p in locked_paths:
                if locked_p in path_str or os.path.basename(locked_p) in path_str:
                    accessed_locked.append(path_str)
            return _make_dummy_df(n_rows=20)

        with patch('src.pipeline.automated_pipeline.pd.read_parquet',
                   side_effect=tracked_read_parquet):
            run_dry_run(config)

        self.assertEqual(len(accessed_locked), 0,
                         f"Dry-run accessed locked datasets: {accessed_locked}")

    @patch('src.pipeline.automated_pipeline.pd.read_parquet')
    @patch('os.path.exists', return_value=True)
    def test_dry_run_returns_data_quality_info(self, mock_exists, mock_read_parquet):
        """Dry-run returns data quality info without error."""
        from src.pipeline.automated_pipeline import run_dry_run
        config = load_pipeline_config('configs/pipeline_config.yaml')
        
        df = _make_dummy_df(n_rows=20)
        mock_read_parquet.return_value = df
        
        result = run_dry_run(config)
        self.assertIn('train_rows', result)
        self.assertIn('val_rows', result)
        self.assertIn('feature_count', result)
        self.assertEqual(result['feature_count'], 21)


class TestMissingInputFailure(unittest.TestCase):
    """Tests for clear failure behavior when required inputs are missing."""

    def test_missing_model_config_fails(self):
        """Pipeline fails clearly if model_config.yaml is missing."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        with patch('os.path.exists') as mock_exists:
            def side_effect(path):
                if 'model_config' in path:
                    return False
                return os.path.exists.__wrapped__(path) if hasattr(os.path.exists, '__wrapped__') else True
            mock_exists.side_effect = side_effect
            with self.assertRaises(FileNotFoundError):
                validate_inputs(config)

    def test_missing_fairness_config_fails(self):
        """Pipeline fails clearly if fairness_config.yaml is missing."""
        config = load_pipeline_config('configs/pipeline_config.yaml')
        original_exists = os.path.exists

        def mock_exists(path):
            if 'fairness_config' in str(path):
                return False
            return original_exists(path)

        with patch('src.pipeline.automated_pipeline.os.path.exists',
                   side_effect=mock_exists):
            with self.assertRaises(FileNotFoundError):
                validate_inputs(config)


if __name__ == '__main__':
    unittest.main()
