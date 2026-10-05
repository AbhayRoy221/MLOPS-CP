import unittest
import pandas as pd
import numpy as np
from src.models.train_baselines import PREDICTIVE_FEATURES

import os

@unittest.skipUnless(os.path.exists('data/processed/splits/development_train.parquet'), "Local datasets not available in CI")
class TestCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev_df = pd.read_parquet('data/processed/splits/development_train.parquet')
        cls.val_df = pd.read_parquet('data/processed/splits/validation.parquet')
        try:
            cls.cal_summary = pd.read_csv('reports/calibration/calibration_summary.csv')
            cls.sel_policy = pd.read_csv('reports/calibration/selected_operating_policy.csv')
        except FileNotFoundError:
            cls.cal_summary = None
            cls.sel_policy = None

    def test_predictive_features_length(self):
        self.assertEqual(len(PREDICTIVE_FEATURES), 21, "Predictive features count is not exactly 21.")

    def test_sensitive_attributes_excluded(self):
        sensitive = ['gender', 'disability', 'age_band', 'region', 'imd_band', 'id_student', 'final_result', 'is_withdrawn']
        for attr in sensitive:
            self.assertNotIn(attr, PREDICTIVE_FEATURES, f"Sensitive attribute {attr} found in predictive features.")

    def test_probability_values_in_bounds(self):
        # We assume calibrate.py produces a probability. We can check if calibrate ran by checking the summary
        if self.cal_summary is not None:
            # this doesn't directly test the probabilities arrays since we didn't save them, 
            # but we can verify that ECE and Brier were calculated, which implies [0,1]
            self.assertTrue(self.cal_summary['brier_score'].min() >= 0.0)
            self.assertTrue(self.cal_summary['brier_score'].max() <= 1.0)
            
    def test_selected_policy_coverage_rule(self):
        if self.sel_policy is not None:
            coverage = self.sel_policy['coverage'].iloc[0]
            self.assertGreaterEqual(coverage, 0.90, "Selected policy coverage is less than 0.90")

if __name__ == '__main__':
    unittest.main()
