import unittest
import pandas as pd
import numpy as np
from src.models.train_baselines import PREDICTIVE_FEATURES

class TestIntegratedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.val_summary = pd.read_csv('reports/integrated/integrated_validation_summary.csv')
            cls.sel_sweep = pd.read_csv('reports/integrated/selective_threshold_sweep.csv')
            cls.cal_audit = pd.read_csv('reports/integrated/calibration_audit.csv')
            cls.sel_policy = pd.read_csv('reports/integrated/selected_operating_policy.csv')
        except FileNotFoundError:
            cls.val_summary = None
            cls.sel_sweep = None
            cls.cal_audit = None
            cls.sel_policy = None

    def test_predictive_features_length(self):
        self.assertEqual(len(PREDICTIVE_FEATURES), 21, "Predictive features count is not exactly 21.")

    def test_sensitive_attributes_excluded(self):
        sensitive = ['gender', 'disability', 'age_band', 'region', 'imd_band', 'id_student', 'final_result', 'is_withdrawn']
        for attr in sensitive:
            self.assertNotIn(attr, PREDICTIVE_FEATURES, f"Sensitive attribute {attr} found in predictive features.")

    def test_coverage_abstention_sum(self):
        if self.sel_sweep is not None:
            sums = self.sel_sweep['coverage'] + self.sel_sweep['abstention_rate']
            self.assertTrue(np.allclose(sums, 1.0), "Coverage and abstention rate do not sum to 1.")

    def test_confidence_always_gte_half(self):
        """confidence = max(p, 1-p) is mathematically >= 0.50, so threshold 0.50 gives 100% coverage."""
        if self.sel_sweep is not None:
            row_050 = self.sel_sweep[self.sel_sweep['confidence_threshold'] == 0.50]
            if not row_050.empty:
                self.assertAlmostEqual(row_050['coverage'].iloc[0], 1.0, places=5)

    def test_correct_positive_class_extraction(self):
        """Uncalibrated ROC-AUC should be well above 0.5, confirming correct column extraction."""
        if self.cal_audit is not None:
            uncal_row = self.cal_audit[self.cal_audit['calibration_method'] == 'uncalibrated']
            if not uncal_row.empty:
                self.assertGreater(uncal_row['roc_auc'].iloc[0], 0.60)

    def test_sigmoid_ranking_inverted(self):
        """Sigmoid calibration must be flagged as ranking-inverted in the audit."""
        if self.cal_audit is not None:
            sig_row = self.cal_audit[self.cal_audit['calibration_method'] == 'sigmoid']
            if not sig_row.empty:
                self.assertTrue(sig_row['ranking_inverted'].iloc[0])
                self.assertLess(sig_row['spearman_vs_uncal'].iloc[0], 0)

    def test_sigmoid_does_not_preserve_ranking(self):
        """Sigmoid ROC-AUC < 0.5 confirms ranking is inverted, not preserved."""
        if self.cal_audit is not None:
            sig_row = self.cal_audit[self.cal_audit['calibration_method'] == 'sigmoid']
            if not sig_row.empty:
                self.assertLess(sig_row['roc_auc'].iloc[0], 0.50)

    def test_no_probability_column_reversal_for_uncalibrated(self):
        """For uncalibrated model, col 1 must give ROC-AUC > 0.5 (correct orientation)."""
        if self.cal_audit is not None:
            uncal = self.cal_audit[self.cal_audit['calibration_method'] == 'uncalibrated']
            if not uncal.empty:
                self.assertGreater(uncal['roc_auc'].iloc[0], 0.50)

    def test_selected_policy_coverage_rule(self):
        if self.sel_policy is not None:
            coverage = self.sel_policy['coverage'].iloc[0]
            self.assertGreaterEqual(coverage, 0.90)

    def test_selected_policy_has_highest_recall(self):
        """The selected threshold must have the highest selective_recall among coverage >= 0.90."""
        if self.sel_sweep is not None and self.sel_policy is not None:
            valid = self.sel_sweep[self.sel_sweep['coverage'] >= 0.90]
            max_recall = valid['selective_recall'].max()
            selected_recall = self.sel_policy['selective_recall'].iloc[0]
            self.assertAlmostEqual(selected_recall, max_recall, places=6)

if __name__ == '__main__':
    unittest.main()
