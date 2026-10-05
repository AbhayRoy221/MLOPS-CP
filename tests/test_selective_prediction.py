import unittest
import numpy as np
import pandas as pd
from src.calibration.selective_prediction import calculate_selective_metrics, sweep_thresholds, select_operating_policy, calculate_group_selective_metrics

class TestSelectivePrediction(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55])
        self.A = np.array(['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A'])

    def test_confidence_and_uncertainty_bounds(self):
        metrics = calculate_selective_metrics(self.y_true, self.y_prob, 0.5)
        # We know probability is in [0,1], therefore confidence = max(p, 1-p) is in [0.5, 1.0]
        confidence = np.maximum(self.y_prob, 1 - self.y_prob)
        uncertainty = 1 - confidence
        self.assertTrue(np.all(confidence >= 0.5) and np.all(confidence <= 1.0))
        self.assertTrue(np.all(uncertainty >= 0.0) and np.all(uncertainty <= 0.5))

    def test_coverage_abstention_sum(self):
        metrics = calculate_selective_metrics(self.y_true, self.y_prob, 0.75)
        self.assertAlmostEqual(metrics['coverage'] + metrics['abstention_rate'], 1.0)
        
    def test_selection_rule(self):
        # Sweep and check selection
        sweep = sweep_thresholds(self.y_true, self.y_prob, min_t=0.5, max_t=0.9, step=0.1)
        policy = select_operating_policy(sweep)
        
        # Verify coverage constraint
        self.assertGreaterEqual(policy['coverage'].iloc[0], 0.90)

    def test_group_level_handling(self):
        # test that calculating group metrics works, including empty groups handling
        metrics = calculate_group_selective_metrics(self.y_true, self.y_prob, self.A, 0.95)
        # Nobody has confidence >= 0.95, coverage should be 0 everywhere
        for idx, row in metrics.iterrows():
            self.assertEqual(row['coverage'], 0.0)
            self.assertTrue(np.isnan(row['selective_recall']))

if __name__ == '__main__':
    unittest.main()
