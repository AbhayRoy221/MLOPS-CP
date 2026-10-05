import unittest
import numpy as np
import pandas as pd
import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.train_baselines import PREDICTIVE_FEATURES
from src.fairness.fairness_metrics import calculate_group_metrics, calculate_disparity_metrics

class TestFairnessAudit(unittest.TestCase):
    def setUp(self):
        with open('configs/fairness_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)['fairness']
            
    def test_all_5_sensitive_attributes_audited(self):
        sens_attrs = self.config['sensitive_attributes']
        self.assertEqual(len(sens_attrs), 5)
        self.assertIn('gender', sens_attrs)
        self.assertIn('disability', sens_attrs)
        self.assertIn('age_band', sens_attrs)
        self.assertIn('region', sens_attrs)
        self.assertIn('imd_band', sens_attrs)
        
    def test_fairness_attributes_absent_from_model_X(self):
        sens_attrs = self.config['sensitive_attributes']
        forbidden = set(sens_attrs).union({
            'id_student', 'final_result', 'is_withdrawn', 
            'date_unregistration', 'missing_registration', 
            'missing_socioeconomic_information', 
            'assessment_submission_before_registration', 
            'no_vle_activity_by_day28'
        })
        for f in PREDICTIVE_FEATURES:
            self.assertNotIn(f, forbidden)
            
    def test_group_counts_sum_correctly_and_metrics_valid(self):
        # Dummy data
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        sensitive_features = pd.DataFrame({
            'gender': ['M', 'M', 'F', 'F', 'M', 'F']
        })
        
        df = calculate_group_metrics(y_true, y_pred, sensitive_features)
        
        # M group: indices 0, 1, 4 -> y_true: [0, 1, 1], y_pred: [0, 0, 1]
        # F group: indices 2, 3, 5 -> y_true: [0, 1, 0], y_pred: [1, 1, 0]
        
        self.assertEqual(df['group_count'].sum(), 6)
        
        for _, row in df.iterrows():
            self.assertEqual(row['positive_count'] + row['negative_count'], row['group_count'])
            if row['group_count'] > 0:
                self.assertTrue(0 <= row['selection_rate'] <= 1)
                
    def test_disparity_metrics_valid(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        sensitive_features = pd.DataFrame({
            'gender': ['M', 'M', 'F', 'F', 'M', 'F']
        })
        
        disp = calculate_disparity_metrics(y_true, y_pred, sensitive_features)
        metrics = disp['gender']
        
        if not np.isnan(metrics['demographic_parity_difference']):
            self.assertGreaterEqual(metrics['demographic_parity_difference'], 0)
        
        if not np.isnan(metrics['demographic_parity_ratio']):
            self.assertTrue(0 <= metrics['demographic_parity_ratio'] <= 1)
            
    def test_missing_unknown_sensitive_values(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        sensitive_features = pd.DataFrame({
            'imd_band': ['0-10%', np.nan, '?']
        })
        
        df = calculate_group_metrics(y_true, y_pred, sensitive_features)
        groups = df['group'].tolist()
        self.assertIn('NaN', groups)
        self.assertIn('?', groups)
        
    def test_reproducibility(self):
        # We ensure calculation is deterministic on same data
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        sensitive_features = pd.DataFrame({
            'gender': ['M', 'M', 'F', 'F', 'M', 'F']
        })
        
        df1 = calculate_group_metrics(y_true, y_pred, sensitive_features)
        df2 = calculate_group_metrics(y_true, y_pred, sensitive_features)
        pd.testing.assert_frame_equal(df1, df2)

if __name__ == "__main__":
    unittest.main()
