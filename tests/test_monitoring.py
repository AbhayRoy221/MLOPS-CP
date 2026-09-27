import unittest
import pandas as pd
import numpy as np
from src.monitoring.prediction_monitor import summarize_predictions
from src.monitoring.drift_detector import calculate_psi, calculate_tvd, classify_drift, detect_drift
from src.monitoring.fairness_monitor import summarize_fairness
from fastapi.testclient import TestClient
from src.api.main import app
import warnings

class TestMonitoring(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings('ignore')
        
    def test_risk_tier_assignment_and_summary(self):
        df = pd.DataFrame({
            'risk_probability': [0.1, 0.25, 0.4, 0.6],
            'risk_tier': ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'],
            'human_review_required': [False, False, True, True]
        })
        
        summary = summarize_predictions(df)
        self.assertEqual(summary['total_predictions'], 4)
        self.assertEqual(summary['risk_tier_distribution']['LOW'], 1)
        self.assertEqual(summary['alert_review_rate'], 0.5)
        
    def test_empty_prediction_batch(self):
        df = pd.DataFrame(columns=['risk_probability', 'risk_tier', 'human_review_required'])
        summary = summarize_predictions(df)
        self.assertEqual(summary['total_predictions'], 0)
        self.assertIsNone(summary['alert_review_rate'])
        
    def test_numerical_psi(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 1000))
        cur = pd.Series(np.random.normal(0.1, 1, 1000))
        psi = calculate_psi(ref, cur)
        self.assertGreaterEqual(psi, 0.0)
        
    def test_categorical_tvd(self):
        ref = pd.Series(['A', 'A', 'B'])
        cur = pd.Series(['A', 'B', 'B'])
        tvd = calculate_tvd(ref, cur)
        self.assertAlmostEqual(tvd, 0.33333333333)
        
    def test_constant_feature_handling(self):
        ref = pd.Series([1.0]*100)
        cur = pd.Series([1.0]*100)
        self.assertEqual(calculate_psi(ref, cur), 0.0)
        
    def test_empty_small_batch_handling(self):
        ref = pd.Series([])
        cur = pd.Series([])
        self.assertEqual(calculate_psi(ref, cur), 0.0)
        self.assertEqual(calculate_tvd(ref, cur), 0.0)
        
    def test_missing_value_handling(self):
        ref = pd.Series([1.0, 2.0, np.nan])
        cur = pd.Series([1.0, 2.0, 2.0])
        # PSI should not crash
        psi = calculate_psi(ref, cur)
        self.assertIsInstance(psi, float)
        
    def test_drift_threshold_classification(self):
        self.assertEqual(classify_drift(0.05, 'psi'), 'no meaningful drift')
        self.assertEqual(classify_drift(0.15, 'psi'), 'moderate drift')
        self.assertEqual(classify_drift(0.3, 'psi'), 'substantial drift')
        self.assertEqual(classify_drift(0.01, 'tvd'), 'no meaningful drift')
        self.assertEqual(classify_drift(0.1, 'tvd'), 'moderate drift')
        self.assertEqual(classify_drift(0.2, 'tvd'), 'substantial drift')
        
    def test_fairness_monitoring_valid_groups(self):
        df = pd.DataFrame({
            'human_review_required': [True]*40 + [False]*40,
            'gender': ['M']*40 + ['F']*40,
            'is_withdrawn': [1]*40 + [0]*40
        })
        summary = summarize_fairness(df, sensitive_attrs=['gender'], target_col='is_withdrawn')
        self.assertIn('gender', summary)
        self.assertEqual(summary['gender']['groups']['M']['selection_rate'], 1.0)
        self.assertEqual(summary['gender']['groups']['F']['selection_rate'], 0.0)
        
    def test_fairness_monitoring_small_groups(self):
        df = pd.DataFrame({
            'human_review_required': [True]*5 + [False]*5,
            'gender': ['M']*5 + ['F']*5,
        })
        summary = summarize_fairness(df, sensitive_attrs=['gender'])
        self.assertIn('size < 30', summary['gender']['groups']['M']['note'])
        
    def test_prometheus_metrics_endpoint(self):
        client = TestClient(app)
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"student_dropout_prediction_requests_total", response.content)
        
    def test_prediction_request_increments(self):
        client = TestClient(app)
        
        # Valid payload matching 21 features
        valid_payload = {
            "highest_education": "A Level or Equivalent",
            "num_of_prev_attempts": 0,
            "studied_credits": 60,
            "registration_day": -10,
            "code_module": "AAA",
            "code_presentation": "2013J",
            "module_presentation_length": 268,
            "asm_submission_count": 2,
            "asm_mean_score": 80.5,
            "asm_score_std": 5.0,
            "asm_failed_count": 0,
            "asm_average_delay": 0.5,
            "vle_total_clicks": 150,
            "vle_active_days": 20,
            "vle_days_since_last_activity": 2,
            "vle_clicks_last_7_days": 50,
            "vle_clicks_last_14_days": 100,
            "vle_forum_clicks": 20,
            "vle_resource_clicks": 100,
            "vle_quiz_clicks": 30,
            "vle_activity_type_diversity": 5
        }
        
        # Make a request
        response = client.post("/predict", json=valid_payload)
        self.assertEqual(response.status_code, 200)
        
        # Check metrics
        metrics_response = client.get("/metrics")
        self.assertIn(b"student_dropout_prediction_requests_total", metrics_response.content)

if __name__ == '__main__':
    unittest.main()
