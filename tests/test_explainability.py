import unittest
import pandas as pd
import numpy as np
from src.explainability.shap_explainer import SHAPExplainer
from src.models.train_baselines import PREDICTIVE_FEATURES

class TestExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.explainer = SHAPExplainer()
        
        # Load a few validation rows for testing
        val_df = pd.read_parquet('data/processed/splits/validation.parquet')
        cls.X_test = val_df[PREDICTIVE_FEATURES].head(5).copy()
        num_cols = [c for c in PREDICTIVE_FEATURES if c not in ['highest_education', 'code_module', 'code_presentation']]
        for c in num_cols:
            cls.X_test[c] = pd.to_numeric(cls.X_test[c], errors='coerce')

    def test_explainer_loads_model(self):
        self.assertIsNotNone(self.explainer.pipeline)
        self.assertIsNotNone(self.explainer.model)
        self.assertIsNotNone(self.explainer.explainer)
        
    def test_validation_input_produces_probability(self):
        probs = self.explainer.pipeline.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        
    def test_shap_output_dimension(self):
        shap_values, feature_names = self.explainer.explain_instances(self.X_test)
        self.assertEqual(shap_values.shape[0], 5)
        self.assertEqual(shap_values.shape[1], len(feature_names))
        
    def test_feature_names_align(self):
        _, feature_names = self.explainer.explain_instances(self.X_test)
        self.assertGreater(len(feature_names), 20) # should include one-hot encoded cols
        
    def test_local_explanation(self):
        row = self.X_test.iloc[[0]]
        explanation = self.explainer.get_local_explanation(row, num_features=3)
        self.assertIn('risk_probability', explanation)
        self.assertIn('risk_tier', explanation)
        self.assertIn('top_positive_contributors', explanation)
        self.assertIn('top_negative_contributors', explanation)
        self.assertTrue(isinstance(explanation['top_positive_contributors'], list))
        
    def test_no_id_student_in_predictive(self):
        _, feature_names = self.explainer.explain_instances(self.X_test)
        self.assertNotIn('id_student', feature_names)
        
    def test_no_fairness_attributes(self):
        _, feature_names = self.explainer.explain_instances(self.X_test)
        for attr in ['gender', 'disability', 'age_band', 'region', 'imd_band']:
            self.assertNotIn(attr, feature_names)
            
if __name__ == '__main__':
    unittest.main()
