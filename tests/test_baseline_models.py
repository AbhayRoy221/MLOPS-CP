import unittest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.train_baselines import PREDICTIVE_FEATURES, get_models

class TestBaselineModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_rows = 50
        
        self.df = pd.DataFrame({f: np.random.randn(n_rows) for f in PREDICTIVE_FEATURES})
        
        # Override categoricals
        self.df['highest_education'] = np.random.choice(['HE', 'A Level', '?'], n_rows)
        self.df['code_module'] = np.random.choice(['A', 'B'], n_rows)
        self.df['code_presentation'] = np.random.choice(['2013J', '2014B'], n_rows)
        
        # Add forbidden cols
        self.df['is_withdrawn'] = np.random.randint(0, 2, n_rows)
        self.df['gender'] = ['M'] * n_rows
        
        self.config = {
            'random_state': 42,
            'logistic_regression': {'max_iter': 10},
            'random_forest': {'n_estimators': 10, 'n_jobs': 1},
            'xgboost': {
                'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1,
                'subsample': 1.0, 'colsample_bytree': 1.0, 'eval_metric': 'logloss', 'n_jobs': 1
            }
        }
        
    def test_predictive_features(self):
        self.assertEqual(len(PREDICTIVE_FEATURES), 21)
        forbidden = {'gender', 'disability', 'is_withdrawn'}
        for f in PREDICTIVE_FEATURES:
            self.assertNotIn(f, forbidden)
            
    def test_pipelines(self):
        models = get_models(self.config)
        X = self.df[PREDICTIVE_FEATURES]
        y = self.df['is_withdrawn']
        
        for name, model in models.items():
            self.assertIsInstance(model, Pipeline)
            
            # Fit works
            model.fit(X, y)
            
            # Predict binary
            preds = model.predict(X)
            self.assertTrue(set(preds).issubset({0, 1}))
            
            # Predict probs in [0,1]
            probs = model.predict_proba(X)
            self.assertTrue((probs >= 0).all() and (probs <= 1).all())

if __name__ == "__main__":
    unittest.main()
