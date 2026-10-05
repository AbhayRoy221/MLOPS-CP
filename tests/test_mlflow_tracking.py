import unittest
import pandas as pd
import numpy as np
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.mlflow_tracking import initialize_mlflow, start_run, add_tags
from src.models.train_baselines import PREDICTIVE_FEATURES

class TestMLflowTracking(unittest.TestCase):
    def setUp(self):
        self.tracking_uri = "sqlite:///test_mlflow.db"
        self.experiment_name = "test_experiment"
        initialize_mlflow(self.tracking_uri, self.experiment_name)
        
    def tearDown(self):
        try:
            if os.path.exists("test_mlflow.db"):
                os.remove("test_mlflow.db")
        except PermissionError:
            pass
            
    def test_experiment_creation(self):
        client = MlflowClient(self.tracking_uri)
        exp = client.get_experiment_by_name(self.experiment_name)
        self.assertIsNotNone(exp)
        
    def test_run_creation(self):
        with start_run("test_run") as run:
            self.assertIsNotNone(run.info.run_id)
            
    def test_tags(self):
        with start_run("test_tags") as run:
            add_tags({'project': 'equitable_student_dropout'})
            client = MlflowClient(self.tracking_uri)
            r = client.get_run(run.info.run_id)
            self.assertEqual(r.data.tags.get('project'), 'equitable_student_dropout')
            
    def test_feature_list_is_21(self):
        self.assertEqual(len(PREDICTIVE_FEATURES), 21)
        
    def test_fairness_attributes_excluded(self):
        forbidden = {'gender', 'disability', 'age_band', 'region', 'imd_band'}
        for f in PREDICTIVE_FEATURES:
            self.assertNotIn(f, forbidden)

if __name__ == "__main__":
    unittest.main()
