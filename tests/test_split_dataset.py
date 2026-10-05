import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.split_dataset import get_splits, validate_feature_roles, build_linear_preprocessor

class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        # Create small synthetic dataframe
        np.random.seed(42)
        n_rows = 100
        self.df = pd.DataFrame({
            'id_student': np.random.randint(1, 30, n_rows),
            'code_module': np.random.choice(['A', 'B', 'C'], n_rows),
            'code_presentation': np.random.choice(['2013J', '2014B'], n_rows),
            'is_withdrawn': np.random.randint(0, 2, n_rows),
            'highest_education': np.random.choice(['HE', 'A Level', '?'], n_rows),
            'num_of_prev_attempts': np.random.randint(0, 3, n_rows),
            'studied_credits': np.random.randint(30, 120, n_rows),
            'asm_mean_score': [np.nan if i % 5 == 0 else 80.0 for i in range(n_rows)],
            'gender': ['M']*n_rows
        })
        
        # Ensure unique composite keys
        self.df = self.df.drop_duplicates(['id_student', 'code_module', 'code_presentation']).copy()
        
        self.config = {
            'n_splits': 7,
            'shuffle': True,
            'random_state': 42,
            'stratify_col': 'is_withdrawn',
            'group_col': 'id_student',
            'test_fold': 0,
            'validation_fold': 1
        }
        
    def test_deterministic_split(self):
        tr1, val1, te1 = get_splits(self.df, self.config)
        tr2, val2, te2 = get_splits(self.df, self.config)
        self.assertTrue(tr1.equals(tr2))
        
    def test_no_student_overlap(self):
        tr, val, te = get_splits(self.df, self.config)
        s_tr = set(tr['id_student'])
        s_val = set(val['id_student'])
        s_te = set(te['id_student'])
        
        self.assertEqual(len(s_tr.intersection(s_val)), 0)
        self.assertEqual(len(s_tr.intersection(s_te)), 0)
        self.assertEqual(len(s_val.intersection(s_te)), 0)
        
    def test_no_missing_rows(self):
        tr, val, te = get_splits(self.df, self.config)
        self.assertEqual(len(tr) + len(val) + len(te), len(self.df))
        
    def test_validate_feature_roles(self):
        good = {
            'highest_education', 'num_of_prev_attempts', 'studied_credits', 'registration_day',
            'code_module', 'code_presentation', 'module_presentation_length',
            'asm_submission_count', 'asm_mean_score', 'asm_score_std', 'asm_failed_count', 'asm_average_delay',
            'vle_total_clicks', 'vle_active_days', 'vle_days_since_last_activity',
            'vle_clicks_last_7_days', 'vle_clicks_last_14_days', 'vle_forum_clicks',
            'vle_resource_clicks', 'vle_quiz_clicks', 'vle_activity_type_diversity'
        }
        validate_feature_roles(list(good))
        
        with self.assertRaises(ValueError):
            validate_feature_roles(list(good) + ['gender'])
            
        with self.assertRaises(ValueError):
            validate_feature_roles(list(good) + ['is_withdrawn'])
            
    def test_preprocessing_fit_only_on_train(self):
        # We simulate the preprocessor behavior
        num_cols = ['num_of_prev_attempts', 'studied_credits', 'asm_mean_score']
        cat_cols = ['highest_education', 'code_module']
        
        prep = build_linear_preprocessor(num_cols, cat_cols)
        
        tr, val, te = get_splits(self.df, self.config)
        
        # We must fit ONLY on train
        tr_transformed = prep.fit_transform(tr)
        
        # Test transform on val works
        val_transformed = prep.transform(val)
        
        self.assertTrue(tr_transformed.shape[0] == len(tr))
        self.assertTrue(val_transformed.shape[0] == len(val))

if __name__ == "__main__":
    unittest.main()
