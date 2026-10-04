import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.build_vle_features import build_vle_features

class TestVleFeatures(unittest.TestCase):

    def setUp(self):
        self.cohort_data = pd.DataFrame({
            'id_student': [1, 2, 3],
            'code_module': ['A', 'A', 'A'],
            'code_presentation': ['B', 'B', 'B']
        })
        
        self.vle_data = pd.DataFrame({
            'id_site': [101, 102, 103, 104],
            'code_module': ['A', 'A', 'A', 'A'],
            'code_presentation': ['B', 'B', 'B', 'B'],
            'activity_type': ['forumng', 'resource', 'quiz', 'externalquiz']
        })

    def run_build_with_mocks(self, mock_vle=None, mock_student_vle_chunks=None, mock_cohort=None):
        if mock_vle is None: mock_vle = self.vle_data
        if mock_cohort is None: mock_cohort = self.cohort_data
        
        with patch('src.data.build_vle_features.pd.read_parquet') as mock_read_parquet, \
             patch('src.data.build_vle_features.pd.read_csv') as mock_read_csv, \
             patch('src.data.build_vle_features.os.path.exists') as mock_exists, \
             patch('src.data.build_vle_features.os.makedirs'), \
             patch('src.data.build_vle_features.pd.DataFrame.to_parquet'):
            
            mock_exists.return_value = True
            mock_read_parquet.return_value = mock_cohort
            
            def mock_csv(path, **kwargs):
                if "vle.csv" in path:
                    return mock_vle
                elif "studentVle.csv" in path:
                    return mock_student_vle_chunks
                return pd.DataFrame()
                
            mock_read_csv.side_effect = mock_csv
            
            return build_vle_features()

    def test_explicit_date_boundaries(self):
        # 14, 15, 21, 22, 28, 29
        student_vle = pd.DataFrame({
            'id_student': [1]*6,
            'code_module': ['A']*6,
            'code_presentation': ['B']*6,
            'id_site': [101]*6,
            'date': [14, 15, 21, 22, 28, 29],
            'sum_click': [1]*6
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[student_vle])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        # Total clicks <= 28
        self.assertEqual(s1['vle_total_clicks'], 5) # 14, 15, 21, 22, 28 included. 29 excluded.
        
        # Last 7 days: 21 < date <= 28
        self.assertEqual(s1['vle_clicks_last_7_days'], 2) # 22, 28
        
        # Last 14 days: 14 < date <= 28
        self.assertEqual(s1['vle_clicks_last_14_days'], 4) # 15, 21, 22, 28

    def test_negative_precourse_dates(self):
        student_vle = pd.DataFrame({
            'id_student': [1]*4,
            'code_module': ['A']*4,
            'code_presentation': ['B']*4,
            'id_site': [101, 102, 103, 101],
            'date': [-2, -1, 0, 1],
            'sum_click': [2, 3, 5, 1]
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[student_vle])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        self.assertEqual(s1['vle_total_clicks'], 11)
        self.assertEqual(s1['vle_active_days'], 4)
        self.assertEqual(s1['vle_activity_type_diversity'], 3) # 101, 102, 103 -> forumng, resource, quiz

    def test_multi_chunk_active_days(self):
        chunk1 = pd.DataFrame({
            'id_student': [1]*3, 'code_module': ['A']*3, 'code_presentation': ['B']*3,
            'id_site': [101]*3, 'date': [1, 2, 3], 'sum_click': [1]*3
        })
        chunk2 = pd.DataFrame({
            'id_student': [1]*3, 'code_module': ['A']*3, 'code_presentation': ['B']*3,
            'id_site': [101]*3, 'date': [2, 3, 4], 'sum_click': [1]*3
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[chunk1, chunk2])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        # Active days should be exactly 4 (1, 2, 3, 4)
        self.assertEqual(s1['vle_active_days'], 4)

    def test_multi_chunk_activity_diversity(self):
        chunk1 = pd.DataFrame({
            'id_student': [1]*2, 'code_module': ['A']*2, 'code_presentation': ['B']*2,
            'id_site': [102, 101], # resource, forumng
            'date': [1, 2], 'sum_click': [1, 1]
        })
        chunk2 = pd.DataFrame({
            'id_student': [1]*2, 'code_module': ['A']*2, 'code_presentation': ['B']*2,
            'id_site': [101, 103], # forumng, quiz
            'date': [3, 4], 'sum_click': [1, 1]
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[chunk1, chunk2])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        # Diversity should be 3 (resource, forumng, quiz)
        self.assertEqual(s1['vle_activity_type_diversity'], 3)

    def test_metadata_duplicate_rejection(self):
        vle_dup = pd.DataFrame({
            'id_site': [101, 101],
            'code_module': ['A', 'A'],
            'code_presentation': ['B', 'B'],
            'activity_type': ['forumng', 'resource']
        })
        
        with self.assertRaisesRegex(ValueError, "VLE metadata key is not unique"):
            self.run_build_with_mocks(mock_vle=vle_dup)

    def test_output_key_uniqueness(self):
        original_duplicated = pd.DataFrame.duplicated
        def side_effect(self_obj, subset=None, keep='first'):
            if subset == ['id_student', 'code_module', 'code_presentation'] and 'no_vle_activity_by_day28' in self_obj.columns:
                return pd.Series([True] * len(self_obj), index=self_obj.index)
            return original_duplicated(self_obj, subset=subset, keep=keep)
            
        with patch('src.data.build_vle_features.pd.DataFrame.duplicated', side_effect=side_effect, autospec=True):
            with self.assertRaisesRegex(ValueError, "Duplicate composite keys"):
                self.run_build_with_mocks(mock_student_vle_chunks=[])

    def test_post_cutoff_leakage(self):
        student_vle = pd.DataFrame({
            'id_student': [1, 1],
            'code_module': ['A', 'A'],
            'code_presentation': ['B', 'B'],
            'id_site': [101, 101],
            'date': [28, 29],
            'sum_click': [10, 1000]
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[student_vle])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        self.assertEqual(s1['vle_total_clicks'], 10)
        self.assertEqual(s1['vle_active_days'], 1)
        self.assertEqual(s1['vle_clicks_last_7_days'], 10)
        self.assertEqual(s1['vle_clicks_last_14_days'], 10)
        self.assertEqual(s1['vle_forum_clicks'], 10)
        self.assertEqual(s1['vle_activity_type_diversity'], 1)
        self.assertEqual(s1['vle_days_since_last_activity'], 0) # 28 - 28

    def test_zero_activity_student(self):
        student_vle = pd.DataFrame({
            'id_student': [1], 'code_module': ['A'], 'code_presentation': ['B'],
            'id_site': [101], 'date': [10], 'sum_click': [5]
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[student_vle])
        # Student 2 is in cohort but has no activity
        s2 = df_out[df_out['id_student'] == 2].iloc[0]
        
        self.assertEqual(s2['vle_total_clicks'], 0)
        self.assertEqual(s2['vle_active_days'], 0)
        self.assertTrue(pd.isna(s2['vle_days_since_last_activity']))
        self.assertEqual(s2['vle_clicks_last_7_days'], 0)
        self.assertEqual(s2['vle_clicks_last_14_days'], 0)
        self.assertEqual(s2['vle_forum_clicks'], 0)
        self.assertEqual(s2['vle_resource_clicks'], 0)
        self.assertEqual(s2['vle_quiz_clicks'], 0)
        self.assertEqual(s2['vle_activity_type_diversity'], 0)
        self.assertEqual(s2['no_vle_activity_by_day28'], 1)

    def test_activity_type_separation(self):
        student_vle = pd.DataFrame({
            'id_student': [1]*4,
            'code_module': ['A']*4,
            'code_presentation': ['B']*4,
            'id_site': [101, 102, 103, 104], # forumng, resource, quiz, externalquiz
            'date': [10]*4,
            'sum_click': [100, 200, 300, 400]
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[student_vle])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        self.assertEqual(s1['vle_forum_clicks'], 100)
        self.assertEqual(s1['vle_resource_clicks'], 200)
        self.assertEqual(s1['vle_quiz_clicks'], 300)
        # 400 clicks from externalquiz should not bleed into quiz
        # The sum of forum+resource+quiz is 600, but total clicks is 1000
        self.assertEqual(s1['vle_total_clicks'], 1000)

    def test_last_activity_feature(self):
        student_vle = pd.DataFrame({
            'id_student': [1]*3,
            'code_module': ['A']*3,
            'code_presentation': ['B']*3,
            'id_site': [101]*3,
            'date': [10, 15, 20],
            'sum_click': [1]*3
        })
        
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[student_vle])
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        
        self.assertEqual(s1['vle_days_since_last_activity'], 8) # 28 - 20

    def test_column_contract(self):
        df_out = self.run_build_with_mocks(mock_student_vle_chunks=[])
        expected_cols = [
            'id_student', 'code_module', 'code_presentation',
            'vle_total_clicks', 'vle_active_days', 'vle_days_since_last_activity',
            'vle_clicks_last_7_days', 'vle_clicks_last_14_days',
            'vle_forum_clicks', 'vle_resource_clicks', 'vle_quiz_clicks',
            'vle_activity_type_diversity', 'no_vle_activity_by_day28'
        ]
        self.assertListEqual(list(df_out.columns), expected_cols)

if __name__ == "__main__":
    unittest.main()
