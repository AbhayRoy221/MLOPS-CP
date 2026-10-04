import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.build_assessment_features import build_assessment_features

class TestAssessmentFeatures(unittest.TestCase):

    def setUp(self):
        # Base cohort
        self.df_cohort = pd.DataFrame({
            'id_student': [1, 2, 3],
            'code_module': ['A', 'A', 'A'],
            'code_presentation': ['B', 'B', 'B'],
            'registration_day': [-10, -5, pd.NA]
        })
        
        # Assessments
        self.df_assessments = pd.DataFrame({
            'id_assessment': [101, 102],
            'code_module': ['A', 'A'],
            'code_presentation': ['B', 'B'],
            'date': ['20', '?'] # Scheduled date
        })
        
        # Student Assessment
        self.df_student_assessment = pd.DataFrame({
            'id_student': [1, 1, 2, 2, 4], # 4 is not in cohort
            'id_assessment': [101, 102, 101, 102, 101],
            'date_submitted': ['15', '29', '-12', '10', '15'],
            'score': ['80', '90', '35', '?', '100']
        })

    @patch('src.data.build_assessment_features.pd.read_parquet')
    @patch('src.data.build_assessment_features.pd.read_csv')
    @patch('src.data.build_assessment_features.os.path.exists')
    @patch('src.data.build_assessment_features.os.makedirs')
    @patch('src.data.build_assessment_features.pd.DataFrame.to_parquet')
    def test_assessment_features(self, mock_to_parquet, mock_makedirs, mock_exists, mock_read_csv, mock_read_parquet):
        mock_exists.return_value = True
        mock_read_parquet.return_value = self.df_cohort
        
        def mock_csv(path):
            if "assessments.csv" in path:
                return self.df_assessments
            elif "studentAssessment.csv" in path:
                return self.df_student_assessment
            return pd.DataFrame()
            
        mock_read_csv.side_effect = mock_csv
        
        df_out = build_assessment_features()
        
        # 1. Output row count equals base cohort
        self.assertEqual(len(df_out), 3)
        
        # 2. No-assessment student (Student 3)
        s3 = df_out[df_out['id_student'] == 3].iloc[0]
        self.assertEqual(s3['asm_submission_count'], 0)
        self.assertTrue(pd.isna(s3['asm_mean_score']))
        self.assertTrue(pd.isna(s3['asm_score_std']))
        self.assertEqual(s3['asm_failed_count'], 0)
        
        # 3. Exclude > 28 submissions
        s1 = df_out[df_out['id_student'] == 1].iloc[0]
        # Student 1 has sub on 15 and 29. The 29 should be ignored.
        self.assertEqual(s1['asm_submission_count'], 1)
        self.assertEqual(s1['asm_mean_score'], 80)
        self.assertEqual(s1['asm_average_delay'], -5) # 15 - 20
        self.assertEqual(s1['assessment_submission_before_registration'], 0) # 15 is not < -10
        
        # 4. Failed assessment count and single-obs score std and pre-reg anomaly
        s2 = df_out[df_out['id_student'] == 2].iloc[0]
        # Student 2 has sub on -12 (score 35) and 10 (score ?). Both <= 28.
        self.assertEqual(s2['asm_submission_count'], 2)
        self.assertEqual(s2['asm_failed_count'], 1) # score 35
        self.assertEqual(s2['asm_mean_score'], 35.0) # ? score is coerced to NaN
        self.assertTrue(pd.isna(s2['asm_score_std'])) # only 1 valid score
        # Delay for id=101 is -12 - 20 = -32. For id=102, scheduled is '?', so NaN. Mean is -32
        self.assertEqual(s2['asm_average_delay'], -32)
        self.assertEqual(s2['assessment_submission_before_registration'], 1) # -12 < -5

if __name__ == "__main__":
    unittest.main()
