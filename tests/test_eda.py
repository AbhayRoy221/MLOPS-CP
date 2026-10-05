import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.eda import run_eda

class TestEDA(unittest.TestCase):
    
    @patch('src.data.eda.pd.read_parquet')
    @patch('src.data.eda.plt.savefig')
    @patch('src.data.eda.pd.DataFrame.to_csv')
    def test_eda(self, mock_to_csv, mock_savefig, mock_read_parquet):
        # Create a mock dataset exactly matching schema 27444
        df = pd.DataFrame({
            'id_student': range(27444),
            'code_module': ['A']*27444,
            'code_presentation': ['B']*27444,
            'highest_education': ['HE']*27444,
            'num_of_prev_attempts': [0]*27444,
            'studied_credits': [60]*27444,
            'registration_day': [-10]*27444,
            'module_presentation_length': [260]*27444,
            'asm_submission_count': [1]*27444,
            'asm_mean_score': [80]*27444,
            'asm_score_std': [5]*27444,
            'asm_failed_count': [0]*27444,
            'asm_average_delay': [0]*27444,
            'vle_total_clicks': [100]*27444,
            'vle_active_days': [5]*27444,
            'vle_days_since_last_activity': [2]*27444,
            'vle_clicks_last_7_days': [10]*27444,
            'vle_clicks_last_14_days': [20]*27444,
            'vle_forum_clicks': [30]*27444,
            'vle_resource_clicks': [40]*27444,
            'vle_quiz_clicks': [30]*27444,
            'vle_activity_type_diversity': [3]*27444,
            'gender': ['M']*27444,
            'disability': ['N']*27444,
            'age_band': ['0-35']*27444,
            'region': ['East']*27444,
            'imd_band': ['0-10%']*27444,
            'is_withdrawn': [0]*20000 + [1]*7444,
            'final_result': ['Pass']*27444,
            'missing_registration': [0]*27444,
            'missing_socioeconomic_information': [0]*27444,
            'assessment_submission_before_registration': [0]*27444,
            'no_vle_activity_by_day28': [0]*27444
        })
        
        mock_read_parquet.return_value = df
        
        # Test runs without raising errors
        run_eda()
        
        self.assertTrue(mock_to_csv.called)
        self.assertTrue(mock_savefig.called)

if __name__ == "__main__":
    unittest.main()
