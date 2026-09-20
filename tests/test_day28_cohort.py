import unittest
import pandas as pd
import io

class TestDay28Cohort(unittest.TestCase):
    def test_day28_rules(self):
        csv_info = "id_student,code_module,code_presentation,final_result,gender,disability,age_band,region,imd_band,highest_education,num_of_prev_attempts,studied_credits\n1,A,B,Withdrawn,M,N,1,1,?,HE,0,60\n2,A,B,Pass,M,N,1,1,20,HE,0,60\n3,A,B,Withdrawn,M,N,1,1,20,HE,0,60\n"
        csv_reg = "id_student,code_module,code_presentation,date_registration,date_unregistration\n1,A,B,?,?\n2,A,B,-10,?\n3,A,B,-10,15\n"
        df_info = pd.read_csv(io.StringIO(csv_info))
        df_reg = pd.read_csv(io.StringIO(csv_reg))
        
        df = pd.merge(df_info, df_reg, on=['id_student', 'code_module', 'code_presentation'], how='left')
        df['is_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)
        
        # Test missing socioeconomic logic
        df['missing_socioeconomic_information'] = (df['imd_band'] == '?').astype(int)
        self.assertEqual(df.loc[df['id_student']==1, 'missing_socioeconomic_information'].iloc[0], 1)
        self.assertEqual(df.loc[df['id_student']==2, 'missing_socioeconomic_information'].iloc[0], 0)
        
        # Drop logic
        df_unreg_valid = df[df['date_unregistration'] != '?'].copy()
        df_unreg_valid['date_unregistration'] = df_unreg_valid['date_unregistration'].astype(float)
        early_withdrawn_mask = (df['date_unregistration'] != '?') & (df_unreg_valid['date_unregistration'] <= 28)
        missing_unreg_withdrawn_mask = (df['is_withdrawn'] == 1) & (df['date_unregistration'] == '?')
        inconsistent_nw_mask = (df['is_withdrawn'] == 0) & (df['date_unregistration'] != '?')
        
        drop_mask = early_withdrawn_mask | missing_unreg_withdrawn_mask | inconsistent_nw_mask
        df_cohort = df[~drop_mask].copy()
        
        # Student 1: Withdrawn + null unreg -> Dropped
        # Student 2: Pass + null unreg -> Kept
        # Student 3: Withdrawn + unreg(15) <= 28 -> Dropped
        self.assertEqual(len(df_cohort), 1)
        self.assertEqual(df_cohort.iloc[0]['id_student'], 2)

if __name__ == "__main__":
    unittest.main()
