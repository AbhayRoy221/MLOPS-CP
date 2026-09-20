import unittest
import pandas as pd
import io

class TestRawValidation(unittest.TestCase):
    def test_duplicate_rejection(self):
        csv_data = "id_student,code_module,code_presentation,final_result\n1,A,B,Pass\n1,A,B,Fail\n"
        df = pd.read_csv(io.StringIO(csv_data))
        dups = df.duplicated(subset=['id_student', 'code_module', 'code_presentation'])
        self.assertTrue(dups.any())
        
    def test_invalid_target(self):
        csv_data = "id_student,code_module,code_presentation,final_result\n1,A,B,Invalid\n"
        df = pd.read_csv(io.StringIO(csv_data))
        valid_targets = {"Pass", "Withdrawn", "Fail", "Distinction"}
        self.assertFalse(df['final_result'].isin(valid_targets).all())

if __name__ == "__main__":
    unittest.main()
