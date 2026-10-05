import unittest
import pandas as pd

import os

@unittest.skipUnless(os.path.exists('data/processed/splits/train.parquet'), "Local datasets not available in CI")
class TestDataSplitGovernance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_parquet('data/processed/splits/train.parquet')
        cls.val = pd.read_parquet('data/processed/splits/validation.parquet')
        cls.test = pd.read_parquet('data/processed/splits/test.parquet')
        cls.dev_train = pd.read_parquet('data/processed/splits/development_train.parquet')
        cls.final_holdout = pd.read_parquet('data/processed/splits/final_holdout.parquet')
        cls.canonical = pd.read_parquet('data/processed/day28_modeling_dataset.parquet')
        
    def test_existing_validation_unchanged(self):
        self.assertEqual(len(self.val), 3917, "Validation set size changed.")
        
    def test_existing_test_unchanged(self):
        self.assertEqual(len(self.test), 3899, "Test set size changed.")
        
    def test_canonical_dataset_unchanged(self):
        self.assertEqual(len(self.canonical), 27444, "Canonical dataset size changed.")
        
    def test_partition_of_old_train(self):
        self.assertEqual(len(self.train), len(self.dev_train) + len(self.final_holdout), "dev_train and final_holdout do not sum up to old train size.")
        # Check identical content when combined
        combined = pd.concat([self.dev_train, self.final_holdout]).sort_values(['id_student', 'code_module', 'code_presentation']).reset_index(drop=True)
        old_train_sorted = self.train.sort_values(['id_student', 'code_module', 'code_presentation']).reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, old_train_sorted, check_like=True)
        
    def test_no_student_overlap(self):
        dev_students = set(self.dev_train['id_student'].unique())
        val_students = set(self.val['id_student'].unique())
        test_students = set(self.test['id_student'].unique())
        holdout_students = set(self.final_holdout['id_student'].unique())
        
        self.assertEqual(len(dev_students.intersection(holdout_students)), 0, "Overlap between dev_train and final_holdout")
        self.assertEqual(len(dev_students.intersection(val_students)), 0, "Overlap between dev_train and val")
        self.assertEqual(len(dev_students.intersection(test_students)), 0, "Overlap between dev_train and test")
        self.assertEqual(len(holdout_students.intersection(val_students)), 0, "Overlap between final_holdout and val")
        self.assertEqual(len(holdout_students.intersection(test_students)), 0, "Overlap between final_holdout and test")

    def test_no_duplicate_composite_keys(self):
        for name, df in zip(['dev_train', 'final_holdout'], [self.dev_train, self.final_holdout]):
            dupes = df.duplicated(subset=['id_student', 'code_module', 'code_presentation']).sum()
            self.assertEqual(dupes, 0, f"Duplicate keys found in {name}")
            
    def test_class_labels_valid(self):
        for name, df in zip(['dev_train', 'final_holdout'], [self.dev_train, self.final_holdout]):
            invalid_labels = df[~df['is_withdrawn'].isin([0, 1])]
            self.assertEqual(len(invalid_labels), 0, f"Invalid labels found in {name}")

if __name__ == '__main__':
    unittest.main()
