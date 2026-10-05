import os
import pandas as pd
import yaml
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

def validate_feature_roles(X_cols):
    forbidden = {
        'gender', 'disability', 'age_band', 'region', 'imd_band',
        'final_result', 'is_withdrawn', 'date_unregistration'
    }
    
    expected_21 = {
        'highest_education', 'num_of_prev_attempts', 'studied_credits', 'registration_day',
        'code_module', 'code_presentation', 'module_presentation_length',
        'asm_submission_count', 'asm_mean_score', 'asm_score_std', 'asm_failed_count', 'asm_average_delay',
        'vle_total_clicks', 'vle_active_days', 'vle_days_since_last_activity',
        'vle_clicks_last_7_days', 'vle_clicks_last_14_days', 'vle_forum_clicks',
        'vle_resource_clicks', 'vle_quiz_clicks', 'vle_activity_type_diversity'
    }
    
    X_set = set(X_cols)
    if X_set != expected_21:
        raise ValueError(f"Features in X do not match exactly. Missing: {expected_21 - X_set}, Extra: {X_set - expected_21}")
    
    if any(f in X_set for f in forbidden):
        raise ValueError("Forbidden features found in X")

def build_linear_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999, add_indicator=True)),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

def build_tree_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999, add_indicator=True))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

def get_splits(df, config):
    sgkf = StratifiedGroupKFold(
        n_splits=config['n_splits'],
        shuffle=config['shuffle'],
        random_state=config['random_state']
    )
    
    folds = np.zeros(len(df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df[config['stratify_col']], df[config['group_col']])):
        folds[val_idx] = fold
        
    test_idx = folds == config['test_fold']
    val_idx = folds == config['validation_fold']
    train_idx = ~(test_idx | val_idx)
    
    return df[train_idx].copy(), df[val_idx].copy(), df[test_idx].copy()

def run_split():
    with open('configs/split_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['split']
        
    df = pd.read_parquet('data/processed/day28_modeling_dataset.parquet')
    
    cat_cols = ['highest_education', 'code_module', 'code_presentation']
    for col in cat_cols:
        if '?' in df[col].values:
            df.loc[df[col] == '?', col] = 'Missing'
            
    train_df, val_df, test_df = get_splits(df, config)
    
    os.makedirs('data/processed/splits', exist_ok=True)
    train_df.to_parquet('data/processed/splits/train.parquet', index=False)
    val_df.to_parquet('data/processed/splits/validation.parquet', index=False)
    test_df.to_parquet('data/processed/splits/test.parquet', index=False)
    
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    
    print(f"Train prevalence: {train_df['is_withdrawn'].mean():.4f}")
    print(f"Val prevalence: {val_df['is_withdrawn'].mean():.4f}")
    print(f"Test prevalence: {test_df['is_withdrawn'].mean():.4f}")
    
    train_students = set(train_df['id_student'])
    val_students = set(val_df['id_student'])
    test_students = set(test_df['id_student'])
    
    print(f"Train students: {len(train_students)}")
    print(f"Validation students: {len(val_students)}")
    print(f"Test students: {len(test_students)}")
    
    overlap1 = len(train_students.intersection(val_students))
    overlap2 = len(train_students.intersection(test_students))
    overlap3 = len(val_students.intersection(test_students))
    
    print(f"Train/Val overlap: {overlap1}")
    print(f"Train/Test overlap: {overlap2}")
    print(f"Val/Test overlap: {overlap3}")
    
    assert overlap1 == 0, "Leakage: Train/Val overlap"
    assert overlap2 == 0, "Leakage: Train/Test overlap"
    assert overlap3 == 0, "Leakage: Val/Test overlap"
    
    # Check duplicate keys inside
    for s_df, name in zip([train_df, val_df, test_df], ['Train', 'Val', 'Test']):
        dups = s_df.duplicated(['id_student', 'code_module', 'code_presentation']).sum()
        print(f"{name} duplicates: {dups}")
        assert dups == 0

if __name__ == "__main__":
    run_split()
