import os
import pandas as pd
import numpy as np

def assemble_day28_dataset():
    # 2. Input artifacts
    base_path = "data/processed/day28_base_cohort.parquet"
    asm_path = "data/processed/day28_assessment_features.parquet"
    vle_path = "data/processed/day28_vle_features.parquet"
    
    if not all(os.path.exists(p) for p in [base_path, asm_path, vle_path]):
        raise FileNotFoundError("One or more input artifacts are missing.")
        
    df_base = pd.read_parquet(base_path)
    df_asm = pd.read_parquet(asm_path)
    df_vle = pd.read_parquet(vle_path)
    
    join_keys = ['id_student', 'code_module', 'code_presentation']
    
    # Validation 1: Uniqueness
    for name, df in zip(['Base', 'Assessment', 'VLE'], [df_base, df_asm, df_vle]):
        if df.duplicated(subset=join_keys).sum() > 0:
            raise ValueError(f"Duplicate keys found in {name} artifact.")
            
    base_rows = len(df_base)
    asm_rows = len(df_asm)
    vle_rows = len(df_vle)
    
    # 3. Join
    df_final = pd.merge(df_base, df_asm, on=join_keys, how='left')
    df_final = pd.merge(df_final, df_vle, on=join_keys, how='left')
    
    final_rows = len(df_final)
    
    if final_rows != 27444 or base_rows != 27444:
        raise ValueError(f"Join explosion! Expected 27444 rows, got {final_rows}")
        
    if df_final.duplicated(subset=join_keys).sum() > 0:
        raise ValueError("Join explosion created duplicate keys!")
        
    # Feature Lists
    identifiers = ['id_student', 'code_module', 'code_presentation']
    
    predictive_features = [
        "highest_education", "num_of_prev_attempts", "studied_credits", "registration_day",
        "code_module", "code_presentation", "module_presentation_length",
        "asm_submission_count", "asm_mean_score", "asm_score_std", "asm_failed_count", "asm_average_delay",
        "vle_total_clicks", "vle_active_days", "vle_days_since_last_activity",
        "vle_clicks_last_7_days", "vle_clicks_last_14_days", "vle_forum_clicks",
        "vle_resource_clicks", "vle_quiz_clicks", "vle_activity_type_diversity"
    ]
    
    fairness_features = ['gender', 'disability', 'age_band', 'region', 'imd_band']
    
    targets = ['final_result', 'is_withdrawn']
    
    diagnostic_flags = [
        'missing_registration', 'missing_socioeconomic_information',
        'assessment_submission_before_registration', 'no_vle_activity_by_day28'
    ]
    
    # Audit for target/leakage in X
    leakage_forbidden = ['date_unregistration', 'is_withdrawn', 'final_result']
    for f in predictive_features:
        if f in leakage_forbidden:
            raise ValueError(f"Leakage Audit Failed: {f} found in predictive features!")
            
    for f in predictive_features:
        if f not in df_final.columns:
            raise ValueError(f"Missing predictive feature: {f}")
            
    # Final target validation
    mismatches = ((df_final['is_withdrawn'] == 1) & (df_final['final_result'] != 'Withdrawn')) | \
                 ((df_final['is_withdrawn'] == 0) & (df_final['final_result'] == 'Withdrawn'))
    if mismatches.sum() > 0:
        raise ValueError(f"Target mismatch! {mismatches.sum()} rows have inconsistent targets.")
        
    # Final dataset column reordering
    all_expected_cols = list(set(identifiers + predictive_features + fairness_features + targets + diagnostic_flags))
    df_final = df_final[all_expected_cols].copy()
    
    # Output Schema Guarantee
    if len(df_final) != 27444:
        raise ValueError("Row count is not 27444")
    if df_final.duplicated(subset=join_keys).sum() > 0:
        raise ValueError("Duplicate keys in final dataset")
        
    # Generate X/A/y logical definition check (no-op data split just to ensure they work)
    X_primary = df_final[predictive_features]
    A_fairness = df_final[fairness_features]
    y = df_final['is_withdrawn']
    
    # Missingness Report
    print("=== MISSINGNESS REPORT ===")
    missing_report = df_final.isna().sum()
    pct_missing = (missing_report / len(df_final)) * 100
    for col in df_final.columns:
        print(f"{col}: {missing_report[col]} missing ({pct_missing[col]:.2f}%) - dtype {df_final[col].dtype}")
        
    # Validation Reports
    print("\n=== VALIDATION REPORTS ===")
    print(f"Base rows: {base_rows}, Assessment rows: {asm_rows}, VLE rows: {vle_rows}")
    print(f"Final rows: {final_rows}")
    print(f"Target counts:\n{df_final['final_result'].value_counts()}")
    print(f"Target is_withdrawn counts:\n{df_final['is_withdrawn'].value_counts()}")
    
    print("\nFairness Attributes:")
    for f in fairness_features:
        print(f"{f}: {df_final[f].nunique()} unique values. Missing: {df_final[f].isna().sum()}")
        
    # Save
    os.makedirs("data/processed", exist_ok=True)
    df_final.to_parquet("data/processed/day28_modeling_dataset.parquet", index=False)
    print("\nSaved day28_modeling_dataset.parquet")
    return df_final

if __name__ == "__main__":
    assemble_day28_dataset()
