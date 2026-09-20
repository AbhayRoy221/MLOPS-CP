import os
import pandas as pd
import yaml

def load_config():
    with open("configs/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_assessment_features():
    config = load_config()
    raw_dir = config["raw_data_dir"]
    cutoff_day = config["cutoff_day"]
    
    # 1. Load Data
    df_assessments = pd.read_csv(os.path.join(raw_dir, "assessments.csv"))
    df_student_assessment = pd.read_csv(os.path.join(raw_dir, "studentAssessment.csv"))
    
    cohort_path = "data/processed/day28_base_cohort.parquet"
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Base cohort not found at {cohort_path}")
    df_cohort = pd.read_parquet(cohort_path)
    
    # 2. Join assessment metadata
    df_asm = pd.merge(df_student_assessment, df_assessments, on='id_assessment', how='inner')
    
    # Check valid date_submitted 
    # some might be '?' but the assignment says "Only keep assessment submissions where date_submitted <= 28"
    df_asm = df_asm[df_asm['date_submitted'] != '?'].copy()
    df_asm['date_submitted'] = df_asm['date_submitted'].astype(float)
    
    # 3. Day-28 filtering (BEFORE aggregation)
    df_asm = df_asm[df_asm['date_submitted'] <= cutoff_day].copy()
    
    # Leakage Audit: ensure max date is <= 28
    if not df_asm.empty and df_asm['date_submitted'].max() > cutoff_day:
        raise ValueError(f"Leakage Audit Failed: Found date_submitted > {cutoff_day}")
    
    # 4. Restrict to modeling cohort
    cohort_keys = df_cohort[['id_student', 'code_module', 'code_presentation', 'registration_day']].drop_duplicates()
    df_asm = pd.merge(df_asm, cohort_keys, on=['id_student', 'code_module', 'code_presentation'], how='inner')
    
    # Calculate Delay
    df_asm['scheduled_date'] = pd.to_numeric(df_asm['date'], errors='coerce') # '?' becomes NaN
    df_asm['submission_delay'] = df_asm['date_submitted'] - df_asm['scheduled_date']
    
    # Failed flag
    df_asm['score'] = pd.to_numeric(df_asm['score'], errors='coerce')
    df_asm['is_failed'] = (df_asm['score'] < 40).astype(int)
    
    # Pre-registration flag
    # Compare with registration_day. If registration_day is missing (NaN), it's False.
    df_asm['registration_day'] = pd.to_numeric(df_asm['registration_day'], errors='coerce')
    df_asm['before_reg'] = (df_asm['date_submitted'] < df_asm['registration_day']).astype(int)
    
    # 5. Aggregate
    # If a student has no qualifying assessments, they won't be in this groupby, handled in join
    agg_funcs = {
        'date_submitted': ['count'],
        'score': ['mean', 'std'],
        'is_failed': ['sum'],
        'submission_delay': ['mean'],
        'before_reg': ['max']
    }
    
    df_agg = df_asm.groupby(['id_student', 'code_module', 'code_presentation']).agg(agg_funcs).reset_index()
    
    # Flatten columns
    df_agg.columns = ['id_student', 'code_module', 'code_presentation', 
                      'asm_submission_count', 'asm_mean_score', 'asm_score_std', 
                      'asm_failed_count', 'asm_average_delay', 'assessment_submission_before_registration']
                      
    # 7. No-assessment students (Left join onto cohort keys)
    df_final = pd.merge(cohort_keys[['id_student', 'code_module', 'code_presentation']], 
                        df_agg, 
                        on=['id_student', 'code_module', 'code_presentation'], 
                        how='left')
                        
    # Impute structural absences
    df_final['asm_submission_count'] = df_final['asm_submission_count'].fillna(0).astype(int)
    df_final['asm_failed_count'] = df_final['asm_failed_count'].fillna(0).astype(int)
    df_final['assessment_submission_before_registration'] = df_final['assessment_submission_before_registration'].fillna(0).astype(int)
    # mean_score, score_std, average_delay stay as NaN
    
    # 9. Row-level guarantee
    if len(df_final) != len(df_cohort):
        raise ValueError(f"Output row count ({len(df_final)}) differs from cohort row count ({len(df_cohort)})")
        
    dups = df_final.duplicated(subset=['id_student', 'code_module', 'code_presentation']).sum()
    if dups > 0:
        raise ValueError("Duplicate keys generated in assessment features!")
        
    # 11. Join validation
    missing_in_cohort = df_final[~df_final[['id_student', 'code_module', 'code_presentation']].apply(tuple, axis=1).isin(
        cohort_keys[['id_student', 'code_module', 'code_presentation']].apply(tuple, axis=1)
    )]
    if not missing_in_cohort.empty:
        raise ValueError("Found keys in assessment features not in base cohort!")
        
    # 12. Leakage audit columns
    if 'final_result' in df_final.columns or 'date_unregistration' in df_final.columns:
        raise ValueError("Leakage Audit Failed: forbidden target/unreg columns in assessment output")
        
    # Stats for reporting
    print(f"Output row count: {len(df_final)}")
    print(f"Unique composite-key count: {len(df_final.drop_duplicates(subset=['id_student', 'code_module', 'code_presentation']))}")
    print(f"Zero-assessment students: {(df_final['asm_submission_count'] == 0).sum()}")
    print(f"Pre-registration anomaly records: {df_final['assessment_submission_before_registration'].sum()}")
    print(f"Max date_submitted used: {df_asm['date_submitted'].max() if not df_asm.empty else 'N/A'}")
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    df_final.to_parquet("data/processed/day28_assessment_features.parquet", index=False)
    print("Saved day28_assessment_features.parquet")
    return df_final

if __name__ == "__main__":
    build_assessment_features()
