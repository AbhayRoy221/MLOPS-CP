import os
import pandas as pd
import yaml

def load_config():
    with open("configs/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_day28_cohort():
    config = load_config()
    raw_dir = config["raw_data_dir"]
    cutoff_day = config["cutoff_day"]
    
    df_info = pd.read_csv(os.path.join(raw_dir, "studentInfo.csv"))
    df_reg = pd.read_csv(os.path.join(raw_dir, "studentRegistration.csv"))
    df_courses = pd.read_csv(os.path.join(raw_dir, "courses.csv"))
    
    df = pd.merge(df_info, df_reg, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df = pd.merge(df, df_courses, on=['code_module', 'code_presentation'], how='left')
    
    df['is_withdrawn'] = (df['final_result'] == 'Withdrawn').astype(int)
    
    df['missing_registration'] = (df['date_registration'] == '?').astype(int)
    df['registration_day'] = df['date_registration'].replace('?', pd.NA)
    
    df['missing_socioeconomic_information'] = (df['imd_band'] == '?').astype(int)
    
    df_unreg_valid = df[df['date_unregistration'] != '?'].copy()
    df_unreg_valid['date_unregistration'] = df_unreg_valid['date_unregistration'].astype(float)
    early_withdrawn_mask = (df['date_unregistration'] != '?') & (df_unreg_valid['date_unregistration'] <= cutoff_day)
    
    missing_unreg_withdrawn_mask = (df['is_withdrawn'] == 1) & (df['date_unregistration'] == '?')
    
    inconsistent_nw_mask = (df['is_withdrawn'] == 0) & (df['date_unregistration'] != '?')
    
    drop_mask = early_withdrawn_mask | missing_unreg_withdrawn_mask | inconsistent_nw_mask
    df_cohort = df[~drop_mask].copy()
    
    cols_to_keep = [
        'id_student', 'code_module', 'code_presentation',
        'registration_day', 'gender', 'disability', 'age_band', 'region', 'imd_band',
        'highest_education', 'num_of_prev_attempts', 'studied_credits',
        'final_result', 'is_withdrawn', 'missing_registration', 'missing_socioeconomic_information',
        'module_presentation_length'
    ]
    df_cohort = df_cohort[cols_to_keep]
    
    total_eligible = len(df_cohort)
    eventual_w = df_cohort['is_withdrawn'].sum()
    eventual_nw = total_eligible - eventual_w
    dups = df_cohort.duplicated(subset=['id_student', 'code_module', 'code_presentation']).sum()
    
    expected_total = 27444
    expected_w = 5016
    
    print(f"Total eligible rows: {total_eligible}")
    print(f"Eventual Withdrawn: {eventual_w}")
    print(f"Eventual Non-Withdrawn: {eventual_nw}")
    print(f"Composite-key duplicates: {dups}")
    print(f"Missing registration count: {df_cohort['missing_registration'].sum()}")
    print(f"Missing socioeconomic info count: {df_cohort['missing_socioeconomic_information'].sum()}")
    
    if total_eligible != expected_total or eventual_w != expected_w:
        raise ValueError(f"Cohort validation failed! Expected {expected_total} rows and {expected_w} withdrawn.")
        
    os.makedirs("data/processed", exist_ok=True)
    df_cohort.to_parquet("data/processed/day28_base_cohort.parquet", index=False)
    print("Saved day28_base_cohort.parquet")
    
if __name__ == "__main__":
    build_day28_cohort()
