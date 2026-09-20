import os
import pandas as pd
import yaml
from collections import defaultdict

def load_config():
    with open("configs/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_vle_features():
    config = load_config()
    raw_dir = config["raw_data_dir"]
    cutoff_day = config["cutoff_day"]
    
    # Load base cohort keys
    cohort_path = "data/processed/day28_base_cohort.parquet"
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Base cohort not found at {cohort_path}")
    df_cohort = pd.read_parquet(cohort_path)
    cohort_keys_df = df_cohort[['id_student', 'code_module', 'code_presentation']].drop_duplicates()
    cohort_keys_set = set(cohort_keys_df.itertuples(index=False, name=None))
    
    # Load VLE metadata
    vle_path = os.path.join(raw_dir, "vle.csv")
    df_vle = pd.read_csv(vle_path)
    dups = df_vle.duplicated(subset=['id_site', 'code_module', 'code_presentation']).sum()
    if dups > 0:
        raise ValueError("VLE metadata key is not unique!")
        
    df_vle = df_vle[['id_site', 'code_module', 'code_presentation', 'activity_type']]
    
    # Process studentVle.csv in chunks
    student_vle_path = os.path.join(raw_dir, "studentVle.csv")
    
    total_raw_rows = 0
    total_filtered_before_cohort = 0
    total_retained = 0
    total_excluded_by_cohort = 0
    pre_course_count = 0
    
    chunksize = 500_000
    
    # To accumulate per-key aggregations
    agg_chunks = []
    
    # To accumulate globally unique values
    active_dates = defaultdict(set)
    active_types = defaultdict(set)
    
    max_date_used = -9999
    
    # We only need specific columns
    usecols = ['id_student', 'code_module', 'code_presentation', 'id_site', 'date', 'sum_click']
    
    for chunk in pd.read_csv(student_vle_path, chunksize=chunksize, usecols=usecols):
        total_raw_rows += len(chunk)
        
        # 1. Filter immediately to date <= 28
        chunk['date'] = pd.to_numeric(chunk['date'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk = chunk[chunk['date'] <= cutoff_day].copy()
        
        total_filtered_before_cohort += len(chunk)
        
        # 2. Restrict to modeling keys
        keys_in_chunk = chunk[['id_student', 'code_module', 'code_presentation']].apply(tuple, axis=1)
        valid_mask = keys_in_chunk.isin(cohort_keys_set)
        
        retained = chunk[valid_mask].copy()
        excluded_count = (~valid_mask).sum()
        
        total_retained += len(retained)
        total_excluded_by_cohort += excluded_count
        
        if retained.empty:
            continue
            
        # Update leakage/stats
        max_date_in_retained = retained['date'].max()
        if max_date_in_retained > max_date_used:
            max_date_used = max_date_in_retained
            
        pre_course_count += (retained['date'] < 0).sum()
        
        # 3. Join metadata
        retained = pd.merge(retained, df_vle, on=['id_site', 'code_module', 'code_presentation'], how='left')
        
        if retained['activity_type'].isna().any():
            raise ValueError("An expected metadata mapping is missing!")
            
        # 4. Compute features for this chunk
        retained['click_7'] = retained['sum_click'].where(retained['date'] > 21, 0)
        retained['click_14'] = retained['sum_click'].where(retained['date'] > 14, 0)
        retained['click_forum'] = retained['sum_click'].where(retained['activity_type'] == 'forumng', 0)
        retained['click_resource'] = retained['sum_click'].where(retained['activity_type'] == 'resource', 0)
        retained['click_quiz'] = retained['sum_click'].where(retained['activity_type'] == 'quiz', 0)
        
        grp = retained.groupby(['id_student', 'code_module', 'code_presentation'])
        
        chunk_agg = grp.agg(
            vle_total_clicks=('sum_click', 'sum'),
            max_date=('date', 'max'),
            vle_clicks_last_7_days=('click_7', 'sum'),
            vle_clicks_last_14_days=('click_14', 'sum'),
            vle_forum_clicks=('click_forum', 'sum'),
            vle_resource_clicks=('click_resource', 'sum'),
            vle_quiz_clicks=('click_quiz', 'sum')
        ).reset_index()
        
        agg_chunks.append(chunk_agg)
        
        # Track globally unique dates and activity types
        for row in retained.itertuples(index=False):
            key = (row.id_student, row.code_module, row.code_presentation)
            active_dates[key].add(row.date)
            active_types[key].add(row.activity_type)
            
    # Combine chunks
    if agg_chunks:
        df_all_agg = pd.concat(agg_chunks, ignore_index=True)
        # Re-aggregate across chunks
        grp_final = df_all_agg.groupby(['id_student', 'code_module', 'code_presentation'])
        df_final_agg = grp_final.agg(
            vle_total_clicks=('vle_total_clicks', 'sum'),
            max_date=('max_date', 'max'),
            vle_clicks_last_7_days=('vle_clicks_last_7_days', 'sum'),
            vle_clicks_last_14_days=('vle_clicks_last_14_days', 'sum'),
            vle_forum_clicks=('vle_forum_clicks', 'sum'),
            vle_resource_clicks=('vle_resource_clicks', 'sum'),
            vle_quiz_clicks=('vle_quiz_clicks', 'sum')
        ).reset_index()
    else:
        # Empty case
        df_final_agg = pd.DataFrame(columns=[
            'id_student', 'code_module', 'code_presentation', 'vle_total_clicks',
            'max_date', 'vle_clicks_last_7_days', 'vle_clicks_last_14_days',
            'vle_forum_clicks', 'vle_resource_clicks', 'vle_quiz_clicks'
        ])
        
    # Apply days_since_last_activity
    df_final_agg['vle_days_since_last_activity'] = 28 - df_final_agg['max_date']
    df_final_agg = df_final_agg.drop(columns=['max_date'])
    
    # Map active days and diversity
    df_final_agg['vle_active_days'] = df_final_agg.apply(
        lambda r: len(active_dates.get((r.id_student, r.code_module, r.code_presentation), set())), axis=1
    )
    df_final_agg['vle_activity_type_diversity'] = df_final_agg.apply(
        lambda r: len(active_types.get((r.id_student, r.code_module, r.code_presentation), set())), axis=1
    )
    
    # Left join onto the base cohort
    out_df = pd.merge(cohort_keys_df, df_final_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')
    
    # Fill structural missing
    out_df['no_vle_activity_by_day28'] = out_df['vle_total_clicks'].isna().astype(int)
    
    fill_0_cols = [
        'vle_total_clicks', 'vle_active_days', 'vle_clicks_last_7_days', 
        'vle_clicks_last_14_days', 'vle_forum_clicks', 'vle_resource_clicks', 
        'vle_quiz_clicks', 'vle_activity_type_diversity'
    ]
    out_df[fill_0_cols] = out_df[fill_0_cols].fillna(0).astype(int)
    
    # Ensure correct column order
    final_cols = [
        'id_student', 'code_module', 'code_presentation',
        'vle_total_clicks', 'vle_active_days', 'vle_days_since_last_activity',
        'vle_clicks_last_7_days', 'vle_clicks_last_14_days',
        'vle_forum_clicks', 'vle_resource_clicks', 'vle_quiz_clicks',
        'vle_activity_type_diversity', 'no_vle_activity_by_day28'
    ]
    out_df = out_df[final_cols]
    
    # VALIDATIONS
    if len(out_df) != len(df_cohort):
        raise ValueError(f"Output row count is {len(out_df)} instead of {len(df_cohort)}")
        
    dups = out_df.duplicated(subset=['id_student', 'code_module', 'code_presentation']).sum()
    if dups > 0:
        raise ValueError("Duplicate composite keys found in VLE features!")
        
    if max_date_used > 28:
        raise ValueError(f"Leakage check failed! max date used: {max_date_used}")
        
    for col in ['final_result', 'is_withdrawn', 'date_unregistration']:
        if col in out_df.columns:
            raise ValueError(f"Leakage check failed: {col} is present")
            
    print(f"Total raw VLE rows: {total_raw_rows}")
    print(f"Total filtered (date <= 28) before cohort restriction: {total_filtered_before_cohort}")
    print(f"Rows retained after cohort restriction: {total_retained}")
    print(f"Rows excluded by cohort restriction: {total_excluded_by_cohort}")
    print(f"Pre-course events (date < 0) retained: {pre_course_count} ({(pre_course_count/total_retained)*100 if total_retained else 0:.2f}%)")
    
    print(f"Zero-activity students: {out_df['no_vle_activity_by_day28'].sum()}")
    print(f"Max date used for aggregation: {max_date_used}")
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    out_df.to_parquet("data/processed/day28_vle_features.parquet", index=False)
    print("Saved day28_vle_features.parquet")
    
    return out_df

if __name__ == "__main__":
    build_vle_features()
