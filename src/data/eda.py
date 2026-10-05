import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_eda():
    os.makedirs('reports/eda', exist_ok=True)
    
    # 2. Load and validate
    df = pd.read_parquet('data/processed/day28_modeling_dataset.parquet')
    
    # Assertions
    assert len(df) == 27444, "Expected 27,444 rows"
    assert df.duplicated(['id_student', 'code_module', 'code_presentation']).sum() == 0, "Duplicate composite keys found"
    
    predictive_features = [
        "highest_education", "num_of_prev_attempts", "studied_credits", "registration_day",
        "code_module", "code_presentation", "module_presentation_length",
        "asm_submission_count", "asm_mean_score", "asm_score_std", "asm_failed_count", "asm_average_delay",
        "vle_total_clicks", "vle_active_days", "vle_days_since_last_activity",
        "vle_clicks_last_7_days", "vle_clicks_last_14_days", "vle_forum_clicks",
        "vle_resource_clicks", "vle_quiz_clicks", "vle_activity_type_diversity"
    ]
    
    fairness_features = ['gender', 'disability', 'age_band', 'region', 'imd_band']
    
    for f in predictive_features + fairness_features + ['is_withdrawn']:
        assert f in df.columns, f"Missing {f}"
        
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 3. Basic dataset profile
    missing_nan = df.isna().sum()
    missing_oulad = (df == "?").sum()
    total_missing = missing_nan + missing_oulad
    
    print("\n=== DETAILED MISSINGNESS ===")
    for col in df.columns:
        if total_missing[col] > 0:
            print(f"{col}: {missing_nan[col]} NaN, {missing_oulad[col]} '?'")
            
    profile = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_nan': missing_nan,
        'missing_oulad_unknown': missing_oulad,
        'total_missing': total_missing,
        'missing_percentage': (total_missing / len(df)) * 100,
        'unique_values': df.nunique()
    })
    profile.to_csv('reports/eda/column_profile.csv')
    
    numeric_features = [f for f in predictive_features if pd.api.types.is_numeric_dtype(df[f])]
    cat_features = [f for f in predictive_features if f not in numeric_features]
    
    # 4. Target analysis
    target_counts = df['is_withdrawn'].value_counts()
    target_pct = df['is_withdrawn'].value_counts(normalize=True) * 100
    
    plt.figure()
    target_counts.plot(kind='bar')
    plt.title('Target Distribution (is_withdrawn)')
    plt.savefig('reports/eda/target_distribution.png')
    plt.close()
    
    print(f"\nTarget is_withdrawn counts:\n{target_counts}")
    print(f"Target is_withdrawn %:\n{target_pct}")
    print(f"Imbalance ratio (Majority/Minority): {target_counts[0] / target_counts[1]:.2f}")
    
    # 5. Numeric feature analysis
    numeric_summary = df[numeric_features].describe(percentiles=[.25, .5, .75, .95]).T
    numeric_summary['missing'] = df[numeric_features].isna().sum()
    numeric_summary.to_csv('reports/eda/numeric_summary.csv')
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_features, 1):
        plt.subplot(4, 5, i)
        df[col].hist(bins=20)
        plt.title(col, fontsize=8)
    plt.tight_layout()
    plt.savefig('reports/eda/numeric_distributions.png')
    plt.close()
    
    # 6. Categorical feature analysis
    for col in cat_features:
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Category Distribution: {col}')
        plt.tight_layout()
        plt.savefig(f'reports/eda/cat_{col}_dist.png')
        plt.close()
        
    # 7. Missingness
    plt.figure()
    missing_pct = profile.loc[predictive_features, 'missing_percentage']
    if not missing_pct[missing_pct > 0].empty:
        missing_pct[missing_pct > 0].plot(kind='bar')
    else:
        plt.text(0.5, 0.5, "No Missing Values", ha='center')
    plt.title('Missingness > 0% in Predictive Features')
    plt.tight_layout()
    plt.savefig('reports/eda/missingness.png')
    plt.close()
    
    # 8. Target vs Numeric
    for col in numeric_features:
        if df[col].nunique() > 1:
            plt.figure()
            df.boxplot(column=col, by='is_withdrawn')
            plt.title(f'{col} by is_withdrawn')
            plt.suptitle('')
            plt.tight_layout()
            plt.savefig(f'reports/eda/num_vs_target_{col}.png')
            plt.close()
            
    # 9. Target vs Categorical
    for col in cat_features:
        rate = df.groupby(col)['is_withdrawn'].mean() * 100
        plt.figure()
        rate.plot(kind='bar')
        plt.title(f'Withdrawal Rate by {col}')
        plt.ylabel('Withdrawal %')
        plt.tight_layout()
        plt.savefig(f'reports/eda/cat_vs_target_{col}.png')
        plt.close()
        
    # 10. Correlation
    corr = df[numeric_features].corr()
    corr.to_csv('reports/eda/numeric_correlation.csv')
    
    plt.figure(figsize=(12, 10))
    plt.matshow(corr, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(numeric_features)), numeric_features, rotation=90)
    plt.yticks(range(len(numeric_features)), numeric_features)
    plt.title('Numeric Feature Correlation', pad=20)
    plt.tight_layout()
    plt.savefig('reports/eda/numeric_correlation.png')
    plt.close()
    
    # Fairness grouping 
    print("\nFairness Group Withdrawal Rates:")
    for f in fairness_features:
        print(f"\n{f}:")
        stats = df.groupby(f)['is_withdrawn'].agg(['count', 'mean'])
        stats['mean'] = stats['mean'] * 100
        print(stats)
        
        plt.figure()
        stats['mean'].plot(kind='bar')
        plt.title(f'Withdrawal Rate by {f}')
        plt.ylabel('Withdrawal %')
        plt.tight_layout()
        plt.savefig(f'reports/eda/fairness_{f}.png')
        plt.close()
        
    # Diagnostics
    print("\nDiagnostic Flags:")
    flags = ['missing_registration', 'missing_socioeconomic_information', 
             'assessment_submission_before_registration', 'no_vle_activity_by_day28']
    for flag in flags:
        if flag in df.columns:
            cnt = df[flag].sum()
            rate = df[df[flag] == 1]['is_withdrawn'].mean() * 100 if cnt > 0 else 0
            print(f"{flag}: {cnt} ({cnt/len(df)*100:.2f}%) - Withdrawal rate: {rate:.2f}%")

if __name__ == "__main__":
    run_eda()
