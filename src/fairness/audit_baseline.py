import os
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.train_baselines import PREDICTIVE_FEATURES
from src.fairness.fairness_metrics import calculate_group_metrics, calculate_disparity_metrics

def set_seed(seed):
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def run_fairness_audit():
    with open('configs/fairness_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['fairness']
        
    set_seed(config['random_state'])
    
    sens_attrs = config['sensitive_attributes']
    target = config['target_attribute']
    
    os.makedirs('reports/fairness', exist_ok=True)
    
    val_df = pd.read_parquet('data/processed/splits/validation.parquet')
    test_df = pd.read_parquet('data/processed/splits/test.parquet')
    
    models = {
        'lr': joblib.load('models/baseline/lr_pipeline.pkl'),
        'rf': joblib.load('models/baseline/rf_pipeline.pkl'),
        'xgb': joblib.load('models/baseline/xgb_pipeline.pkl')
    }
    
    splits = {'validation': val_df, 'test': test_df}
    
    # Check that sensitive attributes are not in PREDICTIVE_FEATURES
    for attr in sens_attrs:
        assert attr not in PREDICTIVE_FEATURES, f"Fairness attribute {attr} leaked into PREDICTIVE_FEATURES!"
    
    summary_data = []
    
    for split_name, df in splits.items():
        X = df[PREDICTIVE_FEATURES].copy()
        
        # fix dtypes for numeric features
        num_cols = [f for f in PREDICTIVE_FEATURES if f not in ['highest_education', 'code_module', 'code_presentation']]
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors='coerce')
            
        y_true = df[target]
        S = df[sens_attrs]
        
        for model_name, model in models.items():
            y_pred = model.predict(X)
            
            # 1. Group metrics
            group_metrics_df = calculate_group_metrics(y_true, y_pred, S)
            group_metrics_df['model'] = model_name
            group_metrics_df['split'] = split_name
            
            # Save group metrics CSV
            csv_path = f"reports/fairness/{model_name}_{split_name}_group_metrics.csv"
            group_metrics_df.to_csv(csv_path, index=False)
            
            # 2. Disparity metrics
            disp_metrics = calculate_disparity_metrics(y_true, y_pred, S)
            
            for attr, m in disp_metrics.items():
                summary_data.append({
                    'model': model_name,
                    'split': split_name,
                    'sensitive_attribute': attr,
                    'n_groups': m['n_groups'],
                    'min_group_count': m['min_group_count'],
                    'demographic_parity_difference': m['demographic_parity_difference'],
                    'demographic_parity_ratio': m['demographic_parity_ratio'],
                    'equalized_odds_difference': m['equalized_odds_difference']
                })
                
            # 3. Plots
            for attr in sens_attrs:
                plot_df = group_metrics_df[group_metrics_df['sensitive_attribute'] == attr]
                
                # We will just plot selection rate, TPR, FPR for each group in this attribute
                plot_metrics = ['selection_rate', 'true_positive_rate', 'false_positive_rate']
                
                # Only plot if we have data
                if len(plot_df) == 0:
                    continue
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"{model_name.upper()} - {split_name.title()} - {attr}", y=1.05)
                
                for i, m_name in enumerate(plot_metrics):
                    sns.barplot(data=plot_df, x='group', y=m_name, ax=axes[i], color='skyblue')
                    axes[i].set_title(m_name.replace('_', ' ').title())
                    axes[i].set_ylim(0, 1.0)
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Annotate sample size
                    for j, bar in enumerate(axes[i].patches):
                        axes[i].annotate(f"n={plot_df.iloc[j]['group_count']}",
                                         (bar.get_x() + bar.get_width() / 2, 0.05),
                                         ha='center', va='bottom', color='black', rotation=90, size=8)
                        
                plt.tight_layout()
                plt.savefig(f"reports/fairness/{model_name}_{split_name}_{attr}_plot.png", bbox_inches='tight')
                plt.close(fig)
                
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('reports/fairness/disparity_summary.csv', index=False)
    print("Baseline fairness audit complete.")

if __name__ == "__main__":
    run_fairness_audit()
