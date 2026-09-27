import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import mlflow
from src.explainability.shap_explainer import SHAPExplainer
from src.models.train_baselines import PREDICTIVE_FEATURES

def main():
    os.makedirs('reports/explainability', exist_ok=True)
    
    # Load Validation Data
    val_df = pd.read_parquet('data/processed/splits/validation.parquet')
    X_val = val_df[PREDICTIVE_FEATURES].copy()
    
    # Ensure types
    num_cols = [c for c in PREDICTIVE_FEATURES if c not in ['highest_education', 'code_module', 'code_presentation']]
    for c in num_cols:
        X_val[c] = pd.to_numeric(X_val[c], errors='coerce')
        
    # We will sample 1000 records deterministically for global explanations if dataset is too large
    # Validation is 3917, calculating TreeSHAP is fast enough, but let's do the full set.
    sample_size = len(X_val)
    
    explainer = SHAPExplainer()
    shap_values, feature_names = explainer.explain_instances(X_val)
    
    X_processed = explainer.preprocessor.transform(X_val)
    if isinstance(X_processed, pd.DataFrame):
        X_matrix = X_processed.values
    else:
        X_matrix = X_processed

    # 1. Global Importance CSV
    # Mean absolute SHAP value across all validation instances
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })
    importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    importance_df.to_csv('reports/explainability/shap_global_importance.csv', index=False)
    
    # 2. Global Plots
    # Summary Bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=X_matrix, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('reports/explainability/shap_summary_bar.png')
    plt.close()
    
    # Beeswarm (standard summary plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=X_matrix, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('reports/explainability/shap_summary_beeswarm.png')
    plt.close()
    
    # 3. Local Explanations
    # Get risk probabilities to find representative cases
    probs = explainer.pipeline.predict_proba(X_val)[:, 1]
    
    # We want one deterministic student from each tier. We'll pick the first valid hit.
    tiers_needed = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
    found = {t: False for t in tiers_needed}
    local_results = []
    
    for idx, (index_val, row) in enumerate(X_val.iterrows()):
        if all(found.values()):
            break
            
        rec = explainer.intervention_engine.generate_recommendation(probs[idx])
        tier = rec['risk_tier']
        
        if tier in found and not found[tier]:
            # Generate explanation
            row_df = X_val.iloc[[idx]]
            local_exp = explainer.get_local_explanation(row_df, num_features=5)
            # Add anonymized ID
            local_exp['student_pseudo_id'] = f"val_student_{idx}"
            
            # Format lists to strings for CSV
            local_exp['top_positive_contributors'] = " | ".join(local_exp['top_positive_contributors'])
            local_exp['top_negative_contributors'] = " | ".join(local_exp['top_negative_contributors'])
            
            local_results.append(local_exp)
            found[tier] = True
            
    local_df = pd.DataFrame(local_results)
    # Reorder columns
    local_df = local_df[['student_pseudo_id', 'risk_probability', 'risk_tier', 'top_positive_contributors', 'top_negative_contributors']]
    local_df.to_csv('reports/explainability/local_explanations.csv', index=False)
    
    # 4. MLflow Logging
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student_dropout_explainability")
    
    with mlflow.start_run(run_name="validation_shap_analysis"):
        mlflow.log_param("shap_version", shap.__version__)
        mlflow.log_param("validation_sample_size", sample_size)
        mlflow.log_param("model_artifact", "models/baseline/xgb_pipeline.pkl")
        
        # Log top 10 features as parameters/metrics for visibility
        top_10 = importance_df.head(10)
        mlflow.log_text("\n".join(top_10['feature'].tolist()), "top_10_features.txt")
        
        mlflow.log_artifact('reports/explainability/shap_global_importance.csv')
        mlflow.log_artifact('reports/explainability/local_explanations.csv')
        mlflow.log_artifact('reports/explainability/shap_summary_bar.png')
        mlflow.log_artifact('reports/explainability/shap_summary_beeswarm.png')

    print("SHAP explainability analysis complete.")

if __name__ == '__main__':
    main()
