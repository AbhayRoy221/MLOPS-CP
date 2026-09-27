import pandas as pd
import numpy as np
import joblib
import os
import yaml
import mlflow
from src.intervention.intervention_engine import InterventionEngine
from src.models.train_baselines import PREDICTIVE_FEATURES
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def main():
    os.makedirs('reports/intervention', exist_ok=True)
    
    # 1. Load data
    val_df = pd.read_parquet('data/processed/splits/validation.parquet')
    
    # Prepare X validation
    X_val = val_df[PREDICTIVE_FEATURES].copy()
    num_cols = [c for c in PREDICTIVE_FEATURES if c not in ['highest_education', 'code_module', 'code_presentation']]
    for c in num_cols:
        X_val[c] = pd.to_numeric(X_val[c], errors='coerce')
        
    y_val = val_df['is_withdrawn'].values
    
    # 2. Load integrated risk model (uncalibrated XGBoost)
    # The pipeline is saved as models/baseline/xgb_pipeline.pkl
    xgb_base = joblib.load('models/baseline/xgb_pipeline.pkl')
    
    # Get uncalibrated probability
    risk_prob = xgb_base.predict_proba(X_val)[:, 1]
    
    # 3. Apply Intervention Engine
    engine = InterventionEngine()
    
    results = []
    for prob in risk_prob:
        rec = engine.generate_recommendation(prob)
        results.append(rec)
        
    res_df = pd.DataFrame(results)
    res_df['is_withdrawn'] = y_val
    
    # 4. Compute Metrics per Tier
    metrics = []
    
    for tier in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']:
        tier_mask = res_df['risk_tier'] == tier
        tier_count = tier_mask.sum()
        
        if tier_count == 0:
            continue
            
        tier_df = res_df[tier_mask]
        withdrawals = tier_df['is_withdrawn'].sum()
        withdrawal_rate = withdrawals / tier_count
        
        # Calculate precision/recall by treating THIS tier as positive prediction
        # To do this, we compare actual y_val against whether prediction falls in this tier or higher
        # But per requirements: "For each tier ... precision, recall, F1, FPR where mathematically meaningful"
        # It's better to compute metrics assuming >= this tier is predicted positive
        
        # However, the user specifically asks for tier-level metrics.
        metrics.append({
            'risk_tier': tier,
            'count': tier_count,
            'percentage': tier_count / len(res_df) * 100,
            'withdrawal_count': withdrawals,
            'withdrawal_rate': withdrawal_rate,
        })
        
    # Also calculate threshold-based metrics (tier and above)
    tier_order = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3, 'CRITICAL': 4}
    res_df['tier_rank'] = res_df['risk_tier'].map(tier_order)
    
    for row in metrics:
        rank = tier_order[row['risk_tier']]
        pred_positive = (res_df['tier_rank'] >= rank).astype(int)
        
        prec = precision_score(y_val, pred_positive, zero_division=0)
        rec = recall_score(y_val, pred_positive, zero_division=0)
        f1 = f1_score(y_val, pred_positive, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_val, pred_positive, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        
        row['precision_cumulative'] = prec
        row['recall_cumulative'] = rec
        row['f1_cumulative'] = f1
        row['fpr_cumulative'] = fpr

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('reports/intervention/validation_risk_tier_summary.csv', index=False)
    
    # 5. Save mapping report
    with open('configs/intervention_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    mapping_data = []
    for tier in ['low', 'moderate', 'high', 'critical']:
        rules = config['intervention_rules'][tier]
        mapping_data.append({
            'risk_tier': tier.upper(),
            'intervention_level': rules['intervention_level'],
            'human_review_required': rules['human_review_required'],
            'recommended_actions': " | ".join(rules['recommended_actions'])
        })
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv('reports/intervention/intervention_mapping.csv', index=False)
    
    # 6. MLflow Logging
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student_dropout_intervention")
    
    with mlflow.start_run(run_name="validation_intervention_evaluation"):
        mlflow.log_artifact('configs/intervention_config.yaml')
        mlflow.log_artifact('reports/intervention/validation_risk_tier_summary.csv')
        mlflow.log_artifact('reports/intervention/intervention_mapping.csv')
        
        for idx, row in metrics_df.iterrows():
            tier = row['risk_tier'].lower()
            mlflow.log_metric(f"{tier}_count", row['count'])
            mlflow.log_metric(f"{tier}_withdrawal_rate", row['withdrawal_rate'])
            
    print("Intervention validation evaluation complete.")
    print(metrics_df)

if __name__ == '__main__':
    main()
