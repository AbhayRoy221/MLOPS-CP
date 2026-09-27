import os
import yaml
import json
import sqlite3
import pandas as pd
import numpy as np
import joblib
import mlflow

from src.monitoring.drift_detector import detect_drift
from src.monitoring.prediction_monitor import summarize_predictions
from src.monitoring.fairness_monitor import summarize_fairness

def load_config(config_path="configs/monitoring_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_evaluation():
    config = load_config()
    monitoring_config = config['monitoring']
    
    # 1. Load Reference Data
    print("Loading reference data...")
    ref_path = monitoring_config['reference_data_path']
    reference_df = pd.read_parquet(ref_path)
    
    # 2. Simulate current batch (using validation data)
    print("Loading simulated current batch (validation set)...")
    val_path = "data/processed/splits/validation.parquet"
    current_df = pd.read_parquet(val_path)
    
    # Exclude non-predictive features for drift detector
    # But keep them for fairness monitor
    predictive_features = [c for c in reference_df.columns if c not in [
        'id_student', 'final_result', 'is_withdrawn', 'gender', 'disability',
        'age_band', 'region', 'imd_band'
    ]]
    
    print(f"Tracking {len(predictive_features)} predictive features.")
    
    # 3. Calculate drift statistics
    print("Calculating data drift...")
    drift_results = detect_drift(reference_df, current_df, predictive_features)
    drift_df = pd.DataFrame([
        {'feature': k, 'metric': v['metric'], 'value': v['value'], 'status': v['status']}
        for k, v in drift_results.items()
    ])
    
    # 4. Calculate prediction statistics using XGB pipeline
    print("Loading ML pipeline...")
    pipeline = joblib.load("models/baseline/xgb_pipeline.pkl")
    
    print("Generating predictions on current batch...")
    # Get raw probabilities
    X_curr = current_df[predictive_features]
    probs = pipeline.predict_proba(X_curr)[:, 1]
    
    # Map to risk tiers based on thresholds
    risk_tiers = []
    human_review_required = []
    
    for p in probs:
        if p >= 0.50:
            tier = 'CRITICAL'
            review = True
        elif p >= 0.30:
            tier = 'HIGH'
            review = True
        elif p >= 0.20:
            tier = 'MODERATE'
            review = False
        else:
            tier = 'LOW'
            review = False
            
        risk_tiers.append(tier)
        human_review_required.append(review)
        
    predictions_df = pd.DataFrame({
        'risk_probability': probs,
        'risk_tier': risk_tiers,
        'human_review_required': human_review_required,
        'gender': current_df['gender'] if 'gender' in current_df.columns else None,
        'disability': current_df['disability'] if 'disability' in current_df.columns else None,
        'age_band': current_df['age_band'] if 'age_band' in current_df.columns else None,
        'region': current_df['region'] if 'region' in current_df.columns else None,
        'imd_band': current_df['imd_band'] if 'imd_band' in current_df.columns else None,
        'is_withdrawn': current_df['is_withdrawn'] if 'is_withdrawn' in current_df.columns else None
    })
    
    print("Summarizing predictions...")
    pred_summary = summarize_predictions(predictions_df)
    
    # 5. Generate fairness summary
    print("Summarizing fairness...")
    sensitive_attrs = monitoring_config['fairness']['sensitive_attributes']
    fairness_results = summarize_fairness(
        predictions_df, 
        sensitive_attrs=sensitive_attrs,
        target_col='is_withdrawn' # Because we use validation set, we have delayed labels
    )
    
    # 6. Save reports
    print("Saving monitoring reports...")
    os.makedirs("reports/monitoring", exist_ok=True)
    
    drift_df.to_csv("reports/monitoring/drift_summary.csv", index=False)
    
    with open("reports/monitoring/prediction_summary.json", "w") as f:
        json.dump(pred_summary, f, indent=4)
        
    with open("reports/monitoring/fairness_monitoring_summary.json", "w") as f:
        json.dump(fairness_results, f, indent=4)
        
    # Generate markdown report
    with open("reports/monitoring/monitoring_report.md", "w") as f:
        f.write("# Monitoring Validation Report (Simulation)\n\n")
        f.write("## 1. Data Drift\n")
        f.write(drift_df.to_markdown(index=False) + "\n\n")
        f.write("## 2. Prediction Summary\n")
        f.write("```json\n" + json.dumps(pred_summary, indent=2) + "\n```\n\n")
        f.write("## 3. Fairness Monitoring\n")
        f.write("```json\n" + json.dumps(fairness_results, indent=2) + "\n```\n")

    # 7. Log to MLflow
    print("Logging to MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student_dropout_monitoring")
    
    with mlflow.start_run(run_name="validation_monitoring_simulation"):
        mlflow.log_dict(monitoring_config, "monitoring_config.yaml")
        mlflow.log_artifact("reports/monitoring/drift_summary.csv")
        mlflow.log_artifact("reports/monitoring/prediction_summary.json")
        mlflow.log_artifact("reports/monitoring/fairness_monitoring_summary.json")
        mlflow.log_artifact("reports/monitoring/monitoring_report.md")
        
        # Log summary metrics
        mlflow.log_metric("total_predictions", pred_summary['total_predictions'])
        mlflow.log_metric("mean_risk_probability", pred_summary['mean_risk_probability'])
        mlflow.log_metric("alert_review_rate", pred_summary['alert_review_rate'])
        
        # Count drifted features
        drifted = drift_df[drift_df['status'] != 'no meaningful drift']
        mlflow.log_metric("drifted_features_count", len(drifted))
        
    print("Monitoring evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
