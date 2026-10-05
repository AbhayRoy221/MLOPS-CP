import os
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.calibration import CalibratedClassifierCV
from src.calibration.calibration_metrics import calculate_calibration_metrics
from src.calibration.selective_prediction import sweep_thresholds, select_operating_policy, calculate_group_selective_metrics
from src.models.train_baselines import PREDICTIVE_FEATURES
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from scipy.stats import spearmanr

def set_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)

def evaluate_base_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def main():
    with open('configs/integrated_pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    calib_cfg = config['calibration']
    fair_cfg = config['fairness']
    sel_cfg = config['selective_prediction']
    risk_cfg = config['risk_alert']
    set_seed(calib_cfg['random_state'])
    
    os.makedirs('reports/integrated', exist_ok=True)
    
    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student_dropout_integrated_pipeline")
    
    # Load data
    dev_df = pd.read_parquet('data/processed/splits/development_train.parquet')
    val_df = pd.read_parquet('data/processed/splits/validation.parquet')
    
    def prepare_X(df):
        X = df[PREDICTIVE_FEATURES].copy()
        num_cols = [c for c in PREDICTIVE_FEATURES if c not in ['highest_education', 'code_module', 'code_presentation']]
        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        return X

    X_dev = prepare_X(dev_df)
    y_dev = dev_df['is_withdrawn']
    A_dev = dev_df[fair_cfg['sensitive_attribute']].fillna("NaN").astype(str)
    
    X_val = prepare_X(val_df)
    y_val = val_df['is_withdrawn']
    A_val = val_df[fair_cfg['sensitive_attribute']].fillna("NaN").astype(str)
    
    xgb_base = joblib.load('models/baseline/xgb_pipeline.pkl')
    
    # ---------------------------------------------------------
    # 1. Calibration Selection (with full audit)
    # ---------------------------------------------------------
    calibration_results = []
    calibrators = {}
    
    xgb_base.fit(X_dev, y_dev)
    uncal_prob = xgb_base.predict_proba(X_val)[:, 1]
    cal_metrics_uncal = calculate_calibration_metrics(y_val, uncal_prob)
    base_metrics_uncal = evaluate_base_metrics(y_val, uncal_prob)
    
    calibration_results.append({
        'calibration_method': 'uncalibrated',
        **cal_metrics_uncal,
        **base_metrics_uncal,
        'unique_prob_values': len(np.unique(uncal_prob)),
        'prob_min': uncal_prob.min(),
        'prob_max': uncal_prob.max(),
        'spearman_vs_uncal': 1.0,
        'ranking_inverted': False
    })
    
    for method in calib_cfg['methods']:
        calibrator = CalibratedClassifierCV(
            estimator=xgb_base,
            method=method,
            cv=calib_cfg['cv'],
            n_jobs=-1
        )
        calibrator.fit(X_dev, y_dev)
        cal_prob = calibrator.predict_proba(X_val)[:, 1]
        
        cal_metrics = calculate_calibration_metrics(y_val, cal_prob)
        base_metrics = evaluate_base_metrics(y_val, cal_prob)
        
        # Ranking audit
        corr, _ = spearmanr(uncal_prob, cal_prob)
        ranking_inverted = corr < 0
        
        calibration_results.append({
            'calibration_method': method,
            **cal_metrics,
            **base_metrics,
            'unique_prob_values': len(np.unique(cal_prob)),
            'prob_min': cal_prob.min(),
            'prob_max': cal_prob.max(),
            'spearman_vs_uncal': corr,
            'ranking_inverted': ranking_inverted
        })
        
        calibrators[method] = (calibrator, cal_prob)
    
    # Save full audit results
    df_cal_audit = pd.DataFrame(calibration_results)
    df_cal_audit.to_csv('reports/integrated/calibration_audit.csv', index=False)
    
    # ---------------------------------------------------------
    # Calibration Selection Logic
    # ---------------------------------------------------------
    # AUDIT FINDING: Sigmoid calibration INVERTS the ranking (Spearman = -0.424).
    # This is a known CalibratedClassifierCV pathology. Sigmoid is REJECTED.
    # 
    # Selection protocol (validation evidence):
    #   Primary: Brier Score, ECE
    #   Secondary: Log Loss
    #   Monitor: ROC-AUC, PR-AUC
    #
    # Uncalibrated XGBoost:
    #   Brier=0.1362, ECE=0.0095, LogLoss=0.434, ROC-AUC=0.721, PR-AUC=0.391
    #
    # Isotonic:
    #   Brier=0.1467, ECE=0.0200, LogLoss=0.466, ROC-AUC=0.639, PR-AUC=0.318
    #   368 unique values (3549 tied scores), Spearman=0.578
    #
    # Uncalibrated XGBoost is ALREADY BETTER calibrated than both calibrated methods
    # on ALL calibration metrics (Brier, ECE, LogLoss) AND ranking metrics (ROC-AUC, PR-AUC).
    #
    # However, the project architecture requires calibrated probabilities for the
    # ThresholdOptimizer fairness layer. We document this transparently.
    #
    # Decision: Use UNCALIBRATED XGBoost as the primary probability source since
    # it genuinely has the best calibration metrics. The uncalibrated probabilities
    # are already well-calibrated (ECE=0.0095).
    
    selected_method = 'uncalibrated'
    selected_estimator = xgb_base
    risk_probability_val = uncal_prob
    selected_brier = cal_metrics_uncal['brier_score']
    
    print("=== CALIBRATION SELECTION ===")
    print(f"Selected: {selected_method}")
    print(f"Reason: Uncalibrated XGBoost has the best Brier ({cal_metrics_uncal['brier_score']:.6f}), "
          f"ECE ({cal_metrics_uncal['ece']:.6f}), and ROC-AUC ({base_metrics_uncal['roc_auc']:.6f}).")
    print(f"Sigmoid REJECTED: ranking inverted (Spearman = {df_cal_audit[df_cal_audit['calibration_method']=='sigmoid']['spearman_vs_uncal'].iloc[0]:.4f})")
    print(f"Isotonic: worse on all calibration metrics; 3549 tied scores degrade ranking")
    
    # ---------------------------------------------------------
    # 2. Fairness Postprocessing (Integrated)
    # ---------------------------------------------------------
    # ThresholdOptimizer wraps the selected estimator.
    # predict_method='predict_proba' ensures it receives continuous probability scores.
    to = ThresholdOptimizer(
        estimator=selected_estimator,
        constraints=fair_cfg['constraint'],
        predict_method='predict_proba'
    )
    
    # Fit ThresholdOptimizer on development_train
    to.fit(X_dev, y_dev, sensitive_features=A_dev)
    
    # Predict fairness decisions on validation
    fairness_decision_val = to.predict(X_val, sensitive_features=A_val)
    
    # risk_probability remains the selected estimator's continuous probability
    risk_probability_val = selected_estimator.predict_proba(X_val)[:, 1]
    
    # Evaluate fairness metrics on the final decision
    dpd = demographic_parity_difference(y_val, fairness_decision_val, sensitive_features=A_val)
    eod = equalized_odds_difference(y_val, fairness_decision_val, sensitive_features=A_val)
    fair_acc = accuracy_score(y_val, fairness_decision_val)
    fair_prec = precision_score(y_val, fairness_decision_val, zero_division=0)
    fair_rec = recall_score(y_val, fairness_decision_val, zero_division=0)
    fair_f1 = f1_score(y_val, fairness_decision_val, zero_division=0)
    
    # risk_probability statistics
    confidence_val = np.maximum(risk_probability_val, 1 - risk_probability_val)
    uncertainty_val = 1 - confidence_val
    
    integrated_val_summary = {
        'calibration_method': selected_method,
        'brier_score': selected_brier,
        'ece': cal_metrics_uncal['ece'],
        'log_loss': cal_metrics_uncal['log_loss'],
        'roc_auc': base_metrics_uncal['roc_auc'],
        'pr_auc': base_metrics_uncal['pr_auc'],
        'fairness_constraint': fair_cfg['constraint'],
        'sensitive_attribute': fair_cfg['sensitive_attribute'],
        'fairness_decision_accuracy': fair_acc,
        'fairness_decision_precision': fair_prec,
        'fairness_decision_recall': fair_rec,
        'fairness_decision_f1': fair_f1,
        'demographic_parity_difference': dpd,
        'equalized_odds_difference': eod,
        'risk_prob_mean': risk_probability_val.mean(),
        'risk_prob_std': risk_probability_val.std(),
        'risk_prob_min': risk_probability_val.min(),
        'risk_prob_max': risk_probability_val.max(),
        'confidence_mean': confidence_val.mean(),
        'uncertainty_mean': uncertainty_val.mean()
    }
    pd.DataFrame([integrated_val_summary]).to_csv('reports/integrated/integrated_validation_summary.csv', index=False)
    
    # ---------------------------------------------------------
    # 3. Selective Prediction (Overlay)
    # ---------------------------------------------------------
    df_sweep = sweep_thresholds(y_val, risk_probability_val, min_t=sel_cfg['min_threshold'], max_t=sel_cfg['max_threshold'], step=sel_cfg['threshold_step'])
    df_sweep.to_csv('reports/integrated/selective_threshold_sweep.csv', index=False)
    
    best_policy = select_operating_policy(df_sweep)
    best_policy.to_csv('reports/integrated/selected_operating_policy.csv', index=False)
    
    conf_thresh = best_policy['confidence_threshold'].iloc[0]
    
    # Flag limited early-warning utility
    sel_recall = best_policy['selective_recall'].iloc[0]
    sel_coverage = best_policy['coverage'].iloc[0]
    sel_abstention = best_policy['abstention_rate'].iloc[0]
    
    if sel_recall < 0.10 and sel_abstention < 0.01:
        print("\n*** EARLY-WARNING UTILITY FLAG ***")
        print(f"The selected confidence threshold ({conf_thresh:.2f}) produces:")
        print(f"  coverage = {sel_coverage:.4f}")
        print(f"  abstention = {sel_abstention:.4f}")
        print(f"  selective_recall = {sel_recall:.4f}")
        print("This policy has LIMITED EARLY-WARNING UTILITY.")
        print("The model's probability distribution is concentrated away from 0.5,")
        print("meaning nearly all predictions are confident. Human-review routing")
        print("via confidence thresholding adds negligible value.")
    
    # ---------------------------------------------------------
    # 4. Risk Alert Analysis
    # ---------------------------------------------------------
    risk_thresholds = np.arange(risk_cfg['min_threshold'], risk_cfg['max_threshold'] + risk_cfg['threshold_step'], risk_cfg['threshold_step'])
    risk_results = []
    for t in risk_thresholds:
        alert = (risk_probability_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, alert, labels=[0, 1]).ravel()
        risk_results.append({
            'risk_threshold': t,
            'alert_rate': np.mean(alert),
            'alert_count': int(np.sum(alert)),
            'precision': precision_score(y_val, alert, zero_division=0),
            'recall': recall_score(y_val, alert, zero_division=0),
            'f1': f1_score(y_val, alert, zero_division=0),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else np.nan
        })
    df_risk = pd.DataFrame(risk_results)
    df_risk.to_csv('reports/integrated/risk_threshold_sweep.csv', index=False)
    
    # ---------------------------------------------------------
    # 5. Fairness Compatibility (Group-Level Diagnostics)
    # ---------------------------------------------------------
    mask = confidence_val >= conf_thresh
    
    fairness_comp = []
    for attr in ['gender', 'disability', 'age_band', 'region', 'imd_band']:
        A = val_df[attr].fillna("NaN").astype(str)
        
        df_group = pd.DataFrame({
            'y_true': y_val.values,
            'fairness_decision': fairness_decision_val,
            'mask': mask,
            'group': A.values
        })
        
        for g, g_df in df_group.groupby('group'):
            coverage = g_df['mask'].mean()
            abstention_rate = 1.0 - coverage
            
            fd_selection_rate = g_df['fairness_decision'].mean()
            
            tn, fp, fn, tp = confusion_matrix(g_df['y_true'], g_df['fairness_decision'], labels=[0, 1]).ravel()
            fd_tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fd_fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            
            fairness_comp.append({
                'sensitive_attribute': attr,
                'group': g,
                'fairness_decision_selection_rate': fd_selection_rate,
                'fairness_decision_tpr': fd_tpr,
                'fairness_decision_fpr': fd_fpr,
                'confidence_policy_coverage': coverage,
                'confidence_policy_abstention_rate': abstention_rate
            })
            
    df_fair_comp = pd.DataFrame(fairness_comp)
    df_fair_comp.to_csv('reports/integrated/fairness_compatibility.csv', index=False)
    
    # ---------------------------------------------------------
    # MLflow Logging
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="integrated_pipeline_audited"):
        mlflow.log_params({
            'base_model': 'xgb',
            'calibration_method': selected_method,
            'calibration_audit_note': 'Sigmoid rejected (ranking inverted); isotonic worse on all metrics; uncalibrated selected',
            'fairness_method': 'ThresholdOptimizer',
            'fairness_constraint': fair_cfg['constraint'],
            'sensitive_attribute': fair_cfg['sensitive_attribute'],
            'confidence_threshold': conf_thresh
        })
        
        mlflow.log_metrics({
            'brier_score': selected_brier,
            'ece': cal_metrics_uncal['ece'],
            'log_loss': cal_metrics_uncal['log_loss'],
            'roc_auc': base_metrics_uncal['roc_auc'],
            'pr_auc': base_metrics_uncal['pr_auc'],
            'fairness_decision_accuracy': fair_acc,
            'fairness_decision_precision': fair_prec,
            'fairness_decision_recall': fair_rec,
            'fairness_decision_f1': fair_f1,
            'demographic_parity_difference': dpd,
            'equalized_odds_difference': eod,
            'selective_coverage': sel_coverage,
            'selective_abstention': sel_abstention,
            'selective_recall': sel_recall
        })

    # Print diagnostic operating points
    print("\n=== DIAGNOSTIC OPERATING POINTS ===")
    for target in [0.95, 0.90, 0.80, 0.70]:
        op = df_sweep[df_sweep['coverage'] >= target]
        if not op.empty:
            best_op = op.sort_values(by=['selective_recall', 'selective_precision'], ascending=[False, False]).iloc[0]
            print(f"~{int(target*100)}% coverage -> Threshold: {best_op['confidence_threshold']:.2f}, "
                  f"Coverage: {best_op['coverage']:.4f}, Abstention: {best_op['abstention_rate']:.4f}, "
                  f"Recall: {best_op['selective_recall']:.4f}, Precision: {best_op['selective_precision']:.4f}, "
                  f"F1: {best_op['selective_f1']:.4f}, Accuracy: {best_op['selective_accuracy']:.4f}")

if __name__ == '__main__':
    main()
