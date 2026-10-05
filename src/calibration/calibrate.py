import os
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from src.calibration.calibration_metrics import calculate_calibration_metrics
from src.calibration.selective_prediction import sweep_thresholds, select_operating_policy, calculate_group_selective_metrics
from src.models.train_baselines import PREDICTIVE_FEATURES
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

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
    with open('configs/calibration_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    calib_cfg = config['calibration']
    sel_cfg = config['selective_prediction']
    set_seed(calib_cfg['random_state'])
    
    os.makedirs('reports/calibration', exist_ok=True)
    
    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("student_dropout_calibration")
    
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
    X_val = prepare_X(val_df)
    y_val = val_df['is_withdrawn']
    
    models = {
        'xgb': joblib.load('models/baseline/xgb_pipeline.pkl'),
        'lr': joblib.load('models/baseline/lr_pipeline.pkl')
    }
    
    calibration_results = []
    selective_results = []
    
    best_overall_brier = float('inf')
    best_config = None
    best_y_prob_val = None
    
    for model_name, estimator in models.items():
        # Evaluate uncalibrated on VAL
        estimator.fit(X_dev, y_dev) # ensure fitted on dev
        uncal_prob = estimator.predict_proba(X_val)[:, 1]
        
        cal_metrics_uncal = calculate_calibration_metrics(y_val, uncal_prob)
        base_metrics_uncal = evaluate_base_metrics(y_val, uncal_prob)
        
        calibration_results.append({
            'model': model_name,
            'calibration_method': 'uncalibrated',
            **cal_metrics_uncal,
            **base_metrics_uncal
        })
        
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(y_val, uncal_prob, n_bins=10, ax=ax, name=f'{model_name}_uncalibrated')
        
        for method in calib_cfg['methods']:
            print(f"Fitting {method} calibration for {model_name}...")
            # Fit calibration on dev_train using CV
            calibrator = CalibratedClassifierCV(
                estimator=estimator,
                method=method,
                cv=calib_cfg['cv'],
                n_jobs=-1
            )
            calibrator.fit(X_dev, y_dev)
            
            cal_prob = calibrator.predict_proba(X_val)[:, 1]
            cal_metrics = calculate_calibration_metrics(y_val, cal_prob)
            base_metrics = evaluate_base_metrics(y_val, cal_prob)
            
            calibration_results.append({
                'model': model_name,
                'calibration_method': method,
                **cal_metrics,
                **base_metrics
            })
            
            CalibrationDisplay.from_predictions(y_val, cal_prob, n_bins=10, ax=ax, name=f'{model_name}_{method}')
            
            # MLflow logging for calibration
            with mlflow.start_run(run_name=f"{model_name}_{method}"):
                mlflow.log_params({
                    'model': model_name,
                    'calibration_method': method,
                    'cv': calib_cfg['cv'],
                    'dataset': 'development_train'
                })
                mlflow.log_metrics({**cal_metrics, **base_metrics})
                
                # Check if this is the best brier
                if cal_metrics['brier_score'] < best_overall_brier:
                    best_overall_brier = cal_metrics['brier_score']
                    best_config = {'model': model_name, 'method': method}
                    best_y_prob_val = cal_prob
            
            # Sweep selective prediction for this calibrated probability
            df_sweep = sweep_thresholds(y_val, cal_prob, min_t=sel_cfg['min_threshold'], max_t=sel_cfg['max_threshold'], step=sel_cfg['threshold_step'])
            df_sweep['model'] = model_name
            df_sweep['calibration_method'] = method
            selective_results.append(df_sweep)
            
            # Histogram
            plt.figure()
            plt.hist(cal_prob, bins=50, alpha=0.7)
            plt.title(f'Probability Histogram - {model_name} {method}')
            plt.savefig(f'reports/calibration/{model_name}_{method}_hist.png')
            plt.close()
            
        ax.set_title(f'Calibration Curves - {model_name}')
        plt.savefig(f'reports/calibration/{model_name}_reliability.png')
        plt.close(fig)
        
    df_cal = pd.DataFrame(calibration_results)
    df_cal.to_csv('reports/calibration/calibration_summary.csv', index=False)
    
    df_sel = pd.concat(selective_results, ignore_index=True)
    df_sel.to_csv('reports/calibration/selective_prediction_summary.csv', index=False)
    
    print("Selected Calibration Configuration based on Validation Brier Score:")
    print(best_config)
    
    # Process the selective prediction rule on the best configuration
    best_sel_df = df_sel[(df_sel['model'] == best_config['model']) & (df_sel['calibration_method'] == best_config['method'])]
    best_policy = select_operating_policy(best_sel_df)
    best_policy.to_csv('reports/calibration/selected_operating_policy.csv', index=False)
    
    # Fairness compatibility descriptive check for the selected policy
    conf_thresh = best_policy['confidence_threshold'].iloc[0]
    fairness_results = []
    sensitive_attrs = ['gender', 'disability', 'age_band', 'region', 'imd_band']
    
    for attr in sensitive_attrs:
        A = val_df[attr].fillna("NaN").astype(str)
        grp_df = calculate_group_selective_metrics(y_val, best_y_prob_val, A, conf_thresh)
        grp_df['sensitive_attribute'] = attr
        fairness_results.append(grp_df)
        
    df_fair = pd.concat(fairness_results, ignore_index=True)
    df_fair.to_csv('reports/calibration/fairness_compatibility.csv', index=False)
    
    # Log the final policy under a specific MLflow run
    with mlflow.start_run(run_name="final_selective_policy"):
        mlflow.log_params({
            'selected_model': best_config['model'],
            'selected_calibration': best_config['method'],
            'confidence_threshold': conf_thresh
        })
        
        # log metrics of the policy
        policy_metrics = best_policy.iloc[0].to_dict()
        for k, v in policy_metrics.items():
            if k not in ['model', 'calibration_method'] and not pd.isna(v):
                mlflow.log_metric(k, v)

if __name__ == '__main__':
    main()
