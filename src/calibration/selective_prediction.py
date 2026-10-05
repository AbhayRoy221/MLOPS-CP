import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_selective_metrics(y_true, y_prob, confidence_threshold):
    """
    Calculate metrics for predictions where confidence >= confidence_threshold.
    confidence = max(p, 1-p)
    """
    confidence = np.maximum(y_prob, 1 - y_prob)
    mask = confidence >= confidence_threshold
    
    coverage = np.mean(mask)
    abstention_rate = 1.0 - coverage
    
    if coverage == 0:
        return {
            'coverage': 0.0,
            'abstention_rate': 1.0,
            'selective_accuracy': np.nan,
            'selective_precision': np.nan,
            'selective_recall': np.nan,
            'selective_f1': np.nan,
            'selective_fpr': np.nan,
            'risk_probability_mean': np.nan,
            'number_automatically_handled': 0,
            'number_routed_to_human_review': len(y_true)
        }
        
    y_true_sel = y_true[mask]
    y_prob_sel = y_prob[mask]
    y_pred_sel = (y_prob_sel >= 0.5).astype(int)
    
    acc = accuracy_score(y_true_sel, y_pred_sel)
    prec = precision_score(y_true_sel, y_pred_sel, zero_division=0)
    rec = recall_score(y_true_sel, y_pred_sel, zero_division=0)
    f1 = f1_score(y_true_sel, y_pred_sel, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true_sel, y_pred_sel, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    return {
        'coverage': coverage,
        'abstention_rate': abstention_rate,
        'selective_accuracy': acc,
        'selective_precision': prec,
        'selective_recall': rec,
        'selective_f1': f1,
        'selective_fpr': fpr,
        'risk_probability_mean': np.mean(y_prob_sel),
        'number_automatically_handled': int(np.sum(mask)),
        'number_routed_to_human_review': int(len(y_true) - np.sum(mask))
    }

def sweep_thresholds(y_true, y_prob, min_t=0.5, max_t=0.99, step=0.01):
    thresholds = np.arange(min_t, max_t + step, step)
    results = []
    for t in thresholds:
        metrics = calculate_selective_metrics(y_true, y_prob, t)
        metrics['confidence_threshold'] = t
        results.append(metrics)
    return pd.DataFrame(results)

def select_operating_policy(df_sweep):
    """
    Selection rule:
    PRIMARY: Choose threshold with highest selective recall among coverage >= 0.90
    SECONDARY TIE-BREAKER: Higher selective precision
    """
    valid = df_sweep[df_sweep['coverage'] >= 0.90].copy()
    if len(valid) == 0:
        # Fallback if no threshold gives >= 90% coverage
        return df_sweep.iloc[0:1]
        
    # Sort by selective recall (desc), then selective precision (desc)
    best = valid.sort_values(by=['selective_recall', 'selective_precision'], ascending=[False, False])
    return best.iloc[0:1]

def calculate_group_selective_metrics(y_true, y_prob, A, confidence_threshold):
    """
    Fairness descriptive audit for selective prediction
    """
    confidence = np.maximum(y_prob, 1 - y_prob)
    mask = confidence >= confidence_threshold
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'mask': mask,
        'group': A
    })
    
    results = []
    for g, g_df in df.groupby('group'):
        coverage = g_df['mask'].mean()
        abstention_rate = 1.0 - coverage
        alert_rate = g_df[g_df['mask']]['y_pred'].mean() if coverage > 0 else np.nan
        
        if coverage > 0:
            sel_df = g_df[g_df['mask']]
            sel_rec = recall_score(sel_df['y_true'], sel_df['y_pred'], zero_division=0)
            tn, fp, fn, tp = confusion_matrix(sel_df['y_true'], sel_df['y_pred'], labels=[0, 1]).ravel()
            sel_fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        else:
            sel_rec = np.nan
            sel_fpr = np.nan
            
        results.append({
            'group': g,
            'coverage': coverage,
            'abstention_rate': abstention_rate,
            'automatic_alert_rate': alert_rate,
            'selective_recall': sel_rec,
            'selective_fpr': sel_fpr
        })
    return pd.DataFrame(results)
