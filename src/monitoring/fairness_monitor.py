import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix

def summarize_fairness(predictions: pd.DataFrame, sensitive_attrs: list, target_col: str = None) -> dict:
    """
    Summarize fairness metrics.
    If target_col is provided and exists in the DataFrame, calculates full outcome metrics (TPR, FPR, etc).
    If target_col is None or missing, only calculates prediction-based monitoring (selection rate).
    """
    if len(predictions) == 0:
        return {}

    has_true_labels = target_col and target_col in predictions.columns
    
    # We assume 'human_review_required' acts as the binary predicted outcome for intervention selection
    if 'human_review_required' not in predictions.columns:
        return {}
        
    y_pred = predictions['human_review_required'].astype(int)
    if has_true_labels:
        y_true = predictions[target_col].astype(int)
        
    results = {}
    
    for attr in sensitive_attrs:
        if attr not in predictions.columns:
            continue
            
        group_results = {}
        counts = predictions[attr].value_counts()
        
        for group, count in counts.items():
            if count < 30: # Small group handling
                group_results[str(group)] = {'note': 'Group size < 30, metrics suppressed for reliability'}
                continue
                
            mask = predictions[attr] == group
            group_pred = y_pred[mask]
            
            # Selection rate (proportion selected for intervention)
            selection_rate = float(group_pred.mean())
            
            metrics = {
                'count': int(count),
                'selection_rate': selection_rate
            }
            
            if has_true_labels:
                group_true = y_true[mask]
                
                # Check if group has only one class in true labels
                if len(group_true.unique()) > 1:
                    tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tpr
                else:
                    tpr = fpr = precision = recall = None
                    
                metrics.update({
                    'tpr': float(tpr) if tpr is not None else None,
                    'fpr': float(fpr) if fpr is not None else None,
                    'precision': float(precision) if precision is not None else None,
                    'recall': float(recall) if recall is not None else None
                })
                
            group_results[str(group)] = metrics
            
        results[attr] = {
            'monitoring_type': 'delayed_outcome' if has_true_labels else 'prediction_only',
            'groups': group_results
        }
        
    return results
