import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)

def calculate_group_metrics(y_true, y_pred, sensitive_features):
    results = []
    
    # Calculate base metrics using MetricFrame for each sensitive attribute separately
    for attr in sensitive_features.columns:
        sf = sensitive_features[attr].copy()
        sf = sf.fillna("NaN").astype(str) # ensure missing values are distinct groups
        
        unique_groups = sf.unique()
        for group in unique_groups:
            mask = (sf == group)
            yt = y_true[mask]
            yp = y_pred[mask]
            
            group_count = len(yt)
            if group_count == 0:
                continue
                
            pos_count = yt.sum()
            neg_count = group_count - pos_count
            
            # Confusion matrix
            if len(np.unique(yt)) > 1 or len(np.unique(yp)) > 1:
                cm = confusion_matrix(yt, yp, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge cases where group only has one class
                tn = sum((yt == 0) & (yp == 0))
                fp = sum((yt == 0) & (yp == 1))
                fn = sum((yt == 1) & (yp == 0))
                tp = sum((yt == 1) & (yp == 1))
            
            selection_rate = yp.mean()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tpr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision > 0 and recall > 0) else np.nan
            
            results.append({
                'sensitive_attribute': attr,
                'group': group,
                'group_count': group_count,
                'positive_count': pos_count,
                'negative_count': neg_count,
                'selection_rate': selection_rate,
                'true_positive_rate': tpr,
                'true_negative_rate': tnr,
                'false_positive_rate': fpr,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
    return pd.DataFrame(results)

def calculate_disparity_metrics(y_true, y_pred, sensitive_features):
    results = {}
    for attr in sensitive_features.columns:
        sf = sensitive_features[attr].copy().fillna("NaN").astype(str)
        try:
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        except Exception:
            dp_diff = np.nan
            
        try:
            dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
        except Exception:
            dp_ratio = np.nan
            
        try:
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
        except Exception:
            eo_diff = np.nan
            
        n_groups = sf.nunique()
        min_group_count = sf.value_counts().min()
        
        results[attr] = {
            'n_groups': n_groups,
            'min_group_count': min_group_count,
            'demographic_parity_difference': dp_diff,
            'demographic_parity_ratio': dp_ratio,
            'equalized_odds_difference': eo_diff
        }
    return results
