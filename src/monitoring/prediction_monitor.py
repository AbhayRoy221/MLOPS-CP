import pandas as pd
import numpy as np

def summarize_predictions(predictions: pd.DataFrame) -> dict:
    """
    Summarize a batch of predictions.
    Assumes dataframe has columns: 'risk_probability', 'risk_tier', 'human_review_required'
    """
    if len(predictions) == 0:
        return {
            'total_predictions': 0,
            'mean_risk_probability': None,
            'median_risk_probability': None,
            'min_risk_probability': None,
            'max_risk_probability': None,
            'risk_tier_distribution': {'LOW': 0, 'MODERATE': 0, 'HIGH': 0, 'CRITICAL': 0},
            'alert_review_rate': None
        }

    probs = predictions['risk_probability']
    
    tier_counts = predictions['risk_tier'].value_counts().to_dict()
    tier_dist = {
        'LOW': tier_counts.get('LOW', 0),
        'MODERATE': tier_counts.get('MODERATE', 0),
        'HIGH': tier_counts.get('HIGH', 0),
        'CRITICAL': tier_counts.get('CRITICAL', 0)
    }

    review_rate = float(predictions['human_review_required'].mean())

    return {
        'total_predictions': len(predictions),
        'mean_risk_probability': float(probs.mean()),
        'median_risk_probability': float(probs.median()),
        'min_risk_probability': float(probs.min()),
        'max_risk_probability': float(probs.max()),
        'risk_tier_distribution': tier_dist,
        'alert_review_rate': review_rate
    }
