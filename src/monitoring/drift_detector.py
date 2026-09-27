import pandas as pd
import numpy as np

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets=10) -> float:
    """
    Calculate Population Stability Index (PSI) for a continuous variable.
    Handles empty arrays, constant features and 0 values securely.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Ensure no NaNs are used in calculation (simplistic approach: dropna)
    expected = expected.dropna()
    actual = actual.dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    if expected.nunique() <= 1 and actual.nunique() <= 1:
        return 0.0
    
    # Calculate quantiles on expected distribution
    try:
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        # Unique breakpoints to prevent duplicate bin edges
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
            
        # Ensure the edges cover all data by slightly expanding limits
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
    except:
        return 0.0

    # Calculate frequencies in each bucket
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    def substitute_zero(val):
        return val if val > 0 else 0.0001
        
    expected_percents = np.array([substitute_zero(x) for x in expected_percents])
    actual_percents = np.array([substitute_zero(x) for x in actual_percents])
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi_value)

def calculate_tvd(expected: pd.Series, actual: pd.Series) -> float:
    """
    Calculate Total Variation Distance (TVD) for categorical variables.
    Formula: 0.5 * sum(|actual_freq - expected_freq|) over all categories
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    expected_dist = expected.value_counts(normalize=True).to_dict()
    actual_dist = actual.value_counts(normalize=True).to_dict()
    
    all_categories = set(expected_dist.keys()) | set(actual_dist.keys())
    
    tvd = 0.0
    for cat in all_categories:
        e = expected_dist.get(cat, 0.0)
        a = actual_dist.get(cat, 0.0)
        tvd += abs(e - a)
        
    return float(0.5 * tvd)

def classify_drift(stat_value: float, threshold_type: str = 'psi') -> str:
    """
    Classify drift based on heuristics.
    """
    if threshold_type == 'psi':
        if stat_value < 0.10:
            return 'no meaningful drift'
        elif stat_value < 0.25:
            return 'moderate drift'
        else:
            return 'substantial drift'
    elif threshold_type == 'tvd':
        # TVD > 0.1 means overall distribution shifted by more than 10%
        if stat_value < 0.05:
            return 'no meaningful drift'
        elif stat_value < 0.15:
            return 'moderate drift'
        else:
            return 'substantial drift'
    return 'unknown'

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list) -> dict:
    """
    Calculate drift for all features.
    """
    results = {}
    if len(reference_df) == 0 or len(current_df) == 0:
        return results
        
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
            
        ref_col = reference_df[feature]
        cur_col = current_df[feature]
        
        # Determine if feature is numerical or categorical
        if pd.api.types.is_numeric_dtype(ref_col):
            stat_value = calculate_psi(ref_col, cur_col)
            classification = classify_drift(stat_value, 'psi')
            metric_type = 'PSI'
        else:
            stat_value = calculate_tvd(ref_col, cur_col)
            classification = classify_drift(stat_value, 'tvd')
            metric_type = 'TVD'
            
        results[feature] = {
            'metric': metric_type,
            'value': round(stat_value, 4),
            'status': classification
        }
        
    return results
