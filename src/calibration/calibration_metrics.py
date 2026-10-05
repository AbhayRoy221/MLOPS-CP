import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Calculate bin counts to weight the ECE
    bins = np.linspace(0., 1., n_bins + 1)
    binned = np.digitize(y_prob, bins) - 1
    
    bin_totals = np.bincount(binned, minlength=len(bins))
    
    # Filter empty bins
    non_empty_bins = bin_totals[bin_totals > 0]
    weights = non_empty_bins / len(y_prob)
    
    # Absolute difference between true fraction of positives and mean predicted probability
    abs_diff = np.abs(prob_true - prob_pred)
    
    # ECE is the weighted average of absolute differences
    ece = np.sum(weights * abs_diff)
    return ece

def calculate_calibration_metrics(y_true, y_prob):
    """
    Calculate a suite of calibration-specific metrics.
    """
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    ll = log_loss(y_true, y_prob)
    return {
        'brier_score': brier,
        'ece': ece,
        'log_loss': ll
    }
