# Integrated Final Model Development (Step 2.8)

This document describes the unified integrated model pipeline for the Equitable Student Dropout Prediction project.

## Pipeline Architecture

```
21 predictive features
        ↓
    XGBoost
        ↓
probability output (risk_probability)
        ↓
ThresholdOptimizer(equalized_odds, sensitive_features=gender)
        ↓
    fairness_decision
```

## Risk Score vs Fairness Decision

> [!IMPORTANT]
> The system does not treat these outputs as interchangeable.

- **`risk_probability`**: The continuous probability estimate reflecting the probability of student dropout. Used for risk ranking, dashboards, and intervention prioritization.
- **`fairness_decision`**: The `ThresholdOptimizer` postprocessed classification decision (hard label). Requires `gender`. 
- **`human_review_required`**: Derived from the categorical risk tier mapping in `intervention_config.yaml` (HIGH and CRITICAL tiers require human review).

> [!WARNING]
> **Operational Constraint**: The `ThresholdOptimizer` requires the sensitive attribute (`gender`) at prediction time.

## Calibration and Selective Prediction Audit

### Calibration Selection
Three calibration variants were evaluated on validation:

| Method | Brier | ECE | LogLoss | ROC-AUC | PR-AUC | Spearman vs Uncal | Status |
|---|---|---|---|---|---|---|---|
| **Uncalibrated** | **0.1362** | **0.0095** | **0.4338** | **0.7205** | **0.3915** | 1.000 | **SELECTED** |
| Sigmoid | 0.1553 | 0.0306 | 0.4932 | 0.4030 | 0.1580 | -0.424 | **REJECTED** |
| Isotonic | 0.1467 | 0.0200 | 0.4659 | 0.6395 | 0.3182 | 0.578 | Not selected |

**Sigmoid was REJECTED** — code-verified root cause:
- Inspection of `_SigmoidCalibration` parameters (`a_`, `b_`) across all 5 CV folds showed that **4 out of 5 folds have `a_ > 0`**. The Platt sigmoid formula is `P = 1/(1+exp(a*f+b))`, so `a > 0` means higher scores produce *lower* calibrated probabilities — the mapping is inverted.
- The root cause: on each CV fold, `CalibratedClassifierCV` refits the base XGBoost pipeline on the fold's training subset (~80% of `development_train`) and then evaluates on the fold's calibration subset (~20%). On 4 of 5 folds, the fold-trained estimator produced **higher mean predict_proba for the negative class than the positive class** on the calibration subset (e.g., Fold 0: `neg_mean=0.0313, pos_mean=0.0277`; Fold 2: `neg_mean=0.6926, pos_mean=0.6042`). Platt scaling correctly fits `a_ > 0` to model this inverted relationship.
- The one fold (Fold 1) with correct discrimination (`pos_mean=0.874 > neg_mean=0.784`) learned `a_ = -5.097` (correct orientation). But since the final output is the **average** of all 5 folds' calibrated probabilities, the 4 inverted folds dominate, producing a net-inverted ranking (Spearman = -0.424).
- This is not a bug in sklearn. It is a consequence of the base XGBoost pipeline losing discriminative power when trained on smaller (~80%) subsets of `development_train` and evaluated on unseen calibration subsets. The full-data-trained model discriminates correctly (ROC-AUC 0.721), but the CV-subset models frequently do not.
- **Conclusion**: Sigmoid calibration via `CalibratedClassifierCV(cv=5)` is unsuitable for this pipeline because the base estimator's discrimination is unstable across CV folds.

**Isotonic was not selected** because:
- It degraded all calibration metrics vs uncalibrated (Brier 0.147 vs 0.136, ECE 0.020 vs 0.010).
- It produced only 368 unique probability values (3549 tied scores out of 3917), degrading ranking resolution.
- Spearman correlation with uncalibrated is 0.578 (positive but weak), confirming the tied scores disrupt ranking.

**Uncalibrated XGBoost was selected** because:
- It genuinely has the best calibration metrics on all primary criteria (Brier, ECE, LogLoss).
- Its ECE of 0.0095 indicates it is already very well calibrated.
- It preserves the full ranking resolution (3911 unique values).
- Calibration assessment: calibrated variants did not improve the validation probability-quality metrics over the uncalibrated XGBoost baseline.

## Data Governance (Strict Four-Way Split)
- **`development_train`**: Used for model fitting and fairness postprocessor fitting.
- **`validation`**: Used strictly for evaluation, calibration selection, and policy threshold selection.
- **`locked test`**: Historical fairness evaluation holdout; permanently locked.
- **`final_holdout`**: Untouched final evaluation dataset for the fully frozen pipeline.

## Fairness Integration
The `ThresholdOptimizer(equalized_odds)` was fit on `development_train` with `gender` as the sensitive attribute. It wraps the uncalibrated XGBoost estimator and uses `predict_method='predict_proba'` to receive the continuous positive-class probability before applying group-specific decision thresholds.

## Operational Logic
```
IF risk_tier in [HIGH, CRITICAL]:
    human_review_required = True
ELSE:
    human_review_required = False
```
The `risk_probability` remains visible for risk ranking, explanations, dashboards, and intervention prioritization regardless of the fairness decision. The `fairness_decision` is evaluated only if a valid `gender` is supplied and authorized.
