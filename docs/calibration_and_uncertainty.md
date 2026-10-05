# Calibration and Selective Prediction

This document outlines the probability calibration and selective prediction strategies implemented for the Equitable Student Dropout Prediction project.

## 1. Why Calibration is Needed
Machine learning models (like XGBoost or uncalibrated Logistic Regression) often produce scores that rank examples well (high ROC-AUC) but do not reflect true probabilities. Calibration ensures that a predicted probability of 0.8 means an 80% chance of dropout, which is essential for defining reliable early-warning thresholds and understanding prediction confidence. Note that calibrated probability is not absolute certainty; it merely aligns the model's output distribution with empirical frequencies.

## 2. Calibration Methods Evaluated
We evaluate:
- **Uncalibrated**: Raw baseline model probabilities.
- **Sigmoid (Platt Scaling)**: Fits a logistic regression model on the outputs.
- **Isotonic Regression**: A non-parametric approach fitting a piecewise constant non-decreasing function.

## 3. Data Partition Policy
- **`development_train.parquet`**: Used exclusively to fit the calibration parameters.
- **`validation.parquet`**: Used exclusively for evaluating calibration quality and selecting the final method and selective-prediction policy.
- **`test.parquet`**: The historical fairness evaluation holdout; strictly locked.
- **`final_holdout.parquet`**: The final evaluation holdout; strictly **untouched** during this phase.

## 4. Calibration Metrics
Calibration is evaluated using:
- **Brier Score**: Mean squared error of probabilities (primary selection metric).
- **Expected Calibration Error (ECE)**: Weighted average of the absolute difference between predicted probability and true fraction of positives across bins.
- **Log Loss**: Secondary metric.
- **Reliability Diagrams**: Visual assessment of calibration curve fit.

### Calibration Selection Protocol
The calibration method was selected using the `validation.parquet` set. We prioritized the lowest Brier Score and ECE, supported by visual reliability curves, while monitoring standard classification metrics (PR-AUC, ROC-AUC, Recall, F1).

## 5. Reliability Interpretation
A well-calibrated model's predictions align closely with the diagonal $y=x$ on a reliability diagram.

## 6. Uncertainty Definition
We use a simple heuristic for binary classification confidence and uncertainty based on calibrated probability $p$:
- **risk_probability** = $p$
- **confidence** = $\max(p, 1 - p)$
- **uncertainty** = $1 - \text{confidence}$

*Limitation*: These uncertainty estimates are a simple heuristic reflecting prediction confidence near the decision boundary. They are not mathematically comprehensive epistemic or aleatoric uncertainty decompositions.

## 7. Selective Prediction
Selective prediction (or abstention) involves setting a confidence threshold $t$. 
- If `confidence >= t`, the model makes an automated prediction.
- If `confidence < t`, the model abstains, reflecting high uncertainty.

## 8. Human-Review Routing & Selection Protocol
Cases where the model abstains are routed to human review. We evaluated a sweep of confidence thresholds on validation data.

### Selective Prediction Selection Protocol
Among operating points with **coverage >= 0.90**, select the threshold with the **highest selective recall**. Use selective precision as the tie-breaker.

## 9. Fairness Compatibility Assessment
We audit whether the chosen selective-prediction policy produces substantially different coverage across sensitive groups (gender, disability, age_band, region, imd_band). This is a descriptive fairness check only. It evaluates coverage, abstention rate, selective recall, and selective FPR per group. It does not introduce new mitigation nor alter the frozen XGBoost ThresholdOptimizer configuration.

## 10. Operational Limitations
- The uncertainty metric only captures proximity to the decision boundary, not out-of-distribution (epistemic) uncertainty.
- Calibration relies on the assumption that the `development_train` distribution matches the deployment distribution.

## 11. Reproducibility
All calibration fitting and threshold sweeping processes use `random_state=42` to ensure deterministic execution.

## 12. Final Holdout Governance
The `final_holdout.parquet` remains completely untouched. It will only be used for the ultimate system evaluation once all pipelines and policies are frozen.
