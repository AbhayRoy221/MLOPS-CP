# Baseline Fairness Audit

## 1. Purpose
The purpose of this audit is to measure and document the baseline fairness characteristics of the three baseline models (Logistic Regression, Random Forest, XGBoost) before any mitigation or algorithmic fairness interventions are applied. This is a descriptive baseline auditing phase only; observed disparities do not inherently prove discriminatory intent but signal areas that require further investigation.

## 2. Protected Attributes
We audited five sensitive attributes from the OULAD dataset:
- `gender`
- `disability`
- `age_band`
- `region`
- `imd_band`

## 3. Dataset/Split Used
The audit relies on the pre-existing, leakage-safe, group-aware split (Day-28 cohort). Metrics are calculated on the `validation` and `test` splits.

## 4. Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost

## 5. Fairness Definitions
We calculate standard Fairlearn disparity metrics:
- **Demographic Parity Difference**: Difference between the highest and lowest group selection rates.
- **Demographic Parity Ratio**: Ratio of the lowest to the highest group selection rates.
- **Equalized Odds Difference**: Maximum disparity across groups in terms of either true positive rate or false positive rate.

## 6. Group-size Handling
Groups are preserved exactly as defined in OULAD. Missing data or "unknown" ('?') groups are treated explicitly as distinct categories. Small sample sizes are noted and metrics for groups with 0 predicted/actual cases return `NaN`. We do not drop small groups.

## 7. Validation Results
Detailed tables for the validation set are available in `reports/fairness/`. Baseline fairness measurements indicate observed disparities across various demographic features, particularly in false-positive and true-positive rates.

## 8. Test Results
The held-out test set is audited to provide an unbiased estimate of real-world disparities. The test set remains strictly isolated and will not be used to tune models or select fairness mitigation strategies.

## 9. Interpretation Guidelines
- **No causal claims**: Differences in predictive performance across groups are observational correlations, often rooted in historical disparities or feature distribution skew.
- **Group-level differences**: Focus on Demographic Parity Difference and Equalized Odds Difference to understand how differently the model treats distinct cohorts.

## 10. Limitations
This audit assumes the OULAD sensitive attribute definitions are accurate. It does not perform intersectional fairness audits (e.g., gender AND age_band simultaneously).

## 11. Reproducibility Instructions
To reproduce the fairness audit:
```bash
python -m src.fairness.audit_baseline
```
All randomness utilizes a fixed seed (`random_state=42`) to guarantee identical outputs.

## 12. Statement of Scope
This is an **audit only, not a mitigation step**. No model evaluated here is considered fair or production-ready based on this report alone. The findings will guide future calibration and mitigation efforts.
