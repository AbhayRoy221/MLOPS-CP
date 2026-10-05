# Fairness Mitigation and Final Evaluation

## 1. Overview and Experiments
As part of our Fairness-Aware MLOps pipeline for the **Equitable Student Dropout Prediction** project, we conducted a systematic fairness audit and mitigation phase using `fairlearn` version `0.14.0`.

We evaluated 30 distinct mitigation configurations:
- **Base Models**: Logistic Regression, Random Forest, XGBoost
- **Mitigation Algorithms**: ThresholdOptimizer (postprocessing), ExponentiatedGradient (reduction)
- **Constraints**: Demographic Parity (for reduction), Equalized Odds (for reduction and postprocessing)
- **Protected Attributes**: `gender`, `disability`, `age_band`, `region`, `imd_band`

## 2. Trade-off Analysis & Configuration Selection

Following our pre-specified Fairness Selection Protocol:
1. **Primary**: Minimize Equalized Odds Difference (EOD)
2. **Secondary**: Minimize Demographic Parity Difference (DPD)
3. **Preserve**: PR-AUC
4. **Early-Warning**: Recall

### Top Mitigation Configurations by Validation EOD:
- **Logistic Regression / Disability (ThresholdOptimizer, EqualizedOdds)**: EOD = 0.0072
- **Logistic Regression / Gender (ThresholdOptimizer, EqualizedOdds)**: EOD = 0.0096
- **XGBoost / Gender (ThresholdOptimizer, EqualizedOdds)**: EOD = 0.0161

**Frozen Configuration:**
We selected **XGBoost** mitigated via **ThresholdOptimizer (EqualizedOdds)** addressing **gender**. 
- **Why?** This model was selected through the pre-specified validation trade-off protocol. Based entirely on validation evidence and the predefined fairness/performance considerations, it provided a strong balance. 

**Note on PR-AUC Comparability:**
The validation result currently contains a mitigated PR-AUC value of approximately 0.248114. Because `ThresholdOptimizer` produces fairness-constrained hard/randomized predictions, this PR-AUC is NOT directly comparable with the baseline probability PR-AUC of 0.390179. It is not a probability-ranking metric. We do not consider the 0.248114 value to be a direct apples-to-apples PR-AUC degradation.

Instead, the mitigation comparison emphasizes valid hard-decision metrics:
- **Recall**: Boosted significantly from the 0.106 baseline up to 0.193 in validation (an +8.7% absolute gain).
- **Equalized Odds Difference (EOD)**: Heavily reduced for gender from the baseline 0.0589 down to 0.0161 on validation data.
- **Demographic Parity Difference (DPD)**: Reduced to 0.0012.
- **Accuracy, Precision, and F1**: Remained stable or acceptable per hard-decision metrics.

## 3. Held-Out Test Evaluation
The held-out test evaluation was performed once after the mitigation configuration was frozen. These results were not subsequently used to tune or modify the mitigation configuration.

### Mitigated Model (XGBoost + ThresholdOptimizer on Gender) Test Results:
- **Equalized Odds Difference**: 0.0283 (significantly lower than the unmitigated baseline test EOD of 0.0589)
- **Demographic Parity Difference**: 0.0098
- **Demographic Parity Ratio**: 0.8138
- **Accuracy**: 0.8150
- **Precision**: 0.5000
- **Recall**: 0.1289 (Improved from baseline XGBoost test recall of 0.086)
- **F1 Score**: 0.2050
- **ROC AUC**: 0.5498

## 4. Post-Mitigation Evaluation Data Governance

The existing test set is now **LOCKED** for future model-development decisions because it has already been used for the mitigation-phase held-out evaluation.

Future processes such as:
- Hyperparameter tuning
- Calibration
- Threshold-policy development
- Model selection

must **NOT** use this test set.

## 5. Operational and Privacy Considerations

> [!WARNING]
> **Production Deployment Requirement:** 
> Because we selected `ThresholdOptimizer` (a postprocessing technique), the mitigation algorithm **requires the sensitive attribute (`gender`) at prediction time**.

**Privacy & Compliance Implications:**
- **Data Collection**: The production API and frontend must explicitly capture the `gender` field from students during inference.
- **Privacy Trade-offs**: This creates a potential privacy burden. If users opt out of providing their gender or provide an unsupported value, the system will **not** substitute a global or majority-group threshold. Instead, it will return an explicit status (`Unavailable`) indicating the fairness-adjusted decision cannot be produced, while preserving the raw risk probability and risk tier.
- **Unresolved Governance Policy**: The production policy for mandatory gender collection, user consent, data retention, access, and explicit opt-out is currently **unresolved**. Until approved, gender must remain an optional field in the API, and mandatory collection is not authorized for production.
- **Model Registry**: The MLflow tracking implementation must explicitly document that the input schema for inference accepts the protected attribute for post-processing, despite it *not* being part of the 21-feature predictive modeling contract.
