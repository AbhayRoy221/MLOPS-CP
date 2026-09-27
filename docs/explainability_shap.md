# Model Explainability with SHAP (Step 2.10)

## 1. Why Explainability is Needed
Explainability helps human advisors interpret why the model generated a specific risk prediction for a student. While the `risk_tier` establishes the appropriate intervention level, the SHAP explanation provides targeted context (e.g., specific missed assessments or low VLE activity) that the advisor can address in their outreach.

## 2. SHAP Methodology
We use SHapley Additive exPlanations (SHAP), specifically the `TreeExplainer`, which is theoretically sound for tree-based models like XGBoost. SHAP values distribute the model's output prediction among its input features.

## 3. Model Being Explained
The authoritative uncalibrated baseline XGBoost model (`models/baseline/xgb_pipeline.pkl`) is explained. The model receives 21 engineered predictive features derived up to Day 28. No new model was trained for this step.

## 4. Preprocessing Handling
The XGBoost estimator operates on transformed features (e.g., OneHotEncoded categorical variables, scaled numericals). The SHAP values are extracted on this transformed feature space to ensure mathematical correctness, and feature names are retrieved dynamically from the `ColumnTransformer`.

## 5. Validation-Only Analysis
The entire SHAP analysis was conducted on the `validation.parquet` dataset (3917 students) exclusively. Both the locked historical fairness `test.parquet` and the pristine `final_holdout.parquet` remain strictly isolated and untouched.

## 6. Global Explanations
Global SHAP values summarize which features most consistently influence the model across the entire validation cohort. The top 10 features sorted by mean absolute SHAP value are tracked in MLflow and in `reports/explainability/shap_global_importance.csv`. Visual summaries are available in `shap_summary_bar.png` and `shap_summary_beeswarm.png`.

## 7. Local Explanations
Local explanations clarify why a single student was assigned a specific risk score. For four deterministic, representative students spanning each of the four Risk Tiers (LOW, MODERATE, HIGH, CRITICAL), the top positive and negative contributing features were documented in `reports/explainability/local_explanations.csv`.

## 8. Relationship to Risk Tiers
Risk Tiers are purely operational thresholds (e.g. HIGH >= 0.30) applied on top of the model's `risk_probability`. The SHAP explainer unpacks the composition of that underlying `risk_probability` without changing the risk tier mapping.

## 9. Relationship to Intervention Recommendations
The Step 2.9 intervention system translates the Risk Tier into a recommended action (e.g., "advisor/tutor outreach"). The SHAP explanation complements this by providing human-readable context. The SHAP explanation does **NOT** automatically change the intervention tier, but rather acts as supportive context for the human advisor reviewing the student.

## 10. Limitations
- **SHAP explains the model, not reality:** It shows what features the model relies on, but does not prove causality.
- **Directional interpretation:** A positive SHAP contribution means the feature pushed the model output toward higher predicted risk relative to the model's baseline/reference. A negative contribution means it pushed the output toward lower predicted risk.
- **No causal guarantee:** Modifying a feature with high SHAP importance (e.g., making a student click more on a VLE resource) will not necessarily prevent them from withdrawing.
- **Supportive only:** Explanations are meant to support human review and outreach, rather than replace human judgment or trigger automated punitive actions.
