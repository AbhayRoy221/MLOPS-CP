# Baseline Modeling

## 1. Modeling Objective
The objective is to establish robust baseline predictive performance for forecasting student withdrawal (`is_withdrawn`) by Day 28, before introducing complex interventions like MLflow tracking, fairness mitigation, or hyperparameter tuning.

## 2. Dataset and Split
* **Train**: ~19,628 rows
* **Validation**: ~3,917 rows
* **Test**: ~3,899 rows

## 3. Group-aware splitting
Splitting was strictly performed using `StratifiedGroupKFold` grouped by `id_student` to prevent data leakage (the same student appearing in both train and validation/test splits).

## 4. Feature Contract
Exactly 21 predictive features are used. Fairness attributes (`gender`, `disability`, `age_band`, `region`, `imd_band`) and diagnostic flags are strictly excluded from the predictive model (`X`).

## 5. Preprocessing
* **Numeric**: Constant imputation (`-999`) with `add_indicator=True`, followed by `RobustScaler`.
* **Categorical**: OULAD unknown markers mapped to `"Missing"`, constant imputation (`"Missing"`), followed by `OneHotEncoder(handle_unknown="ignore")`.
* **Leakage Prevention**: All preprocessing was fit strictly on `X_train`. Validation and test sets are exclusively passed through `transform`.

## 6. Logistic Regression
A standard linear baseline. `max_iter=2000`.

## 7. Random Forest
A robust ensemble baseline using bagging. `n_estimators=300`.

## 8. XGBoost
A gradient boosting baseline configured with a conservative learning rate (`0.05`) and moderate depth (`4`).

## 9. Evaluation Metrics
Evaluated metrics include Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, and Specificity. Due to class imbalance (~18% positive rate), PR-AUC and Recall are primary indicators of minority-class capture. Default classification threshold is `0.5`.

## 10. Validation/Test Methodology
* **Train**: Used exclusively for fitting the pipelines (preprocessing + estimator).
* **Validation**: Used to compare baseline architectures.
* **Test**: Fully held-out until the final evaluation step in this baseline report.

## 11. Overfitting Observations
(To be updated based on metrics)

## 12. Model Comparison Table
Results are stored in `reports/models/baseline_comparison.csv`.

## 13. Feature Importance
Tree-based models export feature importance to:
* `reports/models/random_forest_feature_importance.csv`
* `reports/models/xgboost_feature_importance.csv`
*Note: Feature importance is calculated on the transformed one-hot features output by the ColumnTransformer.*

## 14. Limitations
* Baseline models do not address the ~4.5:1 class imbalance.
* No fairness mitigation has been applied; the models may exhibit biased performance across demographic groups.
* Causality cannot be inferred.

## 15. Why this is a baseline and not final model selection
These models are untrained defaults designed to anchor performance. Subsequent steps will introduce MLflow, optimize hyperparameters, address fairness, and calibrate probabilities.
