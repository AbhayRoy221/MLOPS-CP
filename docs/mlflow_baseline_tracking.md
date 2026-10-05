# MLflow Baseline Tracking

## 1. Why MLflow is used
MLflow is used to ensure robust experiment tracking, parameter versioning, and reproducibility. Baseline experiments are tracked for reproducibility. No model is considered production-ready at this stage.

## 2. Experiment structure
All baseline runs are grouped under a single MLflow experiment: `student_dropout_baseline`. The tracking store is backed locally by SQLite (`sqlite:///mlflow.db`).

## 3. Run structure
Each model architecture (Logistic Regression, Random Forest, XGBoost) is executed as an isolated MLflow run with clear nomenclature (e.g., `baseline_logistic_regression`).

## 4. Parameter tracking
Each run logs the dataset configuration, cohort parameters (e.g. `cutoff_day=28`), feature counts, and the exact estimator hyperparameters (e.g., `max_iter`, `n_estimators`, `max_depth`).

## 5. Metric tracking
Metrics from all three splits (Train, Validation, Test) are logged natively (e.g. `test_accuracy`, `validation_pr_auc`, `train_roc_auc`). Evaluation code calculates these deterministically from the existing baseline functions.

## 6. Artifact tracking
Visualization plots (Confusion Matrices, ROC curves, PR curves) and feature importance CSVs are logged directly as artifacts for the corresponding run.

## 7. Model logging
The complete Scikit-Learn `Pipeline` (including `ColumnTransformer` preprocessing and the estimator) is logged using MLflow's native sklearn flavor. The signature enforces the 21-feature contract.

## 8. Model registry
Each pipeline is registered under stable model names (e.g., `student_dropout_logistic_regression`).

## 9. Reproducibility
The MLflow setup allows exact reconstruction of the training configuration and environments. Because deterministic seeds (`random_state=42`) are used, multiple execution passes result in identical parameter configurations.

## 10. Candidate-vs-production status
All models are currently just baseline candidates. None are production-approved until fairness and calibration interventions have been explored.

## 11. How to start MLflow UI
To inspect the experiment results in the browser, run:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`

## 12. Limitations
Current models do not handle class imbalances and contain no algorithmic fairness mitigation. The registry reflects baseline anchors, not finalized applications.
