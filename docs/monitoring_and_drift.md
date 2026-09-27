# Monitoring and Drift Detection (Step 2.13)

## 1. Why monitoring is required
Deploying a machine learning model is not the end of its lifecycle. Once in production, the data distributions can shift, rendering predictions inaccurate (data drift), and operational characteristics like latency or failure rates can degrade. We must monitor the system to maintain reliability, fairness, and prediction quality over time.

## 2. What is monitored
- **Application health**: Service availability, latency, and error rates.
- **Prediction distribution**: How the model is classifying students across risk tiers.
- **Data drift**: Statistical shifts in the 21 predictive features.
- **Fairness metrics**: Group-wise disparities when sensitive attributes are present in the batch.

## 3. Application metrics
The FastAPI application exposes a `/metrics` endpoint compatible with Prometheus. It tracks:
- `student_dropout_prediction_requests_total`: Total count of inference requests.
- `student_dropout_prediction_errors_total`: Count of failed inference requests.
- `student_dropout_prediction_latency_seconds`: Histogram of prediction request durations.

## 4. Prediction monitoring
Prediction monitoring tracks the distribution of predictions in a given batch. Key metrics include the mean, median, min, and max predicted probabilities, as well as the proportion of students mapped to `LOW`, `MODERATE`, `HIGH`, and `CRITICAL` risk tiers. It also calculates the total human review rate based on the intervention configurations.

## 5. Data drift
Data drift refers to changes in the distribution of the 21 predictive features compared to the reference dataset used during training. We monitor both numerical and categorical features.

## 6. PSI (Population Stability Index)
For numerical features, we calculate the Population Stability Index (PSI). The dataset is divided into 10 buckets based on the reference distribution. The frequency of data points in these buckets is compared against the current batch to produce a single stability score.

## 7. Categorical drift (Total Variation Distance)
For categorical features (e.g., `highest_education`, `code_module`), we calculate the Total Variation Distance (TVD). This represents half the sum of absolute differences in category frequencies between the reference and current datasets.

## 8. Fairness monitoring
Fairness monitoring measures disparities across subgroups based on sensitive attributes (`gender`, `disability`, `age_band`, `region`, `imd_band`). 

## 9. Reference data
The reference dataset for monitoring is strictly `data/processed/splits/development_train.parquet`. The locked `test.parquet` and `final_holdout.parquet` are strictly reserved for final model evaluations and are never used as a baseline for production drift monitoring.

## 10. Drift thresholds
We use the following documented heuristics to flag drift:
- **PSI (Numerical)**:
  - `< 0.10`: No meaningful drift
  - `0.10 – < 0.25`: Moderate drift
  - `>= 0.25`: Substantial drift
- **TVD (Categorical)**:
  - `< 0.05`: No meaningful drift
  - `0.05 – < 0.15`: Moderate drift
  - `>= 0.15`: Substantial drift

## 11. Difference between drift and model degradation
Data drift implies that the inputs to the model have changed structurally. However, **drift does not automatically mean model degradation**. If the relationship between the drifted features and the target variable remains stable, the model's accuracy might still be perfect. Drift acts as an early warning indicator for investigation, not a definitive failure state.

## 12. Difference between prediction-only and delayed-outcome fairness monitoring
- **Prediction-only monitoring**: When the true outcomes (`is_withdrawn`) are not yet known, we can only monitor the **selection rate** (e.g., the proportion of a subgroup flagged for intervention). 
- **Delayed-outcome monitoring**: Once the semester finishes and the true labels become available, we can compute full classification metrics like True Positive Rate (TPR), False Positive Rate (FPR), precision, and recall per subgroup.

## 13. Limitations
- **No Causal Assumptions**: Monitoring correlations and feature drifts does not explain *why* the data shifted.
- **Small Sample Sizes**: Drift and fairness statistics calculated on very small batches are inherently noisy and unreliable. Fairness monitoring specifically suppresses reporting for groups with fewer than 30 samples.
- **Simulation**: The current evaluation script (`evaluate_monitoring.py`) runs a simulated batch using the validation dataset. It does not represent live production traffic.

## 14. Future extensions
This monitoring framework is designed to integrate into standard observability stacks. Application metrics can be scraped by Prometheus and visualized in Grafana. The Python-based batch scripts can be orchestrated via Apache Airflow or CI/CD pipelines to run periodically on new data exports, enabling automated alerting for substantial drift or fairness violations.
