# Day-28 VLE Feature Pipeline

## Purpose
This module processes the Virtual Learning Environment (VLE) interactions up to Day 28 of the course presentation to generate 9 predictive behavioral features. Because the `studentVle.csv` file contains approximately 10.6M records, this pipeline uses chunked pandas processing to enforce strict memory constraints while ensuring mathematically correct global aggregations.

## Input Tables
* `data/raw/studentVle.csv` (Event-level clicks)
* `data/raw/vle.csv` (Metadata mapping `id_site` to `activity_type`)
* `data/processed/day28_base_cohort.parquet` (Base eligible cohort)

## Feature Definitions
For every modeling key (`id_student + code_module + code_presentation`), the following features are calculated from VLE events where `date <= 28`:
1. **`vle_total_clicks`**: Total sum of `sum_click`.
2. **`vle_active_days`**: Count of distinct `date` values across all events globally.
3. **`vle_days_since_last_activity`**: `28 - max(date)`.
4. **`vle_clicks_last_7_days`**: Sum of clicks where `21 < date <= 28`.
5. **`vle_clicks_last_14_days`**: Sum of clicks where `14 < date <= 28`.
6. **`vle_forum_clicks`**: Sum of clicks on `activity_type == "forumng"`.
7. **`vle_resource_clicks`**: Sum of clicks on `activity_type == "resource"`.
8. **`vle_quiz_clicks`**: Sum of clicks on `activity_type == "quiz"`.
9. **`vle_activity_type_diversity`**: Count of unique `activity_type` values interacted with globally.

## Data Quality Flag
* **`no_vle_activity_by_day28`**: A diagnostic flag equal to `1` if the student had exactly zero qualifying VLE interactions, otherwise `0`.

## Leakage Controls
* **Temporal Filtering**: `date <= 28` is enforced strictly before aggregation.
* **Base Cohort Restriction**: VLE events outside the frozen `day28_base_cohort` keys are aggressively excluded early during the chunk loop.
* **Target Isolation**: No modeling targets or missingness features leak into this pipeline.

## Missing-Value Imputation
Zero-activity students are preserved in the artifact through a left join against the base cohort keys. Their numerical click sums and active days are imputed to `0`, while `vle_days_since_last_activity` correctly remains structurally missing (`NaN`).

## Output Artifact
`data/processed/day28_vle_features.parquet`
* Contains exactly 27,444 unique rows matching the baseline cohort keys, guaranteeing row-level consistency for the final ML pipeline merge.
