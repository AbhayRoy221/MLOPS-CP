# Day-28 Final Modeling Dataset

## Purpose
This document specifies the exact structure, constraints, and semantics of the assembled canonical Day-28 modeling dataset for the "Equitable Student Dropout Prediction with Fairness-Aware MLOps" project. It unifies demographic, contextual, assessment, and behavioral features up to the strict temporal boundary of Day 28 of the course presentation, enabling early-warning dropout prediction without longitudinal leakage.

## Input Artifacts
* `data/processed/day28_base_cohort.parquet`
* `data/processed/day28_assessment_features.parquet`
* `data/processed/day28_vle_features.parquet`

## Join Strategy & Grain
* **Join Method**: Left join from the base cohort to the assessment and VLE feature artifacts.
* **Grain**: The composite primary key is `id_student + code_module + code_presentation`.
* **Row-Level Guarantee**: The dataset maintains exactly one row per student-module-presentation enrollment (27,444 unique rows).

## Exact Column-Role Manifest

| Column | Role | Source | Allowed in X? | Used for Fairness? | Target? | Identifier? |
| ------ | ---- | ------ | ------------- | ------------------ | ------- | ----------- |
| `id_student` | Identifier | Base | No | No | No | Yes |
| `code_module` | Categorical Context & Identifier | Base | Yes | No | No | Yes |
| `code_presentation` | Categorical Context & Identifier | Base | Yes | No | No | Yes |
| `highest_education` | Predictive (Categorical) | Base | Yes | No | No | No |
| `num_of_prev_attempts` | Predictive (Numeric) | Base | Yes | No | No | No |
| `studied_credits` | Predictive (Numeric) | Base | Yes | No | No | No |
| `registration_day` | Predictive (Numeric) | Base | Yes | No | No | No |
| `module_presentation_length` | Predictive (Numeric) | Base | Yes | No | No | No |
| `asm_submission_count` | Predictive (Numeric) | Assessment | Yes | No | No | No |
| `asm_mean_score` | Predictive (Numeric) | Assessment | Yes | No | No | No |
| `asm_score_std` | Predictive (Numeric) | Assessment | Yes | No | No | No |
| `asm_failed_count` | Predictive (Numeric) | Assessment | Yes | No | No | No |
| `asm_average_delay` | Predictive (Numeric) | Assessment | Yes | No | No | No |
| `vle_total_clicks` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_active_days` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_days_since_last_activity` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_clicks_last_7_days` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_clicks_last_14_days` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_forum_clicks` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_resource_clicks` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_quiz_clicks` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `vle_activity_type_diversity` | Predictive (Numeric) | VLE | Yes | No | No | No |
| `gender` | Fairness Attribute | Base | No | Yes | No | No |
| `disability` | Fairness Attribute | Base | No | Yes | No | No |
| `age_band` | Fairness Attribute | Base | No | Yes | No | No |
| `region` | Fairness Attribute | Base | No | Yes | No | No |
| `imd_band` | Fairness Attribute | Base | No | Yes | No | No |
| `is_withdrawn` | Binary Target | Base | No | No | Yes | No |
| `final_result` | Multiclass Target (Traceability) | Base | No | No | Yes | No |
| `missing_registration` | Diagnostic Flag | Base | No | No | No | No |
| `missing_socioeconomic_information`| Diagnostic Flag | Base | No | No | No | No |
| `assessment_submission_before_registration`| Diagnostic Flag | Assessment | No | No | No | No |
| `no_vle_activity_by_day28` | Diagnostic Flag | VLE | No | No | No | No |

*Total physical columns: 33*
*Predictive feature columns: 21*
*Fairness columns: 5*
*Target columns: 2*
*Identifier columns: 3 (with 2 overlapping as categorical context)*

**Note**: `code_module` and `code_presentation` function as both primary key identifiers for tracing and categorical context predictors. `id_student` is an identifier only.

## Target Definition
* **`is_withdrawn`**: The primary binary classification target. 1 if the student's `final_result` is "Withdrawn", 0 otherwise.

## Diagnostic Flags
Diagnostic flags highlight structural missingness or extreme data anomalies (like early submissions or lack of engagement). They are excluded from the 21 predictive features but retained for debugging or subset analysis.

## Missingness
* **`asm_mean_score`, `asm_score_std`, `asm_average_delay`**: Null for students with 0 assessment submissions.
* **`vle_days_since_last_activity`**: Null for students with 0 VLE interactions.
* **`imd_band`, `registration_day`**: Native nulls retained from OULAD base metadata.
* No arbitrary imputation is performed in this layer.

## Leakage Controls
* The dataset assembly script programmatically audits the 21 selected features against a forbidden list (e.g., `date_unregistration`, `is_withdrawn`, `final_result`). 
* Post-Day-28 features are absent, guaranteed by upstream temporal filtering during artifact generation.

## Schema
Output: `data/processed/day28_modeling_dataset.parquet`

## Validation Results
* Row count explicitly matches 27,444.
* Unique composite-key count explicitly matches 27,444.
* No duplicate rows generated via join explosion.

## Test Results
Tested locally in `tests/test_final_dataset.py` with mock artifacts mimicking schema conditions.

## Known Limitations
* Categorical features (`highest_education`, `code_module`, `code_presentation`) remain unencoded strings.
* Missing numerical values require active imputation before modeling.
