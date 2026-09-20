# Final Data Contract: Day-28 Early-Warning Model

## Authoritative Modeling Rule
> "At prediction time on Day 28, the model may use only information that would have been observable on or before Day 28 for that student-module-presentation record."

## 1. Modeling Grain
One row represents: `id_student + code_module + code_presentation`.
This composite key must remain strictly unique throughout the final modeling dataset.

## 2. Prediction Cutoff
**Primary prediction cutoff:** Day 28.
Only information observed on or before Day 28 relative to the course presentation start can be used as model input.

## 3. Target
The target is a conceptual binary variable:
* `is_withdrawn = 1` if `final_result == "Withdrawn"`
* `is_withdrawn = 0` otherwise

The original `final_result` string is NOT a model feature.

## 4. Registration Feature Correction
* Feature Name: `registration_day` (mapped directly from `date_registration`).
* **Sign Convention**: 
  * Negative values mean registration occurred before the course started (Day 0).
  * Zero means registration occurred exactly on Day 0.
  * Positive values mean registration occurred after Day 0.
* We preserve the original OULAD sign convention. `date_unregistration` is NEVER a model feature.

## 5. Temporal Events
* **VLE events**: Filtered strictly to `date <= 28`.
* **Assessment submissions**: Filtered strictly to `date_submitted <= 28`. The scheduled assessment date may be used to calculate submission delay (e.g., if a student submitted early), but the actual submission timestamp determines whether the student's information is available by the cutoff.

## 6. Pre-Registration Observations
* **VLE**: Empirical validation showed NO VLE interactions occur before a student's recorded registration date. We retain all VLE events with `date <= 28`.
* **Assessment**: 115 submissions (across 25 students) occur before their recorded registration date. We retain these observed submissions because they are available observations, but we will create a data-quality/anomaly indicator during feature engineering. This anomaly will not be used as a proxy for dropout.

## 7. Fairness Attributes
The following attributes are kept in the dataset for fairness auditing:
* `gender`
* `disability`
* `age_band`
* `region`
* `imd_band`

For the PRIMARY predictive model, they are excluded from the input features (X). This blind approach does not automatically guarantee fairness; fairness will be empirically evaluated and mitigation may be applied later during MLOps validation.

## 8. Predictive Feature Groups
The V1 predictive feature groups are strictly frozen as follows (Total: 21 features):

### Background
* `highest_education`
* `num_of_prev_attempts`
* `studied_credits`

### Registration
* `registration_day`

### Course Context
* `code_module`
* `code_presentation`
* `module_presentation_length`

### Assessment
* `asm_submission_count`
* `asm_mean_score`
* `asm_score_std`
* `asm_failed_count`
* `asm_average_delay`

### VLE / Engagement
* `vle_total_clicks`
* `vle_active_days`
* `vle_days_since_last_activity`
* `vle_clicks_last_7_days`
* `vle_clicks_last_14_days`
* `vle_forum_clicks`
* `vle_resource_clicks`
* `vle_quiz_clicks`
* `vle_activity_type_diversity`

## 9. Identifiers
The following are retained for traceability:
* `id_student`
* `code_module`
* `code_presentation`

`id_student` is strictly an identifier, not a predictive feature. `code_module` and `code_presentation` act as both tracing identifiers and potential categorical contextual predictors.

## 10. Forbidden Inputs
The following are explicitly frozen as forbidden:
* `final_result`
* `is_withdrawn`
* `date_unregistration`
* any VLE observation with `date > 28`
* any assessment submission with `date_submitted > 28`
* any aggregate calculated before temporal filtering
* any statistic derived from the student's future course history
* any post-Day-28 information
* target-derived features

## 11. Join Contract
### Base
`studentInfo`

### Registration
`studentRegistration`
**On**: `id_student + code_module + code_presentation`

### Course context
`courses`
**On**: `code_module + code_presentation`

### Assessment
`studentAssessment` $\rightarrow$ `assessments`
**On**: `id_assessment`
* **Filter**: `studentAssessment` to `date_submitted <= 28`
* **Aggregate by**: `id_student + code_module + code_presentation`

### VLE
`studentVle` $\rightarrow$ `vle`
**On**: `id_site + code_module + code_presentation`
* **Filter**: `studentVle` to `date <= 28`
* **Aggregate by**: `id_student + code_module + code_presentation`

**Final Step**: Left join all aggregated features onto the base cohort.

## 12. Final Row-Level Guarantee
The final modeling table MUST satisfy:
`COUNT(*) == COUNT(unique(id_student, code_module, code_presentation))`
Any violation must cause validation failure rather than silently proceeding.

## 13. Missingness Policy
* Zero VLE activity is structural absence and may become zero.
* No early assessment submission produces missing academic aggregates (NaN).
* `imd_band == "?"` and `date_registration == "?"` are explicit missing/unknown values.
* Final imputation handling will be evaluated during preprocessing; no final decision is made yet.

## 14. Data Quality Flags
We plan to introduce flags later to track data quality issues, including:
* `assessment_submission_before_registration`
* `missing_registration`
* `missing_socioeconomic_information`
* `no_vle_activity_by_day28`
* `no_assessment_submission_by_day28`

These flags should NOT automatically be predictive features. Their eventual treatment will be evaluated during preprocessing.
