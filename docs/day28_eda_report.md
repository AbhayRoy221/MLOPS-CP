# Day-28 Exploratory Data Analysis Report

## 1. Dataset Overview
The dataset contains exactly 27,444 unique rows, representing every `id_student` + `code_module` + `code_presentation` enrollment in the cohort. The dataset spans 33 columns with a memory footprint of approximately 21.37 MB.

## 2. Target Distribution
The target variable `is_withdrawn` has the following distribution:
* **0 (Non-Withdrawn)**: 22,428 (81.72%)
* **1 (Withdrawn)**: 5,016 (18.28%)
* **Imbalance Ratio**: 4.47:1
Withdrawal prevalence in this Day-28 cohort is 18.28%.

## 3. Feature Profile
There are 21 explicitly defined predictive features.
* **Categorical**: `highest_education`, `code_module`, `code_presentation`
* **Numeric**: `num_of_prev_attempts`, `studied_credits`, `registration_day`, `module_presentation_length`, `asm_submission_count`, `asm_mean_score`, `asm_score_std`, `asm_failed_count`, `asm_average_delay`, `vle_total_clicks`, `vle_active_days`, `vle_days_since_last_activity`, `vle_clicks_last_7_days`, `vle_clicks_last_14_days`, `vle_forum_clicks`, `vle_resource_clicks`, `vle_quiz_clicks`, `vle_activity_type_diversity`

## 4. Missingness
Missingness takes two distinct forms: true Numeric NaN, and explicit OULAD unknown markers represented by `"?"`. 
### Numeric NaN
* **`asm_mean_score`**: 7,064 missing. (Students lacking graded submissions).
* **`asm_average_delay`**: 7,058 missing.
* **`asm_score_std`**: 24,206 missing. (Calculation requires >= 2 submissions).
* **`vle_days_since_last_activity`**: 900 missing. (Students with zero VLE clicks).
* **`registration_day`**: 7 missing.

### OULAD unknown markers ("?")
* **`imd_band`**: 1,027 unknown.

## 5. Numeric Analysis
The raw numeric distributions exhibit significant skew. Most engagement metrics (clicks, active days) are highly right-skewed, showing that a small proportion of students generate a massive volume of clicks.

## 6. Categorical Analysis
The categorical variables (`code_module`, `code_presentation`, `highest_education`) demonstrate diverse distributions with structurally large categories representing primary university offerings.

## 7. Assessment Analysis
Over 25% of the student population submitted no assessments by Day 28. The absence of an early assessment score is strongly associated with a higher observed withdrawal rate.

## 8. VLE Analysis
VLE variables display high variance. 900 students had zero clicks by Day 28. Extremely low VLE engagement (clicks, active days) is associated with a higher observed withdrawal rate.

## 9. Target Associations
* Lower early assessment scores and fewer assessment submissions are descriptively associated with a higher withdrawal rate.
* Lower engagement in the VLE (especially `vle_active_days` and `vle_total_clicks`) is associated with higher withdrawal risk.

## 10. Fairness-Group Descriptive Analysis
Fairness attributes (`gender`, `disability`, `age_band`, `region`, `imd_band`) exhibit varying baseline withdrawal rates. 
For instance, records with an `imd_band` of 0-10% (highest deprivation) exhibit a 21.25% withdrawal rate, compared to a 15.85% rate in the 90-100% band. 

## 11. Correlation/Redundancy
Significant correlations exist internally within VLE features:
* `vle_total_clicks` is strongly correlated with `vle_active_days`.
* `vle_clicks_last_14_days` and `vle_clicks_last_7_days` contain highly overlapping signals.

## 12. Data-Quality Observations
Diagnostic flags perfectly capture data exceptions:
* `missing_registration`: 7 (Withdrawal rate: 14.29%)
* `missing_socioeconomic_information`: 1027 (Withdrawal rate: 14.80%)
* `assessment_submission_before_registration`: 25 (Withdrawal rate: 20.00%)
* `no_vle_activity_by_day28`: 900 (Withdrawal rate: 20.33%)

## 13. Modeling Considerations
* **Missing Data**: The missingness in assessment metrics (e.g. 7,064 NaNs for mean score) is heavily structural. Using basic mean imputation would incorrectly inject false signal. Missing values require explicit indicator modeling.
* **Skewness**: Click volume is profoundly skewed; modeling with unscaled distance algorithms or standard linear regression will require robust log-transformation.
* **Categorical Handling**: Features like `imd_band` technically have a missing category (`"?"`) that must be preserved or distinctly encoded during categorical preprocessing.

## 14. Preliminary Preprocessing Recommendations
*(Based purely on observed data facts. No actions taken yet.)*
* **Imputation**: Adopt a constant/out-of-range imputation strategy (e.g. `-1` or `0`) for assessment and VLE NaNs if the model cannot handle them natively.
* **Encoding**: Convert ordinal bands (`imd_band`, `age_band`) logically, retaining `"?"` as an independent indicator class or explicitly treating it as missing.
* **Scaling**: Skewed continuous VLE metrics may benefit from robust scaling.

## 15. Limitations
This is a purely observational Descriptive EDA. No causal relationships (e.g., "low clicks cause dropout") can be asserted. Unobserved, external confounding variables likely influence both behavioral indicators and withdrawal outcomes.
