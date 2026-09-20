# Feature Specification Validation

## 1. Composite-Key Uniqueness
* **Total rows in `studentInfo.csv`**: 32,593
* **Unique composite keys (`id_student + code_module + code_presentation`)**: 32,593
* **Duplicate count**: 0
* **Conclusion**: The composite key is perfectly unique in the base table. Since `studentRegistration` and `courses` share this exact grain, joining them on this key will strictly maintain the 1:1 mapping without duplicate expansion.

## 2. Registration Timing
* **Min**: -322 days
* **Max**: 167 days
* **Median**: -57.0 days
* **Missing/Invalid (`?`)**: 45 values
* **Distribution**: The vast majority of students register well before the course starts (negative values).
* **Canonical Feature Name**: `days_from_registration_to_course_start`.
  * *Formula*: `-1 * date_registration`. By OULAD convention, a negative `date_registration` means they registered *before* the course. Multiplying by -1 makes it a positive integer representing "lead time" (e.g., 57 days lead time), which is more intuitive for ML models than negative dates.

## 3. Pre-Registration VLE Observations
* **VLE Total Interactions**: 10,655,280
* **VLE Interactions before Day 0 (`date < 0`)**: 688,988 (6.47%)
* **VLE Interactions before Registration (`date < date_registration`)**: 0 (0.00%)
* **VLE Interactions on/after Registration**: 10,654,677 (99.99%) *(Excluding the 603 interactions belonging to students with missing registration dates)*
* **Affected unique students (VLE < reg)**: 0

**Recommendation**: **A. Include all VLE events $\le$ Day 28, including pre-registration events.**
*Justification*: The empirical validation proves that *zero* VLE interactions occur before a student's registration date. While 6.47% of interactions happen before the course officially starts (Day 0), students are strictly logging in *after* they have registered. Therefore, there is zero risk of mixing activity from before official registration, and capturing early exploration (before Day 0) is highly useful.

## 4. Pre-Registration Assessment Observations
* **Total Submissions**: 173,912
* **Submissions before Registration (`date_submitted < date_registration`)**: 115 (0.07%)
* **Affected unique students**: 25
* **Recommendation**: Retain them. This is an extremely rare edge case (affecting 25 out of 32,593 students) where a system glitch or administrative correction likely placed their registration date slightly after they began submitting work. Filtering them out would destroy valid academic signals for Day 28.

## 5. Candidate Feature Count
* **Predictive Features**: 17
* **Contextual Features**: 1 (`module_presentation_length`)
* **Fairness/Audit Attributes**: 5 (`gender`, `disability`, `age_band`, `region`, `imd_band`)
* **Identifiers**: 3 (`id_student`, `code_module`, `code_presentation`)
* **Target**: 1 (`final_result` / `is_withdrawn`)
* **Total V1 Features**: 27 explicit columns.

## 6. Canonical Feature Names

| Concept | Canonical Feature Name | Source | Time Rule |
| :--- | :--- | :--- | :--- |
| Registration timing | `days_from_registration_to_course_start` | studentRegistration | Static |
| Total clicks | `vle_total_clicks` | studentVle | `date` $\le 28$ |
| Active days | `vle_active_days` | studentVle | `date` $\le 28$ |
| Last activity | `vle_days_since_last_activity` | studentVle | `date` $\le 28$ |
| Recent 7-day clicks | `vle_clicks_last_7_days` | studentVle | $21 <$ `date` $\le 28$ |
| Recent 14-day clicks | `vle_clicks_last_14_days` | studentVle | $14 <$ `date` $\le 28$ |
| Forum clicks | `vle_forum_clicks` | studentVle + vle | `date` $\le 28$ |
| Resource clicks | `vle_resource_clicks` | studentVle + vle | `date` $\le 28$ |
| Quiz clicks | `vle_quiz_clicks` | studentVle + vle | `date` $\le 28$ |
| Activity diversity | `vle_activity_type_diversity` | studentVle + vle | `date` $\le 28$ |
| Assessment sub count | `asm_submission_count` | studentAssessment | `date_submitted` $\le 28$ |
| Mean score | `asm_mean_score` | studentAssessment | `date_submitted` $\le 28$ |
| Score std | `asm_score_std` | studentAssessment | `date_submitted` $\le 28$ |
| Failed assessments | `asm_failed_count` | studentAssessment | `date_submitted` $\le 28$ |
| Submission delay | `asm_average_delay` | studentAssessment + assessments | `date_submitted` $\le 28$ |

## 7. Join-Key Validation
* **`studentInfo` ↔ `studentRegistration`**: 1:1 match.
* **`studentAssessment` ↔ `assessments`**: Verified that `id_assessment` maps perfectly 1:1 to a specific `code_module` + `code_presentation`.
* **Unmatched `studentAssessment` to `studentInfo`**: 0 unmatched records.
* **Unmatched `studentVle` to `studentInfo`**: 0 unmatched records.
* **Expansion Risk**: Zero. All many-to-one logs match cleanly to the base cohort.

## 8. Course-Context Validation
* `module_presentation_length`, `code_module`, and `code_presentation` are completely static and known long before Day 28. They are 100% safe as contextual inputs.

## 9. Missingness Validation
* **`imd_band`**: 1,111 values are explicitly marked as `?` (structural unknown / refused to provide).
* **`date_registration`**: 45 values are explicitly marked as `?`.
* **VLE Features**: Students with 0 clicks will naturally result in SQL/Pandas `NULL` after a left join. This is structural absence (0 clicks), not missing data.
* **Assessment Features**: Students with 0 submissions $\le$ 28 will naturally result in `NULL` for `asm_mean_score`. This is structural absence (NaN).

## 10. Final Recommendation

### Recommended Temporal Rule
Include ALL events where `date` $\le 28$ and `date_submitted` $\le 28$. Pre-course events (negative dates) are valid and kept.

### Recommended Prediction Features
**18** explicit predictive features (17 behavioral/academic + 1 course length context).

### Fairness Audit Attributes
`gender`, `disability`, `age_band`, `region`, `imd_band`.

### Forbidden Variables
`final_result`, `date_unregistration`, any VLE `date` $> 28$, any Assessment `date_submitted` $> 28$.

### Join Strategy
* `studentRegistration` and `courses` via `id_student`, `code_module`, `code_presentation`.
* `studentAssessment` via `id_assessment` (to get module context), filter `date_submitted` $\le 28$, GROUP BY student/module/presentation, Left Join to base.
* `studentVle` via `id_site` (to get activity type), filter `date` $\le 28$, GROUP BY student/module/presentation, Left Join to base.

### Remaining Methodological Questions
* How does XGBoost handle `asm_mean_score = NaN` (no assessments submitted) vs. actual poor scores in an early-warning context? (To be tested during model training).
* Does `vle_days_since_last_activity` need to be capped/imputed for students with 0 clicks, or left as NaN?
