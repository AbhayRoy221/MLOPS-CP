# Feature and Join Specification: Day-28 Early-Warning Model

## 1. Modeling Grain
**Intended Unique Composite Key**: `id_student` + `code_module` + `code_presentation`

**Why `id_student` alone is NOT the correct key**: 
A single student can take multiple modules concurrently, or retake the same module across different presentations (semesters). Modeling at the student level would incorrectly aggregate interactions across completely separate academic courses, blurring the signals for a specific course withdrawal and causing duplicate target ambiguities. Our prediction is whether a student drops out of a *specific* presentation of a *specific* module.

## 2. Source Tables

| Table | Required | Contribution | Purpose |
| :--- | :--- | :--- | :--- |
| **studentInfo.csv** | Yes | Base cohort definitions, demographics, outcomes. | Model features, Target generation, Fairness auditing. |
| **studentRegistration.csv** | Yes | Registration dates, withdrawal dates (for filtering). | Model features, Context/Eligibility filtering. |
| **courses.csv** | Yes | Presentation lengths. | Context, Model features. |
| **assessments.csv** | Yes | Assessment structure, weights, scheduled dates. | Context, Validation only (to map submissions). |
| **studentAssessment.csv** | Yes | Actual student scores and submission dates. | Model features. |
| **vle.csv** | Yes | VLE material metadata (activity types). | Context (to enrich VLE clicks). |
| **studentVle.csv** | Yes | Granular daily clicks on VLE materials. | Model features. |

## 3. Exact Join Strategy

To preserve exactly ONE row per `id_student + code_module + code_presentation`, we must aggregate the one-to-many behavioral tables before joining them to the base table.

1. **Base Table Generation**: Start with `studentInfo`.
2. **Context Join**: Left join `studentRegistration` and `courses` onto the Base Table using the composite key: `id_student, code_module, code_presentation`.
3. **Assessment Aggregation**: 
   - Inner join `studentAssessment` with `assessments` on `id_assessment`.
   - **Filter**: `date_submitted` $\le 28$.
   - **Aggregate**: Group by `id_student, code_module, code_presentation` to calculate sum, mean, count, etc.
   - Left join aggregated results to the Base Table on `id_student, code_module, code_presentation`.
4. **VLE Aggregation**:
   - Inner join `studentVle` with `vle` on `id_site, code_module, code_presentation`.
   - **Filter**: `date` $\le 28$.
   - **Aggregate**: Group by `id_student, code_module, code_presentation` to calculate click counts, diversity, etc.
   - Left join aggregated results to the Base Table on `id_student, code_module, code_presentation`.

**Leakage Risk**: Joining `studentAssessment` or `studentVle` directly to the base table *without* aggregating first will explode the dataset into a one-to-many format, creating duplicate student-module-presentation rows and invalidating the prediction unit.

## 4. Feature Groups

### A. Enrollment / Background Features
* **gender, region, age_band, imd_band, disability**: Used for Fairness Auditing (and optionally predictive). Rationale: Important for equity analysis to ensure the model doesn't bias against protected groups. Risk: None (static at enrollment).
* **highest_education, num_of_prev_attempts, studied_credits**: Predictive. Rationale: Prior academic history and current workload strongly predict retention. Risk: None.

### B. Registration Features
* **days_from_registration_to_course_start**: Predictive. Derived from `date_registration` (negative values mean they registered before start). Rationale: Late registrants often struggle to catch up. Risk: None.
* *(Forbidden: `date_unregistration`)*

### C. Assessment Features Available by Day 28
*(Aggregated strictly where `date_submitted` $\le 28$)*
* **number_of_assessments_submitted_by_day28**: Count of submissions.
* **mean_score_by_day28**: Average score of submitted assessments.
* **score_std_by_day28**: Variance in early scores.
* **failed_assessment_count**: Count of submissions scoring $< 40$.
* **average_submission_delay**: Mean of (`date_submitted` - scheduled `date`), only calculated if the scheduled `date` is valid and the submission was made $\le$ Day 28. Early submissions yield negative delays.

**Temporal Note**: We ONLY use the actual `date_submitted` to filter data. A student submitting an assignment on Day 25 that was scheduled for Day 40 is valid information. A student submitting an assignment on Day 30 that was scheduled for Day 20 is forbidden information.

### D. VLE / Engagement Features Available by Day 28
*(Aggregated strictly where `date` $\le 28$)*
* **total_clicks_by_day28**: Sum of `sum_click`.
* **active_days_by_day28**: Count of unique `date` values a student logged in.
* **days_since_last_activity**: $28 - \max(date)$.
* **activity_type_diversity**: Count of unique `activity_type` engaged with.
* **clicks_last_7_days**: Sum of clicks where $21 < date \le 28$.
* **clicks_forumng**: Sum of clicks where `activity_type == 'forumng'`.

**Pre-registration observations**: Many VLE events occur before Day 0 and occasionally before `date_registration`. **Rule**: Include them. Early exploration of the VLE (e.g., orientation modules) before the official start is a valid, highly predictive behavioral signal available at Day 28.

### E. Course Context Features
* **module, presentation, module_presentation_length**: Safe to use. Rationale: Different courses have different baselines for dropout rates and lengths. 

## 5. Sensitive Attributes and Fairness Design

* **gender, disability, age_band, region, imd_band**:
  * **Treatment**: Exclude from the primary production predictive features (Blind fairness approach), but RETAIN in the underlying validation dataset as Fairness Audit Attributes.
  * **Rationale**: We want to predict dropout based on academic and behavioral engagement, not demographic destiny. Using them as predictive inputs risks automating historical biases (e.g., flagging students from lower `imd_band` regions automatically). They must be kept to calculate Equalized Odds, Demographic Parity, and perform subgroup analysis during MLOps validation.

## 6. Leakage Rules

| Field / Derived Feature | Allowed? | Reason |
| :--- | :--- | :--- |
| `final_result` | **NO** | The target variable. Total leakage. |
| `date_unregistration` | **NO** | Directly reveals the exact day a student dropped out. |
| `date_registration` | YES | Known at enrollment. |
| `assessments.date` (scheduled) | YES | Known in the syllabus before the course starts. |
| `studentAssessment.date_submitted` | YES | Allowed ONLY to filter submissions $\le$ 28. |
| `studentAssessment.score` | YES | Allowed ONLY if corresponding `date_submitted` $\le$ 28. |
| `studentVle.date` | YES | Allowed ONLY to filter interactions $\le$ 28. |
| `module_presentation_length` | YES | Context known beforehand. |
| Aggregate: `total_course_clicks` | **NO** | Requires data from Day 29 to end of course. |

**Why "aggregate first, filter later" fails**: If you group all VLE data by student, calculate their total clicks, and *then* filter out students who withdrew before Day 28, you have already leaked their Day 29-269 clicks into the `total_clicks` feature. You must filter the raw, unaggregated event rows `date <= 28` *before* executing the `GROUP BY`.

## 7. Temporal Feature Rule
**Final Recommended Temporal Rule**:
`feature = f(events with event_date <= 28)`
This strictly includes pre-course ($<0$) and pre-registration events, capturing all known history up to the midnight of Day 28 relative to the presentation start. 

## 8. Missing Values
* **Structurally Missing**: If a student has no VLE clicks $\le 28$, `total_clicks` will be NULL after the left join. This means 0. Impute with 0.
* **Structurally Missing**: If a student submitted no assessments $\le 28$, `mean_score` will be NULL. Impute with a flag/category (e.g., -1 or NaN handled natively by XGBoost).
* **Genuinely Missing**: `imd_band` and `date_registration` contain actual missing values (e.g., '?'). These should be treated as a distinct "Missing" category, as the lack of data may be correlated with administrative issues or international status.

## 9. Feature Categories for the Final Dataset

| Feature | Source Table | Type | Time Boundary | Predictive/Audit | Leakage Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `highest_education` | studentInfo | Categorical | Static | Predictive | None |
| `num_of_prev_attempts` | studentInfo | Numeric | Static | Predictive | None |
| `studied_credits` | studentInfo | Numeric | Static | Predictive | None |
| `imd_band` | studentInfo | Ordinal | Static | Audit Only | None |
| `gender`, `disability`, `region` | studentInfo | Categorical | Static | Audit Only | None |
| `days_to_reg` | studentRegistration | Numeric | Static | Predictive | None |
| `total_clicks_by_day28` | studentVle | Numeric | $\le$ Day 28 | Predictive | High (if unfiltered) |
| `active_days_by_day28` | studentVle | Numeric | $\le$ Day 28 | Predictive | High (if unfiltered) |
| `clicks_last_7_days` | studentVle | Numeric | Day 21-28 | Predictive | High (if unfiltered) |
| `days_since_last_vle` | studentVle | Numeric | $\le$ Day 28 | Predictive | High (if unfiltered) |
| `forum_clicks_by_day28` | studentVle + vle | Numeric | $\le$ Day 28 | Predictive | High (if unfiltered) |
| `assessments_submitted_by_day28`| studentAssessment | Numeric | $\le$ Day 28 | Predictive | High (if unfiltered) |
| `mean_score_by_day28` | studentAssessment | Numeric | $\le$ Day 28 | Predictive | High (if unfiltered) |

## 10. Recommended First Feature Set
We recommend a lean, robust V1 feature set of ~25 features:
* **Background**: `highest_education`, `num_of_prev_attempts`, `studied_credits`
* **Registration**: `days_to_registration` (derived)
* **Context**: `code_module`, `code_presentation`, `module_presentation_length`
* **Academic ($\le$ 28)**: `submitted_assessments_count`, `mean_score`, `score_std`, `failed_assessments_count`, `average_submission_delay`
* **Engagement ($\le$ 28)**: `total_clicks`, `active_days`, `days_since_last_click`, `clicks_last_7_days`, `clicks_last_14_days`, `forum_clicks`, `resource_clicks`, `quiz_clicks`, `activity_type_diversity`

## 11. Future Ablation Experiments
* **A. Background only**: Tests if demographic/historical data alone determines dropout (baseline).
* **B. Academic only**: Tests if early grades are the sole driver of retention.
* **C. Engagement only**: Tests if VLE behavior alone can predict dropout before grades are available.
* **D. Background + Academic**: Traditional university retention model.
* **E. Background + Academic + Engagement**: Full holistic model (our proposed).
* **F. Full temporal feature set**: Compares Day 28 vs Day 42 vs Day 60 to measure the information gain over time.

## 12. Final Data Pipeline Concept

```
studentInfo + studentRegistration + courses
                      ↓
           [Base Cohort Eligibility Filter] -> Drop edge cases, unreg <= 28
                      ↓
              [Base Population (27,444 rows)]
                      |
                      |    [studentVle + vle] -> Filter date <= 28 -> GroupBy(key) -> [VLE Features]
                      |------------------------------------------------------------------/
                      |
                      |    [studentAssessment + assessments] -> Filter date_submitted <= 28 -> GroupBy(key) -> [Assessment Features]
                      |------------------------------------------------------------------------------------------/
                      ↓
           [Final Modeling Dataset (ONE row per student-module-presentation)]
                      ↓
      [Separate Features (X) and Target/Fairness variables (Y, A)]
```

## 13. Important Research Questions / Decisions Still Open
* **Imputation Strategy**: Does treating missing VLE clicks as 0 perform better than using XGBoost's native NaN handling?
* **Early Registration**: Do students with highly negative registration dates exhibit fundamentally different behaviors than late registrants?
* **Assessment Data Sparsity**: With only 17 assessments officially scheduled $\le 28$, the assessment features will be zero/null for many students. Does VLE engagement fully compensate for missing early grades?
