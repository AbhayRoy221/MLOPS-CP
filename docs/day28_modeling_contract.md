# Day-28 Modeling Contract

## 1. Prediction Unit
ONE student + ONE module + ONE presentation.

## 2. Primary Cutoff
**DAY 28** relative to the presentation start.

## 3. Target Definition
Whether the student eventually has `final_result == "Withdrawn"`.

## 4. Eligibility Definition
A record is eligible for prediction at Day 28 if:
* It belongs to a valid student-module-presentation record.
* It is **not** a confirmed pre-Day-28 withdrawal (`date_unregistration` is either $> 28$ or legitimately null).
* Its future target is retained separately.
* No information occurring after Day 28 will be used as an input feature.
* It does not belong to logically inconsistent edge cases (see sections 5 and 6).

## 5. Treatment of Missing Withdrawal Dates
**Observation**: 93 records have `final_result == "Withdrawn"` but a missing `date_unregistration`.
**Analysis**: Among these 93 records, there is evidence of post-Day-28 activity (11 have VLE activity after Day 28, and 8 have assessment submissions after Day 28). We cannot reliably infer their actual withdrawal date.
**Recommendation**: **Exclude** these 93 records from the primary Day-28 cohort. Retaining them risks introducing temporal-label uncertainty (e.g., treating them as eligible at Day 28 when they might have actually withdrawn on Day 10).

## 6. Treatment of Inconsistent Unregistration Records
**Observation**: 9 records have `final_result != "Withdrawn"` (all are "Fail") but have a recorded `date_unregistration`.
**Analysis**: 8 of them have unregistration dates $\le 28$ (e.g., 0, -4, -7) and 1 has unregistration on day 166. This is logically inconsistent.
**Recommendation**: **Exclude** these 9 records from the primary cohort to maintain label integrity.

## 7. Treatment of Pre-Course Withdrawals
**Observation**: There are 2,678 records with an unregistration date $< 0$ (2,676 Withdrawn, 2 Non-Withdrawn).
**Treatment**: These records are **automatically excluded** by the Day-28 eligibility rule because their withdrawal occurred on or before Day 28 (in fact, before Day 0).

## 8. Allowed Information at Prediction Time
Only information with timestamp/date $\le$ Day 28 may become an input feature.
* Student demographics and registration data (if registered $\le$ 28).
* Aggregate VLE clicks where `date` $\le 28$.
* Assessment scores/counts where `date_submitted` $\le 28$.

## 9. Forbidden Information / Leakage
The following must NEVER be used as input features to avoid target leakage and looking into the future:
* `final_result`
* `date_unregistration`
* any VLE observation after Day 28 (`date` $> 28$)
* any assessment submission after Day 28 (`date_submitted` $> 28$)
* any derived aggregate that uses post-Day-28 observations (e.g., total course clicks, final course score)

## 10. Assessment Timing Rules
**Observation**: 17 assessments are scheduled to be due $\le$ Day 28 (based on `assessments.csv`). However, there are **25,614** actual student assessment submissions occurring $\le$ Day 28 (based on `studentAssessment.csv`).
**Rule**: We must NOT describe the 25,614 early submissions as "25,614 scheduled assessments." For feature engineering, we must strictly filter by the student's actual `date_submitted`, NOT the assessment's scheduled `date`. A student may submit an assessment early (before Day 28) even if it is scheduled for Day 40, which makes that submission a valid feature. Conversely, they may submit late, making it invalid for a Day-28 model.

## 11. VLE Timing Rules
**Rule**: VLE interactions are only eligible if the interaction `date` $\le 28$.
*(Note: There are 29,440 rows in the raw dataset where VLE activity occurs after a student's recorded withdrawal date. These will naturally be filtered out if the withdrawal was $\le$ 28, but for post-28 withdrawals, they remain a known data quality quirk).*

## 12. Final Cohort Statistics
Applying all eligibility rules and exclusions for the Day-28 model:
* **Total original records**: 32,593
* **Excluded**: 5,149 records
  * Withdrew on or before Day 28: 5,047
  * Dropped due to missing unregistration (Withdrawn): 93
  * Dropped due to inconsistent unregistration (Non-Withdrawn): 9
* **Eligible Records**: 27,444
  * **Eventual Withdrawn**: 5,016 (18.28% prevalence)
  * **Eventual Non-Withdrawn**: 22,428

## 13. Remaining Data-Quality Limitations
While building the feature pipelines, we must be aware of the following temporal inconsistencies in the OULAD dataset (though they do not invalidate the prediction task):
* **VLE activity after unregistration**: 29,440 instances.
* **Assessment submissions after unregistration**: 601 instances.
* **Assessment activity before registration**: 115 instances.
* **VLE activity before registration**: 0 instances.
