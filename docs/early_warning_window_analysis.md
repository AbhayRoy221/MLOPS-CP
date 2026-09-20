# Early-Warning Window Temporal Analysis

## Definitions
* **Prediction Unit**: ONE student + ONE module + ONE presentation.
* **Target Definition**: `final_result` == "Withdrawn"
* **Eligibility Definition**: A student is eligible for an early-warning prediction at a given day $D$ if they have *not* already unregistered on or before day $D$. If `date_unregistration` $\le D$, they are excluded to prevent target leakage. If `date_unregistration` is null or $> D$, they are eligible.

## 1. Registration & Withdrawal Analysis
* **Total Withdrawn**: 10,156
* **Withdrawn w/ non-null `date_unregistration`**: 10,063
* **Withdrawn w/ null `date_unregistration`**: 93
* **Unregistration Date Statistics (for Withdrawn)**:
  * Minimum: -365 (Registered & withdrew long before module start)
  * Maximum: 444 (Withdrew long after module ended)
  * Mean: 49.79 days
  * Median: 27 days

## 2. Early-Warning Cutoff Comparison
Total dataset records: 32,593

### Cutoff: Day 14 (2 weeks)
* **Eligible Population Size**: 28,119
* **Eligible Eventual Withdrawn**: 5,690 (20.24% of eligible)
* **Eligible Non-Withdrawn**: 22,429
* **Excluded (Already Withdrawn)**: 4,474 (13.73% of total)

### Cutoff: Day 28 (4 weeks)
* **Eligible Population Size**: 27,538
* **Eligible Eventual Withdrawn**: 5,109 (18.55% of eligible)
* **Eligible Non-Withdrawn**: 22,429
* **Excluded (Already Withdrawn)**: 5,055 (15.51% of total)

### Cutoff: Day 42 (6 weeks)
* **Eligible Population Size**: 27,024
* **Eligible Eventual Withdrawn**: 4,595 (17.00% of eligible)
* **Eligible Non-Withdrawn**: 22,429
* **Excluded (Already Withdrawn)**: 5,569 (17.09% of total)

## 3. Temporal Observations (Day 28)
* **Assessments Scheduled $\le$ Day 28**: 17 out of 206 (8.25%)
* **Assessments Submitted $\le$ Day 28**: 25,614
* **Assessments Submitted $>$ Day 28**: 148,298
* **Total VLE Interactions**: 10,655,280
* **VLE Interactions $\le$ Day 28**: 2,847,851 (26.7%)
* **VLE Interactions $>$ Day 28**: 7,807,429
* **VLE Date Range**: Min -25, Max 269

## 4. Edge Cases & Inconsistencies
* **Withdrawn with NO `date_unregistration`**: 93 records are marked "Withdrawn" but have a null withdrawal date. We must assume they were eligible to avoid target leakage, or drop them. (Recommend dropping or imputing carefully).
* **Non-Withdrawn WITH `date_unregistration`**: 9 records are marked as Pass, Fail, or Distinction but have a recorded `date_unregistration`. This is logically inconsistent.
* **Pre-course Withdrawals**: The median withdrawal date is 27 days, but the minimum is -365. Many students drop out before the course even begins (negative dates).
* **Late Interactions**: VLE and assessment dates occasionally stretch past the maximum presentation length (up to 269 days).

## 5. Recommendation
**Recommended Primary Cutoff: Day 28 (4 weeks)**

**Justification**:
1. **Practical Usefulness**: 4 weeks into a typical 9-month (269-day) module provides sufficient time for human review and academic intervention.
2. **Sufficient Sample Size**: We retain 27,538 eligible students. Only 15.51% of the cohort is excluded for having already withdrawn.
3. **Sufficient Positive Examples**: We retain 5,109 eventual dropouts (18.55% prevalence), yielding a well-balanced minority class that is easy to train on.
4. **Meaningful Information**: By Day 28, we capture over 2.8 million VLE interaction logs (~27% of all activity) and over 25,000 assessment submissions. This provides a strong behavioral signal before making predictions. At Day 14, behavioural signals are significantly weaker.

## 6. Assumptions Needing Verification
* The 93 withdrawn students with missing unregistration dates need to be addressed (likely removed from the modeling set).
* The 9 non-withdrawn students with unregistration dates should be investigated or removed.
