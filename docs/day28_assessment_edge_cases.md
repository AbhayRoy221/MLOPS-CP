# Day-28 Assessment Feature Edge Cases Investigation

## 1. Six-Record Discrepancy Investigation
During validation, `asm_submission_count == 0` occurred for 7,058 records, but `asm_mean_score` was `NaN` for 7,064 records. This discrepancy of exactly 6 records corresponds to the following students:
* 531205 (BBB 2013B)
* 606501 (BBB 2014B)
* 678578 (BBB 2014J)
* 427248 (DDD 2013J)
* 174436 (FFF 2013B)
* 126074 (FFF 2013J)

**Finding**: Option A. All 6 students submitted exactly one assessment before Day 28, but their recorded `score` was `'?'` (missing) in the raw OULAD data. Because there were 0 valid numerical scores to average, the resulting mean score is correctly `NaN` despite having 1 submission. The pipeline logic is completely correct.

## 2. Missing-Score Behavior (Mean Score Denominator)
* Rows with missing scores (`?`) are natively coerced to `NaN` by pandas during conversion.
* When aggregating via `.groupby().mean()`, pandas mathematically excludes these `NaN` values from both the numerator and the denominator. 
* **Conclusion**: Missing scores are cleanly excluded and are NOT artificially penalized as zeros.

## 3. Score Standard Deviation Behavior
The `asm_score_std` behaves as follows natively via pandas sample standard deviation (`ddof=1`):
* **0 valid scores**: `NaN`
* **1 valid score**: `NaN` (standard mathematical definition, as variance requires $n > 1$)
* **2+ valid scores**: Computes the actual standard deviation.
* **Conclusion**: This is mathematically sound and consistent. Future model imputation pipelines can choose how to impute these `NaN` values (e.g., using `0` variance for single submissions).

## 4. Failed-Score Behavior
The failure count is strictly computed as `score < 40`.
* **Missing scores (`?`)**: Coerced to `NaN`. `NaN < 40` evaluates to `False`. Thus, missing scores do not inflate the failure count.
* **Score == 40**: Evaluates to `False` and is correctly not counted as a failure.
* **Conclusion**: The logic perfectly implements the strict mathematical threshold `< 40`.

## 5. Submission-Delay Distribution
* **Minimum**: `-239.0` (indicates extremely early submission, likely an artifact of repeating students or orientation tasks)
* **Maximum**: `16.0`
* **Median**: `-1.0`
* **Mean**: `-13.17`
* **Distribution (among cohort)**: 12,423 Negative, 3,533 Zero, 4,430 Positive.
* **Missing**: 7,058 (exactly aligns with the zero-assessment students who have no submissions to average).
* **Conclusion**: Delay statistics are robust, though extreme negative outliers exist and must be handled during modeling if using linear/distance-based algorithms.

## 6. Scheduled-Date Analysis
For the 206 assessment definitions:
* **Missing (`?`)**: 11
* **Negative**: 0
* **<= 28**: 17
* **> 28**: 178
* **Conclusion**: This is why filtering on `date_submitted` was critical! Only 17 assessments were scheduled by Day 28, but students submitted thousands of assessments that were officially scheduled for later in the presentation (e.g. Day 40).

## 7. Leakage Verification
* `max(date_submitted)` entering the aggregation logic strictly equals `28.0`.
* Post-Day-28 submissions were strictly excluded prior to aggregation.
* Target and registration logic was appropriately handled without leaking.

## 8. Final Recommendation
**Recommendation**: A (Remain unchanged).
The current pipeline flawlessly handles edge cases using robust native pandas behaviors without requiring methodological corrections. Missing values are excluded rather than corrupted, and strict temporal filtering is respected.
