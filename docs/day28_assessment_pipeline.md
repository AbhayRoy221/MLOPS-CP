# Day-28 Assessment Feature Pipeline

## Purpose
This pipeline module extracts and aggregates early academic performance signals from the OULAD assessment tables. It calculates 5 frozen academic features summarizing a student's assessment activity available strictly on or before Day 28 of their course presentation.

## Input Tables
* `data/raw/assessments.csv` (Assessment metadata, weights, scheduled dates)
* `data/raw/studentAssessment.csv` (Actual student submissions and scores)
* `data/processed/day28_base_cohort.parquet` (The strictly filtered base eligible population)

## Join Key
* Assessment mapping: `id_assessment`
* Aggregation/Cohort joining: `id_student + code_module + code_presentation`

## Temporal Filtering
* Only assessment submissions with `date_submitted <= 28` are retained.
* This is applied *before* aggregation to prevent future submissions (e.g., Day 40) from leaking into early-warning aggregates.
* Scheduled dates (`date`) are not used for filtering, as students can submit early or late regardless of the schedule.

## Five Feature Definitions
1. **`asm_submission_count`**: Total number of qualifying submissions ($\le 28$).
2. **`asm_mean_score`**: Mean score of qualifying submissions.
3. **`asm_score_std`**: Standard deviation of qualifying scores (using sample std, yields NaN if only 1 observation).
4. **`asm_failed_count`**: Number of qualifying submissions where `score < 40`.
5. **`asm_average_delay`**: Mean of `(date_submitted - scheduled_date)`. Negative values indicate early submission. Submissions lacking a valid scheduled date are excluded from the delay average.

## Missing-Value Behavior
* Students with zero qualifying assessment submissions are retained in the dataset via a left join.
* `asm_submission_count` and `asm_failed_count` are imputed to `0`.
* `asm_mean_score`, `asm_score_std`, and `asm_average_delay` remain structurally missing (`NaN`). No arbitrary numerical imputation is performed at this stage.

## Anomaly Flag
* **`assessment_submission_before_registration`**: A diagnostic indicator set to `1` if the student submitted an assessment before their recorded registration date. It is not used as a predictive feature and does not trigger row exclusion.

## Leakage Controls
* **Temporal Audits**: The code programmatically asserts that the maximum `date_submitted` entering the aggregation step is $\le 28$.
* **Target Isolation**: `final_result` and `is_withdrawn` are explicitly verified to not exist in the output artifact.
* **Population Guard**: The output artifact is restricted via inner/left joins to perfectly match the length and keys of the `day28_base_cohort`.

## Output Schema
File: `data/processed/day28_assessment_features.parquet`
* `id_student`
* `code_module`
* `code_presentation`
* `asm_submission_count`
* `asm_mean_score`
* `asm_score_std`
* `asm_failed_count`
* `asm_average_delay`
* `assessment_submission_before_registration`

## Test Procedure & Validation Statistics
* Unit tests exist in `tests/test_assessment_features.py` to synthetically verify behavior (exclusion of late submissions, NaN imputation, delay math, and single-observation behavior).
* Validation statistics are printed to standard output during runtime to visually confirm zero-assessment counts, anomaly counts, and composite-key integrity.
