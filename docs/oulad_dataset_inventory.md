# OULAD Dataset Inventory

## A. Download Information
* **Source URL**: `https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip` (UCI Official Mirror for OULAD)
* **Archive Name**: `oulad.zip`
* **Download Size**: 46,748,244 bytes (~44.5 MB)
* **Status**: Download and extraction completed successfully.

## B. File Inventory
| Filename | Extension | Size (Bytes) | Rows | Columns |
| :--- | :--- | :--- | :--- | :--- |
| `assessments.csv` | .csv | 8,211 | 206 | 6 |
| `courses.csv` | .csv | 526 | 22 | 3 |
| `studentAssessment.csv` | .csv | 5,690,483 | 173,912 | 5 |
| `studentInfo.csv` | .csv | 3,462,763 | 32,593 | 12 |
| `studentRegistration.csv` | .csv | 1,132,550 | 32,593 | 5 |
| `studentVle.csv` | .csv | 453,836,331 | 10,655,280 | 6 |
| `vle.csv` | .csv | 270,612 | 6,364 | 6 |

## C. Schema

### `assessments.csv`
* `code_module` (String): Module identification code.
* `code_presentation` (String): Presentation/semester identification code.
* `id_assessment` (Integer): Unique identifier for the assessment.
* `assessment_type` (String): Type of assessment (TMA, CMA, Exam).
* `date` (Integer/String): Final submission date (relative to presentation start).
* `weight` (Float): Weight of the assessment in %.

### `courses.csv`
* `code_module` (String): Module identification code.
* `code_presentation` (String): Presentation/semester identification code.
* `module_presentation_length` (Integer): Length of the module presentation in days.

### `studentAssessment.csv`
* `id_assessment` (Integer): Assessment identifier.
* `id_student` (Integer): Student identifier.
* `date_submitted` (Integer): Day the student submitted the assessment.
* `is_banked` (Boolean/Integer): Whether the result is transferred from a previous attempt.
* `score` (Float): Student's score on the assessment.

### `studentInfo.csv`
* `code_module` (String): Module code.
* `code_presentation` (String): Presentation code.
* `id_student` (Integer): Student identifier.
* `gender` (String): Student's gender.
* `region` (String): Geographic region.
* `highest_education` (String): Student's highest education level at entry.
* `imd_band` (String): Index of Multiple Deprivation band (socioeconomic status).
* `age_band` (String): Student's age band.
* `num_of_prev_attempts` (Integer): Number of times the student previously attempted the module.
* `studied_credits` (Integer): Total credits studied by the student.
* `disability` (String): Student's declared disability status.
* `final_result` (String): Final outcome of the course (Pass, Fail, Withdrawn, Distinction).

### `studentRegistration.csv`
* `code_module` (String): Module code.
* `code_presentation` (String): Presentation code.
* `id_student` (Integer): Student identifier.
* `date_registration` (Integer): Day the student registered.
* `date_unregistration` (Integer): Day the student unregistered (withdrew).

### `studentVle.csv`
* `code_module` (String): Module code.
* `code_presentation` (String): Presentation code.
* `id_student` (Integer): Student identifier.
* `id_site` (Integer): VLE material identifier.
* `date` (Integer): Day of student's interaction with the material.
* `sum_click` (Integer): Number of times the student interacted with the material on that day.

### `vle.csv`
* `id_site` (Integer): VLE material identifier.
* `code_module` (String): Module code.
* `code_presentation` (String): Presentation code.
* `activity_type` (String): Type of VLE activity (e.g., resource, forum, quiz).
* `week_from` (Integer): Week from which the material is planned to be used.
* `week_to` (Integer): Week until which the material is planned to be used.

## D. Relationships
* **Primary Identifiers**: `id_student` (Student), `code_module` & `code_presentation` (Course offering), `id_assessment` (Assessment), `id_site` (VLE Material).
* **Foreign Keys**: 
  * `id_student` connects `studentInfo`, `studentRegistration`, `studentAssessment`, and `studentVle`.
  * `code_module` and `code_presentation` connect `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, and `studentVle`.
  * `id_assessment` connects `assessments` and `studentAssessment`.
  * `id_site` connects `vle` and `studentVle`.
* **Student-Level Data**: `studentInfo`, `studentRegistration`
* **Assessment Information**: `assessments`, `studentAssessment`
* **VLE/Activity Information**: `vle`, `studentVle`
* **Final Outcome Table**: `studentInfo`

## E. Target Investigation
* **Exact Column**: `final_result` (in `studentInfo.csv`)
* **Unique Target Values, Counts, and Percentages**:
  * Pass: 12,361 (37.93%)
  * Withdrawn: 10,156 (31.16%)
  * Fail: 7,052 (21.64%)
  * Distinction: 3,024 (9.28%)

## F. Temporal Information
* **`assessments.csv` -> `date`**: The planned final submission date for an assessment (measured in days relative to the start of the presentation).
* **`courses.csv` -> `module_presentation_length`**: Total duration of the course in days.
* **`studentAssessment.csv` -> `date_submitted`**: The actual day the student submitted the assessment (relative to presentation start).
* **`studentRegistration.csv` -> `date_registration`**: The day the student registered (relative to presentation start, often negative).
* **`studentRegistration.csv` -> `date_unregistration`**: The day the student withdrew (relative to presentation start).
* **`studentVle.csv` -> `date`**: The day the student interacted with a VLE component.
* **`vle.csv` -> `week_from` / `week_to`**: The planned weeks for using specific VLE materials.

## G. Potential Sensitive/Fairness Attributes
* `gender`
* `disability`
* `age_band`
* `region`
* `imd_band` (Index of Multiple Deprivation - Proxy for socioeconomic status)

## H. Potential Leakage Fields
* **`date_unregistration`** (`studentRegistration.csv`): If a student withdraws, this contains the exact date. Using this field or knowing it is not null at prediction time would directly leak that the student has withdrawn.
* **`final_result`** (`studentInfo.csv`): The target variable itself. Must be carefully separated.
* **Assessment scores and late submissions** (`studentAssessment.csv`): Any assessment submitted *after* our temporal cutoff point for an early-warning prediction must be excluded, otherwise we are looking into the future.
* **VLE interactions** (`studentVle.csv`): Any interaction date *after* the prediction cutoff point must be excluded.

## I. Important Observations
* **Class Imbalance**: "Withdrawn" is heavily represented (~31%), but "Distinction" is quite small (~9%). If combining (Pass/Distinction vs Fail/Withdrawn), balance is better.
* **Missing Values**: `imd_band`, `date_unregistration`, `date`, `week_from`, `week_to` likely contain missing values based on typical OULAD characteristics (e.g., `date_unregistration` is null if they didn't withdraw).
* **Suspicious Columns / Leakage**: As noted, `date_unregistration` is a massive leakage source for a withdrawal prediction task.
* **Large File Sizes**: `studentVle.csv` is very large (~453 MB) and will require efficient processing (chunking or specialized tools) when creating temporal features.
* **Duplicate Identifiers**: `id_student` is unique per module-presentation in `studentInfo`, but a single student can take multiple courses, meaning `id_student` appears multiple times across the whole dataset.
