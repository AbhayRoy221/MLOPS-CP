# Day-28 Data Pipeline Implementation

## 1. Raw Data Validation (`src/data/validate_raw.py`)
This module strictly enforces the required OULAD structural schema and constraints before building any cohorts:
* Uses `pandas` to validate required CSV inputs.
* Validates composite-key uniqueness inside `studentInfo.csv`.
* Validates valid values for `final_result`.
* Uses `chunksize` iterator to validate all 10,655,280 `studentVle.csv` interactions without crashing memory, ensuring all `id_site` map correctly to the `vle.csv` context table.

## 2. Base Cohort Builder (`src/data/build_day28_cohort.py`)
This script executes the frozen Day-28 methodology using efficient vectorized pandas merges:
1. **Joins**: Joins `studentInfo` with `studentRegistration` and `courses` on the composite key (`id_student`, `code_module`, `code_presentation`).
2. **Exclusions**: 
   - Drops early withdrawals (`date_unregistration <= 28`).
   - Drops Withdrawn students with missing `date_unregistration` (`?`).
   - Drops inconsistent Non-Withdrawn students with a recorded `date_unregistration`.
3. **Target**: Derives `is_withdrawn` (1 if Withdrawn else 0).
4. **Registration Day**: `date_registration` is carried over preserving the original OULAD sign convention. `?` is converted to standard pandas missing values, and a boolean flag `missing_registration` is mapped.
5. **Socioeconomic Flag**: Identifies missing `imd_band` values (`?`) using `missing_socioeconomic_information`.
6. **Output Generation**: Exports genuine Parquet output preserving columns exactly as requested:
   - `id_student`, `code_module`, `code_presentation`
   - `registration_day`, `missing_registration`
   - `gender`, `disability`, `age_band`, `region`, `imd_band`, `missing_socioeconomic_information`
   - `highest_education`, `num_of_prev_attempts`, `studied_credits`
   - `final_result`, `is_withdrawn`
   - `module_presentation_length`

## 3. Data Check Assertions
The cohort builder strictly guarantees exactly 27,444 final eligible rows and 5,016 eventual Withdrawn target outputs, halting execution otherwise. 

## 4. Current Output Artifact
* **Path**: `data/processed/day28_base_cohort.parquet`
* **Format**: Genuine Parquet (PyArrow Engine)
* **Shape**: 27,444 rows, 17 columns.
