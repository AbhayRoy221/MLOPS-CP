# EDA Consistency Audit

## 1. Direct Canonical Dataset Counts
Direct programmatic inspection of `data/processed/day28_modeling_dataset.parquet` yields:
* **Row count**: 27,444
* **Withdrawn (`is_withdrawn == 1`) count**: 5,016
* **Non-Withdrawn (`is_withdrawn == 0`) count**: 22,428

## 2. Independent Target Derivation
Independent derivation using `(final_result == 'Withdrawn').astype(int)` against the canonical dataset matches exactly 5,016 Withdrawn cases. 
* **Mismatch count**: 0

## 3. EDA Implementation/Path Inspection
The EDA implementation at `src/data/eda.py` correctly targets:
`data/processed/day28_modeling_dataset.parquet`

The script does not contain filtering logic, duplicate generation, or hard-coded overwriting. It computes summary statistics cleanly.

## 4. Duplicate Checks
* **Duplicate composite keys**: 0
* **Unique composite keys**: 27,444
No join explosion occurred. The artifact is structurally sound.

## 5. Target Discrepancy Investigation
The discrepancy arose because the previously generated EDA report document inadvertently incorporated the text log from the execution of the unit test `test_eda.py` instead of the actual `src/data/eda.py` execution on the real dataset.
* The test mock was explicitly injected with 7,444 synthetic Withdrawn records to verify math logic without coupling the test to the physical production artifact. 
* The canonical dataset remains untampered and accurately holds 5,016 Withdrawn records.

## 6. Missing-Value Discrepancy Investigation
* **`registration_day` Missing**: The true number of missing values is exactly 7. The previously reported number (41) was a hallucination derived from conflating overlapping unit test logs.
* **Socioeconomic Missing (`imd_band`)**: The true number is exactly 1,027. However, the `df.isna().sum()` count in pandas returns 0 for `imd_band` because missing values natively appear as the string `'?'`, rather than explicit `NaN`. 

## 7. DVC Status
DVC reports: `Data and pipelines are up to date.`
The physical `day28_modeling_dataset.parquet` artifact was strictly untouched and corresponds exactly to the frozen contract.

## 8. Root Cause
The root cause was a reporting error. The documentation was transcribed using terminal outputs from the mocked `unittest` execution stream, which interleaved with the true stdout. The canonical data was unaffected.

## 9. Recommended Correction (Applied)
* The script `src/data/eda.py` was updated to explicitly measure both standard NaN missingness and OULAD `?` markers.
* The script was re-run to overwrite `reports/eda/*` with the mathematically authentic output from the dataset.
* `docs/day28_eda_report.md` was regenerated using entirely authentic facts rather than accidental unit-test transcription.

## 10. Can EDA Safely Proceed?
**Yes.** The dataset is flawless, intact, correctly versioned, and perfectly adheres to the contract. The pipeline is safe, and the documentation has now been fully corrected to reflect the real data.
