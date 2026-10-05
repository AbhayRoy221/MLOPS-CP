# Day-28 Split Strategy

## Why row-wise random splitting is inappropriate
The primary entity of analysis is an enrollment (`id_student` + `code_module` + `code_presentation`), rather than just an independent student. Because the same student can enroll in multiple modules or presentations across time, using standard row-wise random splitting (`train_test_split`) would leak data. A student's behavioral patterns could theoretically appear in the training set for one module and the validation set for another, causing severe target leakage.

## Grouping variable
`id_student` is the strictly enforced grouping variable. No single student is allowed to span multiple data partitions.

## Stratification variable
`is_withdrawn` is used as the stratification variable to ensure the ~18.28% withdrawal prevalence remains relatively consistent across all splits.

## Splitter
`StratifiedGroupKFold`

### Parameters
* `n_splits = 7`
* `shuffle = True`
* `random_state = 42`

### Fold assignment
The deterministic mapping across 7 folds is:
* **Test set**: Fold 0 (approx 1/7 of data)
* **Validation set**: Fold 1 (approx 1/7 of data)
* **Training set**: Folds 2 through 6 (approx 5/7 of data)

## Final split statistics
* **Train rows**: 19,628 (Prevalence: 18.13%)
* **Validation rows**: 3,917 (Prevalence: 18.79%)
* **Test rows**: 3,899 (Prevalence: 18.49%)

*(Note: Actual row counts derived from dynamic fold boundaries: exact numbers are printed by `split_dataset.py` during execution. Overlaps are strictly zero.)*

## Preprocessing strategy
The model input `X` strictly comprises the frozen 21 predictive features. Fairness attributes are wholly excluded from `X` to maintain a blind model. The target is omitted.

### Numeric features
* **Imputation**: `SimpleImputer` using a constant (`-999`) and `add_indicator=True` to explicitly flag structurally missing assessments or VLE behaviors.
* **Scaling**: `RobustScaler` due to extreme right-skewness identified in EDA (e.g., thousands of VLE clicks). Log-transformations remain a candidate future experiment.

### Categorical features
* **Imputation**: Constant imputation converting missing/unknown markers (such as `"?"`) to a dedicated `"Missing"` category.
* **Encoding**: `OneHotEncoder(handle_unknown='ignore')` to ensure unseen inference contexts do not crash pipelines.

### Train-only fitting
The entire preprocessing pipeline (`ColumnTransformer`) is fit strictly and exclusively on the training split. The validation and test sets are exclusively passed through `.transform()` to entirely prevent distribution leakage.
