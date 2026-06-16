# Feature Store Design

## Objective

The feature store provides modeling-ready datasets for all machine learning experiments.

The objective is to create reproducible training, validation, and testing datasets while preventing target leakage.

---

# Input Dataset

Source:

master_dataset_2018.parquet

Shape:

* 1,285,434 loans
* 49 columns

---

# Temporal Splitting Strategy

To mimic real-world deployment conditions, temporal validation was adopted.

Train Dataset:

* 2018Q1
* 2018Q2

Validation Dataset:

* 2018Q3

Test Dataset:

* 2018Q4

Resulting Dataset Sizes:

| Dataset    | Records |
| ---------- | ------: |
| Train      | 661,318 |
| Validation | 336,669 |
| Test       | 287,447 |

---

# Stress Rate Validation

| Dataset    | Stress Rate (%) |
| ---------- | --------------- |
| Train      | 13.97           |
| Validation | 13.74           |
| Test       | 13.39           |

Observation:

Financial stress rates remain stable across all temporal partitions.

The variation between the highest and lowest stress rate is less than one percentage point, indicating limited temporal drift.

---

# Leakage Prevention

The following variables were excluded from model training because they represent future borrower outcomes.

Removed Columns:

* max_delinquency
* ever_ra
* ever_modified
* ever_assistance

Target Variables:

* stress_flag
* bssi

These variables remain available for evaluation but are not used as predictors.

---

# Removed Features

The following variables were removed due to excessive missingness or high cardinality.

| Feature                       | Reason                                      |
| ----------------------------- | ------------------------------------------- |
| special_eligibility_program   | 98.98% missing                              |
| pre_harp_loan_sequence_number | 96.99% missing                              |
| postal_code                   | Extremely high cardinality (889 categories) |

---

# Baseline Dataset

Purpose:

Evaluate predictive performance using traditional mortgage origination characteristics.

Characteristics:

* Raw origination variables only
* No serviceability framework variables

Output Files:

* train_baseline.parquet
* valid_baseline.parquet
* test_baseline.parquet

---

# BSS Enhanced Dataset

Purpose:

Evaluate whether the proposed Borrower Serviceability Score improves predictive performance.

Additional Features:

* bss
* bss_bucket
* bss_level

Output Files:

* train_bss.parquet
* valid_bss.parquet
* test_bss.parquet

---

# Research Hypothesis

H0:

The Borrower Serviceability Score does not improve prediction of future borrower financial stress.

H1:

The Borrower Serviceability Score provides incremental predictive value beyond traditional origination characteristics.

The baseline and BSS-enhanced datasets were created to evaluate this hypothesis.
