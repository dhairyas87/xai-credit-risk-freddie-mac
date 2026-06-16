# Schema and Feature Store Validation Summary

## Objective

A comprehensive validation was performed on the feature store prior to preprocessing and model development.

The purpose of this validation was to:

* Verify schema correctness
* Detect potential column mapping issues
* Identify missing values
* Review categorical feature distributions
* Detect constant features
* Validate temporal consistency
* Finalize preprocessing requirements

---

# Datasets Validated

The following modeling datasets were validated:

| Dataset    | Records | Features |
| ---------- | ------: | -------: |
| Train      | 661,318 |       34 |
| Validation | 336,669 |       34 |
| Test       | 287,447 |       34 |

Total Records:

1,285,434

---

# Validation Outcome

## No Major Schema Shift Detected

Initial investigation suggested a possible schema mapping issue due to unusual values observed within:

* interest_only_indicator
* mortgage_insurance_type
* vantagescore_4

Further validation indicated that core origination variables exhibited expected distributions and business behavior.

Validated Features:

* credit_score
* dti
* cltv
* ltv
* interest_rate
* occupancy_status
* property_type
* loan_purpose
* seller_name

Conclusion:

No evidence of a large-scale schema shift was identified.

Observed values are more likely attributable to Freddie Mac coding conventions and special indicator values.

---

# Constant Feature Analysis

The following features contain little or no variation and provide negligible predictive value.

Identified Constant Features:

* prepayment_penalty_indicator
* amortization_type
* property_valuation_method
* mortgage_insurance_type
* year

Observations:

prepayment_penalty_indicator

* Predominantly "N"

amortization_type

* Predominantly "FRM"

property_valuation_method

* Extremely sparse
* Near constant

mortgage_insurance_type

* Effectively constant

year

* All observations originate from 2018

Decision:

Remove these features prior to model training.

---

# Metadata Features

The following variables are retained for auditing and temporal analysis but will not be used for prediction.

Metadata Columns:

* loan_identifier
* year
* quarter
* quarter_num

Decision:

Exclude from feature matrix.

---

# Target Leakage Review

The following variables contain future borrower outcome information.

Leakage Features:

* max_delinquency
* ever_ra
* ever_modified
* ever_assistance
* bssi

Decision:

Exclude from predictor variables.

Target Variable:

stress_flag

---

# Date Feature Review

Original Features:

* first_payment_date
* maturity_date

Observed Format:

YYYYMM

Examples:

* 201803
* 204802

Issue:

Raw date values do not provide meaningful information to machine learning models.

Decision:

Create:

remaining_term_months

Definition:

Difference between maturity date and first payment date expressed in months.

Following feature creation:

* Drop first_payment_date
* Drop maturity_date

---

# Numerical Feature Review

Retained Numerical Features:

* credit_score
* mortgage_insurance_pct
* num_units
* cltv
* dti
* original_upb
* ltv
* interest_rate
* original_loan_term
* num_borrowers
* remaining_term_months

Preprocessing Strategy:

* Median imputation
* Standard scaling

---

# Categorical Feature Review

Retained Categorical Features:

* msa
* first_time_homebuyer_indicator
* occupancy_status
* channel
* property_state
* property_type
* loan_purpose
* seller_name
* super_conforming_flag
* harp_indicator
* interest_only_indicator
* vantagescore_4

Preprocessing Strategy:

* Missing category imputation
* One-hot encoding

---

# Special Feature Investigation

## MSA

Observed:

* Numeric data type
* 447 unique values

Interpretation:

MSA represents a metropolitan area code rather than a continuous numerical measure.

Decision:

Treat as categorical.

---

## Interest Only Indicator

Observed Values:

* 7
* 1

Interpretation:

Feature behaves as a coded indicator variable rather than a continuous numerical measure.

Decision:

Convert to categorical.

---

## VantageScore 4

Observed Values:

* 7
* N
* Y

Interpretation:

Feature does not contain actual credit score values.

Most likely represents a Freddie Mac indicator variable rather than a traditional VantageScore measure.

Decision:

Treat as categorical.

Further Freddie Mac documentation review may be conducted during final model interpretation.

---

# Temporal Split Validation

Stress Rates:

| Dataset    | Stress Rate (%) |
| ---------- | --------------: |
| Train      |           13.97 |
| Validation |           13.74 |
| Test       |           13.39 |

Observation:

Stress prevalence remains stable across all temporal datasets.

Difference between highest and lowest observed stress rate:

0.58 percentage points

Conclusion:

Temporal validation strategy remains appropriate.

---

# Final Preprocessing Decisions

Features Removed:

* loan_identifier
* year
* quarter
* quarter_num
* bssi
* prepayment_penalty_indicator
* amortization_type
* property_valuation_method
* mortgage_insurance_type

Date Engineering:

* Create remaining_term_months

Numerical Processing:

* Median imputation
* Standard scaling

Categorical Processing:

* Missing category imputation
* One-hot encoding

Target Variable:

* stress_flag

---


# Schema Correction and Post-Rebuild Validation

## Background

During preprocessing validation, an inconsistency was identified within the categorical feature encodings.

The issue was initially detected when inspecting one-hot encoded feature names generated by the preprocessing pipeline.

Unexpected categories were observed within the following feature:

```text
super_conforming_flag
```

Examples included:

```text
ROCKET MORTGAGE
MATRIX FINANCIAL SERVICES
MR. COOPER
WELLS FARGO
```

These values represent mortgage servicers and should not appear within a binary indicator field.

This observation suggested a schema mismatch within the origination data ingestion process.

---

# Root Cause Analysis

The acquisition file schema implemented within the project was compared against the official Freddie Mac Single-Family Loan-Level Dataset (SFLD) documentation for pre-July 2026 historical datasets.

The implemented schema incorrectly assumed the following fields:

* special_eligibility_program
* mortgage_insurance_type
* vantagescore_4

However, the official Freddie Mac acquisition layout contains:

* servicer_name
* program_indicator
* mortgage_insurance_cancellation_indicator

As a result, several columns in the tail section of the acquisition dataset were incorrectly labeled.

---

# Corrected Acquisition Layout

The following fields were validated against the official Freddie Mac specification.

| Position | Correct Field Name                        |
| -------- | ----------------------------------------- |
| 24       | seller_name                               |
| 25       | servicer_name                             |
| 26       | super_conforming_flag                     |
| 27       | pre_harp_loan_sequence_number             |
| 28       | program_indicator                         |
| 29       | harp_indicator                            |
| 30       | property_valuation_method                 |
| 31       | interest_only_indicator                   |
| 32       | mortgage_insurance_cancellation_indicator |

The previous fields:

* special_eligibility_program
* mortgage_insurance_type
* vantagescore_4

were removed from the schema.

---

# Pipeline Regeneration

Following schema correction, all downstream datasets were regenerated through the project pipeline.

Regenerated artifacts:

* loan_perf_2018Q1.parquet

* loan_perf_2018Q2.parquet

* loan_perf_2018Q3.parquet

* loan_perf_2018Q4.parquet

* master_dataset_2018Q1.parquet

* master_dataset_2018Q2.parquet

* master_dataset_2018Q3.parquet

* master_dataset_2018Q4.parquet

* combined_master_dataset.parquet

* train_baseline.parquet

* valid_baseline.parquet

* test_baseline.parquet

* train_bss.parquet

* valid_bss.parquet

* test_bss.parquet

---

# Post-Correction Feature Validation

## Seller Name

Unique Categories:

25

Observation:

Represents originating lenders and mortgage sellers.

Examples:

* Wells Fargo
* JPMorgan Chase
* Quicken Loans
* AmeriHome Mortgage

Decision:

Retained for modeling.

---

## Servicer Name

Unique Categories:

24

Observation:

Represents mortgage servicing organizations.

Examples:

* AmeriHome Mortgage
* Mr. Cooper
* Truist
* U.S. Bank

Decision:

Retained within the feature store and preprocessing pipeline.

---

## Super Conforming Flag

Distribution:

| Value   |   Count |
| ------- | ------: |
| Missing | 639,733 |
| Y       |  21,585 |

Observation:

The feature now exhibits expected business behavior.

Decision:

Retained.

---

## Program Indicator

Distribution:

| Value |   Count |
| ----- | ------: |
| 9     | 586,551 |
| H     |  71,625 |
| F     |   3,142 |

Observation:

Represents Freddie Mac program classifications.

Decision:

Retained.

---

## HARP Indicator

Distribution:

| Value   |   Count |
| ------- | ------: |
| Missing | 652,655 |
| Y       |   8,663 |

Observation:

Low-frequency but valid indicator.

Decision:

Retained.

---

## Mortgage Insurance Cancellation Indicator

Distribution:

| Value |   Count |
| ----- | ------: |
| 7     | 449,033 |
| N     | 183,580 |
| Y     |  28,705 |

Observation:

Valid categorical feature with meaningful variation.

Decision:

Retained.

---

## Interest Only Indicator

Distribution:

| Value |   Count |
| ----- | ------: |
| N     | 661,318 |

Observation:

The feature contains a single value across the entire training dataset.

No variance is present.

Decision:

Remove from model development.

Reason:

Constant features provide no predictive information and unnecessarily increase dimensionality.

---

# Final Validation Outcome

The schema correction successfully resolved the acquisition file mapping issue.

Post-rebuild validation confirmed:

* Correct separation of seller and servicer information
* Correct encoding of Freddie Mac program indicators
* Removal of obsolete schema fields
* Consistent feature distributions
* No remaining evidence of schema misalignment

The feature store is now considered validated and suitable for preprocessing and predictive modeling.

---

# Impact on Research

This validation exercise highlighted the importance of schema governance and feature-level auditing within mortgage risk modeling projects.

Without schema validation, downstream models would have been trained on incorrectly mapped acquisition attributes, potentially leading to misleading interpretations and feature importance estimates.

The corrected schema improves both model reliability and research reproducibility.
