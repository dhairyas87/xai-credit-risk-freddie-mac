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

# Status

Schema Validation: COMPLETE

Feature Store Validation: COMPLETE

Preprocessing Design: FINALIZED

Next Step:

Build src/preprocessing.py and implement the preprocessing pipeline for baseline and BSS-enhanced modeling datasets.
