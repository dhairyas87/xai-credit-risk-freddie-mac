# Preprocessing Pipeline Design

## Objective

The purpose of the preprocessing pipeline is to transform the feature store into a model-ready dataset while preventing target leakage and ensuring consistent treatment of numerical and categorical variables.

This document records all preprocessing decisions made prior to model training.

---

# Modeling Objective

Target Variable:

stress_flag

Definition:

* 0 = No observed borrower stress
* 1 = Borrower experienced financial stress

The objective is to predict future borrower stress using information available at loan origination.

---

# Leakage Prevention

The following variables were excluded from model training because they are derived from future borrower outcomes.

Removed Variables:

* max_delinquency
* ever_ra
* ever_modified
* ever_assistance
* bssi

Justification:

These variables contain information that would not be available at loan origination and would therefore result in target leakage.

---

# Metadata Variables

The following variables are retained for tracking purposes but excluded from model training.

Removed From Features:

* loan_identifier
* year
* quarter
* quarter_num

Justification:

These variables uniquely identify observations or support temporal analysis but do not represent borrower characteristics.

---

# Date Feature Engineering

Original Variables:

* first_payment_date
* maturity_date

Observed Format:

YYYYMM

Examples:

* 201803
* 204802

Issue:

Directly using these variables would introduce artificial numerical relationships that do not reflect borrower characteristics.

Solution:

A derived feature will be created:

remaining_term_months

Definition:

Number of months between first payment date and maturity date.

Rationale:

Remaining loan term is more interpretable and economically meaningful than raw date values.

Following feature creation, the original date variables will be removed.

---

# Numerical Features

The following variables will be treated as continuous numerical predictors.

* credit_score
* mortgage_insurance_pct
* cltv
* dti
* original_upb
* ltv
* interest_rate
* original_loan_term
* num_borrowers
* remaining_term_months

Preprocessing:

* Missing value imputation using median values
* Standard scaling for models requiring normalization

---

# Categorical Features

The following variables will be treated as categorical predictors.

* msa
* first_time_homebuyer_indicator
* occupancy_status
* channel
* prepayment_penalty_indicator
* amortization_type
* property_state
* property_type
* loan_purpose
* seller_name
* super_conforming_flag
* harp_indicator
* property_valuation_method
* mortgage_insurance_type

Preprocessing:

* Missing values imputed using "Missing"
* One-hot encoding

---

# Investigation: VantageScore 4

Observed Values:

| Value |   Count |
| ----- | ------: |
| 7     | 449,033 |
| N     | 183,580 |
| Y     |  28,705 |

Observation:

The variable does not appear to contain actual VantageScore values.

Instead, it behaves as a categorical indicator variable.

Decision:

Treat as a categorical feature pending further Freddie Mac documentation review.

---

# Investigation: Interest Only Indicator

Observed Values:

| Value |   Count |
| ----- | ------: |
| 7     | 643,329 |
| 1     |  17,989 |

Observation:

The variable behaves as a categorical indicator rather than a continuous numerical variable.

Decision:

Treat as a categorical feature.

---

# Feature Store Variants

## Baseline Dataset

Purpose:

Traditional mortgage risk prediction.

Characteristics:

* Origination features only
* No serviceability framework variables

Outputs:

* train_baseline.parquet
* valid_baseline.parquet
* test_baseline.parquet

---

## BSS Enhanced Dataset

Purpose:

Evaluate the predictive contribution of the Borrower Serviceability Score.

Additional Variables:

* bss
* bss_bucket
* bss_level

Outputs:

* train_bss.parquet
* valid_bss.parquet
* test_bss.parquet

---

# Research Hypothesis

H0:

The Borrower Serviceability Score does not improve prediction of future borrower financial stress.

H1:

The Borrower Serviceability Score provides additional predictive value beyond traditional origination characteristics.

The preprocessing pipeline has been designed to ensure a fair comparison between baseline and BSS-enhanced models.
