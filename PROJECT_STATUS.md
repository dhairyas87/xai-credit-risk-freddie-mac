# Explainable Credit Risk Management using Freddie Mac Data

## Project Overview

This project aims to develop an explainable credit risk management framework using Freddie Mac mortgage loan data.

The primary objective is to predict borrower financial stress using origination characteristics while introducing a novel Borrower Serviceability framework.

---

# Research Objectives

1. Predict future borrower financial stress using loan origination characteristics.
2. Develop a Borrower Serviceability Stress Index (BSSI).
3. Develop a Borrower Serviceability Score (BSS).
4. Compare traditional credit risk models with serviceability-enhanced models.
5. Apply Explainable AI techniques (SHAP) to interpret predictions.

---

# Dataset

Source:

* Freddie Mac Single Family Loan-Level Dataset

Period Used:

* 2018 Q1
* 2018 Q2
* 2018 Q3
* 2018 Q4

Total Loans:

* 1,285,434 loans

---

# Project Structure

data/

* raw/
* processed/
* modeling/

src/

* schema.py
* data_loader.py
* validation.py
* target_generation.py
* feature_engineering.py

scripts/

* build_target_dataset.py
* build_master_dataset.py
* build_combined_dataset.py
* pipeline.py

notebooks/

* 01_data_exploration.ipynb
* 02_target_creation.ipynb
* 03_feature_engineering.ipynb
* 04_dataset_diagnostics.ipynb
* 05_modeling_preparation.ipynb (planned)

---

# Work Completed

## 1. Data Exploration

Completed analysis of:

* Credit Score
* DTI
* LTV
* CLTV
* Interest Rate
* Original UPB
* Occupancy Status
* Loan Purpose
* Property Type
* Number of Borrowers

Key Findings:

* Lower credit scores are associated with higher borrower stress.
* Higher DTI ratios are associated with higher borrower stress.
* Multiple borrowers appear more resilient.
* Interest rate exhibits a meaningful relationship with future stress.

---

## 2. Target Creation

Created loan-level performance dataset.

Generated:

### max_delinquency

Maximum delinquency observed during loan performance history.

### ever_modified

Whether the loan was ever modified.

### ever_assistance

Whether the borrower received assistance.

### ever_ra

Whether the loan entered severe resolution conditions.

---

# Borrower Serviceability Stress Index (BSSI)

Created custom stress severity framework.

Definition:

BSSI = 0

* No stress

BSSI = 1

* Delinquency
* Assistance
* Modification

BSSI = 2

* Significant borrower stress

BSSI = 3

* Severe stress / Resolution event

Distribution:

* BSSI 0 = 86.22%
* BSSI 1 = 10.26%
* BSSI 2 = 3.46%
* BSSI 3 = 0.057%

---

# Stress Flag

Primary modeling target.

Definition:

stress_flag = 1

if:

BSSI > 0

otherwise:

stress_flag = 0

Observed Stress Rate:

Approximately 14%.

---

## 3. Feature Engineering

Created Borrower Serviceability Score (BSS).

Purpose:

Estimate borrower repayment capacity at loan origination.

Features Used:

* Credit Score
* DTI
* Interest Rate
* LTV
* Number of Borrowers

Normalization:

Min-Max scaling.

Current Weights:

* Credit Score = 40%
* DTI = 25%
* Interest Rate = 20%
* Number of Borrowers = 10%
* LTV = 5%

---

# BSS Validation

Average BSS by BSSI:

* BSSI 0 = 53.08
* BSSI 1 = 48.54
* BSSI 2 = 45.36
* BSSI 3 = 44.40

Result:

BSS decreases monotonically as borrower stress increases.

---

# BSS Buckets

Created serviceability categories:

* Very Low
* Low
* Medium
* High
* Very High

Validation:

Stress Rate by BSS Bucket:

* Very Low = 25.93%
* Low = 17.16%
* Medium = 13.03%
* High = 9.29%
* Very High = 5.70%

Finding:

Borrowers in the Very Low Serviceability bucket are approximately 4.5x more likely to experience future financial stress than borrowers in the Very High Serviceability bucket.

---

## 4. Dataset Diagnostics

Combined all quarters.

Final Dataset:

master_dataset_2018.parquet

Shape:

* 1,285,434 rows
* 49 columns

Quarter Distribution:

* Q1 = 296,816
* Q2 = 364,502
* Q3 = 336,669
* Q4 = 287,447

Stress Rate by Quarter:

* Q1 = 14.41%
* Q2 = 13.61%
* Q3 = 13.74%
* Q4 = 13.39%

Observation:

Stress rates are stable across quarters.

---

# Pipeline Architecture

Current Pipeline:

Raw Data

↓

Target Generation

↓

loan_perf_YYYYQX.parquet

↓

Feature Engineering

↓

master_dataset_YYYYQX.parquet

↓

Combine Quarters

↓

master_dataset_2018.parquet

---

# Scripts Completed

build_target_dataset.py

Creates:

* loan_perf_YYYYQX.parquet

build_master_dataset.py

Creates:

* master_dataset_YYYYQX.parquet

build_combined_dataset.py

Creates:

* master_dataset_2018.parquet

pipeline.py

Orchestrates all stages.

Supports:

* Quarter-wise execution
* Rebuilds
* Stage control

---

# Next Steps

## Feature Store

Create:

* train.parquet
* validation.parquet
* test.parquet

Temporal Split:

Train:

* 2018Q1
* 2018Q2

Validation:

* 2018Q3

Test:

* 2018Q4

---

## Modeling

Model A:

Traditional Origination Features

Model B:

Traditional Features + BSS

Objective:

Determine whether BSS provides incremental predictive value.

---

## Explainability

Planned:

* SHAP
* Feature Importance
* Risk Driver Analysis

---

# Current Status

Data Engineering: COMPLETE

Target Engineering: COMPLETE

Feature Engineering: COMPLETE

Dataset Diagnostics: COMPLETE

Feature Store: IN PROGRESS

Modeling: NOT STARTED

Explainability: NOT STARTED
