# Explainable Credit Risk Management

## Overview

This project develops machine learning models for credit risk assessment using Freddie Mac Single-Family Loan-Level Dataset.

The objective is to predict loan delinquency/default risk and provide transparent explanations using Explainable AI techniques.

## Research Area

Explainable Artificial Intelligence for Financial Risk Management

## Objectives

- Predict credit risk using machine learning
- Compare Logistic Regression, Random Forest and XGBoost
- Explain model decisions using SHAP and LIME
- Support governance and compliance requirements

## Dataset

Freddie Mac Single-Family Loan-Level Dataset

## Technologies

- Python
- Pandas
- Scikit-Learn
- XGBoost
- SHAP
- LIME
- Google Colab

## Dissertation

M.Tech Artificial Intelligence and Machine Learning

## LoanFit Web Application

The repository includes a one-page React interface and FastAPI prediction
service for estimating a loan amount and standard loan term. The UI lets you
choose between four model styles:

- Original mixture (v4): CatBoost + XGBoost stacked ensemble blended with a
  Ridge meta-model.
- Original CatBoost (v4): direct CatBoost lending model trained on the
  underwriting feature set.
- Stable smart model (v5): non-linear tree model with project-owned
  preprocessing to avoid scikit-learn pickle compatibility issues.
- Stable simple model (v5): faster linear-style baseline for comparison.

See `docs/LOANFIT_MODEL_SUMMARY.md` for a presentation-friendly summary of
all four model options.

Run the end-to-end project pipeline from raw Freddie Mac text files through
feature stores and LoanFit model creation:

```bash
python scripts/pipeline.py all
```

This full run includes the original v4 CatBoost/XGBoost research models and
the v5 UI-safe comparison models. If your environment does not currently have
CatBoost/XGBoost installed, skip the original v4 training while still creating
the runnable v5 UI models:

```bash
python scripts/pipeline.py all --skip-original-models
```

If the processed master dataset already exists and you only want to rebuild
feature stores/models:

```bash
python scripts/pipeline.py all --skip-raw
```

To rebuild only the original v4 research models:

```bash
python scripts/pipeline.py original-models
```

For a quick smoke test during development:

```bash
python scripts/pipeline.py lending-datasets
python scripts/pipeline.py lending-models --sample-rows 50000
```

The LoanFit model artifacts use a small project-owned preprocessing bundle so
they are less likely to fail when scikit-learn patch versions differ between
training and inference environments.

Build and run the one-page web app:

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000`. For separate frontend development, run
`npm run dev` inside `frontend`; Vite proxies API requests to port 8000.
