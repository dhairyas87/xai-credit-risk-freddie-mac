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
service for estimating a loan amount and standard loan term.

```bash
python scripts/train_loan_term.py
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app:app --reload
```

Open `http://127.0.0.1:8000`. For separate frontend development, run
`npm run dev` inside `frontend`; Vite proxies API requests to port 8000.
