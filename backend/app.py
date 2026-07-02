"""FastAPI service for the loan recommendation interface."""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AMOUNT_MODEL_PATH = (
    PROJECT_ROOT 
    / "models/v4/loan_amount/catboost_loan_amount.pkl"
)
TERM_MODEL_PATH = (
    PROJECT_ROOT 
    / "models/v4/loan_term/hist_gradient_boosting.pkl"
)
RATE_MODEL_PATH = (
    PROJECT_ROOT 
    / "models/v4/interest_rate/catboost_interest_rate.pkl"
)
FRONTEND_DIST = PROJECT_ROOT / "frontend/dist"

app = FastAPI(title="LoanFit API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173", 
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_amount_model = None
_term_model = None
_rate_model = None

class LoanApplication(BaseModel):
    credit_score: int = Field(740, ge=300, le=850)
    dti: float = Field(35, ge=0, le=65)
    ltv: float = Field(80, ge=1, le=110)
    cltv: float = Field(80, ge=1, le=150)
    mortgage_insurance_pct: float = Field(0, ge=0, le=55)
    num_borrowers: int = Field(1, ge=1, le=5)
    num_units: int = Field(1, ge=1, le=4)
    first_time_homebuyer: bool = False
    occupancy_status: Literal["P", "S", "I"] = "P"
    property_type: Literal["SF", "PU", "CO", "MH", "CP"] = "SF"
    property_state: str = Field("CA", min_length=2, max_length=2)
    msa: str = Field("0", min_length=1, max_length=8)
    channel: Literal["R", "C", "B"] = "R"
    loan_purpose: Literal["P", "C", "N"] = "P"

def load_models():
    global _amount_model, _term_model, _rate_model
    if _amount_model is None:
        if not AMOUNT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing amount model: {AMOUNT_MODEL_PATH}"
            )
        _amount_model = joblib.load(AMOUNT_MODEL_PATH)
    if _term_model is None:
        if not TERM_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Missing classification term model"
            )
        _term_model = joblib.load(TERM_MODEL_PATH)
    if _rate_model is None:
        if not RATE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing interest rate model: {RATE_MODEL_PATH}"
            )
        _rate_model = joblib.load(RATE_MODEL_PATH)
    return _amount_model, _term_model, _rate_model

def model_frame(application: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "credit_score": application.credit_score,
                "dti": application.dti,
                "num_borrowers": application.num_borrowers,
                "num_units": application.num_units,
                "ltv": application.ltv,
                "cltv": application.cltv,
                "mortgage_insurance_pct": (
                    application.mortgage_insurance_pct
                ),
                "quarter_num": 4,
                "first_time_homebuyer_indicator": (
                    "Y" if application.first_time_homebuyer else "N"
                ),
                "occupancy_status": application.occupancy_status,
                "property_type": application.property_type,
                "property_state": application.property_state.upper(),
                "msa": str(application.msa),
                "channel": application.channel,
                "loan_purpose": application.loan_purpose,
                "program_indicator": "9",
                "property_valuation_method": "7",
                "interest_only_indicator": "N",
                "mortgage_insurance_cancellation_indicator": "7",
                "quarter": "2018Q4",
            }
        ]
    )

@app.get("/api/health")
def health():
    try:
        load_models()
        return {
            "status": "ready", 
            "models": ["loan_amount", "loan_term", "interest_rate"]
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503, 
            detail=str(exc)
        ) from exc

@app.post("/api/predict")
def predict(application: LoanApplication):
    try:
        amount_model, term_model, rate_model = load_models()
        features = model_frame(application)

        # DEFINITIONS LAYER
        NUMERICAL_FEATURES = [
            "credit_score", "dti", "num_borrowers", 
            "num_units", "ltv", "cltv", 
            "mortgage_insurance_pct", "quarter_num"
        ]
        
        CATEGORICAL_FEATURES = [
            "first_time_homebuyer_indicator", "occupancy_status", 
            "property_type", "property_state", "msa", "channel", 
            "loan_purpose", "program_indicator", 
            "property_valuation_method", "interest_only_indicator",
            "mortgage_insurance_cancellation_indicator", "quarter"
        ]

        # FIXED COLUMN ORDER SEQUENCE: Exactly matches training list addition layout
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        features = features[all_features]

        # Clean strings for CatBoost
        for col in CATEGORICAL_FEATURES:
            features[col] = features[col].astype(str)

                # 1. Predict log loan amount (Since v3 feature store data is log-transformed)
        raw_amount_pred = amount_model.predict(features)
        
        if hasattr(raw_amount_pred, "item"):
            log_amount = float(raw_amount_pred.item())
        elif isinstance(raw_amount_pred, (list, np.ndarray)):
            log_amount = float(raw_amount_pred)
        else:
            log_amount = float(raw_amount_pred)

        # DECODE THE LOG VALUE: Converts 12.xx back into real dollar values
        estimated_amount = float(np.expm1(log_amount))

        # 2. Predict loan term category (outputs 15, 30, or 99)
        raw_term_pred = term_model.predict(features)
        if hasattr(raw_term_pred, "item"):
            predicted_class = int(raw_term_pred.item())
        elif isinstance(raw_term_pred, (list, np.ndarray)):
            predicted_class = int(np.array(raw_term_pred).ravel())
        else:
            predicted_class = int(raw_term_pred)

        if predicted_class == 99:
            estimated_months = 360 
            estimated_years = 30
        else:
            estimated_months = predicted_class * 12
            estimated_years = predicted_class

        # 3. Predict continuous interest rate
        raw_rate_pred = rate_model.predict(features)
        if hasattr(raw_rate_pred, "item"):
            estimated_rate = float(raw_rate_pred.item())
        elif isinstance(raw_rate_pred, (list, np.ndarray)):
            estimated_rate = float(raw_rate_pred)
        else:
            estimated_rate = float(raw_rate_pred)

        # Return clean real-world dollar amounts
        return {
            "estimated_loan_amount": round(estimated_amount),
            "estimated_loan_term_months": estimated_months,
            "estimated_loan_term_years": estimated_years,
            "estimated_interest_rate": round(estimated_rate, 2),
            "currency": "USD",
            "disclaimer": (
                "Research estimate based on historical Freddie Mac loans; "
                "not a lending decision or financial advice."
            ),
        }


    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {exc}"
        ) from exc


if FRONTEND_DIST.exists():
    app.mount(
        "/", 
        StaticFiles(directory=FRONTEND_DIST, html=True), 
        name="frontend"
    )
