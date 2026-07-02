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
AMOUNT_MODEL_PATH = PROJECT_ROOT / "models/v3/loan_amount/linear_regression.pkl"
TERM_MODEL_PATH = PROJECT_ROOT / "models/v3/loan_term/hist_gradient_boosting.pkl"
FRONTEND_DIST = PROJECT_ROOT / "frontend/dist"

app = FastAPI(title="LoanFit API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_amount_model = None
_term_model = None


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
    global _amount_model, _term_model
    if _amount_model is None:
        if not AMOUNT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing amount model: {AMOUNT_MODEL_PATH}")
        _amount_model = joblib.load(AMOUNT_MODEL_PATH)
    if _term_model is None:
        if not TERM_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Missing term model. Run: python scripts/train_loan_term.py"
            )
        _term_model = joblib.load(TERM_MODEL_PATH)
    return _amount_model, _term_model


def model_frame(application: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "credit_score": application.credit_score,
                "first_time_homebuyer_indicator": (
                    "Y" if application.first_time_homebuyer else "N"
                ),
                "dti": application.dti,
                "num_borrowers": application.num_borrowers,
                "occupancy_status": application.occupancy_status,
                "property_type": application.property_type,
                "property_state": application.property_state.upper(),
                "msa": str(application.msa),
                "num_units": application.num_units,
                "ltv": application.ltv,
                "cltv": application.cltv,
                "mortgage_insurance_pct": application.mortgage_insurance_pct,
                "channel": application.channel,
                "loan_purpose": application.loan_purpose,
                "program_indicator": "9",
                "property_valuation_method": "7",
                "interest_only_indicator": "N",
                "mortgage_insurance_cancellation_indicator": "7",
                "quarter": "2018Q4",
                "quarter_num": 4,
            }
        ]
    )


@app.get("/api/health")
def health():
    try:
        load_models()
        return {"status": "ready", "models": ["loan_amount", "loan_term"]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/predict")
def predict(application: LoanApplication):
    try:
        amount_model, term_model = load_models()
        features = model_frame(application)

        predicted_log_amount = float(amount_model.predict(features)[0])
        estimated_amount = float(np.expm1(predicted_log_amount))
        estimated_amount = float(np.clip(estimated_amount, 10_000, 2_000_000))

        raw_term = float(term_model.predict(features)[0])
        standard_terms = np.array([120, 180, 240, 300, 360])
        estimated_term = int(standard_terms[np.abs(standard_terms - raw_term).argmin()])

        return {
            "estimated_loan_amount": round(estimated_amount / 1000) * 1000,
            "estimated_loan_term_months": estimated_term,
            "estimated_loan_term_years": estimated_term // 12,
            "raw_term_estimate_months": round(raw_term, 1),
            "currency": "USD",
            "disclaimer": (
                "Research estimate based on historical Freddie Mac loans; "
                "not a lending decision or financial advice."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
