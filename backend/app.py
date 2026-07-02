"""FastAPI service for the loan recommendation interface."""

from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_AMOUNT_DIR = PROJECT_ROOT / "models/v4/loan_amount"
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cb_model = None
_xgb_model = None
_meta_blender = None
_encoder = None
_term_model = None
_rate_model = None

def load_models():
    global _cb_model, _xgb_model, _meta_blender, _encoder, _term_model, _rate_model
    if _cb_model is None:
        _cb_model = joblib.load(
            MODEL_AMOUNT_DIR / "stack_base_catboost.pkl"
        )
        _xgb_model = joblib.load(
            MODEL_AMOUNT_DIR / "stack_base_xgb.pkl"
        )
        _meta_blender = joblib.load(
            MODEL_AMOUNT_DIR / "stack_meta_blender.pkl"
        )
        _encoder = joblib.load(
            MODEL_AMOUNT_DIR / "stack_target_encoder.pkl"
        )
    if _term_model is None:
        _term_model = joblib.load(TERM_MODEL_PATH)
    if _rate_model is None:
        _rate_model = joblib.load(RATE_MODEL_PATH)
    return (
        _cb_model, _xgb_model, _meta_blender, 
        _encoder, _term_model, _rate_model
    )

class LoanApplication(pd.BaseModel if hasattr(pd, "BaseModel") else dict):
    # Standard Pydantic structure fallback validation layout
    pass

try:
    from pydantic import BaseModel, Field
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
except ImportError:
    pass

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
        (cb_model, xgb_model, meta_blender, 
         encoder, term_model, rate_model) = load_models()
        
        base_data = {
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
        df = pd.DataFrame([base_data])
        
        # 1. Re-apply leakage-free interaction terms
                # 1. Base Column Definitions (Matches original shapes perfectly)
        original_num_cols = [
            "credit_score", "dti", "num_borrowers", "num_units", 
            "ltv", "cltv", "mortgage_insurance_pct", "quarter_num"
        ]
        cat_cols = [
            "first_time_homebuyer_indicator", "occupancy_status", 
            "property_type", "property_state", "msa", "channel", 
            "loan_purpose", "program_indicator",
            "property_valuation_method", "interest_only_indicator",
            "mortgage_insurance_cancellation_indicator", "quarter"
        ]
        
        # 2. CREATE CLEAN ORIGINAL FRAME (Exactly 20 columns for Term & Rate models)
        original_features_order = original_num_cols + cat_cols
        features_original = df[original_features_order].copy()
        for col in cat_cols:
            features_original[col] = features_original[col].astype(str)

        # 3. CREATE EXTENDED FEATURE LAYERS FOR THE STACKED LOAN AMOUNT MODEL
        df['dti_ltv_interaction'] = df['dti'] * df['ltv']
        df['credit_risk_multiplier'] = (
            df['credit_score'] / (df['dti'] + 1.0)
        )
        
        stacked_num_cols = original_num_cols + [
            "dti_ltv_interaction", 
            "credit_risk_multiplier"
        ]
        
        # Aligned CatBoost features matrix sequence for loan amount
        cb_features_order = stacked_num_cols + cat_cols
        features_cb = df[cb_features_order].copy()
        for col in cat_cols:
            features_cb[col] = features_cb[col].astype(str)
            
        # Target Encoded features matrix sequence for XGBoost
        encoded_cats = encoder.transform(df[cat_cols])
        encoded_cat_cols = [f"{col}_encoded" for col in cat_cols]
        encoded_df = pd.DataFrame(
            encoded_cats, 
            columns=encoded_cat_cols, 
            index=df.index
        )
        
        features_xgb_raw = pd.concat([df[stacked_num_cols], encoded_df], axis=1)
        xgb_features_order = stacked_num_cols + encoded_cat_cols
        features_xgb = features_xgb_raw[xgb_features_order].copy()
        
        # 4. Generate Predictions safely using correct aligned inputs
        pred_cb_raw = cb_model.predict(features_cb)
        pred_xgb_raw = xgb_model.predict(features_xgb)

        def safe_scalar(value):
            if hasattr(value, "item"):
                return float(value.item())
            arr = np.asarray(value).ravel()
            if len(arr) > 0:
                return float(arr[0])
            return float(value)

        pred_cb = safe_scalar(pred_cb_raw)
        pred_xgb = safe_scalar(pred_xgb_raw)
        
        # Blend Layer Final Estimation
        meta_X = pd.DataFrame(
            {"pred_cb": [pred_cb], "pred_xgb": [pred_xgb]}
        )
        final_log_amount = safe_scalar(meta_blender.predict(meta_X))
        estimated_amount = float(np.expm1(final_log_amount))

        # 5. Handle classified loan term categories (Using original 20-col layout!)
        raw_term = term_model.predict(features_original)
        p_class = int(safe_scalar(raw_term))
        estimated_years = 30 if p_class == 99 else p_class

        # 6. Handle interest rate regression (Using original 20-col layout!)
        raw_rate = rate_model.predict(features_original)
        estimated_rate = safe_scalar(raw_rate)

        return {
            "estimated_loan_amount": round(estimated_amount),
            "estimated_loan_term_months": estimated_years * 12,
            "estimated_loan_term_years": estimated_years,
            "estimated_interest_rate": round(estimated_rate, 2),
            "currency": "USD",
            "disclaimer": (
                "Research estimate based on historical Freddie Mac loans; "
                "not a lending decision or financial advice."
            ),
        }
    except Exception as exc:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {exc}"
        )

if FRONTEND_DIST.exists():
    app.mount(
        "/", 
        StaticFiles(directory=FRONTEND_DIST, html=True), 
        name="frontend"
    )
