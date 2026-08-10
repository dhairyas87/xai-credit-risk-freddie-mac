"""FastAPI service for the LoanFit one-page UI."""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.loanfit_modeling import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
V4_ROOT = PROJECT_ROOT / "models" / "v4"
V5_ROOT = PROJECT_ROOT / "models" / "v5"

MODEL_OPTIONS = {
    "v4_stacked": {
        "label": "Original v4 stacked model",
        "version": "v4",
        "family": "CatBoost + XGBoost mixture",
        "summary": "Original advanced mixture model: CatBoost and XGBoost predictions are blended by a Ridge meta-model.",
    },
    "v4_catboost": {
        "label": "Original v4 CatBoost model",
        "version": "v4",
        "family": "CatBoost",
        "summary": "Original CatBoost model trained directly on borrower, property, and loan-structure features.",
    },
    "smart_tree": {
        "label": "Stable smart tree model",
        "version": "v5",
        "family": "Scikit-learn histogram gradient boosting",
        "summary": "New UI-safe tree model that captures non-linear relationships without fragile ColumnTransformer pickles.",
    },
    "fast_linear": {
        "label": "Stable simple model",
        "version": "v5",
        "family": "Linear-style baseline",
        "summary": "New UI-safe baseline model for quick comparison and easier explanation.",
    },
}

ModelName = Literal["v4_stacked", "v4_catboost", "smart_tree", "fast_linear"]
STANDARD_TERMS = np.array([120, 180, 240, 300, 360])

app = FastAPI(title="LoanFit API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict[str, object] = {}
_shap_explainers: dict[str, object] = {}

FEATURE_LABELS = {
    "credit_score": "Credit score",
    "dti": "Debt-to-income ratio",
    "num_borrowers": "Number of borrowers",
    "num_units": "Property units",
    "ltv": "Loan-to-value ratio",
    "cltv": "Combined loan-to-value ratio",
    "mortgage_insurance_pct": "Mortgage insurance",
    "quarter_num": "Origination quarter",
    "first_time_homebuyer_indicator": "First-time homebuyer",
    "occupancy_status": "Occupancy",
    "property_type": "Property type",
    "property_state": "Property state",
    "msa": "Metro area",
    "channel": "Origination channel",
    "loan_purpose": "Loan purpose",
    "program_indicator": "Program indicator",
    "property_valuation_method": "Property valuation method",
    "interest_only_indicator": "Interest-only indicator",
    "mortgage_insurance_cancellation_indicator": "Mortgage insurance cancellation",
    "quarter": "Origination quarter",
    "dti_ltv_interaction": "DTI × LTV interaction",
    "credit_risk_multiplier": "Credit-risk multiplier",
}


class LoanApplication(BaseModel):
    model_selection: ModelName = "v4_stacked"
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


def safe_scalar(value) -> float:
    arr = np.asarray(value).ravel()
    if len(arr) == 0:
        return float(value)
    return float(arr[0])


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return path


def load_model_option(model_name: ModelName):
    if model_name in _models:
        return _models[model_name]

    try:
        if model_name == "fast_linear":
            bundle = {
                "amount": joblib.load(require_file(V5_ROOT / "loan_amount" / "fast_linear.pkl")),
                "term": joblib.load(require_file(V5_ROOT / "loan_term" / "fast_linear.pkl")),
                "rate": joblib.load(require_file(V5_ROOT / "interest_rate" / "fast_linear.pkl")),
            }
        elif model_name == "smart_tree":
            bundle = {
                "amount": joblib.load(require_file(V5_ROOT / "loan_amount" / "smart_tree.pkl")),
                "term": joblib.load(require_file(V5_ROOT / "loan_term" / "smart_tree.pkl")),
                "rate": joblib.load(require_file(V5_ROOT / "interest_rate" / "smart_tree.pkl")),
            }
        elif model_name == "v4_catboost":
            bundle = {
                "amount": joblib.load(require_file(V4_ROOT / "loan_amount" / "catboost_loan_amount.pkl")),
                "term": joblib.load(require_file(V4_ROOT / "loan_term" / "hist_gradient_boosting.pkl")),
                "rate": joblib.load(require_file(V4_ROOT / "interest_rate" / "catboost_interest_rate.pkl")),
            }
        elif model_name == "v4_stacked":
            bundle = {
                "catboost": joblib.load(require_file(V4_ROOT / "loan_amount" / "stack_base_catboost.pkl")),
                "xgboost": joblib.load(require_file(V4_ROOT / "loan_amount" / "stack_base_xgb.pkl")),
                "meta": joblib.load(require_file(V4_ROOT / "loan_amount" / "stack_meta_blender.pkl")),
                "encoder": joblib.load(require_file(V4_ROOT / "loan_amount" / "stack_target_encoder.pkl")),
                "term": joblib.load(require_file(V4_ROOT / "loan_term" / "hist_gradient_boosting.pkl")),
                "rate": joblib.load(require_file(V4_ROOT / "interest_rate" / "catboost_interest_rate.pkl")),
            }
        else:
            raise ValueError(f"Unsupported model option: {model_name}")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{MODEL_OPTIONS[model_name]['label']} needs the optional package `{exc.name}`. "
            "Install the project requirements in this environment, then restart the backend."
        ) from exc

    _models[model_name] = bundle
    return bundle


def build_model_frame(application: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "credit_score": application.credit_score,
                "dti": application.dti,
                "num_borrowers": application.num_borrowers,
                "num_units": application.num_units,
                "ltv": application.ltv,
                "cltv": application.cltv,
                "mortgage_insurance_pct": application.mortgage_insurance_pct,
                "quarter_num": 4,
                "first_time_homebuyer_indicator": "Y" if application.first_time_homebuyer else "N",
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


def clean_amount(log_prediction: float) -> int:
    amount = float(np.expm1(log_prediction))
    amount = float(np.clip(amount, 10_000, 2_000_000))
    return int(round(amount / 1_000) * 1_000)


def clean_term_months(prediction: float, model_name: ModelName) -> int:
    raw_value = int(round(float(prediction)))
    if model_name in {"v4_catboost", "v4_stacked"}:
        if raw_value == 15:
            return 180
        if raw_value == 30:
            return 360
        if raw_value == 99:
            return 360

    nearest_index = int(np.argmin(np.abs(STANDARD_TERMS - raw_value)))
    return int(STANDARD_TERMS[nearest_index])


def original_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype(str).fillna("UNKNOWN")
    for col in NUMERICAL_FEATURES:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
    return features


def stacked_features(df: pd.DataFrame, encoder) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = original_features(df)
    working["dti_ltv_interaction"] = working["dti"] * working["ltv"]
    working["credit_risk_multiplier"] = working["credit_score"] / (working["dti"] + 1.0)

    engineered_numeric = NUMERICAL_FEATURES + ["dti_ltv_interaction", "credit_risk_multiplier"]
    catboost_features = working[engineered_numeric + CATEGORICAL_FEATURES].copy()

    encoded_input = working[CATEGORICAL_FEATURES].copy()
    encoded_input.loc[encoded_input["msa"] == "0", "msa"] = "UNKNOWN"
    encoded_values = encoder.transform(encoded_input)
    encoded_cols = [f"{col}_encoded" for col in CATEGORICAL_FEATURES]
    encoded_df = pd.DataFrame(encoded_values, columns=encoded_cols, index=working.index)
    xgboost_features = pd.concat([working[engineered_numeric], encoded_df], axis=1)

    return catboost_features, xgboost_features


def predict_amount_log(model_name: ModelName, model_bundle, features: pd.DataFrame) -> float:
    if model_name in {"fast_linear", "smart_tree"}:
        return safe_scalar(model_bundle["amount"].predict(features))

    if model_name == "v4_catboost":
        return safe_scalar(model_bundle["amount"].predict(original_features(features)))

    catboost_features, xgboost_features = stacked_features(features, model_bundle["encoder"])
    pred_catboost = safe_scalar(model_bundle["catboost"].predict(catboost_features))
    pred_xgboost = safe_scalar(model_bundle["xgboost"].predict(xgboost_features))
    meta_features = pd.DataFrame({"pred_cb": [pred_catboost], "pred_xgb": [pred_xgboost]})
    return safe_scalar(model_bundle["meta"].predict(meta_features))


def predict_term_months(model_name: ModelName, model_bundle, features: pd.DataFrame) -> int:
    if model_name in {"fast_linear", "smart_tree"}:
        raw_term = safe_scalar(model_bundle["term"].predict(features))
    else:
        raw_term = safe_scalar(model_bundle["term"].predict(original_features(features)))
    return clean_term_months(raw_term, model_name)


def predict_interest_rate(model_name: ModelName, model_bundle, features: pd.DataFrame) -> float:
    if model_name in {"fast_linear", "smart_tree"}:
        raw_rate = safe_scalar(model_bundle["rate"].predict(features))
    else:
        raw_rate = safe_scalar(model_bundle["rate"].predict(original_features(features)))
    return round(float(np.clip(raw_rate, 0.0, 20.0)), 2)


def shap_module():
    try:
        import shap

        return shap
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SHAP explanations need the optional package `shap`. "
            "Install the project requirements, then restart the backend."
        ) from exc


def base_feature_name(feature_name: str) -> str:
    if "__" in feature_name:
        return feature_name.split("__", 1)[0]
    if feature_name.endswith("_encoded"):
        return feature_name.removesuffix("_encoded")
    return feature_name


def friendly_feature_name(feature_name: str) -> str:
    return FEATURE_LABELS.get(base_feature_name(feature_name), feature_name.replace("_", " ").title())


def aggregate_shap_values(feature_names: list[str], shap_values: np.ndarray) -> list[dict[str, object]]:
    grouped: dict[str, float] = {}
    for feature_name, value in zip(feature_names, shap_values):
        base_name = base_feature_name(feature_name)
        grouped[base_name] = grouped.get(base_name, 0.0) + float(value)

    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    return [
        {
            "feature": feature,
            "label": friendly_feature_name(feature),
            "value": round(score, 4),
            "direction": "increases estimate" if score >= 0 else "decreases estimate",
        }
        for feature, score in ranked
    ]


def explain_v5_model(model_name: ModelName, model_bundle, features: pd.DataFrame) -> dict[str, object]:
    shap = shap_module()
    amount_bundle = model_bundle["amount"]
    transformed = amount_bundle.transform(features)
    background = amount_bundle.background_features
    if background is None:
        background = transformed

    explainer_key = f"{model_name}:amount"
    if explainer_key not in _shap_explainers:
        _shap_explainers[explainer_key] = shap.Explainer(amount_bundle.estimator.predict, background)

    explanation = _shap_explainers[explainer_key](transformed)
    values = np.asarray(explanation.values)
    if values.ndim > 1:
        values = values[0]

    return {
        "target": "loan_amount",
        "method": "SHAP on the selected v5 loan amount model",
        "drivers": aggregate_shap_values(list(transformed.columns), values),
    }


def explain_v4_catboost(model_name: ModelName, model, feature_frame: pd.DataFrame, feature_names: list[str]) -> dict[str, object]:
    shap = shap_module()
    explainer_key = f"{model_name}:catboost_amount"
    if explainer_key not in _shap_explainers:
        _shap_explainers[explainer_key] = shap.TreeExplainer(model)

    explanation = _shap_explainers[explainer_key](feature_frame)
    values = np.asarray(explanation.values)
    if values.ndim > 1:
        values = values[0]

    return {
        "target": "loan_amount",
        "method": "SHAP TreeExplainer on the CatBoost loan amount component",
        "drivers": aggregate_shap_values(feature_names, values),
    }


def explain_prediction(model_name: ModelName, model_bundle, features: pd.DataFrame) -> dict[str, object]:
    if model_name in {"fast_linear", "smart_tree"}:
        return explain_v5_model(model_name, model_bundle, features)

    if model_name == "v4_catboost":
        feature_frame = original_features(features)
        return explain_v4_catboost(
            model_name,
            model_bundle["amount"],
            feature_frame,
            list(feature_frame.columns),
        )

    catboost_features, _xgboost_features = stacked_features(features, model_bundle["encoder"])
    return explain_v4_catboost(
        model_name,
        model_bundle["catboost"],
        catboost_features,
        list(catboost_features.columns),
    )


@app.get("/api/models")
def models():
    return {
        "default_model": "v4_stacked",
        "model_options": [
            {"key": key, **value}
            for key, value in MODEL_OPTIONS.items()
        ],
    }


@app.get("/api/health")
def health():
    available = {}
    for key in MODEL_OPTIONS:
        try:
            load_model_option(key)
            available[key] = "ready"
        except Exception as exc:
            available[key] = f"unavailable: {exc}"

    return {
        "status": "ready",
        "default_model": "v4_stacked",
        "model_options": list(MODEL_OPTIONS),
        "availability": available,
        "targets": ["loan_amount", "loan_term", "interest_rate"],
    }


@app.post("/api/predict")
def predict(application: LoanApplication):
    try:
        model_bundle = load_model_option(application.model_selection)
        features = build_model_frame(application)

        amount_log_prediction = predict_amount_log(application.model_selection, model_bundle, features)
        estimated_amount = clean_amount(amount_log_prediction)
        estimated_term_months = predict_term_months(application.model_selection, model_bundle, features)
        estimated_interest_rate = predict_interest_rate(application.model_selection, model_bundle, features)
        try:
            shap_explanation = explain_prediction(application.model_selection, model_bundle, features)
        except RuntimeError as shap_error:
            shap_explanation = {
                "target": "loan_amount",
                "method": str(shap_error),
                "drivers": [],
            }

        selected_meta = MODEL_OPTIONS[application.model_selection]
        return {
            "model_selection": application.model_selection,
            "model_label": selected_meta["label"],
            "model_version": selected_meta["version"],
            "model_family": selected_meta["family"],
            "model_summary": selected_meta["summary"],
            "estimated_loan_amount": estimated_amount,
            "estimated_loan_term_months": estimated_term_months,
            "estimated_loan_term_years": round(estimated_term_months / 12, 1),
            "estimated_interest_rate": estimated_interest_rate,
            "shap_explanation": shap_explanation,
            "currency": "USD",
            "disclaimer": (
                "Research estimate based on historical Freddie Mac loans; "
                "not a lending decision or financial advice."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
