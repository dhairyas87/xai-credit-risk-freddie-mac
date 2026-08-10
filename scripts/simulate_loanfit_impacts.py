"""Simulate LoanFit model behavior and feature impacts.

Outputs:
    reports/dissertation_results/loanfit_simulation/model_performance.csv
    reports/dissertation_results/loanfit_simulation/feature_impacts.csv
    docs/LOANFIT_SIMULATION_RESULTS.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import (  # noqa: E402
    MODEL_OPTIONS,
    load_model_option,
    original_features,
    stacked_features,
)
from src.loanfit_modeling import ALL_FEATURES  # noqa: E402


MODEL_KEYS = ["v4_stacked", "v4_catboost", "smart_tree", "fast_linear"]
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
}


def predict_batch(model_key: str, bundle, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return raw-dollar amount and interest-rate predictions for a batch."""
    if model_key in {"fast_linear", "smart_tree"}:
        amount_log = np.asarray(bundle["amount"].predict(frame)).ravel()
        rate = np.asarray(bundle["rate"].predict(frame)).ravel()
        return np.expm1(amount_log), np.clip(rate, 0.0, 20.0)

    if model_key == "v4_catboost":
        base_features = original_features(frame)
        amount_log = np.asarray(bundle["amount"].predict(base_features)).ravel()
        rate = np.asarray(bundle["rate"].predict(base_features)).ravel()
        return np.expm1(amount_log), np.clip(rate, 0.0, 20.0)

    catboost_features, xgboost_features = stacked_features(frame, bundle["encoder"])
    pred_catboost = np.asarray(bundle["catboost"].predict(catboost_features)).ravel()
    pred_xgboost = np.asarray(bundle["xgboost"].predict(xgboost_features)).ravel()
    meta_features = pd.DataFrame({"pred_cb": pred_catboost, "pred_xgb": pred_xgboost})
    amount_log = np.asarray(bundle["meta"].predict(meta_features)).ravel()
    rate = np.asarray(bundle["rate"].predict(original_features(frame))).ravel()
    return np.expm1(amount_log), np.clip(rate, 0.0, 20.0)


def load_test_frames(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    amount_test = pd.read_parquet(root / "data/feature_store/v3/loan_amount/test.parquet")
    rate_test = pd.read_parquet(root / "data/feature_store/v3/interest_rate/test.parquet")

    features = amount_test[ALL_FEATURES].copy()
    amount_target = np.expm1(amount_test["target"])
    rate_target = rate_test["target"]
    return features, amount_target, rate_target


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, target: str) -> dict[str, float | str]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    metrics: dict[str, float | str] = {
        "target": target,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    if target == "loan_amount":
        metrics["mape_pct"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return metrics


def evaluate_models(features: pd.DataFrame, amount_y: pd.Series, rate_y: pd.Series, sample_rows: int) -> pd.DataFrame:
    sample = features.sample(min(sample_rows, len(features)), random_state=2026)
    amount_true = amount_y.loc[sample.index]
    rate_true = rate_y.loc[sample.index]
    records = []

    for model_key in MODEL_KEYS:
        bundle = load_model_option(model_key)
        amount_predictions, rate_predictions = predict_batch(model_key, bundle, sample)

        model_meta = MODEL_OPTIONS[model_key]
        for metrics in [
            regression_metrics(amount_true, amount_predictions, "loan_amount"),
            regression_metrics(rate_true, rate_predictions, "interest_rate"),
        ]:
            records.append(
                {
                    "model_key": model_key,
                    "model_label": model_meta["label"],
                    "model_family": model_meta["family"],
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def feature_impacts(features: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    sample = features.sample(min(sample_rows, len(features)), random_state=2027).reset_index(drop=True)
    shuffled = sample.copy()
    records = []

    for model_key in MODEL_KEYS:
        bundle = load_model_option(model_key)

        base_amount_arr, base_rate_arr = predict_batch(model_key, bundle, sample)

        for feature in ALL_FEATURES:
            changed = shuffled.copy()
            changed[feature] = changed[feature].sample(frac=1, random_state=hash((model_key, feature)) % 2**32).to_numpy()

            amount_changed, rate_changed = predict_batch(model_key, bundle, changed)
            amount_delta = amount_changed - base_amount_arr
            rate_delta = rate_changed - base_rate_arr
            records.extend(
                [
                    {
                        "model_key": model_key,
                        "model_label": MODEL_OPTIONS[model_key]["label"],
                        "target": "loan_amount",
                        "feature": feature,
                        "feature_label": FEATURE_LABELS.get(feature, feature),
                        "mean_absolute_impact": float(np.mean(np.abs(amount_delta))),
                        "mean_signed_impact": float(np.mean(amount_delta)),
                        "unit": "USD",
                    },
                    {
                        "model_key": model_key,
                        "model_label": MODEL_OPTIONS[model_key]["label"],
                        "target": "interest_rate",
                        "feature": feature,
                        "feature_label": FEATURE_LABELS.get(feature, feature),
                        "mean_absolute_impact": float(np.mean(np.abs(rate_delta))),
                        "mean_signed_impact": float(np.mean(rate_delta)),
                        "unit": "percentage points",
                    },
                ]
            )

    return pd.DataFrame(records)


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: list[str]) -> str:
    table = df[columns].copy()
    for col in float_cols:
        table[col] = table[col].map(lambda value: f"{value:,.3f}")
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(performance: pd.DataFrame, impacts: pd.DataFrame, output_path: Path) -> None:
    amount_perf = performance[performance["target"] == "loan_amount"].sort_values("mae")
    rate_perf = performance[performance["target"] == "interest_rate"].sort_values("mae")

    amount_impacts = (
        impacts[impacts["target"] == "loan_amount"]
        .sort_values(["model_key", "mean_absolute_impact"], ascending=[True, False])
        .groupby("model_key")
        .head(5)
    )
    rate_impacts = (
        impacts[impacts["target"] == "interest_rate"]
        .sort_values(["model_key", "mean_absolute_impact"], ascending=[True, False])
        .groupby("model_key")
        .head(5)
    )

    best_amount = amount_perf.iloc[0]
    best_rate = rate_perf.iloc[0]

    report = f"""# LoanFit Simulation Results

This simulation used held-out v3 lending test rows and all four UI model
choices. Feature impact is measured by permuting one input feature at a time
and observing the average absolute movement in the model prediction.

## Best model by target

- Best loan amount model by MAE: **{best_amount['model_label']}**
  with MAE **${best_amount['mae']:,.0f}**.
- Best interest-rate model by MAE: **{best_rate['model_label']}**
  with MAE **{best_rate['mae']:.3f} percentage points**.

## Loan amount performance

{markdown_table(amount_perf, ['model_label', 'model_family', 'mae', 'rmse', 'r2', 'mape_pct'], ['mae', 'rmse', 'r2', 'mape_pct'])}

## Interest-rate performance

{markdown_table(rate_perf, ['model_label', 'model_family', 'mae', 'rmse', 'r2'], ['mae', 'rmse', 'r2'])}

## Top loan amount feature impacts by model

{markdown_table(amount_impacts, ['model_label', 'feature_label', 'mean_absolute_impact', 'mean_signed_impact', 'unit'], ['mean_absolute_impact', 'mean_signed_impact'])}

## Top interest-rate feature impacts by model

{markdown_table(rate_impacts, ['model_label', 'feature_label', 'mean_absolute_impact', 'mean_signed_impact', 'unit'], ['mean_absolute_impact', 'mean_signed_impact'])}

## Interpretation note

Permutation impact is not the same as a causal effect. It answers: “When this
feature is disrupted while the rest of the profile stays the same, how much
does the model prediction move on average?”
"""

    output_path.write_text(report)


def main() -> None:
    report_dir = PROJECT_ROOT / "reports/dissertation_results/loanfit_simulation"
    report_dir.mkdir(parents=True, exist_ok=True)

    features, amount_y, rate_y = load_test_frames(PROJECT_ROOT)
    performance = evaluate_models(features, amount_y, rate_y, sample_rows=400)
    impacts = feature_impacts(features, sample_rows=120)

    performance.to_csv(report_dir / "model_performance.csv", index=False)
    impacts.to_csv(report_dir / "feature_impacts.csv", index=False)
    write_report(performance, impacts, PROJECT_ROOT / "docs/LOANFIT_SIMULATION_RESULTS.md")

    print("Saved simulation results:")
    print(report_dir / "model_performance.csv")
    print(report_dir / "feature_impacts.csv")
    print(PROJECT_ROOT / "docs/LOANFIT_SIMULATION_RESULTS.md")


if __name__ == "__main__":
    main()
