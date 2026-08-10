"""Train runnable lending recommendation models for the LoanFit UI.

This script trains two model options for each UI prediction target:

1. fast_linear - simpler, faster, easier to inspect.
2. smart_tree  - non-linear gradient boosting model.

The source data is the v3 lending feature store produced by
scripts/build_lending_recommendation_feature_store.py and
scripts/build_lending_modeling_datasets.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from src.loanfit_modeling import make_bundle, prepare_frame


TARGET = "target"

def _load_split(feature_store_root: Path, objective: str, version: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = feature_store_root / version / objective
    required = [root / "train.parquet", root / "valid.parquet", root / "test.parquet"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing modeling dataset(s): "
            + ", ".join(missing)
            + ". Run `python scripts/pipeline.py all --skip-raw` after the processed master dataset exists."
        )
    return tuple(pd.read_parquet(path) for path in required)


def _maybe_sample(df: pd.DataFrame, sample_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if sample_rows is None or len(df) <= sample_rows:
        return df
    return df.sample(n=sample_rows, random_state=random_state)


def _amount_metrics(y_log_true: pd.Series, y_log_pred: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    return {
        "rmse_dollars": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_dollars": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "r2_log": float(r2_score(y_log_true, y_log_pred)),
    }


def _term_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_int = y_true.astype(int)
    y_pred_int = pd.Series(y_pred).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_int, y_pred_int)),
        "mae_months": float(mean_absolute_error(y_true_int, y_pred_int)),
    }


def _rate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_rate": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_rate": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save_metrics(metrics: list[dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(report_dir / "metrics.csv", index=False)


def train_lending_models(
    feature_store_root: Path,
    model_root: Path,
    report_root: Path,
    data_version: str = "v3",
    model_version: str = "v5",
    sample_rows: int | None = None,
) -> None:
    model_version_root = model_root / model_version
    report_version_root = report_root / model_version

    print("=" * 60)
    print("TRAINING LOANFIT LENDING MODELS")
    print("=" * 60)
    if sample_rows:
        print(f"Using up to {sample_rows:,} training rows per target for a faster run.")

    # -------------------------------------------------
    # Loan amount models
    # -------------------------------------------------
    amount_train, _amount_valid, amount_test = _load_split(feature_store_root, "loan_amount", data_version)
    amount_train = _maybe_sample(amount_train, sample_rows)
    amount_test = _maybe_sample(amount_test, sample_rows, random_state=43)

    y_amount_train = amount_train[TARGET]
    X_amount_test = prepare_frame(amount_test)
    y_amount_test = amount_test[TARGET]

    amount_models = {
        "fast_linear": (
            "linear",
            Ridge(alpha=10.0),
        ),
        "smart_tree": (
            "tree",
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.08,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    }

    amount_model_dir = model_version_root / "loan_amount"
    amount_model_dir.mkdir(parents=True, exist_ok=True)
    amount_metrics = []

    for name, (mode, estimator) in amount_models.items():
        print(f"\nTraining loan amount model: {name}")
        train_for_bundle = prepare_frame(amount_train)
        train_for_bundle[TARGET] = y_amount_train.values
        bundle = make_bundle(mode, estimator, train_for_bundle)
        predictions = bundle.predict(X_amount_test)
        metrics = _amount_metrics(y_amount_test, predictions)
        metrics["model"] = name
        amount_metrics.append(metrics)
        joblib.dump(bundle, amount_model_dir / f"{name}.pkl")
        print(metrics)

    _save_metrics(amount_metrics, report_version_root / "loan_amount")

    # -------------------------------------------------
    # Loan term models
    # -------------------------------------------------
    term_train, _term_valid, term_test = _load_split(feature_store_root, "loan_term", data_version)
    term_train = _maybe_sample(term_train, sample_rows)
    term_test = _maybe_sample(term_test, sample_rows, random_state=43)

    y_term_train = term_train[TARGET].astype(int)
    X_term_test = prepare_frame(term_test)
    y_term_test = term_test[TARGET].astype(int)

    term_models = {
        "fast_linear": (
            "linear",
            SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "smart_tree": (
            "tree",
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.08,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=42,
            ),
        ),
    }

    term_model_dir = model_version_root / "loan_term"
    term_model_dir.mkdir(parents=True, exist_ok=True)
    term_metrics = []

    for name, (mode, estimator) in term_models.items():
        print(f"\nTraining loan term model: {name}")
        train_for_bundle = prepare_frame(term_train)
        train_for_bundle[TARGET] = y_term_train.values
        bundle = make_bundle(mode, estimator, train_for_bundle)
        predictions = bundle.predict(X_term_test)
        metrics = _term_metrics(y_term_test, predictions)
        metrics["model"] = name
        term_metrics.append(metrics)
        joblib.dump(bundle, term_model_dir / f"{name}.pkl")
        print(metrics)

    _save_metrics(term_metrics, report_version_root / "loan_term")

    # -------------------------------------------------
    # Interest rate models
    # -------------------------------------------------
    rate_train, _rate_valid, rate_test = _load_split(feature_store_root, "interest_rate", data_version)
    rate_train = _maybe_sample(rate_train, sample_rows)
    rate_test = _maybe_sample(rate_test, sample_rows, random_state=43)

    y_rate_train = rate_train[TARGET]
    X_rate_test = prepare_frame(rate_test)
    y_rate_test = rate_test[TARGET]

    rate_models = {
        "fast_linear": (
            "linear",
            Ridge(alpha=10.0),
        ),
        "smart_tree": (
            "tree",
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.08,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
    }

    rate_model_dir = model_version_root / "interest_rate"
    rate_model_dir.mkdir(parents=True, exist_ok=True)
    rate_metrics = []

    for name, (mode, estimator) in rate_models.items():
        print(f"\nTraining interest rate model: {name}")
        train_for_bundle = prepare_frame(rate_train)
        train_for_bundle[TARGET] = y_rate_train.values
        bundle = make_bundle(mode, estimator, train_for_bundle)
        predictions = bundle.predict(X_rate_test)
        metrics = _rate_metrics(y_rate_test, predictions)
        metrics["model"] = name
        rate_metrics.append(metrics)
        joblib.dump(bundle, rate_model_dir / f"{name}.pkl")
        print(metrics)

    _save_metrics(rate_metrics, report_version_root / "interest_rate")

    print("\nLoanFit model training complete.")
    print(f"Models saved under: {model_version_root}")
    print(f"Metrics saved under: {report_version_root}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_lending_models(
        feature_store_root=project_root / "data" / "feature_store",
        model_root=project_root / "models",
        report_root=project_root / "reports" / "dissertation_results",
    )
