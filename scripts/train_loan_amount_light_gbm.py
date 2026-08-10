from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from lightgbm import LGBMRegressor

TARGET = "target"

CATEGORICAL_FEATURES = [
    "first_time_homebuyer_indicator",
    "occupancy_status",
    "property_type",
    "property_state",
    "msa",
    "channel",
    "loan_purpose",
    "program_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mortgage_insurance_cancellation_indicator",
    "quarter"
]

NUMERICAL_FEATURES = [
    "credit_score",
    "dti",
    "num_borrowers",
    "num_units",
    "ltv",
    "cltv",
    "mortgage_insurance_pct",
    "quarter_num"
]

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    return pd.DataFrame({
        "Metric": ["RMSE (Raw Dollars)", "MAE (Raw Dollars)", "MAPE (%)", "R2 Score"],
        "Value": [rmse, mae, mape, r2]
    })

def clean_and_format(df):
    """
    Ensures optimal datatypes for LightGBM execution.
    Categorical columns are cast to pandas 'category' dtype, which LightGBM
    consumes natively (equivalent role to CatBoost's cat_features list).
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("UNKNOWN").astype("category")
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def train_loan_amount(feature_store_root, model_root, report_root, version="v5"):
    print("=" * 60)
    print("TRAINING LOAN AMOUNT REGRESSOR (LIGHTGBM)")
    print("=" * 60)

    # Reading features using your established v3 pipeline configurations
    root = Path(feature_store_root) / "v3" / "loan_amount"

    train_df = pd.read_parquet(root / "train.parquet")
    valid_df = pd.read_parquet(root / "valid.parquet")
    test_df = pd.read_parquet(root / "test.parquet")

    print(f"Train: {train_df.shape} | Valid: {valid_df.shape} | Test : {test_df.shape}")

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    # Clean and sequence features identically
    X_train = clean_and_format(train_df[all_features])
    y_train = train_df[TARGET]

    X_valid = clean_and_format(valid_df[all_features])
    y_valid = valid_df[TARGET]

    X_test = clean_and_format(test_df[all_features])
    y_test = test_df[TARGET]

    # LightGBM gradient boosting regressor, leaf-wise growth (vs CatBoost's
    # symmetric/oblivious trees), generally faster and can capture sharper
    # local splits on tabular data like this.
    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        objective="regression",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining LightGBM Model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=None
    )

    print("\nEvaluating Model on Test Data Matrix...")
    predictions = model.predict(X_test)
    metrics_df = evaluate_regression(y_test, predictions)

    # Model tracking setup across standard directory configurations
    model_dir = Path(model_root) / version / "loan_amount"
    report_dir = Path(report_root) / version / "loan_amount"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lightgbm_loan_amount.pkl"
    joblib.dump(model, model_path)
    metrics_df.to_csv(report_dir / "lightgbm_loan_amount_metrics.csv", index=False)

    print("\nFinal Test Metrics (Loan Amount):")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved clean artifact to: {model_path}")

    return model_path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_loan_amount(
        feature_store_root=project_root / "data" / "feature_store",
        model_root=project_root / "models",
        report_root=project_root / "reports" / "dissertation_results",
        version="v4"
    )