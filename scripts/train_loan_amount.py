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
from catboost import CatBoostRegressor

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
    Ensures optimal datatypes for CatBoost execution.
    Natively formats strings and fills missing states without manual steps.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def train_loan_amount(feature_store_root, model_root, report_root, version="v4"):
    print("=" * 60)
    print("TRAINING ADVANCED LOAN AMOUNT REGRESSOR (CATBOOST)")
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
    
    # Initialize CatBoost optimized specifically for continuous asset valuation metrics
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
        verbose=100
    )
    
    print("\nTraining CatBoost Model...")
    model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50
    )
    
    print("\nEvaluating Model on Test Data Matrix...")
    predictions = model.predict(X_test)
    metrics_df = evaluate_regression(y_test, predictions)
    
    # Model tracking setup across standard directory configurations
    model_dir = Path(model_root) / version / "loan_amount"
    report_dir = Path(report_root) / version / "loan_amount"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "catboost_loan_amount.pkl"
    joblib.dump(model, model_path)
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    
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
