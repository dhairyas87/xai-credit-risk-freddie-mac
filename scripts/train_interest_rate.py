from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

TARGET = "target"
CATEGORICAL_FEATURES = [
    "first_time_homebuyer_indicator", "occupancy_status", "property_type",
    "property_state", "msa", "channel", "loan_purpose", "program_indicator",
    "property_valuation_method", "interest_only_indicator",
    "mortgage_insurance_cancellation_indicator", "quarter"
]
NUMERICAL_FEATURES = [
    "credit_score", "dti", "num_borrowers", "num_units", "ltv", "cltv", 
    "mortgage_insurance_pct", "quarter_num"
]

def clean_and_format(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def train_interest_rate(feature_store_root, model_root, report_root, version="v4"):
    print("=" * 60)
    print("TRAINING INTEREST RATE REGRESSION MODEL (CATBOOST)")
    print("=" * 60)
    
    # Adjust pathway targets to match your feature store structure
    root = Path(feature_store_root) / "v3" / "interest_rate"
    train_df = pd.read_parquet(root / "train.parquet")
    valid_df = pd.read_parquet(root / "valid.parquet")
    test_df = pd.read_parquet(root / "test.parquet")
    
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X_train = clean_and_format(train_df[all_features])
    y_train = train_df[TARGET]
    
    X_valid = clean_and_format(valid_df[all_features])
    y_valid = valid_df[TARGET]
    
    X_test = clean_and_format(test_df[all_features])
    y_test = test_df[TARGET]
    
    # Custom hyperparameters to handle cyclical macro interest shifts
    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
        verbose=100
    )
    
    model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=50
    )
    
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE (Rate %)", "MAE (Rate %)", "R2 Score"],
        "Value": [rmse, mae, r2]
    })
    
    model_dir = Path(model_root) / version / "interest_rate"
    report_dir = Path(report_root) / version / "interest_rate"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_dir / "catboost_interest_rate.pkl")
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    
    print("\nTest Metrics (Interest Rate):")
    print(metrics_df.to_string(index=False))
    return model_dir / "catboost_interest_rate.pkl"

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_interest_rate(
        feature_store_root=project_root / "data" / "feature_store",
        model_root=project_root / "models",
        report_root=project_root / "reports" / "dissertation_results",
    )
