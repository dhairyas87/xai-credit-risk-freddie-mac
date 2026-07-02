from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

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

def map_to_classes(y_series):
    """Maps continuous months directly to clean discrete classification targets using a dict."""
    y_int = y_series.astype(int)
    mapping = {180: 15, 360: 30}
    return y_int.map(mapping).fillna(99).astype(int).values

def clean_and_format(df):
    """Natively formats data types for fast CatBoost multi-threaded processing."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def train_loan_term(feature_store_root: Path, model_root: Path, report_root: Path, version: str = "v4"):
    print("=" * 60)
    print("TRAINING LOAN TERM CLASSIFICATION MODEL (CATBOOST)")
    print("=" * 60)
    
    data_root = feature_store_root / "v3" / "loan_term"
    train_df = pd.read_parquet(data_root / "train.parquet")
    test_df = pd.read_parquet(data_root / "test.parquet")

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X_train = clean_and_format(train_df[all_features])
    y_train = map_to_classes(train_df[TARGET])
    
    X_test = clean_and_format(test_df[all_features])
    y_test = map_to_classes(test_df[TARGET])

    # Initialise CatBoost Classifier for fast parallel training execution
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.08,
        depth=6,
        loss_function='MultiClass',
        random_seed=42,
        verbose=100,
        thread_count=-1 # Forces usage of all available CPU cores to prevent hanging
    )

    print("\nTraining CatBoost Classifier...")
    model.fit(
        X_train, y_train,
        cat_features=CATEGORICAL_FEATURES
    )

    predictions = model.predict(X_test)
    # Flatten array shape output by CatBoost
    predictions = predictions.flatten()
    
    print("\nClassification Report Results:")
    print(classification_report(y_test, predictions, zero_division=0))
    
    acc = accuracy_score(y_test, predictions)
    metrics_df = pd.DataFrame({"Metric": ["Accuracy"], "Value": [acc]})

    model_dir = model_root / version / "loan_term"
    report_dir = report_root / version / "loan_term"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "hist_gradient_boosting.pkl")
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    print(f"Saved classification model: {model_dir / 'hist_gradient_boosting.pkl'}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_loan_term(
        feature_store_root=project_root / "data" / "feature_store",
        model_root=project_root / "models",
        report_root=project_root / "reports" / "dissertation_results",
    )
