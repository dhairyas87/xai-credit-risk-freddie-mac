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
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

TARGET = "target"
NUMERICAL_FEATURES = [
    "credit_score", "dti", "num_borrowers", "num_units", "ltv", "cltv", 
    "mortgage_insurance_pct", "quarter_num"
]
CATEGORICAL_FEATURES = [
    "first_time_homebuyer_indicator", "occupancy_status", "property_type",
    "property_state", "msa", "channel", "loan_purpose", "program_indicator",
    "property_valuation_method", "interest_only_indicator",
    "mortgage_insurance_cancellation_indicator", "quarter"
]

def clean_and_format(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).fillna("UNKNOWN")
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # Leakage-Free Interaction terms (no target references used!)
    df['dti_ltv_interaction'] = df['dti'] * df['ltv']
    df['credit_risk_multiplier'] = df['credit_score'] / (df['dti'] + 1.0)
    return df

def train_stacked_ensemble(feature_store_root, model_root, version="v4"):
    print("=" * 60)
    print("TRAINING CORRECTED LEAKAGE-FREE STACKING PIPELINE")
    print("=" * 60)
    
    root = Path(feature_store_root) / "v3" / "loan_amount"
    train_df = pd.read_parquet(root / "train.parquet")
    valid_df = pd.read_parquet(root / "valid.parquet")
    test_df = pd.read_parquet(root / "test.parquet")
    
    # 1. Clean Data Matrices
    train_df = clean_and_format(train_df)
    valid_df = clean_and_format(valid_df)
    test_df = clean_and_format(test_df)
    
    # 2. Advanced Cross-Model Target Encoding for High-Cardinality Fields
    # This securely transforms text columns like MSA into steady risk floats
    encoder = TargetEncoder(smooth="auto", cv=5, random_state=42)
    
    train_encoded_cats = encoder.fit_transform(train_df[CATEGORICAL_FEATURES], train_df[TARGET])
    valid_encoded_cats = encoder.transform(valid_df[CATEGORICAL_FEATURES])
    test_encoded_cats = encoder.transform(test_df[CATEGORICAL_FEATURES])
    
    encoded_cat_cols = [f"{col}_encoded" for col in CATEGORICAL_FEATURES]
    
    train_enc_df = pd.DataFrame(train_encoded_cats, columns=encoded_cat_cols, index=train_df.index)
    valid_enc_df = pd.DataFrame(valid_encoded_cats, columns=encoded_cat_cols, index=valid_df.index)
    test_enc_df = pd.DataFrame(test_encoded_cats, columns=encoded_cat_cols, index=test_df.index)
    
    # Merge engineered tables back into main sets
    train_df = pd.concat([train_df, train_enc_df], axis=1)
    valid_df = pd.concat([valid_df, valid_enc_df], axis=1)
    test_df = pd.concat([test_df, test_enc_df], axis=1)
    
    ENGINEERED_NUM_FEATURES = NUMERICAL_FEATURES + ['dti_ltv_interaction', 'credit_risk_multiplier']
    
    # Define features for both base models
    cb_features = ENGINEERED_NUM_FEATURES + CATEGORICAL_FEATURES
    xgb_features = ENGINEERED_NUM_FEATURES + encoded_cat_cols
    
    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]
    
    # 3. Train Base Model 1: CatBoost Regressor (handles raw strings beautifully)
    print("\nTraining Base Model 1: CatBoost...")
    model_cb = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6, random_seed=42, verbose=0
    )
    model_cb.fit(
        train_df[cb_features], y_train, 
        cat_features=CATEGORICAL_FEATURES, 
        eval_set=(valid_df[cb_features], y_valid), 
        early_stopping_rounds=40
    )
    
    # 4. Train Base Model 2: XGBoost Regressor (handles encoded floats perfectly)
    print("Training Base Model 2: XGBoost...")
    model_xgb = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6, 
        tree_method="hist", random_state=42
    )
    model_xgb.fit(
        train_df[xgb_features], y_train, 
        eval_set=[(valid_df[xgb_features], y_valid)],
        verbose=False
    )
    
    # 5. Build Level-1 Meta Feature Matrix using Validation Set
    val_pred_cb = model_cb.predict(valid_df[cb_features])
    val_pred_xgb = model_xgb.predict(valid_df[xgb_features])
    meta_X_val = pd.DataFrame({"pred_cb": val_pred_cb, "pred_xgb": val_pred_xgb})
    
    print("Training Meta-Blender Layer...")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_val, y_valid)
    
    # 6. Evaluate Stacking System against Test Set
    test_pred_cb = model_cb.predict(test_df[cb_features])
    test_pred_xgb = model_xgb.predict(test_df[xgb_features])
    meta_X_test = pd.DataFrame({"pred_cb": test_pred_cb, "pred_xgb": test_pred_xgb})
    
    final_log_predictions = meta_model.predict(meta_X_test)
    y_test_raw = np.expm1(test_df[TARGET])
    final_preds_raw = np.expm1(final_log_predictions)
    
    print("\n" + "="*40)
    print("UPDATED LEAKAGE-FREE STACKING PIPELINE METRICS:")
    print(f"MAE (Dollars): ${mean_absolute_error(y_test_raw, final_preds_raw):,.2f}")
    print(f"MAPE (%): {mean_absolute_percentage_error(y_test_raw, final_preds_raw)*100:.3f}%")
    print(f"R2 Variance Score (Log Space): {r2_score(test_df[TARGET], final_log_predictions):.4f}")
    print("="*40 + "\n")
    
    # Save Pipeline Assets safely
    model_dir = Path(model_root) / version / "loan_amount"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_cb, model_dir / "stack_base_catboost.pkl")
    joblib.dump(model_xgb, model_dir / "stack_base_xgb.pkl")
    joblib.dump(meta_model, model_dir / "stack_meta_blender.pkl")
    joblib.dump(encoder, model_dir / "stack_target_encoder.pkl")
    print(f"Saved stacked ensemble configurations inside: {model_dir}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_stacked_ensemble(
        feature_store_root=project_root / "data" / "feature_store",
        model_root=project_root / "models"
    )
