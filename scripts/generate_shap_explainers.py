"""Script to generate native SHAP explainability charts for the dissertation."""

from pathlib import Path
import numpy as np
import pandas as pd
import shap
import catboost
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models/v4/loan_amount"

def extract_shap_weights():
    print("Initializing Native Tree SHAP Explainer...")
    
    # 1. Load the exact CatBoost model asset
    model_path = MODEL_DIR / "stack_base_catboost.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing target model artifact: {model_path}")
    cb_model = joblib.load(model_path)
    
    # 2. Re-create the baseline feature data payload row
    sample_borrower = {
        "credit_score": 740, 
        "dti": 35.0, 
        "num_borrowers": 1, 
        "num_units": 1,
        "ltv": 80.0, 
        "cltv": 80.0, 
        "mortgage_insurance_pct": 0.0, 
        "quarter_num": 4,
        "first_time_homebuyer_indicator": "N", 
        "occupancy_status": "P",
        "property_type": "SF", 
        "property_state": "CA", 
        "msa": "0",
        "channel": "R", 
        "loan_purpose": "P", 
        "program_indicator": "9",
        "property_valuation_method": "7", 
        "interest_only_indicator": "N",
        "mortgage_insurance_cancellation_indicator": "7", 
        "quarter": "2018Q4",
    }
    df = pd.DataFrame([sample_borrower])
    
    # Apply identical Feature Engineering values
    df['dti_ltv_interaction'] = df['dti'] * df['ltv']
    df['credit_risk_multiplier'] = df['credit_score'] / (df['dti'] + 1.0)
    df['msa_baseline_median'] = 12.3
    df['home_value_tier'] = "5"
    
    original_num_cols = [
        "credit_score", "dti", "num_borrowers", "num_units", 
        "ltv", "cltv", "mortgage_insurance_pct", "quarter_num"
    ]
    cat_cols = [
        "first_time_homebuyer_indicator", "occupancy_status", "property_type",
        "property_state", "msa", "channel", "loan_purpose", "program_indicator",
        "property_valuation_method", "interest_only_indicator",
        "mortgage_insurance_cancellation_indicator", "quarter"
    ]
    
    stacked_num_cols = original_num_cols + [
        "dti_ltv_interaction", 
        "credit_risk_multiplier", 
        "msa_baseline_median"
    ]
    stacked_cat_cols = cat_cols + ["home_value_tier"]
    
    # Establish strict column grouping order (All numbers first, text second)
    cb_features_order = stacked_num_cols + stacked_cat_cols
    features_cb = df[cb_features_order].copy()
    
    for col in stacked_cat_cols:
        features_cb[col] = features_cb[col].astype(str)

    # 3. Wrap features directly in a CatBoost Pool structure
    shap_pool = catboost.Pool(
        data=features_cb,
        cat_features=stacked_cat_cols
    )

    # 4. Extract local attributions using the direct TreeExplainer
    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer.shap_values(shap_pool)
    
    # 5. FIXED: Transform raw vectors into an explanation layout structure
    # This prevents the plotting function from trying to slice the un-subscriptable Pool
    explanation = shap.Explanation(
        values=shap_values[0], # slices out row element safely
        base_values=explainer.expected_value,
        data=features_cb.iloc[0].values,
        feature_names=cb_features_order
    )
    
    # 6. Render academic publication-ready plot visualization configuration
    plt.figure(figsize=(10, 6))
    shap.plots.bar(explanation, show=False)
    plt.title(
        "XAI Report: Local SHAP Feature Attributions (Stacked Architecture)", 
        fontsize=12, 
        fontweight='bold', 
        pad=20
    )
    
    # Handle save pathways securely
    output_path = (
        PROJECT_ROOT 
        / "reports/dissertation_results/loan_amount_shap_bar.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("SUCCESS! NATIVE SHAP ATTRIBUTION GRAPH GENERATED")
    print(f"Artifact location saved to: {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    extract_shap_weights()
