from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models/v4/loan_amount"

def generate_pdp_chart():
    print("Initializing PDP Simulation Grid Axis...")
    
    # Load Stacked Pipeline Assets
    cb_model = joblib.load(MODEL_DIR / "stack_base_catboost.pkl")
    xgb_model = joblib.load(MODEL_DIR / "stack_base_xgb.pkl")
    meta_blender = joblib.load(MODEL_DIR / "stack_meta_blender.pkl")
    encoder = joblib.load(MODEL_DIR / "stack_target_encoder.pkl")
    
    # Simulate LTV continuum bounds
    ltv_axis = np.linspace(20, 100, 50)
    simulated_records = []
    
    for ltv in ltv_axis:
        simulated_records.append({
            "credit_score": 740, "dti": 35.0, "num_borrowers": 1, "num_units": 1,
            "ltv": ltv, "cltv": ltv, "mortgage_insurance_pct": 0.0, "quarter_num": 4,
            "first_time_homebuyer_indicator": "N", "occupancy_status": "P",
            "property_type": "SF", "property_state": "CA", "msa": "0",
            "channel": "R", "loan_purpose": "P", "program_indicator": "9",
            "property_valuation_method": "7", "interest_only_indicator": "N",
            "mortgage_insurance_cancellation_indicator": "7", "quarter": "2018Q4",
        })
        
    df = pd.DataFrame(simulated_records)
    
    # 1. Feature Engineering
    df['dti_ltv_interaction'] = df['dti'] * df['ltv']
    df['credit_risk_multiplier'] = df['credit_score'] / (df['dti'] + 1.0)
    
    num_cols = [
        "credit_score", "dti", "num_borrowers", "num_units", "ltv", "cltv", 
        "mortgage_insurance_pct", "quarter_num", "dti_ltv_interaction", "credit_risk_multiplier"
    ]
    cat_cols = [
        "first_time_homebuyer_indicator", "occupancy_status", "property_type",
        "property_state", "msa", "channel", "loan_purpose", "program_indicator",
        "property_valuation_method", "interest_only_indicator",
        "mortgage_insurance_cancellation_indicator", "quarter"
    ]
    
    # Format CatBoost Matrix Order
    cb_features_order = num_cols + cat_cols
    features_cb = df[cb_features_order].copy()
    for col in cat_cols:
        features_cb[col] = features_cb[col].astype(str)
        
    # Format XGBoost Target Encoded Matrix Order
    encoded_cats = encoder.transform(df[cat_cols])
    encoded_cat_cols = [f"{col}_encoded" for col in cat_cols]
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=df.index)
    
    features_xgb_raw = pd.concat([df[num_cols], encoded_df], axis=1)
    xgb_features_order = num_cols + encoded_cat_cols
    features_xgb = features_xgb_raw[xgb_features_order].copy()
    
    # 2. Extract Layer Predictions
    preds_cb = cb_model.predict(features_cb)
    preds_xgb = xgb_model.predict(features_xgb)
    
    # 3. Process blender matrix inputs
    meta_df = pd.DataFrame({"pred_cb": preds_cb, "pred_xgb": preds_xgb})
    final_log_amounts = meta_blender.predict(meta_df)
    final_dollar_amounts = np.expm1(final_log_amounts)
    
    # Render Academic Line Visualisation Chart
    plt.figure(figsize=(9, 5))
    plt.plot(ltv_axis, final_dollar_amounts, color='#174f3a', linewidth=2.5, label='Stacked Ensemble Model')
    plt.title('Partial Dependence Plot (PDP): Loan-to-Value (LTV) Risk Monotonicity', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Loan-to-Value (LTV % Axis Range)', fontsize=10)
    plt.ylabel('Predicted Portfolio Loan Value (USD $)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    output_path = PROJECT_ROOT / "reports/dissertation_results/loan_amount_ltv_pdp_stacked.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Academic line chart saved directly to: {output_path}")

if __name__ == "__main__":
    generate_pdp_chart()
