from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models/v4/loan_amount"

def generate_pdp():
    print("Generating Partial Dependence Plot metrics...")
    
    # Load assets
    model_cb = joblib.load(MODEL_DIR / "stack_base_catboost.pkl")
    msa_map = joblib.load(MODEL_DIR / "stack_msa_map.pkl")
    
    # Synthesize a standard baseline borrower sequence across an LTV axis grid
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
    
    # Apply identical Feature Engineering transformers
    df['dti_ltv_interaction'] = df['dti'] * df['ltv']
    df['credit_risk_multiplier'] = df['credit_score'] / (df['dti'] + 1.0)
    df['msa_baseline_median'] = 12.3 # constant regional anchor alignment
    df['home_value_tier'] = "5"
    
    num_cols = [
        "credit_score", "dti", "num_borrowers", "num_units", "ltv", "cltv", 
        "mortgage_insurance_pct", "quarter_num", "dti_ltv_interaction", 
        "credit_risk_multiplier", "msa_baseline_median"
    ]
    cat_cols = [
        "first_time_homebuyer_indicator", "occupancy_status", "property_type",
        "property_state", "msa", "channel", "loan_purpose", "program_indicator",
        "property_valuation_method", "interest_only_indicator",
        "mortgage_insurance_cancellation_indicator", "quarter", "home_value_tier"
    ]
    
    df = df[num_cols + cat_cols]
    for col in cat_cols:
        df[col] = df[col].astype(str)
        
    # Extract structural predictions across the continuous grid axis
    log_predictions = model_cb.predict(df)
    dollar_predictions = np.expm1(log_predictions)
    
    # Render academic plot figure
    plt.figure(figsize=(9, 5))
    plt.plot(ltv_axis, dollar_predictions, color='#174f3a', linewidth=2.5, label='Engineered Stacking Model')
    plt.title('Partial Dependence Plot: Impact of LTV on Predicted Loan Amount', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Loan-to-Value (LTV %)', fontsize=10)
    plt.ylabel('Predicted Loan Amount ($)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    output_path = PROJECT_ROOT / "reports/dissertation_results/loan_amount_ltv_pdp.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Academic line graph chart successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_pdp()
