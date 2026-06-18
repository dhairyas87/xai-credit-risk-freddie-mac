import pandas as pd
import numpy as np

from pathlib import Path

# =====================================================

# UNDERWRITING FEATURES

# =====================================================

UNDERWRITING_FEATURES = [


# ---------------------------------
# Borrower
# ---------------------------------

"credit_score",

"first_time_homebuyer_indicator",

"dti",

"num_borrowers",

# ---------------------------------
# Property / Occupancy
# ---------------------------------

"occupancy_status",

"property_type",

"property_state",

"msa",

"num_units",

# ---------------------------------
# Loan Structure
# ---------------------------------

"ltv",

"cltv",

"mortgage_insurance_pct",

"channel",

"loan_purpose",

"program_indicator",

"property_valuation_method",

"interest_only_indicator",

"mortgage_insurance_cancellation_indicator",

# ---------------------------------
# Engineered Features
# ---------------------------------


# ---------------------------------
# Time
# ---------------------------------

"quarter",

"quarter_num"


]

# =====================================================

# FEATURE STORE

# =====================================================

def build_lending_recommendation_feature_store(


master_dataset_path,

output_root,

version="v3"


):

    print("=" * 60)
    print(
        "BUILDING LENDING RECOMMENDATION FEATURE STORE"
    )
    print("=" * 60)
    
    # ---------------------------------
    # Load Master Dataset
    # ---------------------------------
    
    df = pd.read_parquet(
        master_dataset_path
    )
    
    print(
        f"Input Shape: {df.shape}"
    )
    
    root = (
        Path(output_root)
        / version
    )
    
    root.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # =================================================
    # LOAN AMOUNT
    # =================================================
    
    print(
        "\nCreating Loan Amount Dataset..."
    )
    
    loan_amount_df = df[
        UNDERWRITING_FEATURES
    ].copy()
    
    loan_amount_df["target"] = (
        np.log1p(
            df["original_upb"]
        )
    )
    
    output_dir = (
        root
        / "loan_amount"
    )
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    loan_amount_df.to_parquet(
    
        output_dir
        / "feature_store.parquet",
    
        index=False
    
    )
    
    print(
        f"Loan Amount Shape: "
        f"{loan_amount_df.shape}"
    )
    
    # =================================================
    # INTEREST RATE
    # =================================================
    
    print(
        "\nCreating Interest Rate Dataset..."
    )
    
    interest_rate_df = df[
        UNDERWRITING_FEATURES
    ].copy()
    
    interest_rate_df["target"] = (
        df["interest_rate"]
    )
    
    output_dir = (
        root
        / "interest_rate"
    )
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    interest_rate_df.to_parquet(
    
        output_dir
        / "feature_store.parquet",
    
        index=False
    
    )
    
    print(
        f"Interest Rate Shape: "
        f"{interest_rate_df.shape}"
    )
    
    # =================================================
    # LOAN TERM
    # =================================================
    
    print(
        "\nCreating Loan Term Dataset..."
    )
    
    loan_term_df = df[
        UNDERWRITING_FEATURES
    ].copy()
    
    loan_term_df["target"] = (
        df["original_loan_term"]
    )
    
    output_dir = (
        root
        / "loan_term"
    )
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    loan_term_df.to_parquet(
    
        output_dir
        / "feature_store.parquet",
    
        index=False
    
    )
    
    print(
        f"Loan Term Shape: "
        f"{loan_term_df.shape}"
    )
    
    print("\n" + "=" * 60)
    print(
        "LENDING FEATURE STORE COMPLETE"
    )
    print("=" * 60)
    
    return {
    
        "loan_amount":
        loan_amount_df,
    
        "interest_rate":
        interest_rate_df,
    
        "loan_term":
        loan_term_df
    
    }

