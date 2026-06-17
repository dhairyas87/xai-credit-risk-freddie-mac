"""
Feature Engineering Module

Creates:
- Clean origination dataset
- Master dataset
- Borrower Serviceability Score (BSS)
- BSS Buckets
- BSS Levels
"""

import numpy as np
import pandas as pd


def clean_origination_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace Freddie Mac special values with NaN.
    """

    df = df.copy()

    df["credit_score"] = df["credit_score"].replace(9999, np.nan)
    df["dti"] = df["dti"].replace(999, np.nan)
    df["ltv"] = df["ltv"].replace(999, np.nan)
    df["cltv"] = df["cltv"].replace(999, np.nan)

    drop_columns = [
    
        "pre_harp_loan_sequence_number",
    
        "harp_indicator",
    
        "super_conforming_flag"
    
    ]

    df = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    return df


def merge_origination_and_target(
    orig_df: pd.DataFrame,
    loan_perf_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge origination and target datasets.
    """

    return orig_df.merge(
        loan_perf_df,
        on="loan_identifier",
        how="inner"
    )


def create_bss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Borrower Serviceability Score.
    """

    df = df.copy()

    # -----------------------------
    # Normalized Features
    # -----------------------------

    df["credit_score_norm"] = (
        (df["credit_score"] - df["credit_score"].min())
        /
        (df["credit_score"].max() - df["credit_score"].min())
    )

    df["dti_norm"] = 1 - (
        (df["dti"] - df["dti"].min())
        /
        (df["dti"].max() - df["dti"].min())
    )

    df["interest_rate_norm"] = 1 - (
        (df["interest_rate"] - df["interest_rate"].min())
        /
        (df["interest_rate"].max() - df["interest_rate"].min())
    )

    df["ltv_norm"] = 1 - (
        (df["ltv"] - df["ltv"].min())
        /
        (df["ltv"].max() - df["ltv"].min())
    )

    df["num_borrowers_norm"] = (
        (df["num_borrowers"] - df["num_borrowers"].min())
        /
        (df["num_borrowers"].max() - df["num_borrowers"].min())
    )

    # -----------------------------
    # Borrower Serviceability Score
    # -----------------------------

    df["bss"] = (
        0.40 * df["credit_score_norm"]
        + 0.25 * df["dti_norm"]
        + 0.20 * df["interest_rate_norm"]
        + 0.10 * df["num_borrowers_norm"]
        + 0.05 * df["ltv_norm"]
    ) * 100

    # -----------------------------
    # Serviceability Buckets
    # -----------------------------

    df["bss_bucket"] = pd.qcut(
        df["bss"],
        q=5,
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    df["bss_level"] = pd.qcut(
        df["bss"],
        q=5,
        labels=[1, 2, 3, 4, 5]
    )

    return df


def validate_master_dataset(df: pd.DataFrame):
    """
    Validate final master dataset.
    """

    required_columns = [
        "loan_identifier",
        "stress_flag",
        "bssi",
        "bss",
        "bss_bucket",
        "bss_level"
    ]

    missing_cols = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    if len(df) == 0:
        raise ValueError(
            "Master dataset is empty."
        )

    print("Master dataset validation passed.")



def create_performance_features(
    df
):
    df["delinquency_intensity"] = (

        df["months_delinquent"]
    
        /
    
        df["loan_age_max"]
        .replace(0, 1)
    
    )
    df["modification_intensity"] = (
    
        df["modification_count"]
    
        /
    
        df["loan_age_max"]
        .replace(0, 1)
    
    )
    df["assistance_intensity"] = (

        df["assistance_count"]
    
        /
    
        df["loan_age_max"]
        .replace(0, 1)
    
    )
    df["eltv_drift"] = (
    
        df["estimated_ltv_avg"]
    
        -
    
        df["ltv"]
    
    )
    return df
    