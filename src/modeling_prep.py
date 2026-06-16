# src/modeling_prep.py

"""
Modeling Preparation Utilities

Creates:

- Baseline feature dataset
- BSS-enhanced feature dataset
- Temporal train/validation/test splits
"""

import pandas as pd


# =====================================================
# COLUMN GROUPS
# =====================================================

METADATA_COLUMNS = [
    "loan_identifier",
    "year",
    "quarter",
    "quarter_num"
]

TARGET_COLUMNS = [
    "stress_flag",
    "bssi"
]

LEAKAGE_COLUMNS = [
    "max_delinquency",
    "ever_ra",
    "ever_modified",
    "ever_assistance"
]

DROP_COLUMNS = [
    "special_eligibility_program",
    "pre_harp_loan_sequence_number",
    "postal_code"
]

ENGINEERED_COLUMNS = [
    "credit_score_norm",
    "dti_norm",
    "interest_rate_norm",
    "ltv_norm",
    "num_borrowers_norm",
    "bss",
    "bss_bucket",
    "bss_level"
]


# =====================================================
# CLEAN DATASET
# =====================================================

def remove_unusable_columns(df):

    df = df.copy()

    columns_to_remove = (
        LEAKAGE_COLUMNS
        + DROP_COLUMNS
    )

    existing_cols = [
        col
        for col in columns_to_remove
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_cols
    )

    return df


# =====================================================
# BASELINE FEATURES
# =====================================================

def create_baseline_dataset(df):

    df = remove_unusable_columns(
        df
    )

    baseline_remove = [
        col
        for col in ENGINEERED_COLUMNS
        if col in df.columns
    ]

    baseline_df = df.drop(
        columns=baseline_remove
    )

    return baseline_df


# =====================================================
# BSS FEATURES
# =====================================================

def create_bss_dataset(df):

    df = remove_unusable_columns(
        df
    )

    remove_cols = [
        "credit_score_norm",
        "dti_norm",
        "interest_rate_norm",
        "ltv_norm",
        "num_borrowers_norm"
    ]

    existing_cols = [
        col
        for col in remove_cols
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_cols
    )

    return df


# =====================================================
# TEMPORAL SPLITS
# =====================================================

def create_temporal_splits(df):

    train_df = df[
        df["quarter"].isin(
            [
                "2018Q1",
                "2018Q2"
            ]
        )
    ].copy()

    valid_df = df[
        df["quarter"] == "2018Q3"
    ].copy()

    test_df = df[
        df["quarter"] == "2018Q4"
    ].copy()

    return (
        train_df,
        valid_df,
        test_df
    )


# =====================================================
# SAVE DATASETS
# =====================================================

def save_datasets(
    train_df,
    valid_df,
    test_df,
    prefix,
    output_dir
):

    train_df.to_parquet(
        f"{output_dir}/train_{prefix}.parquet",
        index=False
    )

    valid_df.to_parquet(
        f"{output_dir}/valid_{prefix}.parquet",
        index=False
    )

    test_df.to_parquet(
        f"{output_dir}/test_{prefix}.parquet",
        index=False
    )

    print(
        f"Saved {prefix} datasets."
    )
