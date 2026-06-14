"""
Build Master Dataset Pipeline

Input:
    Origination TXT
    +
    Loan Performance Parquet

Output:
    Master Dataset Parquet
"""

import pandas as pd

from src.data_loader import load_origination

from src.feature_engineering import (
    clean_origination_data,
    merge_origination_and_target,
    create_bss,
    validate_master_dataset
)


def build_master_dataset(
    origination_path: str,
    target_path: str,
    output_path: str
):
    """
    Build complete master dataset.
    """

    print("=" * 60)
    print("BUILDING MASTER DATASET")
    print("=" * 60)

    # ----------------------------------
    # Load Data
    # ----------------------------------

    print("\nLoading origination data...")

    orig_df = load_origination(
        origination_path
    )

    print(
        f"Origination shape: {orig_df.shape}"
    )

    print("\nLoading target dataset...")

    loan_perf_df = pd.read_parquet(
        target_path
    )

    print(
        f"Target shape: {loan_perf_df.shape}"
    )

    # ----------------------------------
    # Clean Data
    # ----------------------------------

    print("\nCleaning origination data...")

    orig_df = clean_origination_data(
        orig_df
    )

    # ----------------------------------
    # Merge
    # ----------------------------------

    print("\nMerging datasets...")

    master_df = merge_origination_and_target(
        orig_df,
        loan_perf_df
    )

    print(
        f"Master shape after merge: {master_df.shape}"
    )

    # ----------------------------------
    # Create BSS
    # ----------------------------------

    print("\nCreating Borrower Serviceability Score...")

    master_df = create_bss(
        master_df
    )

    # ----------------------------------
    # Validate
    # ----------------------------------

    print("\nValidating dataset...")

    validate_master_dataset(
        master_df
    )

    # ----------------------------------
    # Save
    # ----------------------------------

    print("\nSaving dataset...")

    master_df.to_parquet(
        output_path,
        index=False
    )

    print("\nDataset saved successfully.")
    print(f"Final shape: {master_df.shape}")

    return master_df