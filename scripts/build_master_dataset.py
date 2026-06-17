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
from pathlib import Path

from src.feature_engineering import (
    clean_origination_data,
    merge_origination_and_target,
    create_bss,
    validate_master_dataset,
    create_performance_features
)


def build_master_dataset(
    origination_path: str,
    target_path: str,
    performance_features_path: str,
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

    print("\nLoading performance features...")

    performance_df = pd.read_parquet(
        performance_features_path
    )
    
    print(
        f"Performance Features shape: {performance_df.shape}"
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

    # ----------------------------------
    # Merge Origination + Target
    # ----------------------------------
    
    master_df = merge_origination_and_target(
        orig_df,
        loan_perf_df
    )
    
    print(
        f"After target merge: {master_df.shape}"
    )
    
    # ----------------------------------
    # Merge Performance Features
    # ----------------------------------
    
    master_df = master_df.merge(
        performance_df,
        on="loan_identifier",
        how="left"
    )

    # ----------------------------------
    # Resolve duplicate columns
    # ----------------------------------
    
    master_df = master_df.drop(
        columns=[
            "max_delinquency_x",
            "ever_modified_x",
            "ever_assistance_x"
        ],
        errors="ignore"
    )
    
    master_df = master_df.rename(
        columns={
            "max_delinquency_y": "max_delinquency",
            "ever_modified_y": "ever_modified",
            "ever_assistance_y": "ever_assistance"
        }
    )
        
    print(
        f"After performance merge: {master_df.shape}"
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

    master_df = create_performance_features(
        master_df
    )

    quarter = (
    Path(origination_path)
    .stem
    .split("_")[-1]
    )
    
    master_df["quarter"] = quarter
    
    master_df["year"] = int(
        quarter[:4]
    )
    
    master_df["quarter_num"] = int(
        quarter[-1]
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