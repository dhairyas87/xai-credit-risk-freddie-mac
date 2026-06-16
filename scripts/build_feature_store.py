"""
Build Feature Store

Input:
    master_dataset_2018.parquet

Outputs:

Baseline:
    train_baseline.parquet
    valid_baseline.parquet
    test_baseline.parquet

BSS:
    train_bss.parquet
    valid_bss.parquet
    test_bss.parquet
"""

from pathlib import Path

import pandas as pd

from src.modeling_prep import (
    create_baseline_dataset,
    create_bss_dataset,
    create_temporal_splits,
    save_datasets
)


def build_feature_store(
    input_path,
    output_dir
):

    print("=" * 60)
    print("BUILDING FEATURE STORE")
    print("=" * 60)

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading dataset...")

    df = pd.read_parquet(
        input_path
    )

    print(
        f"Dataset Shape: {df.shape}"
    )

    # =================================================
    # BASELINE DATASET
    # =================================================

    print("\nCreating baseline dataset...")

    baseline_df = create_baseline_dataset(
        df
    )

    train_df, valid_df, test_df = (
        create_temporal_splits(
            baseline_df
        )
    )

    save_datasets(
        train_df,
        valid_df,
        test_df,
        prefix="baseline",
        output_dir=output_dir
    )

    print(
        f"Baseline Shape: {baseline_df.shape}"
    )

    # =================================================
    # BSS DATASET
    # =================================================

    print("\nCreating BSS dataset...")

    bss_df = create_bss_dataset(
        df
    )

    train_df, valid_df, test_df = (
        create_temporal_splits(
            bss_df
        )
    )

    save_datasets(
        train_df,
        valid_df,
        test_df,
        prefix="bss",
        output_dir=output_dir
    )

    print(
        f"BSS Shape: {bss_df.shape}"
    )

    print("\nFeature store creation complete.")
