from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


def create_random_split(df):

    # =====================================
    # First Split
    # 80% Train
    # 20% Temp
    # =====================================

    train_df, temp_df = train_test_split(

        df,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )

    # =====================================
    # Second Split
    # 10% Valid
    # 10% Test
    # =====================================

    valid_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=42,

        shuffle=True

    )

    return (

        train_df,

        valid_df,

        test_df

    )


def save_split(

    train_df,

    valid_df,

    test_df,

    output_dir

):

    train_df.to_parquet(

        output_dir
        / "train.parquet",

        index=False

    )

    valid_df.to_parquet(

        output_dir
        / "valid.parquet",

        index=False

    )

    test_df.to_parquet(

        output_dir
        / "test.parquet",

        index=False

    )

    print(
        f"\nSaved datasets to {output_dir}"
    )

    print(
        f"Train: {train_df.shape}"
    )

    print(
        f"Valid: {valid_df.shape}"
    )

    print(
        f"Test : {test_df.shape}"
    )


def build_lending_modeling_datasets(

    feature_store_root,

    version="v3"

):

    print("=" * 60)
    print(
        "BUILDING LENDING MODELING DATASETS"
    )
    print("=" * 60)

    root = (
        Path(feature_store_root)
        / version
    )

    objectives = [

        "loan_amount",

        "interest_rate",

        "loan_term"

    ]

    for objective in objectives:

        print(
            f"\nProcessing {objective}"
        )

        feature_store_path = (

            root
            / objective
            / "feature_store.parquet"

        )

        df = pd.read_parquet(
            feature_store_path
        )

        train_df, valid_df, test_df = (
            create_random_split(df)
        )

        save_split(

            train_df,

            valid_df,

            test_df,

            root / objective

        )

    print(
        "\nModeling datasets created."
    )