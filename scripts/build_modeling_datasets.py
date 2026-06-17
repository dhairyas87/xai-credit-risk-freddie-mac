from pathlib import Path
import pandas as pd

def create_temporal_split(df):
    
    
    train_df = df[
        df["quarter"].isin(
            ["2018Q1", "2018Q2"]
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


def save_split(
train_df,
valid_df,
test_df,
output_dir
):


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    train_df.to_parquet(
        output_dir / "train.parquet",
        index=False
    )
    
    valid_df.to_parquet(
        output_dir / "valid.parquet",
        index=False
    )
    
    test_df.to_parquet(
        output_dir / "test.parquet",
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


def build_modeling_datasets(


feature_store_root,

version="v1"


):


    print("=" * 60)
    print("BUILDING MODELING DATASETS")
    print("=" * 60)
    
    root = (
        Path(feature_store_root)
        / version
    )
    
    objectives = [
    
        "stress_prediction",
    
        "stress_severity",
    
        "serviceability"
    
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
            create_temporal_split(df)
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
