import pandas as pd

from src.serviceability import (
    calculate_serviceability_score
)


def build_feature_store(

    master_dataset_path,

    performance_features_path,

    output_path

):

    print("=" * 60)
    print("BUILDING FEATURE STORE")
    print("=" * 60)

    master_df = pd.read_parquet(
        master_dataset_path
    )

    perf_df = pd.read_parquet(
        performance_features_path
    )

    print(
        f"Master: {master_df.shape}"
    )

    print(
        f"Performance: {perf_df.shape}"
    )

    feature_store = master_df.merge(

        perf_df,

        on="loan_identifier",

        how="left"

    )

    feature_store = (
        calculate_serviceability_score(
            feature_store
        )
    )

    feature_store.to_parquet(

        output_path,

        index=False

    )

    print(
        f"Feature Store Shape: "
        f"{feature_store.shape}"
    )

    return feature_store