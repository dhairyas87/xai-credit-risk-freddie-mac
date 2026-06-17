# scripts/build_performance_features.py

import pandas as pd

from src.data_loader import load_performance


def build_performance_features(perf_df):

    perf_df = perf_df.copy()

    perf_df[
        "current_loan_delinquency_status"
    ] = pd.to_numeric(
        perf_df[
            "current_loan_delinquency_status"
        ],
        errors="coerce"
    ).fillna(0)

    perf_df["modified_flag"] = (
        perf_df["modification_flag"]
        .notna()
    ).astype(int)

    perf_df["assistance_flag"] = (
        perf_df["borrower_assistance_status"]
        .notna()
    ).astype(int)

    perf_df["zero_balance_flag"] = (
        perf_df["zero_balance_code"]
        .notna()
    ).astype(int)

    features = (
        perf_df
        .groupby("loan_identifier")
        .agg(
            max_delinquency=(
                "current_loan_delinquency_status",
                "max"
            ),
            avg_delinquency=(
                "current_loan_delinquency_status",
                "mean"
            ),
            months_delinquent=(
                "current_loan_delinquency_status",
                lambda x: (x > 0).sum()
            ),
            loan_age_max=("loan_age", "max"),
            loan_age_avg=("loan_age", "mean"),
            remaining_term_avg=(
                "remaining_months_to_maturity",
                "mean"
            ),
            current_rate_avg=(
                "current_interest_rate",
                "mean"
            ),
            current_rate_max=(
                "current_interest_rate",
                "max"
            ),
            current_rate_min=(
                "current_interest_rate",
                "min"
            ),
            estimated_ltv_avg=(
                "estimated_ltv",
                "mean"
            ),
            estimated_ltv_max=(
                "estimated_ltv",
                "max"
            ),
            ever_modified=(
                "modified_flag",
                "max"
            ),
            modification_count=(
                "modified_flag",
                "sum"
            ),
            ever_assistance=(
                "assistance_flag",
                "max"
            ),
            assistance_count=(
                "assistance_flag",
                "sum"
            ),
            ever_zero_balance=(
                "zero_balance_flag",
                "max"
            )
        )
        .reset_index()
    )

    features["rate_volatility"] = (
        features["current_rate_max"]
        -
        features["current_rate_min"]
    )

    return features


def build_performance_features_file(
    perf_path,
    output_path
):

    perf_df = load_performance(
        perf_path
    )

    features = build_performance_features(
        perf_df
    )

    features.to_parquet(
        output_path,
        index=False
    )

    return features