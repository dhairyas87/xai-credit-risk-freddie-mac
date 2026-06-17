# scripts/build_target_dataset.py

from pathlib import Path

from src.data_loader import (
    load_origination,
    load_performance
)

from src.validation import (
    validate_loan_counts
)

from src.target_generation import (
    create_loan_level_performance,
    create_bssi,
    create_stress_flag,
    create_stress_level
)


def build_target_dataset(
    orig_path,
    perf_path,
    output_path
):
    
    print("Loading data...")

    orig_df = load_origination(orig_path)

    perf_df = load_performance(perf_path)

    print("Running validation...")

    validate_loan_counts(
        orig_df,
        perf_df
    )

    print("Creating loan-level dataset...")

    loan_perf = (
        create_loan_level_performance(
            perf_df
        )
    )

    loan_perf = create_bssi(
        loan_perf
    )

    loan_perf = create_stress_flag(
        loan_perf
    )

    loan_perf = create_stress_level(
        loan_perf
    )

    print("Saving parquet...")

    loan_perf.to_parquet(
        output_path,
        index=False
    )

    print(
        loan_perf[
            [
                "bssi",
                "stress_flag",
                "stress_level"
            ]
        ].head()
    )
    
    print(
        loan_perf["stress_level"]
        .value_counts()
    )

    print("Done")