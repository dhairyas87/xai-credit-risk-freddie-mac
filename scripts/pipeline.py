"""
Project Pipeline Orchestrator

Stages:

1. target_generation
2. feature_engineering

Future:
3. feature_store
4. model_training
5. explainability
"""

from pathlib import Path

from scripts.build_target_dataset import (
    build_target_dataset
)

from scripts.build_master_dataset import (
    build_master_dataset
)


PIPELINE_STAGES = [
    "target_generation",
    "feature_engineering"
]


def run_pipeline(
    quarter: str,
    start_stage: str = "target_generation",
    force_rebuild: bool = False
):
    """
    Run pipeline from any stage.

    Example:

    run_pipeline(
        quarter="2018Q1"
    )

    run_pipeline(
        quarter="2018Q2",
        start_stage="feature_engineering"
    )
    """

    start_idx = PIPELINE_STAGES.index(
        start_stage
    )

    stages_to_run = PIPELINE_STAGES[
        start_idx:
    ]

    print("=" * 60)
    print(f"RUNNING PIPELINE : {quarter}")
    print("=" * 60)

    raw_origination = (
        f"../data/raw/historical_data_{quarter}.txt"
    )

    raw_performance = (
        f"../data/raw/historical_data_time_{quarter}.txt"
    )

    target_dataset = (
        f"../data/processed/loan_perf_{quarter}.parquet"
    )

    master_dataset = (
        f"../data/processed/master_dataset_{quarter}.parquet"
    )

    # --------------------------------------------------
    # TARGET GENERATION
    # --------------------------------------------------

    if "target_generation" in stages_to_run:

        if (
            Path(target_dataset).exists()
            and not force_rebuild
        ):
            print(
                f"Target dataset already exists: {target_dataset}"
            )

        else:

            print("\nGenerating target dataset...")

            build_target_dataset(
                orig_path=raw_origination,
                perf_path=raw_performance,
                output_path=target_dataset
            )

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    if "feature_engineering" in stages_to_run:

        if (
            Path(master_dataset).exists()
            and not force_rebuild
        ):
            print(
                f"Master dataset already exists: {master_dataset}"
            )

        else:

            print(
                "\nGenerating master dataset..."
            )

            build_master_dataset(
                origination_path=raw_origination,
                target_path=target_dataset,
                output_path=master_dataset
            )

    print("\nPipeline completed.")