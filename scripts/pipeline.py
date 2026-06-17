# scripts/pipeline.py

from pathlib import Path

from scripts.build_target_dataset import (
build_target_dataset
)

from scripts.build_master_dataset import (
build_master_dataset
)

from scripts.build_performance_features import (
build_performance_features_file
)

from scripts.combine_quarters import (
combine_quarters
)

from scripts.build_feature_store import (
    build_feature_store
)
from scripts.build_modeling_datasets import (
    build_modeling_datasets
)
from scripts.run_diagnostics import (
    run_diagnostics
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUARTERS = [
    "2018Q1",
    "2018Q2",
    "2018Q3",
    "2018Q4"
    ]

def run_pipeline(
    stage,
    quarter=None,
    version="v1",
    description=""
):


    # =====================================
    # QUARTER PROCESSING
    # =====================================
    
    if stage == "quarter":
    
        if quarter is None:
    
            raise ValueError(
                "quarter must be provided"
            )
    
        print(
            f"\nProcessing {quarter}"
        )
    
        orig_file = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / f"historical_data_{quarter}.txt"
        )
    
        perf_file = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / f"historical_data_time_{quarter}.txt"
        )
    
        target_output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"loan_perf_{quarter}.parquet"
        )
    
        perf_feature_output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"performance_features_{quarter}.parquet"
        )
    
        master_output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"master_dataset_{quarter}.parquet"
        )
    
        # ------------------------
        # Target Dataset
        # ------------------------
    
        build_target_dataset(
    
            orig_file,
    
            perf_file,
    
            target_output
    
        )
    
        # ------------------------
        # Performance Features
        # ------------------------
    
        build_performance_features_file(
    
            perf_file,
    
            perf_feature_output
    
        )
    
        # ------------------------
        # Master Dataset
        # ------------------------
    
        build_master_dataset(
    
            orig_file,
    
            target_output,
    
            perf_feature_output,
    
            master_output
    
        )
    
        print(
            f"\nFinished {quarter}"
        )
    
        return
    
    # =====================================
    # COMBINE
    # =====================================
    
    if stage == "combine":
    
        processed_dir = (
            PROJECT_ROOT
            / "data"
            / "processed"
        )
    
        files = sorted(
    
            processed_dir.glob(
                "master_dataset_2018Q*.parquet"
            )
    
        )
    
        if len(files) == 0:
    
            raise ValueError(
                "No quarter datasets found."
            )
    
        print(
            "\nFound:"
        )
    
        for f in files:
    
            print(
                f.name
            )
    
        combine_quarters(
    
            input_dir=processed_dir,
    
            output_file=
            processed_dir
            / "master_dataset_2018.parquet"
    
        )
    
        print(
            "\nCombined dataset created."
        )
    
        return
    
    # =====================================
    # FEATURE STORE
    # =====================================
    
    if stage == "feature_store":

        build_feature_store(
    
            input_path=
            PROJECT_ROOT
            / "data"
            / "processed"
            / "master_dataset_2018.parquet",
    
            output_root=
            PROJECT_ROOT
            / "data"
            / "feature_store",
    
            version=version,
    
            description=description
    
        )
    
        return
    
    # =====================================
    # MODELING DATASETS
    # =====================================
    
    if stage == "modeling":
    
        build_modeling_datasets(

            feature_store_root=
            PROJECT_ROOT
            / "data"
            / "feature_store",
    
            version=version

        )

        return

    # =====================================
    # DIAGNOSTICS
    # =====================================
    
    if stage == "diagnostics":
    
        run_diagnostics(
    
            feature_store_root=
            PROJECT_ROOT
            / "data"
            / "feature_store",
    
            report_root=
            PROJECT_ROOT
            / "reports"
            / "dissertation_results",
    
            version=version
    
        )
    
        return
    
    # =====================================
    # TRAINING
    # =====================================
    
    if stage == "training":
    
        print(
            "\nImplement training stage"
        )
    
        return
    
    raise ValueError(
        f"Unknown stage: {stage}"
    )

