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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


QUARTERS = [
"2018Q1",
"2018Q2",
"2018Q3",
"2018Q4"
]

def run_pipeline(
quarter="all"
):

    quarters = (
        QUARTERS
        if quarter == "all"
        else [quarter]
    )
    
    for q in quarters:
    
        print(f"\nProcessing {q}")
    
        

        orig_file = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / f"historical_data_{q}.txt"
        )
        
        perf_file = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / f"historical_data_time_{q}.txt"
        )
    
        target_output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"loan_perf_{q}.parquet"
        )
        
        master_output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"master_dataset_{q}.parquet"
        )
        
        perf_feature_output = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"performance_features_{q}.parquet"
        )
    
        build_target_dataset(
            orig_file,
            perf_file,
            target_output
        )
        build_performance_features_file(
            perf_file,
            perf_feature_output
        )
    
        build_master_dataset(
            orig_file,
            target_output,
            perf_feature_output,
            master_output
        )
    
        
    
        print(
            "\nQuarter processing complete."
        )


    # =================================================
    # COMBINED DATASET
    # =================================================

    if "combine_quarters" in stages_to_run:

        print(
            "\nCombining all quarters..."
        )

        # combine_quarters()

    # =================================================
    # FEATURE STORE
    # =================================================

    if "feature_store" in stages_to_run:

        print(
            "\nBuilding feature store..."
        )

        # build_feature_store()

    # =================================================
    # SERVICEABILITY
    # =================================================

    if "serviceability" in stages_to_run:

        print(
            "\nBuilding serviceability features..."
        )

    # =================================================
    # MODEL TRAINING
    # =================================================

    if "model_training" in stages_to_run:

        print(
            "\nTraining models..."
        )

    # =================================================
    # EVALUATION
    # =================================================

    if "evaluation" in stages_to_run:

        print(
            "\nEvaluating models..."
        )

    # =================================================
    # EXPLAINABILITY
    # =================================================

    if "explainability" in stages_to_run:

        print(
            "\nRunning explainability..."
        )

    print(
        "\nPipeline Complete."
    )