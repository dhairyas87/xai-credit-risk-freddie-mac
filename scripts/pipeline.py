"""End-to-end runnable project pipeline.

Examples:
    python scripts/pipeline.py all
    python scripts/pipeline.py all --skip-raw
    python scripts/pipeline.py original-models
    python scripts/pipeline.py lending-models --sample-rows 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QUARTERS = ["2018Q1", "2018Q2", "2018Q3", "2018Q4"]


def _paths() -> dict[str, Path]:
    return {
        "raw": PROJECT_ROOT / "data" / "raw",
        "processed": PROJECT_ROOT / "data" / "processed",
        "feature_store": PROJECT_ROOT / "data" / "feature_store",
        "models": PROJECT_ROOT / "models",
        "reports": PROJECT_ROOT / "reports" / "dissertation_results",
    }


def process_quarter(quarter: str) -> None:
    from scripts.build_master_dataset import build_master_dataset
    from scripts.build_performance_features import build_performance_features_file
    from scripts.build_target_dataset import build_target_dataset

    paths = _paths()
    print(f"\nProcessing raw quarter: {quarter}")

    orig_file = paths["raw"] / f"historical_data_{quarter}.txt"
    perf_file = paths["raw"] / f"historical_data_time_{quarter}.txt"

    missing = [str(path) for path in [orig_file, perf_file] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raw file(s): " + ", ".join(missing))

    target_output = paths["processed"] / f"loan_perf_{quarter}.parquet"
    perf_feature_output = paths["processed"] / f"performance_features_{quarter}.parquet"
    master_output = paths["processed"] / f"master_dataset_{quarter}.parquet"

    build_target_dataset(orig_file, perf_file, target_output)
    build_performance_features_file(perf_file, perf_feature_output)
    build_master_dataset(orig_file, target_output, perf_feature_output, master_output)
    print(f"Finished quarter: {quarter}")


def combine_processed_quarters() -> None:
    from scripts.combine_quarters import combine_quarters

    paths = _paths()
    files = sorted(paths["processed"].glob("master_dataset_2018Q*.parquet"))
    if not files:
        raise FileNotFoundError(
            "No processed quarter datasets found in data/processed. "
            "Run `python scripts/pipeline.py raw` first."
        )

    print("\nCombining processed quarterly master datasets:")
    for file in files:
        print(f"- {file.name}")

    combine_quarters(
        input_dir=paths["processed"],
        output_file=paths["processed"] / "master_dataset_2018.parquet",
    )


def build_credit_risk_datasets(version: str, description: str) -> None:
    from scripts.build_feature_store import build_feature_store
    from scripts.build_modeling_datasets import build_modeling_datasets

    paths = _paths()
    master_path = paths["processed"] / "master_dataset_2018.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing combined master dataset: {master_path}")

    build_feature_store(
        input_path=master_path,
        output_root=paths["feature_store"],
        version=version,
        description=description,
    )
    build_modeling_datasets(
        feature_store_root=paths["feature_store"],
        version=version,
    )


def build_lending_datasets(version: str) -> None:
    from scripts.build_lending_modeling_datasets import build_lending_modeling_datasets
    from scripts.build_lending_recommendation_feature_store import build_lending_recommendation_feature_store

    paths = _paths()
    master_path = paths["processed"] / "master_dataset_2018.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing combined master dataset: {master_path}")

    build_lending_recommendation_feature_store(
        master_dataset_path=master_path,
        output_root=paths["feature_store"],
        version=version,
    )
    build_lending_modeling_datasets(
        feature_store_root=paths["feature_store"],
        version=version,
    )


def train_ui_models(data_version: str, model_version: str, sample_rows: int | None) -> None:
    from scripts.train_lending_models import train_lending_models

    paths = _paths()
    train_lending_models(
        feature_store_root=paths["feature_store"],
        model_root=paths["models"],
        report_root=paths["reports"],
        data_version=data_version,
        model_version=model_version,
        sample_rows=sample_rows,
    )


def train_original_models(data_version: str = "v3") -> None:
    from scripts.train_interest_rate import train_interest_rate
    from scripts.train_loan_amount import train_loan_amount
    from scripts.train_loan_term import train_loan_term
    from scripts.train_stacked_loan_amount import train_stacked_ensemble

    if data_version != "v3":
        print("Original v4 trainers use the v3 lending feature store.")

    paths = _paths()
    train_loan_amount(
        feature_store_root=paths["feature_store"],
        model_root=paths["models"],
        report_root=paths["reports"],
        version="v4",
    )
    train_stacked_ensemble(
        feature_store_root=paths["feature_store"],
        model_root=paths["models"],
        version="v4",
    )
    train_loan_term(
        feature_store_root=paths["feature_store"],
        model_root=paths["models"],
        report_root=paths["reports"],
        version="v4",
    )
    train_interest_rate(
        feature_store_root=paths["feature_store"],
        model_root=paths["models"],
        report_root=paths["reports"],
        version="v4",
    )


def run_pipeline(
    stage: str,
    quarter: str | None = None,
    risk_version: str = "v2",
    lending_data_version: str = "v3",
    model_version: str = "v5",
    description: str = "End-to-end generated feature store",
    skip_raw: bool = False,
    skip_original_models: bool = False,
    sample_rows: int | None = None,
) -> None:
    if stage == "quarter":
        if quarter is None:
            raise ValueError("quarter must be provided when stage is 'quarter'")
        process_quarter(quarter)
        return

    if stage == "raw":
        for selected_quarter in QUARTERS:
            process_quarter(selected_quarter)
        return

    if stage == "combine":
        combine_processed_quarters()
        return

    if stage == "risk-datasets":
        build_credit_risk_datasets(risk_version, description)
        return

    if stage == "diagnostics":
        from scripts.run_diagnostics import run_diagnostics

        run_diagnostics(
            feature_store_root=_paths()["feature_store"],
            report_root=_paths()["reports"],
            version=risk_version,
        )
        return

    if stage == "lending-datasets":
        build_lending_datasets(lending_data_version)
        return

    if stage == "lending-models":
        train_ui_models(lending_data_version, model_version, sample_rows)
        return

    if stage == "original-models":
        train_original_models(lending_data_version)
        return

    if stage == "all":
        if not skip_raw:
            for selected_quarter in QUARTERS:
                process_quarter(selected_quarter)
        combine_processed_quarters()
        build_credit_risk_datasets(risk_version, description)
        build_lending_datasets(lending_data_version)
        if not skip_original_models:
            train_original_models(lending_data_version)
        train_ui_models(lending_data_version, model_version, sample_rows)
        print("\nFull pipeline complete.")
        return

    raise ValueError(f"Unknown stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Freddie Mac credit-risk and LoanFit pipeline.")
    parser.add_argument(
        "stage",
        choices=[
            "all",
            "raw",
            "quarter",
            "combine",
            "risk-datasets",
            "diagnostics",
            "lending-datasets",
            "lending-models",
            "original-models",
        ],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--quarter", choices=QUARTERS, help="Quarter to process for the 'quarter' stage.")
    parser.add_argument("--risk-version", default="v2", help="Output version for credit-risk feature stores.")
    parser.add_argument("--lending-data-version", default="v3", help="Output version for lending UI feature stores.")
    parser.add_argument("--model-version", default="v5", help="Output version for UI lending models.")
    parser.add_argument("--description", default="End-to-end generated feature store")
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Use existing processed master data and skip raw text parsing.",
    )
    parser.add_argument(
        "--skip-original-models",
        action="store_true",
        help="Skip CatBoost/XGBoost v4 research model training during the 'all' stage.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional cap on training rows per target for faster smoke-test runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        stage=args.stage,
        quarter=args.quarter,
        risk_version=args.risk_version,
        lending_data_version=args.lending_data_version,
        model_version=args.model_version,
        description=args.description,
        skip_raw=args.skip_raw,
        skip_original_models=args.skip_original_models,
        sample_rows=args.sample_rows,
    )
