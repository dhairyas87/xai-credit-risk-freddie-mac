from pathlib import Path
from datetime import datetime
import pandas as pd
import json

from src.serviceability import (
calculate_serviceability_score
)

# =====================================================

# CONFIG

# =====================================================

IDENTIFIER_COLUMNS = [
"loan_identifier"
]

# -----------------------------------------------------

# Research Objective 1

# Stress Prediction

# -----------------------------------------------------

STRESS_PREDICTION_DROP = [


"stress_level",
"bssi",

"bss",
"bss_bucket",
"bss_level",

# Future leakage

"max_delinquency",
"avg_delinquency",
"months_delinquent",

"ever_modified",
"ever_assistance",

"modification_count",
"assistance_count",

"delinquency_intensity",
"modification_intensity",
"assistance_intensity",

"ever_ra",
"ever_zero_balance"


]

# -----------------------------------------------------

# Research Objective 2

# Stress Severity

# -----------------------------------------------------

STRESS_SEVERITY_DROP = [


"stress_flag",

"bssi",

"bss",
"bss_bucket",
"bss_level"


]

# -----------------------------------------------------

# Research Objective 3

# Serviceability

# -----------------------------------------------------

SERVICEABILITY_DROP = [


"stress_flag",
"stress_level",
"bssi",

"bss",
"bss_bucket"


]

# -----------------------------------------------------

# Explainability Dataset

# -----------------------------------------------------

EXPLAINABILITY_DROP = []

# =====================================================

# SAVE DATASET

# =====================================================

def save_feature_store(
df,
output_dir
):


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    output_file = (
        output_dir
        / "feature_store.parquet"
    )
    
    df.to_parquet(
        output_file,
        index=False
    )
    
    print(
        f"Saved: {output_file}"
    )
    
    print(
        f"Shape: {df.shape}"
    )


# =====================================================

# BUILD FEATURE STORE

# =====================================================

def build_feature_store(


input_path,

output_root,

version="v1",

description=""

):


    print("=" * 60)
    print("BUILDING FEATURE STORE")
    print("=" * 60)
    
    df = pd.read_parquet(
        input_path
    )
    
    print(
        f"Input Shape: {df.shape}"
    )
    
    # ==========================================
    # Serviceability Features
    # ==========================================
    
    try:
    
        df = calculate_serviceability_score(
            df
        )
    
        print(
            "Serviceability score calculated."
        )
    
    except Exception as e:
    
        print(
            f"Serviceability score skipped: {e}"
        )
    
    # ==========================================
    # Version Folder
    # ==========================================
    
    feature_root = (
        Path(output_root)
        / version
    )
    
    feature_root.mkdir(
        parents=True,
        exist_ok=True
    )
    
    # ==========================================
    # Stress Prediction
    # ==========================================
    
    stress_prediction_df = df.drop(
    
        columns=
        IDENTIFIER_COLUMNS
        + STRESS_PREDICTION_DROP,
    
        errors="ignore"
    
    )
    
    save_feature_store(
    
        stress_prediction_df,
    
        feature_root
        / "stress_prediction"
    
    )
    
    # ==========================================
    # Stress Severity
    # ==========================================
    
    stress_severity_df = df.drop(
    
        columns=
        IDENTIFIER_COLUMNS
        + STRESS_SEVERITY_DROP,
    
        errors="ignore"
    
    )
    
    save_feature_store(
    
        stress_severity_df,
    
        feature_root
        / "stress_severity"
    
    )
    
    # ==========================================
    # Serviceability
    # ==========================================
    
    serviceability_df = df.drop(
    
        columns=
        IDENTIFIER_COLUMNS
        + SERVICEABILITY_DROP,
    
        errors="ignore"
    
    )
    
    save_feature_store(
    
        serviceability_df,
    
        feature_root
        / "serviceability"
    
    )
    
    # ==========================================
    # Explainability
    # ==========================================
    
    explainability_df = df.drop(
    
        columns=
        IDENTIFIER_COLUMNS
        + EXPLAINABILITY_DROP,
    
        errors="ignore"
    
    )
    
    save_feature_store(
    
        explainability_df,
    
        feature_root
        / "explainability"
    
    )
    
    # ==========================================
    # Metadata
    # ==========================================
    
    metadata = {
    
        "version": version,
    
        "created_on":
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    
        "source_dataset":
        Path(input_path).name,
    
        "rows":
        int(df.shape[0]),
    
        "columns":
        int(df.shape[1]),
    
        "description":
        description
    
    }
    
    with open(
    
        feature_root
        / "metadata.json",
    
        "w"
    
    ) as f:
    
        json.dump(
            metadata,
            f,
            indent=4
        )
    
    # ==========================================
    # Version Manifest
    # ==========================================
    
    manifest_path = (
    
        Path(output_root)
    
        / "version_manifest.json"
    
    )
    
    if manifest_path.exists():
    
        with open(
            manifest_path,
            "r"
        ) as f:
    
            manifest = json.load(f)
    
    else:
    
        manifest = {
    
            "latest_version": version,
    
            "versions": {}
    
        }
    
    manifest[
        "latest_version"
    ] = version
    
    manifest[
        "versions"
    ][version] = metadata
    
    with open(
        manifest_path,
        "w"
    ) as f:
    
        json.dump(
            manifest,
            f,
            indent=4
        )
    
    print(
        f"\nFeature Store Version Created: {version}"
    )
    
    return df

