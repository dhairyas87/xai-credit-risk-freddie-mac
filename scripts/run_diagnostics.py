from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import (
mutual_info_classif
)

# =====================================================

# HELPERS

# =====================================================

def save_dataset_summary(
df,
output_dir
):


    summary = pd.DataFrame({
    
        "Metric": [
    
            "Rows",
            "Columns",
            "Numeric Columns",
            "Categorical Columns",
            "Missing Cells"
    
        ],
    
        "Value": [
    
            len(df),
    
            len(df.columns),
    
            len(
                df.select_dtypes(
                    include=np.number
                ).columns
            ),
    
            len(
                df.select_dtypes(
                    exclude=np.number
                ).columns
            ),
    
            int(
                df.isna()
                .sum()
                .sum()
            )
    
        ]
    
    })
    
    summary.to_csv(
    
        output_dir
        / "dataset_summary.csv",
    
        index=False
    
    )


def save_missing_values(
df,
output_dir
):


    missing = pd.DataFrame({
    
        "column":
        df.columns,
    
        "missing_count":
        df.isna().sum(),
    
        "missing_pct":
        (
            df.isna().mean() * 100
        )
    
    })
    
    missing = (
        missing
        .sort_values(
            "missing_pct",
            ascending=False
        )
    )
    
    missing.to_csv(
    
        output_dir
        / "missing_values.csv",
    
        index=False
    
    )

def save_numeric_summary(
df,
output_dir
):


    numeric = (
        df.select_dtypes(
            include=np.number
        )
        .describe()
        .T
    )
    
    numeric.to_csv(
    
        output_dir
        / "numeric_summary.csv"
    
    )


def save_categorical_cardinality(
df,
output_dir
):


    cat_cols = (
        df.select_dtypes(
            exclude=np.number
        )
        .columns
    )
    
    rows = []
    
    for col in cat_cols:
    
        rows.append({
    
            "column":
            col,
    
            "unique_values":
            df[col].nunique()
    
        })
    
    pd.DataFrame(
        rows
    ).to_csv(
    
        output_dir
        / "categorical_cardinality.csv",
    
        index=False
    
    )


def save_target_distribution(
df,
target,
output_dir
):


    dist = (
        df[target]
        .value_counts()
        .reset_index()
    )
    
    dist.columns = [
        target,
        "count"
    ]
    
    dist["pct"] = (
        dist["count"]
        / dist["count"].sum()
    )
    
    dist.to_csv(
    
        output_dir
        / "target_distribution.csv",
    
        index=False
    
    )
    
    plt.figure(
        figsize=(8, 5)
    )
    
    dist.plot(
    
        x=target,
    
        y="count",
    
        kind="bar"
    
    )
    
    plt.title(
        f"{target} Distribution"
    )
    
    plt.tight_layout()
    
    plt.savefig(
    
        output_dir
        / "target_distribution.png"
    
    )
    
    plt.close()


def save_correlation(
df,
output_dir
):


    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
    )
    
    corr = (
        numeric_df
        .corr()
    )
    
    corr.to_csv(
    
        output_dir
        / "correlation_matrix.csv"
    
    )
    
    plt.figure(
        figsize=(14, 10)
    )
    
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0
    )
    
    plt.title(
        "Correlation Matrix"
    )
    
    plt.tight_layout()
    
    plt.savefig(
    
        output_dir
        / "correlation_heatmap.png"
    
    )
    
    plt.close()


def save_outliers(
df,
output_dir
):


    numeric_cols = (
        df.select_dtypes(
            include=np.number
        )
        .columns
    )
    
    rows = []
    
    for col in numeric_cols:
    
        q1 = (
            df[col]
            .quantile(0.25)
        )
    
        q3 = (
            df[col]
            .quantile(0.75)
        )
    
        iqr = q3 - q1
    
        lower = (
            q1
            - 1.5 * iqr
        )
    
        upper = (
            q3
            + 1.5 * iqr
        )
    
        outliers = (
    
            (
                df[col]
                < lower
            )
    
            |
    
            (
                df[col]
                > upper
            )
    
        ).sum()
    
        rows.append({
    
            "column": col,
    
            "outlier_count":
            int(outliers),
    
            "outlier_pct":
            round(
                outliers
                / len(df)
                * 100,
                2
            )
    
        })
    
    pd.DataFrame(
        rows
    ).to_csv(
    
        output_dir
        / "outlier_summary.csv",
    
        index=False
    
    )


def save_mutual_information(
    df,
    target,
    output_dir
):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
        .copy()
    )

    if target not in numeric_df.columns:
        return

    X = numeric_df.drop(
        columns=[target],
        errors="ignore"
    )

    y = numeric_df[target]

    # =====================================
    # Remove rows with missing target
    # =====================================

    valid_idx = y.notna()

    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    print(
        f"{target}: removed "
        f"{(~valid_idx).sum()} rows "
        f"with missing target"
    )

    # =====================================
    # Fill missing feature values
    # =====================================

    X = X.fillna(
        X.median()
    )

    # =====================================
    # Ensure classification labels
    # =====================================

    y = y.astype(int)

    # =====================================
    # Mutual Information
    # =====================================

    scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi = pd.DataFrame({

        "feature": X.columns,

        "score": scores

    })

    mi = (
        mi
        .sort_values(
            "score",
            ascending=False
        )
    )

    mi.to_csv(

        output_dir
        / "feature_relevance.csv",

        index=False

    )

def save_leakage_check(
columns,
objective,
output_dir
):


    leakage_features = [
    
        "max_delinquency",
        "avg_delinquency",
        "months_delinquent",
    
        "modification_count",
    
        "assistance_count",
    
        "delinquency_intensity"
    
    ]
    
    rows = []
    
    for col in leakage_features:
    
        rows.append({
    
            "feature":
            col,
    
            "present":
            col in columns
    
        })
    
    pd.DataFrame(
        rows
    ).to_csv(
    
        output_dir
        / "leakage_check.csv",
    
        index=False
    
    )


# =====================================================

# MAIN

# =====================================================

def run_diagnostics(


feature_store_root,

report_root,

version="v1"


):


    objectives = {
    
        "stress_prediction":
        "stress_flag",
    
        "stress_severity":
        "stress_level",
    
        "serviceability":
        "bss_level"
    
    }
    
    for objective, target in (
        objectives.items()
    ):
    
        print(
            f"\nDiagnostics: {objective}"
        )
    
        data_path = (
    
            Path(feature_store_root)
    
            / version
    
            / objective
    
            / "train.parquet"
    
        )
    
        df = pd.read_parquet(
            data_path
        )
    
        output_dir = (
    
            Path(report_root)
    
            / version
    
            / "diagnostics"
    
            / objective
    
        )
    
        output_dir.mkdir(
    
            parents=True,
    
            exist_ok=True
    
        )
    
        save_dataset_summary(
            df,
            output_dir
        )
    
        save_missing_values(
            df,
            output_dir
        )
    
        save_numeric_summary(
            df,
            output_dir
        )
    
        save_categorical_cardinality(
            df,
            output_dir
        )
    
        save_target_distribution(
            df,
            target,
            output_dir
        )
    
        save_correlation(
            df,
            output_dir
        )
    
        save_outliers(
            df,
            output_dir
        )
    
        save_mutual_information(
            df,
            target,
            output_dir
        )
    
        save_leakage_check(
            df.columns,
            objective,
            output_dir
        )
    
        print(
            f"Saved diagnostics to {output_dir}"
        )

