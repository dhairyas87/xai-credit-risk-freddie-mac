"""Train the loan-term estimator used by the web application."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET = "target"

CATEGORICAL_FEATURES = [
    "first_time_homebuyer_indicator",
    "occupancy_status",
    "property_type",
    "property_state",
    "msa",
    "channel",
    "loan_purpose",
    "program_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mortgage_insurance_cancellation_indicator",
    "quarter",
]

NUMERICAL_FEATURES = [
    "credit_score",
    "dti",
    "num_borrowers",
    "num_units",
    "ltv",
    "cltv",
    "mortgage_insurance_pct",
    "quarter_num",
]


def build_pipeline() -> Pipeline:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERICAL_FEATURES),
                        ("categorical", categorical, CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.08,
                    max_iter=180,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def train_loan_term(
    feature_store_root: Path,
    model_root: Path,
    report_root: Path,
    version: str = "v3",
) -> Path:
    data_root = feature_store_root / version / "loan_term"
    train_df = pd.read_parquet(data_root / "train.parquet")
    test_df = pd.read_parquet(data_root / "test.parquet")

    model = build_pipeline()
    model.fit(train_df.drop(columns=TARGET), train_df[TARGET])

    predictions = model.predict(test_df.drop(columns=TARGET))
    actual = test_df[TARGET]
    metrics = pd.DataFrame(
        {
            "Metric": ["RMSE_months", "MAE_months", "R2"],
            "Value": [
                np.sqrt(mean_squared_error(actual, predictions)),
                mean_absolute_error(actual, predictions),
                r2_score(actual, predictions),
            ],
        }
    )

    model_dir = model_root / version / "loan_term"
    report_dir = report_root / version / "loan_term"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "hist_gradient_boosting.pkl"
    joblib.dump(model, model_path)
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Saved model: {model_path}")
    return model_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    train_loan_term(
        feature_store_root=project_root / "data" / "feature_store",
        model_root=project_root / "models",
        report_root=project_root / "reports" / "dissertation_results",
    )
