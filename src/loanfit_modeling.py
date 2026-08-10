"""Stable preprocessing helpers for LoanFit model artifacts.

The LoanFit API loads model files across local Python environments. To keep
those artifacts less fragile, this module avoids pickling scikit-learn's
private ColumnTransformer internals and performs the small amount of tabular
preprocessing with plain pandas/numpy instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("string").fillna("UNKNOWN")
    for col in NUMERICAL_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


@dataclass
class LoanFitPreprocessor:
    mode: str
    min_category_count: int = 20
    numeric_medians: dict[str, float] = field(default_factory=dict)
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "LoanFitPreprocessor":
        frame = prepare_frame(df)
        numeric = frame[NUMERICAL_FEATURES].copy()

        self.numeric_medians = numeric.median().fillna(0).to_dict()
        numeric = numeric.fillna(self.numeric_medians)

        self.numeric_means = numeric.mean().to_dict()
        stds = numeric.std(ddof=0).replace(0, 1).fillna(1)
        self.numeric_stds = stds.to_dict()

        self.categories = {}
        for col in CATEGORICAL_FEATURES:
            counts = frame[col].value_counts(dropna=False)
            if self.mode == "linear":
                selected = counts[counts >= self.min_category_count].index.astype(str).tolist()
            else:
                selected = counts.index.astype(str).tolist()
            self.categories[col] = sorted(selected)

        if self.mode == "linear":
            self.feature_names = list(NUMERICAL_FEATURES)
            for col in CATEGORICAL_FEATURES:
                self.feature_names.extend([f"{col}__{value}" for value in self.categories[col]])
        elif self.mode == "tree":
            self.feature_names = ALL_FEATURES
        else:
            raise ValueError("mode must be either 'linear' or 'tree'")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = prepare_frame(df)
        numeric = frame[NUMERICAL_FEATURES].fillna(self.numeric_medians)

        if self.mode == "linear":
            numeric = (numeric - pd.Series(self.numeric_means)) / pd.Series(self.numeric_stds)
            columns = {col: numeric[col].to_numpy(dtype=float) for col in NUMERICAL_FEATURES}
            for col in CATEGORICAL_FEATURES:
                values = frame[col].astype(str)
                for category in self.categories[col]:
                    columns[f"{col}__{category}"] = (values == category).to_numpy(dtype=float)
            output = pd.DataFrame(columns, index=frame.index)
            return output.reindex(columns=self.feature_names, fill_value=0).astype(float)

        output = numeric.copy()
        for col in CATEGORICAL_FEATURES:
            mapping = {value: index for index, value in enumerate(self.categories[col])}
            output[col] = frame[col].astype(str).map(mapping).fillna(-1).astype(float)
        return output.reindex(columns=self.feature_names, fill_value=-1).astype(float)


@dataclass
class LoanFitModelBundle:
    preprocessor: LoanFitPreprocessor
    estimator: object
    background_features: pd.DataFrame | None = None

    def predict(self, df: pd.DataFrame):
        features = self.preprocessor.transform(df)
        return self.estimator.predict(features)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(df)


def make_bundle(mode: str, estimator: object, train_df: pd.DataFrame) -> LoanFitModelBundle:
    preprocessor = LoanFitPreprocessor(mode=mode).fit(train_df)
    features = preprocessor.transform(train_df)
    estimator.fit(features, train_df["target"])
    background_size = min(100, len(features))
    background_features = features.sample(background_size, random_state=42)
    return LoanFitModelBundle(
        preprocessor=preprocessor,
        estimator=estimator,
        background_features=background_features,
    )
