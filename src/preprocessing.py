"""
Preprocessing Utilities

Purpose:
    Transform feature store datasets into
    model-ready datasets.

Responsibilities:
    - Remove leakage
    - Remove metadata
    - Create remaining_term_months
    - Define feature groups
    - Build sklearn preprocessing pipeline
    - Create X and y datasets
"""

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# =====================================================
# TARGET
# =====================================================

TARGET = "stress_flag"


# =====================================================
# REMOVE COLUMNS
# =====================================================

REMOVE_COLUMNS = [

    "loan_identifier",

    "year",
    "quarter",
    "quarter_num",

    "bssi",

    "prepayment_penalty_indicator",
    "amortization_type",
    "property_valuation_method",

    "interest_only_indicator"
]


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_remaining_term_feature(df):

    """
    Convert:

        first_payment_date
        maturity_date

    into:

        remaining_term_months
    """

    df = df.copy()

    fp_year = (
        df["first_payment_date"] // 100
    )

    fp_month = (
        df["first_payment_date"] % 100
    )

    mat_year = (
        df["maturity_date"] // 100
    )

    mat_month = (
        df["maturity_date"] % 100
    )

    df["remaining_term_months"] = (

        (mat_year - fp_year) * 12

        +

        (mat_month - fp_month)

    )

    return df


# =====================================================
# DROP UNUSED FEATURES
# =====================================================

def drop_unused_columns(df):

    """
    Remove metadata,
    leakage,
    and constant columns.
    """

    df = df.copy()

    existing_cols = [

        col

        for col in REMOVE_COLUMNS

        if col in df.columns

    ]

    df = df.drop(
        columns=existing_cols
    )

    return df


# =====================================================
# FINAL FEATURE CLEANUP
# =====================================================

def prepare_features(df):

    """
    Full preprocessing
    before sklearn pipeline.
    """

    df = create_remaining_term_feature(
        df
    )

    df = drop_unused_columns(
        df
    )

    df = df.drop(
        columns=[
            "first_payment_date",
            "maturity_date"
        ]
    )

    return df


# =====================================================
# FEATURE GROUPS
# =====================================================

NUMERICAL_FEATURES = [

    "credit_score",

    "mortgage_insurance_pct",

    "num_units",

    "cltv",

    "dti",

    "original_upb",

    "ltv",

    "interest_rate",

    "original_loan_term",

    "num_borrowers",

    "remaining_term_months"
]


CATEGORICAL_FEATURES = [

    "msa",

    "first_time_homebuyer_indicator",

    "occupancy_status",

    "channel",

    "property_state",

    "property_type",

    "loan_purpose",

    "seller_name",

    "servicer_name",

    "super_conforming_flag",

    "program_indicator",

    "harp_indicator",

    "mortgage_insurance_cancellation_indicator"
]


# =====================================================
# PREPROCESSOR
# =====================================================

def build_preprocessor():

    """
    Create sklearn
    preprocessing pipeline.
    """

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )

        ]

    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]

    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                NUMERICAL_FEATURES
            ),

            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )

        ]

    )

    return preprocessor


# =====================================================
# X AND Y
# =====================================================

def prepare_xy(df):

    """
    Create:

        X
        y
    """

    df = prepare_features(
        df
    )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y
