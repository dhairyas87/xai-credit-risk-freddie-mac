from pathlib import Path

import pandas as pd
import numpy as np

import joblib

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (


mean_absolute_error,

mean_squared_error,

mean_absolute_percentage_error,

r2_score


)

from catboost import (
    CatBoostRegressor,
    Pool
)

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

"quarter"


]

NUMERICAL_FEATURES = [


"credit_score",

"dti",

"num_borrowers",

"num_units",

"ltv",

"cltv",

"mortgage_insurance_pct",

"quarter_num"


]

def evaluate_regression(


y_true,

y_pred


):


    rmse = np.sqrt(
    
        mean_squared_error(
    
            y_true,
    
            y_pred
    
        )
    
    )
    
    mae = mean_absolute_error(
    
        y_true,
    
        y_pred
    
    )
    
    mape = (
    
        mean_absolute_percentage_error(
    
            y_true,
    
            y_pred
    
        )
    
        * 100
    
    )
    
    r2 = r2_score(
    
        y_true,
    
        y_pred
    
    )
    
    return pd.DataFrame({
    
        "Metric": [
    
            "RMSE",
    
            "MAE",
    
            "MAPE",
    
            "R2"
    
        ],
    
        "Value": [
    
            rmse,
    
            mae,
    
            mape,
    
            r2
    
        ]
    
    })


def train_loan_amount(


feature_store_root,

model_root,

report_root,

version="v3"


):


    print("=" * 60)
    print("TRAINING LOAN AMOUNT MODEL")
    print("=" * 60)
    
    root = (
    
        Path(feature_store_root)
    
        / version
    
        / "loan_amount"
    
    )
    
    train_df = pd.read_parquet(
    
        root
    
        / "train.parquet"
    
    )
    
    valid_df = pd.read_parquet(
    
        root
    
        / "valid.parquet"
    
    )
    
    test_df = pd.read_parquet(
    
        root
    
        / "test.parquet"
    
    )
    
    print(
        f"Train: {train_df.shape}"
    )
    
    print(
        f"Valid: {valid_df.shape}"
    )
    
    print(
        f"Test : {test_df.shape}"
    )
    
    X_train = train_df.drop(
    columns=[TARGET]
    )
    
    y_train = train_df[TARGET]
    
    X_test = test_df.drop(
        columns=[TARGET]
    )
    
    y_test = test_df[TARGET]
    
    X_train, X_test = (
    
        clean_categorical_features(
    
            X_train,
    
            X_test,
    
            CATEGORICAL_FEATURES
    
        )
    
    )

    print(
        "\nChecking Data Types..."
    )
    
    for col in CATEGORICAL_FEATURES:
    
        print(
            col,
            X_train[col].dtype
        )
    
    numeric_transformer = Pipeline(
    
        steps=[
    
            (
    
                "imputer",
    
                SimpleImputer(
    
                    strategy="median"
    
                )
    
            )
    
        ]
    
    )
    
    categorical_transformer = Pipeline(

        steps=[
    
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
    
                "num",
    
                numeric_transformer,
    
                NUMERICAL_FEATURES
    
            ),
    
            (
    
                "cat",
    
                categorical_transformer,
    
                CATEGORICAL_FEATURES
    
            )
    
        ]
    
    )
    
    model = Pipeline(
    
        steps=[
    
            (
    
                "preprocessor",
    
                preprocessor
    
            ),
    
            (
    
                "model",
    
                LinearRegression()
    
            )
    
        ]
    
    )
    
    print(
        "\nTraining Linear Regression..."
    )
    
    model.fit(
    
        X_train,
    
        y_train
    
    )
    
    preds = model.predict(
    
        X_test
    
    )
    
    metrics = evaluate_regression(
    
        y_test,
    
        preds
    
    )
    
    print("\nResults")
    print(metrics)
    
    model_dir = (
    
        Path(model_root)
    
        / version
    
        / "loan_amount"
    
    )
    
    model_dir.mkdir(
    
        parents=True,
    
        exist_ok=True
    
    )
    
    report_dir = (
    
        Path(report_root)
    
        / version
    
        / "loan_amount"
    
    )
    
    report_dir.mkdir(
    
        parents=True,
    
        exist_ok=True
    
    )
    
    joblib.dump(
    
        model,
    
        model_dir
    
        / "linear_regression.pkl"
    
    )
    
    metrics.to_csv(
    
        report_dir
    
        / "linear_metrics.csv",
    
        index=False
    
    )

    
    
    prediction_df = pd.DataFrame({
    
        "actual_log":
    
        y_test,
    
        "predicted_log":
    
        preds,
    
        "actual_upb":
    
        np.expm1(y_test),
    
        "predicted_upb":
    
        np.expm1(preds)
    
    })
    
    prediction_df.to_csv(
    
        report_dir
    
        / "predictions.csv",
    
        index=False
    
    )

    train_catboost(

        train_df,
    
        valid_df,
    
        test_df,
    
        report_dir,
    
        model_dir
    
    )
    
    print(
        "\nSaved model and reports."
    )
    
    return model


def clean_categorical_features(

    train_df,

    test_df,

    categorical_features,

    valid_df=None

):

    for col in categorical_features:

        train_df[col] = (
            train_df[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

        test_df[col] = (
            test_df[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

        if valid_df is not None:

            valid_df[col] = (
                valid_df[col]
                .fillna("UNKNOWN")
                .astype(str)
            )

    if valid_df is not None:

        return (

            train_df,

            valid_df,

            test_df

        )

    return (

        train_df,

        test_df

    )

def train_catboost(

train_df,

valid_df,

test_df,

report_dir,

model_dir


):


    print("\n" + "=" * 60)
    print("TRAINING CATBOOST")
    print("=" * 60)
    
    X_train = train_df.drop(
        columns=["target"]
    )
    
    y_train = train_df["target"]
    
    X_valid = valid_df.drop(
        columns=["target"]
    )
    
    y_valid = valid_df["target"]
    
    X_test = test_df.drop(
        columns=["target"]
    )
    
    y_test = test_df["target"]
    
    # ------------------------------------
    # Clean Categoricals
    # ------------------------------------
    
    X_train, X_valid, X_test = (
    
        clean_categorical_features(
    
            X_train,
    
            X_test,
    
            CATEGORICAL_FEATURES,
    
            valid_df=X_valid
    
        )
    
    )
    
    # ------------------------------------
    # Fill Numeric Missing Values
    # ------------------------------------
    
    for col in NUMERICAL_FEATURES:
    
        median_val = (
    
            X_train[col]
            .median()
    
        )
    
        X_train[col] = (
            X_train[col]
            .fillna(median_val)
        )
    
        X_valid[col] = (
            X_valid[col]
            .fillna(median_val)
        )
    
        X_test[col] = (
            X_test[col]
            .fillna(median_val)
        )
    
    # ------------------------------------
    # CatBoost Pools
    # ------------------------------------
    
    cat_indices = [
    
        X_train.columns.get_loc(col)
    
        for col in CATEGORICAL_FEATURES
    
    ]
    
    train_pool = Pool(
    
        X_train,
    
        y_train,
    
        cat_features=cat_indices
    
    )
    
    valid_pool = Pool(
    
        X_valid,
    
        y_valid,
    
        cat_features=cat_indices
    
    )
    
    test_pool = Pool(
    
        X_test,
    
        y_test,
    
        cat_features=cat_indices
    
    )
    
    # ------------------------------------
    # Model
    # ------------------------------------
    
    model = CatBoostRegressor(
    
        iterations=1000,
    
        depth=8,
    
        learning_rate=0.03,
    
        loss_function="RMSE",
    
        eval_metric="RMSE",
    
        random_seed=42,
    
        verbose=100
    
    )
    
    model.fit(
    
        train_pool,
    
        eval_set=valid_pool,
    
        early_stopping_rounds=100,
    
        use_best_model=True
    
    )
    
    preds = model.predict(
        test_pool
    )
    
    metrics = evaluate_regression(
    
        y_test,
    
        preds
    
    )
    
    print("\nCatBoost Results")
    print(metrics)
    
    # ------------------------------------
    # Save Metrics
    # ------------------------------------
    
    metrics.to_csv(
    
        report_dir
        / "catboost_metrics.csv",
    
        index=False
    
    )
    
    # ------------------------------------
    # Save Model
    # ------------------------------------
    
    model.save_model(
    
        str(
            model_dir
            / "catboost.cbm"
        )
    )
    
    # ------------------------------------
    # Save Predictions
    # ------------------------------------
    
    prediction_df = pd.DataFrame({
    
        "actual_log":
    
        y_test,
    
        "predicted_log":
    
        preds,
    
        "actual_upb":
    
        np.expm1(y_test),
    
        "predicted_upb":
    
        np.expm1(preds)
    
    })
    
    prediction_df.to_csv(
    
        report_dir
        / "catboost_predictions.csv",
    
        index=False
    
    )
    
    # ------------------------------------
    # Feature Importance
    # ------------------------------------
    
    importance = pd.DataFrame({
    
        "feature":
    
        X_train.columns,
    
        "importance":
    
        model.get_feature_importance()
    
    })
    
    importance = (
    
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    
    )
    
    importance.to_csv(
    
        report_dir
        / "feature_importance.csv",
    
        index=False
    
    )
    
    print(
        "\nTop Features"
    )
    
    print(
        importance.head(15)
    )
    
    return model

