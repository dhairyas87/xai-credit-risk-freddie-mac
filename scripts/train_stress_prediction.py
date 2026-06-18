import pandas as pd
import numpy as np

from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
OneHotEncoder,
StandardScaler
)

from sklearn.linear_model import (
LogisticRegression
)


from pandas.api.types import is_numeric_dtype
from catboost import CatBoostClassifier

from sklearn.metrics import (


accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
classification_report


)

# =====================================================

# LOAD DATA

# =====================================================

def load_data(feature_store_root, version):


    root = (
        Path(feature_store_root)
        / version
        / "stress_prediction"
    )
    
    train_df = pd.read_parquet(
        root / "train.parquet"
    )
    
    valid_df = pd.read_parquet(
        root / "valid.parquet"
    )
    
    test_df = pd.read_parquet(
        root / "test.parquet"
    )
    
    return (
        train_df,
        valid_df,
        test_df
    )


# =====================================================

# PREPARE DATA

# =====================================================

def prepare_data(df):


    df = df.copy()
    
    target = "stress_flag"
    
    DROP_COLUMNS = [
    
        "quarter",
        "year",
        "quarter_num"
    
    ]
    
    X = df.drop(
    
        columns=
        [target]
        + DROP_COLUMNS,
    
        errors="ignore"
    
    )
    
    y = df[target]
    
    return X, y


# =====================================================

# LOGISTIC REGRESSION

# =====================================================

def train_logistic(


train_df,

valid_df,

test_df,

model_dir,

report_dir


):


    print(
        "\nTraining Logistic Regression..."
    )
    
    X_train, y_train = (
        prepare_data(train_df)
    )
    
    X_valid, y_valid = (
        prepare_data(valid_df)
    )
    
    X_test, y_test = (
        prepare_data(test_df)
    )
    
    numeric_cols = (
    
        X_train
    
        .select_dtypes(
            include=np.number
        )
    
        .columns
    
    )
    
    categorical_cols = (
    
        X_train
    
        .select_dtypes(
            exclude=np.number
        )
    
        .columns
    
    )
    
    numeric_pipeline = Pipeline([
    
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
    
    ])
    
    categorical_pipeline = Pipeline([
    
        (
    
            "imputer",
    
            SimpleImputer(
    
                strategy="constant",
    
                fill_value="UNKNOWN"
    
            )
    
        ),
    
        (
    
            "encoder",
    
            OneHotEncoder(
    
                handle_unknown="ignore"
    
            )
    
        )
    
    ])
    
    preprocessor = ColumnTransformer([
    
        (
    
            "num",
    
            numeric_pipeline,
    
            numeric_cols
    
        ),
    
        (
    
            "cat",
    
            categorical_pipeline,
    
            categorical_cols
    
        )
    
    ])
    
    model = Pipeline([
    
        (
    
            "preprocessor",
    
            preprocessor
    
        ),
    
        (
    
            "classifier",
    
            LogisticRegression(
    
                max_iter=1000,
    
                class_weight="balanced",
    
                random_state=42
    
            )
    
        )
    
    ])
    
    model.fit(
        X_train,
        y_train
    )
    
    preds = model.predict(
        X_test
    )
    
    probs = model.predict_proba(
        X_test
    )[:, 1]
    
    metrics = {
    
        "Accuracy":
        accuracy_score(
            y_test,
            preds
        ),
    
        "Precision":
        precision_score(
            y_test,
            preds
        ),
    
        "Recall":
        recall_score(
            y_test,
            preds
        ),
    
        "F1":
        f1_score(
            y_test,
            preds
        ),
    
        "ROC_AUC":
        roc_auc_score(
            y_test,
            probs
        )
    
    }
    
    metrics_df = pd.DataFrame(
    
        metrics.items(),
    
        columns=[
            "Metric",
            "Value"
        ]
    
    )
    
    metrics_df.to_csv(
    
        report_dir
        / "logistic_metrics.csv",
    
        index=False
    
    )
    
    with open(
    
        report_dir
        / "logistic_classification_report.txt",
    
        "w"
    
    ) as f:
    
        f.write(
    
            classification_report(
    
                y_test,
    
                preds
    
            )
    
        )
    
    joblib.dump(
    
        model,
    
        model_dir
        / "logistic.pkl"
    
    )
    
    print(metrics_df)
    
    return metrics_df
    

# =====================================================

# MAIN

# =====================================================

def train_stress_prediction(


feature_store_root,

model_root,

report_root,

version="v2"


):


    train_df, valid_df, test_df = (
    
        load_data(
    
            feature_store_root,
    
            version
    
        )
    
    )
    
    model_dir = (
    
        Path(model_root)
    
        / version
    
        / "stress_prediction"
    
    )
    
    report_dir = (
    
        Path(report_root)
    
        / version
    
        / "stress_prediction"
    
    )
    
    model_dir.mkdir(
    
        parents=True,
    
        exist_ok=True
    
    )
    
    report_dir.mkdir(
    
        parents=True,
    
        exist_ok=True
    
    )
    
    train_logistic(
    
        train_df,
    
        valid_df,
    
        test_df,
    
        model_dir,
    
        report_dir
    
    )

    train_catboost(

        train_df,
        valid_df,
        test_df,
        model_dir,
        report_dir
    
    )




def train_catboost(

train_df,
valid_df,
test_df,
model_dir,
report_dir

):
    print("\n" + "=" * 60)
    print("TRAINING CATBOOST")
    print("=" * 60)
    
    X_train, y_train = prepare_data(
        train_df
    )
    
    X_valid, y_valid = prepare_data(
        valid_df
    )
    
    X_test, y_test = prepare_data(
        test_df
    )
    
    # ==================================
    # Remove rows with missing target
    # ==================================
    
    valid_rows = y_train.notna()
    
    X_train = X_train.loc[
        valid_rows
    ]
    
    y_train = y_train.loc[
        valid_rows
    ]
    
    # ==================================
    # Handle Missing Values
    # ==================================
    
    for col in X_train.columns:
    
        if is_numeric_dtype(
            X_train[col]
        ):
    
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
    
        else:
    
            X_train[col] = (
                X_train[col]
                .fillna("UNKNOWN")
                .astype(str)
            )
    
            X_valid[col] = (
                X_valid[col]
                .fillna("UNKNOWN")
                .astype(str)
            )
    
            X_test[col] = (
                X_test[col]
                .fillna("UNKNOWN")
                .astype(str)
            )
    
    # ==================================
    # Identify Categorical Columns
    # ==================================
    
    categorical_features = [
    
        idx
    
        for idx, col in enumerate(
            X_train.columns
        )
    
        if not is_numeric_dtype(
            X_train[col]
        )
    
    ]
    
    print(
        f"Rows Train: {len(X_train):,}"
    )
    
    print(
        f"Rows Valid: {len(X_valid):,}"
    )
    
    print(
        f"Rows Test : {len(X_test):,}"
    )
    
    print(
        f"Categorical Features: "
        f"{len(categorical_features)}"
    )
    
    # ==================================
    # Class Weight
    # ==================================
    
    class_weights = [
    
        1.0,
    
        (
            len(y_train)
            /
            (
                2
                *
                y_train.sum()
            )
        )
    
    ]
    
    print(
        f"Class Weights: "
        f"{class_weights}"
    )
    
    # ==================================
    # Model
    # ==================================
    
    model = CatBoostClassifier(
    
        iterations=500,
    
        depth=10,
    
        learning_rate=0.05,
    
        loss_function="Logloss",
    
        eval_metric="AUC",
    
        random_seed=42,
    
        class_weights=class_weights,
    
        verbose=50
    
    )
    
    model.fit(
    
        X_train,
    
        y_train,
    
        cat_features=
        categorical_features,
    
        eval_set=(
            X_valid,
            y_valid
        ),
    
        use_best_model=True,
    
        early_stopping_rounds=30
    
    )
    
    # ==================================
    # Predictions
    # ==================================
    
    preds = model.predict(
        X_test
    )
    
    probs = model.predict_proba(
        X_test
    )[:, 1]
    
    metrics = {
    
        "Accuracy":
        accuracy_score(
            y_test,
            preds
        ),
    
        "Precision":
        precision_score(
            y_test,
            preds
        ),
    
        "Recall":
        recall_score(
            y_test,
            preds
        ),
    
        "F1":
        f1_score(
            y_test,
            preds
        ),
    
        "ROC_AUC":
        roc_auc_score(
            y_test,
            probs
        )
    
    }
    
    metrics_df = pd.DataFrame(
    
        metrics.items(),
    
        columns=[
            "Metric",
            "Value"
        ]
    
    )
    
    metrics_df.to_csv(
    
        report_dir
        / "catboost_metrics.csv",
    
        index=False
    
    )
    
    with open(
    
        report_dir
        / "catboost_classification_report.txt",
    
        "w"
    
    ) as f:
    
        f.write(
    
            classification_report(
                y_test,
                preds
            )
    
        )
    
    joblib.dump(
    
        model,
    
        model_dir
        / "catboost.pkl"
    
    )
    
    print("\nCatBoost Results")
    
    print(metrics_df)
    
    return model

