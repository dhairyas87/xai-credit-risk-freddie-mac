# src/data_loader.py

import pandas as pd
from src.schema import (
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS
)

def load_origination(path):

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        low_memory=False
    )

    if len(df.columns) != len(ORIGINATION_COLUMNS):
        raise ValueError(
            f"Expected {len(ORIGINATION_COLUMNS)} columns, "
            f"found {len(df.columns)}"
        )

    df.columns = ORIGINATION_COLUMNS

    return df


def load_performance(path):

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        low_memory=False
    )

    if len(df.columns) != len(PERFORMANCE_COLUMNS):
        raise ValueError(
            f"Expected {len(PERFORMANCE_COLUMNS)} columns, "
            f"found {len(df.columns)}"
        )

    df.columns = PERFORMANCE_COLUMNS

    return df