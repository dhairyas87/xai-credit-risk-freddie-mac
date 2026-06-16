# scripts/combine_quarters.py

from pathlib import Path
import pandas as pd

def combine_quarters(
input_dir="../data/processed",
output_file="../data/processed/master_dataset_combined.parquet"
):


quarters = [
    "2018Q1",
    "2018Q2",
    "2018Q3",
    "2018Q4"
]

dfs = []

for quarter in quarters:

    file_path = (
        Path(input_dir)
        / f"master_dataset_{quarter}.parquet"
    )

    print(
        f"Loading {file_path.name}"
    )

    df = pd.read_parquet(
        file_path
    )

    dfs.append(df)

combined_df = pd.concat(
    dfs,
    ignore_index=True
)

print(
    f"Combined Shape: {combined_df.shape}"
)

combined_df.to_parquet(
    output_file,
    index=False
)

print(
    f"Saved: {output_file}"
)

return combined_df
