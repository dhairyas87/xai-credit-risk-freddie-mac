"""
Combine quarterly master datasets
and preserve temporal information.
"""

import pandas as pd


def build_combined_dataset(
    input_files,
    output_path
):

    datasets = []

    for file in input_files:

        print(f"Loading {file}")

        df = pd.read_parquet(file)

        quarter = (
            file.split("_")[-1]
            .replace(".parquet", "")
        )

        # master_dataset_2018Q1.parquet
        # -> 2018Q1

        year = quarter[:4]

        quarter_num = int(
            quarter[-1]
        )

        df["year"] = int(year)

        df["quarter"] = quarter

        df["quarter_num"] = quarter_num

        datasets.append(df)

    combined_df = pd.concat(
        datasets,
        ignore_index=True
    )

    print(
        f"Combined Shape: {combined_df.shape}"
    )

    combined_df.to_parquet(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    return combined_df