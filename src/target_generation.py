import pandas as pd


def create_loan_level_performance(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly performance records into a single
    loan-level performance summary.

    Parameters
    ----------
    perf_df : pd.DataFrame
        Freddie Mac monthly performance dataset.

    Returns
    -------
    pd.DataFrame
        One row per loan.
    """

    loan_perf = (
        perf_df.groupby("loan_identifier")
        .agg(
            max_delinquency=(
                "current_loan_delinquency_status",
                lambda x: max(
                    [
                        int(v)
                        for v in x.astype(str)
                        if v != "RA"
                    ]
                )
            ),
            ever_ra=(
                "current_loan_delinquency_status",
                lambda x: (x.astype(str) == "RA").any()
            ),
            ever_modified=(
                "modification_flag",
                lambda x: x.notna().any()
            ),
            ever_assistance=(
                "borrower_assistance_status",
                lambda x: x.notna().any()
            )
        )
        .reset_index()
    )

    return loan_perf


def assign_bssi(row):
    """
    Borrower Serviceability Stress Index (BSSI)

    Level 0 : Healthy
    Level 1 : Moderate Stress
    Level 2 : High Stress
    Level 3 : Severe Distress
    """

    if row["ever_ra"]:
        return 3

    if row["max_delinquency"] >= 6:
        return 2

    if row["ever_modified"]:
        return 2

    if row["ever_assistance"]:
        return 1

    if row["max_delinquency"] >= 1:
        return 1

    return 0


def create_bssi(loan_perf: pd.DataFrame) -> pd.DataFrame:
    """
    Create Borrower Serviceability Stress Index.
    """

    loan_perf = loan_perf.copy()

    loan_perf["bssi"] = loan_perf.apply(
        assign_bssi,
        axis=1
    )

    return loan_perf

def create_stress_flag(df):
    
    df = df.copy()

    df["stress_flag"] = (
        df["bssi"] > 0
    ).astype(int)

    return df