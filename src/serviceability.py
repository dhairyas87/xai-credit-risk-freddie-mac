import pandas as pd


def calculate_serviceability_score(
    df
):

    df = df.copy()

    credit_score_component = (
        df["credit_score"] / 850
    ) * 40

    dti_component = (

        1 -

        (
            df["dti"] / 60
        ).clip(
            upper=1
        )

    ) * 30

    ltv_component = (

        1 -

        (
            df["ltv"] / 100
        )

    ) * 20

    borrower_component = (

        df["num_borrowers"] / 4

    ) * 10

    df["serviceability_score"] = (

        credit_score_component

        + dti_component

        + ltv_component

        + borrower_component

    )

    return df