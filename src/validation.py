# src/validation.py

def validate_loan_counts(
    orig_df,
    perf_df
):
    """
    Ensure origination and performance
    datasets contain the same loans.
    """

    orig_loans = orig_df["loan_identifier"].nunique()
    perf_loans = perf_df["loan_identifier"].nunique()

    print(f"Origination Loans : {orig_loans}")
    print(f"Performance Loans : {perf_loans}")

    if orig_loans != perf_loans:
        raise ValueError(
            "Loan count mismatch between datasets."
        )

    print("Validation Passed")