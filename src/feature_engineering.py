import pandas as pd


def create_customer_features(df):
    """
    Create basic customer-level features
    from transaction data.
    """

    df = df.copy()

    # Convert purchase amount to numeric
    df["Purchase_Amount"] = pd.to_numeric(
        df["Purchase_Amount"],
        errors="coerce"
    )

    # Customer-level aggregation
    customer_features = df.groupby(
        "Customer_ID"
    ).agg(
        total_transactions=("Transaction_ID", "count"),
        total_spent=("Purchase_Amount", "sum"),
        average_purchase=("Purchase_Amount", "mean")
    ).reset_index()

    return customer_features