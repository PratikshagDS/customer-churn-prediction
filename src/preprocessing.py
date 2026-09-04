import pandas as pd


def load_data(file_path):
    """
    Load an Excel or CSV dataset.
    """
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)

    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    else:
        raise ValueError("Unsupported file format")


def check_missing_values(df):
    """
    Return missing-value count for each column.
    """
    return df.isnull().sum()


def fill_numeric_missing_values(df):
    """
    Fill missing numeric values with median.
    """
    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    return df