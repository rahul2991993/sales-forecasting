import numpy as np
import pandas as pd


def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "Date",
        "store_id",
        "sales"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    processed_df = df.copy()

    processed_df["Date"] = pd.to_datetime(
        processed_df["Date"]
    )

    processed_df["store_id"] = (
        processed_df["store_id"]
        .astype(str)
    )

    processed_df["sales"] = pd.to_numeric(
        processed_df["sales"],
        errors="coerce"
    )

    processed_df["sales_was_imputed"] = (
        processed_df["sales"].isna().astype(int)
    )

    processed_df["sales"] = (
        processed_df["sales"]
        .fillna(0)
        .clip(lower=0)
    )

    processed_df["sales_log"] = np.log1p(
        processed_df["sales"]
    )

    processed_df = (
        processed_df
        .sort_values(["store_id", "Date"])
        .drop_duplicates(
            ["store_id", "Date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return processed_df