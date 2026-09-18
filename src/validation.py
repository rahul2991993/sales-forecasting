import pandas as pd


REQUIRED_SALES_COLUMNS = {
    "Date",
    "store_id",
    "sales",
}


def validate_sales_data(
        sales_df: pd.DataFrame,
) -> None:
    if sales_df.empty:
        raise ValueError(
            "The BigQuery sales query returned no rows."
        )

    missing_columns = (
        REQUIRED_SALES_COLUMNS
        - set(sales_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if sales_df["Date"].isna().any():
        raise ValueError(
            "Null values were found in Date."
        )

    if sales_df["store_id"].isna().any():
        raise ValueError(
            "Null values were found in store_id."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        sales_df["Date"]
    ):
        raise TypeError(
            "Date must be a datetime column."
        )

    if not pd.api.types.is_numeric_dtype(
        sales_df["sales"]
    ):
        raise TypeError(
            "Sales must be numeric."
        )

    duplicate_count = sales_df.duplicated(
        subset=["Date", "store_id"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "store-date rows."
        )

    print("Sales data validation passed.")
    print("Rows:", len(sales_df))
    print(
        "Stores:",
        sales_df["store_id"].nunique()
    )
    print(
        "Date range:",
        sales_df["Date"].min(),
        "to",
        sales_df["Date"].max(),
    )
    print(
        "Missing sales:",
        sales_df["sales"].isna().sum()
    )