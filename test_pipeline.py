from src.config import get_bigquery_config
from src.data_ingestion import load_sales_data
from src.preprocessing import preprocess_sales_data
from src.features import create_features


def main() -> None:
    config = get_bigquery_config()

    sales_df = load_sales_data(
        config=config
    )

    processed_df = preprocess_sales_data(
        sales_df
    )

    feature_df = create_features(
        processed_df
    )

    print(feature_df.head())
    print("Shape:", feature_df.shape)
    print("Columns:", feature_df.columns.tolist())
    print(
        "Date range:",
        feature_df["Date"].min(),
        "to",
        feature_df["Date"].max(),
    )


if __name__ == "__main__":
    main()