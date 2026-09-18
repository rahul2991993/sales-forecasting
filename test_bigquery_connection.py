from datetime import date

from src.config import get_bigquery_config
from src.data_ingestion import load_sales_data
from src.validation import validate_sales_data


def main() -> None:
    config = get_bigquery_config()

    sales_df = load_sales_data(
        config=config,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 4, 30),
    )

    validate_sales_data(sales_df)

    print(sales_df.head())
    print(sales_df.dtypes)


if __name__== "__main__":
    main()