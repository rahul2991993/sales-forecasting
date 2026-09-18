from datetime import date
from typing import Optional

import pandas as pd
from google.cloud import bigquery

from src.config import BigQueryConfig


def create_bigquery_client(
        config: BigQueryConfig,
) -> bigquery.Client:
    return bigquery.Client(
        project=config.project_id,
        location=config.location
    )


def load_sales_data(
    config: BigQueryConfig,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    client = create_bigquery_client(config)

    filters = []
    query_parameters = []

    if start_date is not None:
        filters.append("Date(Date) >= @start_date")
        query_parameters.append(
                bigquery.ScalarQueryParameter(
                    "start_date",
                    "DATE",
                    start_date,
                )
        )

    if end_date is not None:
        filters.append("DATE(Date) <= @end_date")
        query_parameters.append(
            bigquery.ScalarQueryParameter(
                    "end_date",
                    "DATE",
                    end_date,
            )
        )

    where_clause = ""

    if filters:
            where_clause = (
                "WHERE " + " AND ".join(filters)
                )

    query = f"""
        SELECT
            DATE(Date) AS Date,
            CAST(store_id AS STRING) AS store_id,
            CAST(sales AS FLOAT64) AS sales
        FROM `{config.sales_table_id}`
        {where_clause}
        ORDER BY
            Date,
            store_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=query_parameters
    )

    query_job = client.query(
         query,
         job_config=job_config,
         location=config.location,
    )

    sales_df = query_job.result().to_dataframe(
         create_bqstorage_client=True
    )

    sales_df["Date"] = pd.to_datetime(
         sales_df["Date"]
    )
    sales_df["store_id"] = (
         sales_df["store_id"].astype(str)
    )
    sales_df["sales"] = pd.to_numeric(
         sales_df["sales"],
         errors="coerce",
    )

    return sales_df