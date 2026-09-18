from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()

@dataclass(frozen=True)
class BigQueryConfig:
    project_id: str
    dataset_id: str
    sales_table: str
    location: str = "US"

    @property
    def sales_table_id(self) -> str:
        return(
            f"{self.project_id}."
            f"{self.dataset_id}."
            f"{self.sales_table}"
        )


def get_bigquery_config() -> BigQueryConfig:
    required_variables = [
        "GCP_PROJECT_ID",
        "BQ_DATASET_ID",
        "BQ_SALES_TABLE",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing_variables)
        )

    return BigQueryConfig(
        project_id=os.environ["GCP_PROJECT_ID"],
        dataset_id=os.environ["BQ_DATASET_ID"],
        sales_table=os.environ["BQ_SALES_TABLE"],
        location=os.getenv("BQ_LOCATION", "US")
    )