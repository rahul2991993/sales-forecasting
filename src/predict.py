import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import get_bigquery_config
from src.data_ingestion import load_sales_data
from src.preprocessing import preprocess_sales_data
from src.features import create_features
from src.supervised import add_target_date_features


BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_DIRECTORY = (
    BASE_DIR / "artifacts"
)


def load_prediction_artifacts():
    model = joblib.load(
        ARTIFACT_DIRECTORY
        / "final_model.joblib"
    )

    with open(
        ARTIFACT_DIRECTORY
        / "feature_columns.json",
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    with open(
        ARTIFACT_DIRECTORY
        / "store_categories.json",
        "r",
        encoding="utf-8",
    ) as file:
        store_categories = json.load(file)

    with open(
        ARTIFACT_DIRECTORY
        / "model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        model_metadata = json.load(file)

    return (
        model,
        feature_columns,
        store_categories,
        model_metadata,
    )


def create_forecast_rows(
    feature_df: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    horizon: int,
    store_categories: list[str],
) -> pd.DataFrame:

    origin_df = feature_df[
        feature_df["Date"]
        == forecast_origin
    ].copy()

    if origin_df.empty:
        raise ValueError(
            "No feature data found for "
            f"{forecast_origin.date()}"
        )

    origin_df = origin_df.rename(
        columns={
            "Date": "origin_date",
            "sales": "origin_sales",
            "sales_log": "origin_sales_log",
        }
    )

    forecast_frames = []

    holiday_dates: set[pd.Timestamp] = set()

    for forecast_horizon in range(
        1,
        horizon + 1,
    ):
        horizon_df = (
            add_target_date_features(
                frame=origin_df,
                horizon=forecast_horizon,
                holiday_dates=holiday_dates,
            )
        )

        forecast_frames.append(
            horizon_df
        )

    forecast_df = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    forecast_df[
        "store_id_original"
    ] = pd.Categorical(
        forecast_df[
            "store_id_original"
        ].astype(str),
        categories=store_categories,
    )

    return forecast_df


def generate_forecast(
    model,
    feature_columns,
    store_categories,
    forecast_origin: str,
    horizon: int = 31,
) -> pd.DataFrame:

    if horizon < 1 or horizon > 31:
        raise ValueError(
            "Horizon must be between 1 and 31."
        )

    origin_date = pd.Timestamp(
        forecast_origin
    )

    config = get_bigquery_config()

    # Load historical data only up to
    # forecast origin.
    sales_df = load_sales_data(
        config=config,
        end_date=origin_date.date(),
    )

    processed_df = (
        preprocess_sales_data(
            sales_df
        )
    )

    feature_df = create_features(
        processed_df
    )

    forecast_df = create_forecast_rows(
        feature_df=feature_df,
        forecast_origin=origin_date,
        horizon=horizon,
        store_categories=store_categories,
    )

    missing_features = (
        set(feature_columns)
        - set(forecast_df.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing features: "
            f"{sorted(missing_features)}"
        )

    prediction_log = model.predict(
        forecast_df[
            feature_columns
        ]
    )

    forecast_df[
        "predicted_sales"
    ] = np.clip(
        np.expm1(prediction_log),
        0,
        None,
    )

    result_df = forecast_df[
        [
            "store_id_original",
            "origin_date",
            "target_date",
            "forecast_horizon",
            "predicted_sales",
        ]
    ].copy()

    result_df = result_df.rename(
        columns={
            "store_id_original":
                "store_id",
            "target_date":
                "forecast_date",
        }
    )

    result_df["store_id"] = (
        result_df["store_id"]
        .astype(str)
    )

    return result_df