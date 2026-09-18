from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_target_date_features(
    frame: pd.DataFrame,
    horizon: int,
    holiday_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    """Add features known for the future target date."""
    result = frame.copy()

    result["forecast_horizon"] = horizon
    result["target_date"] = (
        result["origin_date"] + pd.Timedelta(days=horizon)
    )

    result["target_store_age_days"] = (
        result["store_age_days"] + horizon
    )
    result["target_month"] = result["target_date"].dt.month
    result["target_day"] = result["target_date"].dt.day
    result["target_day_of_week"] = (
        result["target_date"].dt.dayofweek
    )
    result["target_week_of_year"] = (
        result["target_date"]
        .dt.isocalendar()
        .week.astype(int)
    )
    result["target_quarter"] = (
        result["target_date"].dt.quarter
    )
    result["target_is_weekend"] = (
        result["target_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )
    result["target_is_month_start"] = (
        result["target_date"]
        .dt.is_month_start.astype(int)
    )
    result["target_is_month_end"] = (
        result["target_date"]
        .dt.is_month_end.astype(int)
    )
    result["target_is_holiday"] = (
        result["target_date"]
        .dt.normalize()
        .isin(holiday_dates)
        .astype(int)
    )

    result["target_dow_sin"] = np.sin(
        2 * np.pi * result["target_day_of_week"] / 7
    )
    result["target_dow_cos"] = np.cos(
        2 * np.pi * result["target_day_of_week"] / 7
    )
    result["target_month_sin"] = np.sin(
        2 * np.pi * result["target_month"] / 12
    )
    result["target_month_cos"] = np.cos(
        2 * np.pi * result["target_month"] / 12
    )

    return result


def create_supervised_multi_horizon_table(
    feature_df: pd.DataFrame,
    origin_feature_columns: Sequence[str],
    holiday_dates: set[pd.Timestamp],
    horizons: Sequence[int] = range(1, 32),
) -> pd.DataFrame:
    """
    Create the long supervised table used by one global model.

    One historical origin row is repeated for every forecast horizon.
    The target is sales at origin_date + forecast_horizon.
    """
    required_columns = {
        "Date",
        "store_id_original",
        "sales",
        "sales_log",
        "sales_was_imputed",
        *origin_feature_columns,
    }

    missing_columns = required_columns - set(feature_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns for supervised table: {sorted(missing_columns)}"
        )

    target_lookup = feature_df[
        [
            "store_id_original",
            "Date",
            "sales",
            "sales_log",
            "sales_was_imputed",
        ]
    ].rename(
        columns={
            "Date": "target_date",
            "sales": "target_sales",
            "sales_log": "target_sales_log",
            "sales_was_imputed": "target_sales_was_imputed",
        }
    )

    origin_base = feature_df[
        [
            "Date",
            "sales",
            "sales_log",
            *origin_feature_columns,
        ]
    ].copy()

    origin_base = origin_base.rename(
        columns={
            "Date": "origin_date",
            "sales": "origin_sales",
            "sales_log": "origin_sales_log",
        }
    )

    horizon_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        frame = add_target_date_features(
            frame=origin_base,
            horizon=horizon,
            holiday_dates=holiday_dates,
        )

        frame = frame.merge(
            target_lookup,
            on=["store_id_original", "target_date"],
            how="inner",
            validate="many_to_one",
        )

        horizon_frames.append(frame)

    supervised_df = pd.concat(
        horizon_frames,
        ignore_index=True,
    )

    store_categories = sorted(
        feature_df["store_id_original"]
        .astype(str)
        .unique(),
        key=int,
    )

    supervised_df["store_id_original"] = pd.Categorical(
        supervised_df["store_id_original"].astype(str),
        categories=store_categories,
    )

    duplicate_count = supervised_df.duplicated(
        subset=[
            "store_id_original",
            "origin_date",
            "forecast_horizon",
        ]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate supervised rows."
        )

    return supervised_df