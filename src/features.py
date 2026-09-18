import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    feature_df = df.copy()

    feature_df["year"] = feature_df["Date"].dt.year
    feature_df["month"] = feature_df["Date"].dt.month
    feature_df["day"] = feature_df["Date"].dt.day

    feature_df["day_of_week"] = (
        feature_df["Date"].dt.dayofweek
    )

    feature_df["week_of_year"] = (
        feature_df["Date"]
        .dt.isocalendar()
        .week.astype(int)
    )

    feature_df["quarter"] = (
        feature_df["Date"].dt.quarter
    )

    feature_df["is_weekend"] = (
        feature_df["day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    feature_df["is_month_start"] = (
        feature_df["Date"]
        .dt.is_month_start
        .astype(int)
    )

    feature_df["is_month_end"] = (
        feature_df["Date"]
        .dt.is_month_end
        .astype(int)
    )

    feature_df["dow_sin"] = np.sin(
        2 * np.pi
        * feature_df["day_of_week"] / 7
    )

    feature_df["dow_cos"] = np.cos(
        2 * np.pi
        * feature_df["day_of_week"] / 7
    )

    feature_df["month_sin"] = np.sin(
        2 * np.pi
        * feature_df["month"] / 12
    )

    feature_df["month_cos"] = np.cos(
        2 * np.pi
        * feature_df["month"] / 12
    )

    return feature_df


def add_store_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    feature_df = df.copy()

    # Required by your LightGBM pipeline.
    feature_df["store_id_original"] = (
        feature_df["store_id"].astype(str)
    )

    # First available date for each store.
    store_opening_dates = (
        feature_df
        .groupby("store_id_original")["Date"]
        .min()
    )

    feature_df["store_age_days"] = (
        feature_df["Date"]
        - feature_df["store_id_original"]
        .map(store_opening_dates)
    ).dt.days

    return feature_df


def add_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    feature_df = (
        df.sort_values(
            ["store_id_original", "Date"]
        )
        .copy()
    )

    lag_days = [
        1, 2, 3, 7, 14, 21, 28, 35, 56
    ]

    for lag in lag_days:
        feature_df[f"sales_lag_{lag}"] = (
            feature_df
            .groupby(
                "store_id_original"
            )["sales_log"]
            .shift(lag)
        )

    return feature_df


def add_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    feature_df = (
        df.sort_values(
            ["store_id_original", "Date"]
        )
        .copy()
    )

    grouped_sales = (
        feature_df.groupby(
            "store_id_original"
        )["sales_log"]
    )

    for window in [7, 14, 28, 60]:
        feature_df[
            f"rolling_mean_{window}"
        ] = grouped_sales.transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window,
                    min_periods=1,
                )
                .mean()
            )
        )

    for window in [7, 14, 28]:
        feature_df[
            f"rolling_std_{window}"
        ] = grouped_sales.transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window,
                    min_periods=2,
                )
                .std()
            )
        )

    return feature_df


def add_same_weekday_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    feature_df = df.copy()

    feature_df["same_weekday_mean_4"] = (
        feature_df
        .groupby(
            [
                "store_id_original",
                "day_of_week",
            ]
        )["sales_log"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    4,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    feature_df["same_weekday_mean_8"] = (
        feature_df
        .groupby(
            [
                "store_id_original",
                "day_of_week",
            ]
        )["sales_log"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    8,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    return feature_df


def add_placeholder_external_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Temporary placeholders.

    Replace these with real holiday/weather data
    before final production deployment.
    """
    feature_df = df.copy()

    if "is_holiday" not in feature_df.columns:
        feature_df["is_holiday"] = 0

    if (
        "temperature_2m_mean (°C)"
        not in feature_df.columns
    ):
        feature_df[
            "temperature_2m_mean (°C)"
        ] = 0.0

    if (
        "precipitation_sum (mm)"
        not in feature_df.columns
    ):
        feature_df[
            "precipitation_sum (mm)"
        ] = 0.0

    if (
        "wind_speed_10m_max (km/h)"
        not in feature_df.columns
    ):
        feature_df[
            "wind_speed_10m_max (km/h)"
        ] = 0.0

    if "humidity_mean" not in feature_df.columns:
        feature_df["humidity_mean"] = 0.0

    return feature_df


def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    feature_df = add_calendar_features(df)

    feature_df = add_store_features(
        feature_df
    )

    feature_df = add_lag_features(
        feature_df
    )

    feature_df = add_rolling_features(
        feature_df
    )

    feature_df = add_same_weekday_features(
        feature_df
    )

    feature_df = add_placeholder_external_features(
        feature_df
    )

    return feature_df


def add_target_date_features(
    df: pd.DataFrame,
    horizon: int,
    holiday_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    output_df = df.copy()

    output_df["forecast_horizon"] = horizon

    output_df["target_date"] = (
        output_df["origin_date"]
        + pd.Timedelta(days=horizon)
    )

    output_df["target_store_age_days"] = (
        output_df["store_age_days"]
        + horizon
    )

    output_df["target_month"] = (
        output_df["target_date"].dt.month
    )

    output_df["target_day"] = (
        output_df["target_date"].dt.day
    )

    output_df["target_day_of_week"] = (
        output_df["target_date"].dt.dayofweek
    )

    output_df["target_week_of_year"] = (
        output_df["target_date"]
        .dt.isocalendar()
        .week.astype(int)
    )

    output_df["target_quarter"] = (
        output_df["target_date"].dt.quarter
    )

    output_df["target_is_weekend"] = (
        output_df["target_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    output_df["target_is_month_start"] = (
        output_df["target_date"]
        .dt.is_month_start
        .astype(int)
    )

    output_df["target_is_month_end"] = (
        output_df["target_date"]
        .dt.is_month_end
        .astype(int)
    )

    output_df["target_is_holiday"] = (
        output_df["target_date"]
        .dt.normalize()
        .isin(holiday_dates)
        .astype(int)
    )

    output_df["target_dow_sin"] = np.sin(
        2
        * np.pi
        * output_df["target_day_of_week"]
        / 7
    )

    output_df["target_dow_cos"] = np.cos(
        2
        * np.pi
        * output_df["target_day_of_week"]
        / 7
    )

    output_df["target_month_sin"] = np.sin(
        2
        * np.pi
        * output_df["target_month"]
        / 12
    )

    output_df["target_month_cos"] = np.cos(
        2
        * np.pi
        * output_df["target_month"]
        / 12
    )

    return output_df