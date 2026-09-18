import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import get_bigquery_config
from src.data_ingestion import load_sales_data
from src.features import create_features
from src.model_utils import (
    calculate_regression_metrics,
    make_store_balanced_weights,
)
from src.preprocessing import preprocess_sales_data
from src.supervised import create_supervised_multi_horizon_table
from src.validation import validate_sales_data


LOGGER = logging.getLogger(__name__)
ARTIFACT_DIRECTORY = Path("artifacts")

TRAIN_END_DATE = pd.Timestamp("2026-03-31")
VALIDATION_ORIGINS = pd.to_datetime(
    [
        "2026-03-01",
        "2026-03-05",
        "2026-03-10",
        "2026-03-15",
        "2026-03-20",
    ]
)
VALIDATION_START_DATE = pd.Timestamp("2026-04-01")
VALIDATION_END_DATE = pd.Timestamp("2026-04-12")
TEST_ORIGIN_DATE = pd.Timestamp("2026-03-31")
TEST_START_DATE = pd.Timestamp("2026-04-13")
TEST_END_DATE = pd.Timestamp("2026-04-30")
FINAL_CUTOFF = pd.Timestamp("2026-04-30")


LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 56]

LAG_FEATURE_COLUMNS = [
    f"sales_lag_{lag}" for lag in LAG_DAYS
]

ROLLING_FEATURE_COLUMNS = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_60",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "same_weekday_mean_4",
    "same_weekday_mean_8",
]

WEATHER_FEATURE_COLUMNS = [
    "temperature_2m_mean (°C)",
    "precipitation_sum (mm)",
    "wind_speed_10m_max (km/h)",
    "humidity_mean",
]

ORIGIN_FEATURE_COLUMNS = [
    "store_id_original",
    "store_age_days",
    "sales_was_imputed",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_holiday",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    *WEATHER_FEATURE_COLUMNS,
    *LAG_FEATURE_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
]

TARGET_KNOWN_FEATURE_COLUMNS = [
    "forecast_horizon",
    "target_store_age_days",
    "target_month",
    "target_day",
    "target_day_of_week",
    "target_week_of_year",
    "target_quarter",
    "target_is_weekend",
    "target_is_month_start",
    "target_is_month_end",
    "target_is_holiday",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
]

FEATURE_COLUMNS = (
    ORIGIN_FEATURE_COLUMNS
    + TARGET_KNOWN_FEATURE_COLUMNS
)

TARGET_COLUMN = "target_sales_log"


LGBM_PARAMETERS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "random_state": 42,
    "n_jobs": -1,
}


def split_supervised_data(
    supervised_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = supervised_df[
        supervised_df["target_date"] <= TRAIN_END_DATE
    ].copy()

    validation_df = supervised_df[
        supervised_df["origin_date"].isin(VALIDATION_ORIGINS)
        & supervised_df["target_date"].between(
            VALIDATION_START_DATE,
            VALIDATION_END_DATE,
        )
    ].copy()

    test_df = supervised_df[
        (supervised_df["origin_date"] == TEST_ORIGIN_DATE)
        & supervised_df["target_date"].between(
            TEST_START_DATE,
            TEST_END_DATE,
        )
    ].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if validation_df.empty:
        raise ValueError("Validation split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    return train_df, validation_df, test_df


def train_validation_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> lgb.LGBMRegressor:
    train_weights = make_store_balanced_weights(train_df)
    validation_weights = make_store_balanced_weights(
        validation_df
    )

    model = lgb.LGBMRegressor(**LGBM_PARAMETERS)

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        sample_weight=train_weights,
        categorical_feature=["store_id_original"],
        eval_set=[
            (
                validation_df[FEATURE_COLUMNS],
                validation_df[TARGET_COLUMN],
            )
        ],
        eval_sample_weight=[validation_weights],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=500,
                verbose=True,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    return model


def evaluate_model(
    model: lgb.LGBMRegressor,
    test_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    predicted_log = model.predict(
        test_df[FEATURE_COLUMNS],
        num_iteration=model.best_iteration_,
    )

    predicted_sales = np.clip(
        np.expm1(predicted_log),
        0,
        None,
    )

    comparison_df = test_df[
        [
            "store_id_original",
            "origin_date",
            "target_date",
            "forecast_horizon",
            "target_sales_was_imputed",
            "origin_sales",
            "target_sales",
        ]
    ].copy()

    comparison_df = comparison_df.rename(
        columns={"target_sales": "actual_sales"}
    )
    comparison_df["predicted_sales"] = predicted_sales

    metrics = calculate_regression_metrics(
        comparison_df["actual_sales"],
        comparison_df["predicted_sales"],
    )

    return metrics, comparison_df


def retrain_final_model(
    supervised_df: pd.DataFrame,
    best_iteration: int,
) -> lgb.LGBMRegressor:
    final_training_df = supervised_df[
        supervised_df["target_date"] <= FINAL_CUTOFF
    ].copy()

    final_weights = make_store_balanced_weights(
        final_training_df
    )

    final_parameters = {
        **LGBM_PARAMETERS,
        "n_estimators": best_iteration,
    }

    final_model = lgb.LGBMRegressor(
        **final_parameters
    )

    final_model.fit(
        final_training_df[FEATURE_COLUMNS],
        final_training_df[TARGET_COLUMN],
        sample_weight=final_weights,
        categorical_feature=["store_id_original"],
    )

    return final_model


def save_artifacts(
    validation_model: lgb.LGBMRegressor,
    final_model: lgb.LGBMRegressor,
    comparison_df: pd.DataFrame,
    test_metrics: dict[str, float],
    store_categories: list[str],
) -> None:
    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        validation_model,
        ARTIFACT_DIRECTORY / "validation_model.joblib",
    )
    joblib.dump(
        final_model,
        ARTIFACT_DIRECTORY / "final_model.joblib",
    )

    comparison_df.to_csv(
        ARTIFACT_DIRECTORY / "test_predictions.csv",
        index=False,
    )

    with open(
        ARTIFACT_DIRECTORY / "feature_columns.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)

    with open(
        ARTIFACT_DIRECTORY / "store_categories.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(store_categories, file, indent=2)

    metadata = {
        "model_name": "single_global_multi_horizon_lightgbm",
        "model_version": "1.0.0",
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_target_cutoff": str(
            TRAIN_END_DATE.date()
        ),
        "final_training_cutoff": str(
            FINAL_CUTOFF.date()
        ),
        "forecast_horizons": list(range(1, 32)),
        "target_transformation": "log1p",
        "target_column": TARGET_COLUMN,
        "feature_count": len(FEATURE_COLUMNS),
        "best_iteration": int(
            validation_model.best_iteration_
            or validation_model.n_estimators
        ),
        "test_metrics": test_metrics,
    }

    with open(
        ARTIFACT_DIRECTORY / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    config = get_bigquery_config()
    sales_df = load_sales_data(config=config)
    validate_sales_data(sales_df)

    processed_df = preprocess_sales_data(sales_df)
    feature_df = create_features(processed_df)

    # Replace this with dates loaded from your holiday table.
    holiday_dates: set[pd.Timestamp] = set()

    supervised_df = create_supervised_multi_horizon_table(
        feature_df=feature_df,
        origin_feature_columns=ORIGIN_FEATURE_COLUMNS,
        holiday_dates=holiday_dates,
        horizons=range(1, 32),
    )

    train_df, validation_df, test_df = (
        split_supervised_data(supervised_df)
    )

    validation_model = train_validation_model(
        train_df=train_df,
        validation_df=validation_df,
    )

    test_metrics, comparison_df = evaluate_model(
        model=validation_model,
        test_df=test_df,
    )

    best_iteration = int(
        validation_model.best_iteration_
        or validation_model.n_estimators
    )

    final_model = retrain_final_model(
        supervised_df=supervised_df,
        best_iteration=best_iteration,
    )

    store_categories = sorted(
        feature_df["store_id_original"]
        .astype(str)
        .unique(),
        key=int,
    )

    save_artifacts(
        validation_model=validation_model,
        final_model=final_model,
        comparison_df=comparison_df,
        test_metrics=test_metrics,
        store_categories=store_categories,
    )

    LOGGER.info("Training completed successfully.")
    LOGGER.info("Test metrics: %s", test_metrics)
    LOGGER.info("Best iteration: %s", best_iteration)


if __name__ == "__main__":
    main()