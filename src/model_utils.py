from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_store_balanced_weights(
    frame: pd.DataFrame,
    store_column: str = "store_id_original",
) -> np.ndarray:
    """Give each store approximately equal total training weight."""
    store_counts = (
        frame.groupby(store_column, observed=True)
        .size()
    )

    weights = (
        frame[store_column]
        .map(1.0 / store_counts)
        .astype(float)
        .to_numpy()
    )

    return weights / weights.mean()


def calculate_regression_metrics(
    actual: Any,
    predicted: Any,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(
        actual_array,
        predicted_array,
    )
    rmse = np.sqrt(
        mean_squared_error(
            actual_array,
            predicted_array,
        )
    )

    denominator = np.abs(actual_array).sum()

    wape = (
        np.abs(actual_array - predicted_array).sum()
        / denominator
        * 100
        if denominator != 0
        else np.nan
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape": float(wape),
        "number_of_records": int(len(actual_array)),
    }