from typing import Sequence

import pandas as pd


def create_multi_output_targets(
    feature_df: pd.DataFrame,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """
    Create one row per store and origin date with separate
    future-sales target columns for every forecast horizon.
    """

    required_columns = {
        "Date",
        "store_id",
        "sales",
        "sales_log",
        "sales_was_imputed",
    }

    missing_columns = (
        required_columns - set(feature_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing target columns: {sorted(missing_columns)}"
        )

    output_df = feature_df.copy()

    output_df = output_df.rename(
        columns={
            "Date": "origin_date",
            "sales": "origin_sales",
            "sales_log": "origin_sales_log",
        }
    )

    for horizon in horizons:
        target_lookup = feature_df[
            [
                "store_id",
                "Date",
                "sales",
                "sales_log",
                "sales_was_imputed",
            ]
        ].copy()

        target_lookup["origin_date"] = (
            target_lookup["Date"]
            - pd.Timedelta(days=horizon)
        )

        target_lookup = target_lookup.rename(
            columns={
                "sales": f"target_sales_h{horizon}",
                "sales_log": (
                    f"target_sales_log_h{horizon}"
                ),
                "sales_was_imputed": (
                    f"target_imputed_h{horizon}"
                ),
            }
        )

        target_lookup = target_lookup[
            [
                "store_id",
                "origin_date",
                f"target_sales_h{horizon}",
                f"target_sales_log_h{horizon}",
                f"target_imputed_h{horizon}",
            ]
        ]

        output_df = output_df.merge(
            target_lookup,
            on=["store_id", "origin_date"],
            how="left",
            validate="one_to_one",
        )

    return output_df