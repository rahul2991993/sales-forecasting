from src.predict import (
    generate_forecast,
    load_prediction_artifacts,
)


def main():
    (
        model,
        feature_columns,
        store_categories,
        metadata,
    ) = load_prediction_artifacts()

    print(
        "Model version:",
        metadata["model_version"],
    )

    forecast_df = generate_forecast(
        model=model,
        feature_columns=feature_columns,
        store_categories=store_categories,
        forecast_origin="2026-04-30",
        horizon=31,
    )

    print(forecast_df.head())

    print(
        "Rows:",
        len(forecast_df),
    )

    print(
        "Stores:",
        forecast_df[
            "store_id"
        ].nunique(),
    )

    print(
        "Dates:",
        forecast_df[
            "forecast_date"
        ].nunique(),
    )


if __name__ == "__main__":
    main()