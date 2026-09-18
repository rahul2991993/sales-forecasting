import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import (
    generate_forecast,
    load_prediction_artifacts
)


LOGGER = logging.getLogger(__name__)

model = None
feature_columns = None
store_categories = None
model_metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global feature_columns
    global store_categories
    global model_metadata

    LOGGER.info("Loading model artifacts.")

    (
        model,
        feature_columns,
        store_categories,
        model_metadata
    ) = load_prediction_artifacts()

    LOGGER.info("Model loaded successfully.")

    yield

    LOGGER.info("Application shutting down.")


app = FastAPI(
    title="Store Sales Forecasting API",
    description=(
        "Generates multi-horizon sales forecasts "
        "using one global LightGBM model."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Defining request and response schemas.
class ForecastRequest(BaseModel):
    forecast_origin: date
    horizon: int = Field(
        default=31,
        ge=1,
        le=31
    )


class ForecastRecord(BaseModel):
    store_id: str
    origin_date: date
    forecast_date: date
    forecast_horizon: int
    predicted_sales: float


class ForecastResponse(BaseModel):
    model_version: str
    forecast_origin: date
    horizon: int
    number_of_predictions: int
    predictions: list[ForecastRecord]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Add API endpoints.
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.get("/model-info")
def get_model_info() -> dict[str, Any]:
    if model_metadata is None:
        raise HTTPException(
            status_code=503,
            details="Model is not loaded."
        )

    return model_metadata

@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast_sales(
    request: ForecastRequest,
) -> ForecastResponse:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        result_df = generate_forecast(
            model=model,
            feature_columns=feature_columns,
            store_categories=store_categories,
            forecast_origin=(
                request.forecast_origin.isoformat()
            ),
            horizon=request.horizon,
        )

        predictions = [
            ForecastRecord(
                store_id=str(row.store_id),
                origin_date=row.origin_date.date(),
                forecast_date=(
                    row.forecast_date.date()
                ),
                forecast_horizon=int(
                    row.forecast_horizon
                ),
                predicted_sales=float(
                    row.predicted_sales
                ),
            )
            for row in result_df.itertuples(
                index=False
            )
        ]

        return ForecastResponse(
            model_version=model_metadata.get(
                "model_version",
                "unknown",
            ),
            forecast_origin=(
                request.forecast_origin
            ),
            horizon=request.horizon,
            number_of_predictions=len(
                predictions
            ),
            predictions=predictions,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        LOGGER.exception(
            "Forecast generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Forecast generation failed.",
        ) from error