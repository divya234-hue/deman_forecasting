from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.routes.auth import require_roles
from backend.routes.dataset import UPLOAD_DIR
from ml.feature_engineering.features import create_features
from ml.prediction.predictor import load_model_bundle, predict_sales

ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(
    prefix="/api/v1",
    tags=["Forecasting & Inventory"],
    dependencies=[Depends(require_roles("admin", "analyst", "viewer"))],
)


class ForecastInput(BaseModel):
    daily_forecasts: list[float] | None = Field(default=None, min_length=1, max_length=365)
    features: list[dict[str, float]] | None = Field(default=None, min_length=1, max_length=365)
    version: str | None = None
    horizon_days: int = Field(default=7, ge=1, le=365)
    lead_time_days: int = Field(default=0, ge=0, le=365)
    current_stock: float = Field(default=0, ge=0)
    service_level_z: float = Field(default=1.65, ge=0, le=5)
    daily_demand_std: float = Field(default=0, ge=0)
    unit_cost: float = Field(default=0, ge=0)
    holding_cost_rate: float = Field(default=0.2, ge=0, le=1)
    order_quantity: float = Field(default=0, ge=0)
    price_change_pct: float = Field(default=0, ge=-100, le=500)
    promotion_pct: float = Field(default=0, ge=0, le=1000)


def _forecast_from_dataset(version: str, horizon_days: int) -> dict:
    try:
        bundle = load_model_bundle(version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_id = bundle.get("dataset_id")
    if not dataset_id or len(dataset_id) != 32 or any(c not in "0123456789abcdef" for c in dataset_id):
        raise HTTPException(status_code=422, detail="This model bundle has no valid dataset ID. Retrain using an uploaded dataset.")

    metadata_path = UPLOAD_DIR / f"{dataset_id}.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="The dataset used for this model was not found.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    csv_path = UPLOAD_DIR / metadata["filename"]
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="The uploaded CSV file was not found.")

    frame = pd.read_csv(csv_path)
    mapping = metadata.get("column_mapping", {})
    frame = frame.rename(columns={source: target for target, source in mapping.items()})

    required = {"date", "product_id", "sales"}
    if not required.issubset(frame.columns):
        raise HTTPException(status_code=422, detail=f"Dataset must contain mapped columns: {sorted(required)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["date", "product_id", "sales"]).copy()
    if frame.empty:
        raise HTTPException(status_code=422, detail="Dataset has no valid dated sales rows.")

    for column in ("price", "discount", "promotion", "inventory"):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    frame = frame.sort_values(["product_id", "date"], kind="stable").reset_index(drop=True)
    last_date = frame["date"].max()
    products = frame.groupby("product_id", sort=False).tail(1).copy()
    products = products.sort_values("product_id", kind="stable")

    # Keep historical rows and append one predicted day at a time so lag features
    # for later forecast days can use earlier predictions.
    working = frame.copy()
    daily_totals = []
    product_forecasts = []

    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        raise HTTPException(status_code=422, detail="The trained model bundle has no feature-column metadata.")

    model = bundle["model"]

    for step in range(1, horizon_days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)
        new_rows = products.copy()
        new_rows["date"] = forecast_date
        new_rows["sales"] = float("nan")
        working = pd.concat([working, new_rows], ignore_index=True)

        engineered = create_features(working)
        future = engineered[engineered["date"] == forecast_date].copy()
        future = future.sort_values("product_id", kind="stable")

        X_future = future.reindex(columns=feature_columns)
        if X_future.isna().any().any():
            missing = X_future.columns[X_future.isna().any()].tolist()
            raise HTTPException(status_code=422, detail=f"Missing forecast features: {missing}. More historical data may be required.")

        predictions = model.predict(X_future)
        predictions = [max(0.0, float(value)) for value in predictions]

        # Write predicted values back to their matching appended rows.
        future_products = future["product_id"].tolist()
        latest_mask = (working["date"] == forecast_date)
        for product_id, prediction in zip(future_products, predictions):
            row_mask = latest_mask & (working["product_id"] == product_id)
            working.loc[row_mask, "sales"] = prediction
            product_forecasts.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "product_id": str(product_id),
                "predicted_sales": round(prediction, 2),
            })

        daily_totals.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_sales": round(sum(predictions), 2),
        })

    return {
        "version": version,
        "dataset_id": dataset_id,
        "model_name": bundle.get("model_name"),
        "horizon_days": horizon_days,
        "daily_forecasts": daily_totals,
        "product_forecasts": product_forecasts,
        "total_demand": round(sum(item["predicted_sales"] for item in daily_totals), 2),
        "average_daily_demand": round(
            sum(item["predicted_sales"] for item in daily_totals) / len(daily_totals), 2
        ),
    }


@router.post("/forecast")
def forecast(payload: ForecastInput):
    if payload.features:
        try:
            predictions = [predict_sales(row, payload.version) for row in payload.features]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not forecast from supplied feature rows: {exc}") from exc

        return {
            "version": payload.version,
            "periods": len(predictions),
            "forecasts": predictions,
            "total_demand": sum(predictions),
            "average_daily_demand": sum(predictions) / len(predictions),
        }

    if payload.daily_forecasts:
        forecasts = payload.daily_forecasts
        return {
            "periods": len(forecasts),
            "forecasts": forecasts,
            "total_demand": sum(forecasts),
            "average_daily_demand": sum(forecasts) / len(forecasts),
        }

    if not payload.version:
        raise HTTPException(status_code=422, detail="Provide a trained model version to forecast from the uploaded dataset.")

    return _forecast_from_dataset(payload.version, payload.horizon_days)


@router.post("/inventory/optimize")
def inventory(payload: ForecastInput):
    if not payload.daily_forecasts:
        raise HTTPException(status_code=422, detail="daily_forecasts are required for inventory optimization.")
    demand = max(0, sum(payload.daily_forecasts[:payload.lead_time_days]))
    safety = payload.service_level_z * payload.daily_demand_std * math.sqrt(payload.lead_time_days)
    reorder = demand + safety
    order = max(0, math.ceil(reorder - payload.current_stock))
    return {
        "expected_lead_time_demand": demand,
        "safety_stock": safety,
        "reorder_point": reorder,
        "recommended_order_quantity": order,
        "stockout_risk": payload.current_stock < demand,
    }


@router.post("/what-if")
def what_if(payload: ForecastInput):
    if not payload.daily_forecasts:
        raise HTTPException(status_code=422, detail="daily_forecasts are required for what-if analysis.")
    factor = max(0, 1 - payload.price_change_pct / 100 + payload.promotion_pct / 100)
    adjusted = [max(0, value * factor) for value in payload.daily_forecasts]
    return {
        "assumption": "unit elasticity for price and promotion",
        "baseline_total": sum(payload.daily_forecasts),
        "scenario_total": sum(adjusted),
        "scenario_forecasts": adjusted,
    }
