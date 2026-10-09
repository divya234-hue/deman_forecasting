from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import math
from backend.routes.auth import require_roles
from ml.prediction.predictor import predict_sales

router=APIRouter(prefix="/api/v1",tags=["Forecasting & Inventory"],dependencies=[Depends(require_roles("admin","analyst","viewer"))])

class ForecastInput(BaseModel):
    daily_forecasts:list[float]|None=Field(default=None,min_length=1,max_length=365)
    features:list[dict[str,float]]|None=Field(default=None,min_length=1,max_length=365)
    version:str|None=None
    lead_time_days:int=Field(default=0,ge=0,le=365)
    current_stock:float=Field(default=0,ge=0)
    service_level_z:float=Field(default=1.65,ge=0,le=5)
    daily_demand_std:float=Field(default=0,ge=0)
    unit_cost:float=Field(default=0,ge=0)
    holding_cost_rate:float=Field(default=0.2,ge=0,le=1)
    order_quantity:float=Field(default=0,ge=0)
    price_change_pct:float=Field(default=0,ge=-100,le=500)
    promotion_pct:float=Field(default=0,ge=0,le=1000)

@router.post("/forecast")
def forecast(payload:ForecastInput):
    if payload.features:
        try: predictions=[predict_sales(row,payload.version) for row in payload.features]
        except FileNotFoundError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
        except Exception as exc: raise HTTPException(status_code=422,detail=f"Could not forecast from the supplied feature rows: {exc}") from exc
        return {"version":payload.version,"periods":len(predictions),"forecasts":predictions,"total_demand":sum(predictions),"average_daily_demand":sum(predictions)/len(predictions)}
    if not payload.daily_forecasts: raise HTTPException(status_code=422,detail="Provide either future feature rows or daily_forecasts.")
    forecasts=payload.daily_forecasts
    return {"periods":len(forecasts),"forecasts":forecasts,"total_demand":sum(forecasts),"average_daily_demand":sum(forecasts)/len(forecasts)}

@router.post("/inventory/optimize")
def inventory(payload:ForecastInput):
    if not payload.daily_forecasts: raise HTTPException(status_code=422,detail="daily_forecasts are required for inventory optimization.")
    demand=max(0,sum(payload.daily_forecasts[:payload.lead_time_days]))
    safety=payload.service_level_z*payload.daily_demand_std*math.sqrt(payload.lead_time_days)
    reorder=demand+safety; order=max(0,math.ceil(reorder-payload.current_stock))
    return {"expected_lead_time_demand":demand,"safety_stock":safety,"reorder_point":reorder,"recommended_order_quantity":order,"stockout_risk":payload.current_stock<demand}

@router.post("/what-if")
def what_if(payload:ForecastInput):
    if not payload.daily_forecasts: raise HTTPException(status_code=422,detail="daily_forecasts are required for what-if analysis.")
    # Unit elasticity is a transparent scenario assumption, not a fitted causal effect.
    factor=max(0,1-payload.price_change_pct/100+payload.promotion_pct/100)
    adjusted=[max(0,value*factor) for value in payload.daily_forecasts]
    return {"assumption":"unit elasticity for price and promotion","baseline_total":sum(payload.daily_forecasts),"scenario_total":sum(adjusted),"scenario_forecasts":adjusted}

