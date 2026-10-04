from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml.prediction.predictor import predict_sales
from backend.routes.dataset import router as dataset_router
from backend.routes.training import router as training_router
from backend.routes.analytics import router as analytics_router
from backend.routes.auth import router as auth_router
from backend.routes.explain import router as explain_router

app=FastAPI(title="Demand Forecasting Platform",version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:5174","http://127.0.0.1:5174"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
for router in (dataset_router,training_router,analytics_router,auth_router,explain_router): app.include_router(router)

class PredictionRequest(BaseModel):
    day:int; month:int; quarter:int; day_of_week:int; week_of_year:int; is_weekend:int
    lag_1:float; lag_7:float; lag_14:float; lag_30:float
    rolling_mean_7:float; rolling_mean_14:float; rolling_mean_30:float
    price:float; discount:float; promotion:int; inventory:float
    version:str|None=None

@app.get("/")
def root(): return {"message":"Demand Forecasting API is running","docs":"/docs"}
@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/api/v1/predict")
def predict(request:PredictionRequest):
    data=request.model_dump(); version=data.pop("version")
    try: return {"predicted_sales":predict_sales(data,version)}
    except FileNotFoundError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc


