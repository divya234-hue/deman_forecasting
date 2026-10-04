from fastapi import APIRouter, HTTPException, Depends
from backend.routes.auth import require_roles
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from backend.database import list_model_versions

router = APIRouter(prefix="/api/v1", tags=["Explainability"], dependencies=[Depends(require_roles("admin", "analyst", "viewer"))])
class ExplainRequest(BaseModel):
    version: str
    features: dict[str, float]

@router.post("/explain")
def explain(payload: ExplainRequest):
    model_path = Path(__file__).resolve().parents[2] / "models" / "company_1" / payload.version / "model_bundle.pkl"
    if not model_path.is_file(): raise HTTPException(status_code=404, detail="Model version not found.")
    bundle=joblib.load(model_path); columns=bundle["feature_columns"]
    unknown=set(payload.features)-set(columns)
    if unknown: raise HTTPException(status_code=422, detail={"unknown_features":sorted(unknown)})
    estimator=bundle["model"]; sample=pd.DataFrame([{c:payload.features.get(c,0) for c in columns}],columns=columns)
    try:
        import shap
        values=shap.TreeExplainer(estimator).shap_values(sample)[0]
        method="SHAP TreeExplainer"
    except (ImportError, Exception) as exc:
        importance=getattr(estimator,"feature_importances_",None)
        if importance is None: raise HTTPException(status_code=501, detail="Install shap to explain this model type.") from exc
        values=importance; method="global feature importance (SHAP unavailable)"
    return {"version":payload.version,"method":method,"contributions":{c:float(v) for c,v in zip(columns,values)}}


