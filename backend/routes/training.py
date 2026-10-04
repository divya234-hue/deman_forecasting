from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from backend.routes.auth import require_roles
from backend.routes.dataset import UPLOAD_DIR
from backend.database import save_run, get_run, save_model_version, list_model_versions
from ml.feature_engineering.features import create_features
from ml.training.train import FEATURE_COLUMNS, train_models
from ml.models.model_saver import save_model

router=APIRouter(prefix="/api/v1",tags=["Training"],dependencies=[Depends(require_roles("admin","analyst","viewer"))])
RUNS:dict[str,dict]={}

def _dataset(dataset_id:str):
    meta_path=UPLOAD_DIR/f"{dataset_id}.json"
    if not meta_path.is_file(): raise HTTPException(status_code=404,detail="Dataset not found.")
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    frame=pd.read_csv(UPLOAD_DIR/meta["filename"])
    frame=frame.rename(columns={source:target for target,source in meta.get("column_mapping",{}).items()})
    return meta,frame

def _run_training(dataset_id:str,run_id:str):
    run=RUNS[run_id]
    run.update({"status":"running","started_at":datetime.now(timezone.utc).isoformat()})
    save_run(run)
    try:
        _,raw=_dataset(dataset_id)
        for column,default in (("price",0),("discount",0),("promotion",0),("inventory",0)):
            if column not in raw.columns: raw[column]=default
        frame=create_features(raw).dropna(subset=FEATURE_COLUMNS+["sales"]).sort_values("date",kind="stable")
        if len(frame)<10: raise ValueError("At least 10 rows with complete lag features are required.")
        unique_dates=frame["date"].drop_duplicates().sort_values().to_numpy()
        if len(unique_dates)<2: raise ValueError("At least two distinct dates are required for chronological validation.")
        cutoff=unique_dates[max(1,min(len(unique_dates)-1,int(len(unique_dates)*.8)))]
        train,test=frame[frame["date"]<cutoff],frame[frame["date"]>=cutoff]
        if train.empty or test.empty: raise ValueError("Chronological split produced an empty partition.")
        scores=train_models(train[FEATURE_COLUMNS],test[FEATURE_COLUMNS],train["sales"],test["sales"])
        best=min(scores,key=lambda name:scores[name]["rmse"])
        version="model_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+run_id[:6]
        legacy_path=save_model(scores[best]["model"],company_id="company_1",version=version)
        import joblib
        bundle_path=Path(legacy_path).with_name("model_bundle.pkl")
        joblib.dump({"model":scores[best]["model"],"feature_columns":FEATURE_COLUMNS,"model_name":best,"dataset_id":dataset_id,"version":version},bundle_path)
        comparison={name:{key:float(value) for key,value in result.items() if key!="model"} for name,result in scores.items()}
        save_model_version(version,best,str(bundle_path),comparison)
        run.update({"status":"completed","completed_at":datetime.now(timezone.utc).isoformat(),"best_model":best,"version":version,"metrics":comparison,"model_path":str(bundle_path)})
    except Exception as exc:
        run.update({"status":"failed","error":str(exc),"completed_at":datetime.now(timezone.utc).isoformat()})
    RUNS[run_id]=run
    save_run(run)

@router.post("/datasets/{dataset_id}/train",status_code=202)
def train_dataset(dataset_id:str,background_tasks:BackgroundTasks,user=Depends(require_roles("admin","analyst"))):
    _dataset(dataset_id)
    run_id=uuid.uuid4().hex
    run={"run_id":run_id,"dataset_id":dataset_id,"status":"queued","created_at":datetime.now(timezone.utc).isoformat()}
    RUNS[run_id]=run; save_run(run)
    background_tasks.add_task(_run_training,dataset_id,run_id)
    return run

@router.get("/training/{run_id}")
def training_status(run_id:str):
    run=RUNS.get(run_id) or get_run(run_id)
    if not run: raise HTTPException(status_code=404,detail="Training run not found.")
    return run

@router.get("/models")
def model_versions(): return {"versions":list_model_versions()}
