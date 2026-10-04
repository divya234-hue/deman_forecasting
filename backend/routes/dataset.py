from __future__ import annotations
import io, json, re, uuid
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from backend.routes.auth import require_roles
from ml.data.validator import validate_dataset

ROOT=Path(__file__).resolve().parents[2]
UPLOAD_DIR=ROOT/"data"/"uploads"
UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
MAX_UPLOAD_BYTES=25*1024*1024
router=APIRouter(prefix="/api/v1",tags=["Dataset"],dependencies=[Depends(require_roles("admin","analyst"))])

def _safe_stem(filename:str)->str:
    return re.sub(r"[^A-Za-z0-9._-]+","_",Path(filename).name).strip("._")[:80] or "dataset.csv"

@router.post("/upload-dataset")
async def upload_dataset(file:UploadFile=File(...)):
    if not file.filename or Path(file.filename).suffix.lower()!=".csv":
        raise HTTPException(status_code=400,detail="Only CSV files are allowed.")
    raw=await file.read(MAX_UPLOAD_BYTES+1)
    if len(raw)>MAX_UPLOAD_BYTES: raise HTTPException(status_code=413,detail="CSV exceeds the 25 MB upload limit.")
    if not raw: raise HTTPException(status_code=400,detail="Uploaded CSV is empty.")
    try: frame=pd.read_csv(io.BytesIO(raw))
    except Exception as exc: raise HTTPException(status_code=400,detail=f"Could not parse CSV: {exc}") from exc
    if frame.empty or len(frame.columns)==0: raise HTTPException(status_code=422,detail="CSV must contain a header and at least one data row.")
    has_standard_columns={"date","product_id","sales"}.issubset(frame.columns)
    if has_standard_columns:
        valid,errors=validate_dataset(frame)
        if not valid: raise HTTPException(status_code=422,detail={"status":"invalid","errors":errors})
    dataset_id=uuid.uuid4().hex
    filename=f"{dataset_id}_{_safe_stem(file.filename)}"
    (UPLOAD_DIR/filename).write_bytes(raw)
    metadata={"dataset_id":dataset_id,"filename":filename,"original_filename":Path(file.filename).name,
              "columns":list(frame.columns),"rows":int(len(frame)),
              "column_mapping":{"date":"date","product_id":"product_id","sales":"sales"} if has_standard_columns else {},
              "needs_mapping":not has_standard_columns}
    (UPLOAD_DIR/f"{dataset_id}.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    preview=frame.head(5).astype(object).where(pd.notna(frame.head(5)),None).to_dict(orient="records")
    return {"status":"success" if has_standard_columns else "needs_mapping",**metadata,"preview":preview}

@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id:str):
    metadata_path=UPLOAD_DIR/f"{dataset_id}.json"
    if not re.fullmatch(r"[a-f0-9]{32}",dataset_id) or not metadata_path.is_file(): raise HTTPException(status_code=404,detail="Dataset not found.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))

@router.put("/datasets/{dataset_id}/mapping")
def map_columns(dataset_id:str,mapping:dict[str,str]):
    metadata_path=UPLOAD_DIR/f"{dataset_id}.json"
    if not re.fullmatch(r"[a-f0-9]{32}",dataset_id) or not metadata_path.is_file(): raise HTTPException(status_code=404,detail="Dataset not found.")
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    if set(mapping)!={"date","product_id","sales"} or len(set(mapping.values()))!=len(mapping):
        raise HTTPException(status_code=422,detail="Map exactly date, product_id, and sales to distinct source columns.")
    missing=set(mapping.values())-set(metadata["columns"])
    if missing: raise HTTPException(status_code=422,detail={"missing_source_columns":sorted(missing)})
    frame=pd.read_csv(UPLOAD_DIR/metadata["filename"]).rename(columns={source:target for target,source in mapping.items()})
    valid,errors=validate_dataset(frame)
    if not valid: raise HTTPException(status_code=422,detail={"mapping_errors":errors})
    metadata["column_mapping"]=mapping; metadata["needs_mapping"]=False
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return {"dataset_id":dataset_id,"column_mapping":mapping,"status":"validated"}
