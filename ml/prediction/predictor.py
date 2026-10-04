from pathlib import Path
import joblib
import pandas as pd
from backend.database import list_model_versions

ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODEL_PATH = ROOT / "models" / "company_1" / "model_v1" / "model.pkl"

def load_model(version: str | None = None):
    if version:
        bundle_path=ROOT / "models" / "company_1" / version / "model_bundle.pkl"
        if bundle_path.is_file():
            bundle=joblib.load(bundle_path); return bundle["model"], bundle.get("feature_columns")
        if version != "model_v1": raise FileNotFoundError(f"Model version not found: {version}")
    else:
        versions=list_model_versions()
        if versions:
            bundle_path=Path(versions[0]["model_path"])
            if bundle_path.is_file():
                bundle=joblib.load(bundle_path); return bundle["model"], bundle.get("feature_columns")
    if LEGACY_MODEL_PATH.is_file(): return joblib.load(LEGACY_MODEL_PATH), None
    raise FileNotFoundError("No trained model is available.")

def predict_sales(features: dict, version: str | None = None):
    model, columns=load_model(version)
    input_data=pd.DataFrame([features])
    if columns: input_data=input_data.reindex(columns=columns,fill_value=0)
    return float(model.predict(input_data)[0])
