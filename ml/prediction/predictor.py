from pathlib import Path

import joblib
import pandas as pd

from backend.database import list_model_versions

ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODEL_PATH = ROOT / "models" / "company_1" / "model_v1" / "model.pkl"


def load_model_bundle(version: str | None = None) -> dict:
    """Load a trained model and its metadata."""
    if version:
        bundle_path = (
            ROOT / "models" / "company_1" / version / "model_bundle.pkl"
        )

        if bundle_path.is_file():
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict) and "model" in bundle:
                return bundle

        if version != "model_v1":
            raise FileNotFoundError(f"Model version not found: {version}")

    else:
        versions = list_model_versions()

        for item in versions:
            bundle_path = Path(item["model_path"])

            if bundle_path.is_file():
                bundle = joblib.load(bundle_path)
                if isinstance(bundle, dict) and "model" in bundle:
                    return bundle

    if LEGACY_MODEL_PATH.is_file():
        return {
            "model": joblib.load(LEGACY_MODEL_PATH),
            "feature_columns": None,
            "dataset_id": None,
            "version": "model_v1",
        }

    raise FileNotFoundError("No trained model is available.")


def load_model(version: str | None = None):
    """Backward-compatible loader for existing API routes."""
    bundle = load_model_bundle(version)
    return bundle["model"], bundle.get("feature_columns")


def predict_sales(features: dict, version: str | None = None) -> float:
    """Predict sales for one feature row."""
    bundle = load_model_bundle(version)
    model = bundle["model"]
    columns = bundle.get("feature_columns")

    input_data = pd.DataFrame([features])

    if columns:
        input_data = input_data.reindex(columns=columns, fill_value=0)

    return float(model.predict(input_data)[0])
