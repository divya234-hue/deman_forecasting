import asyncio
import io
from starlette.datastructures import UploadFile

import backend.routes.dataset as dataset
import backend.routes.training as training
import backend.database as database
import backend.routes.auth as auth
from backend.routes.analytics import ForecastInput, forecast, inventory, what_if


def test_auth_roles_and_forecast():
    auth.SECRET = "test-secret"
    auth.USERS["analyst"]["password"] = "pass"
    token = auth.login(auth.Login(username="analyst", password="pass"))["access_token"]
    user = auth.current_user(auth.HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    assert user["role"] == "analyst"
    try:
        auth.require_roles("admin")(user)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("analyst must not pass the admin-only policy")
    payload = ForecastInput(daily_forecasts=[3, 4, 5], lead_time_days=2, current_stock=2)
    assert forecast(payload)["total_demand"] == 12
    assert inventory(payload)["recommended_order_quantity"] > 0
    scenario = what_if(ForecastInput(daily_forecasts=[10, 20], lead_time_days=2, current_stock=10, price_change_pct=10))
    assert scenario["scenario_total"] < scenario["baseline_total"]


def test_upload_custom_headers_and_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "UPLOAD_DIR", tmp_path)
    source = b"day,sku,units\n2025-01-01,A,3\n2025-01-02,A,4\n"
    uploaded = UploadFile(file=io.BytesIO(source), filename="sales.csv")
    result = asyncio.run(dataset.upload_dataset(uploaded))
    assert result["status"] == "needs_mapping"
    mapped = dataset.map_columns(result["dataset_id"], {"date": "day", "product_id": "sku", "sales": "units"})
    assert mapped["status"] == "validated"
    assert dataset.get_dataset(result["dataset_id"])["needs_mapping"] is False


def test_training_and_versions_are_persisted(tmp_path, monkeypatch):
    import csv
    from datetime import date, timedelta
    import backend.routes.dataset as dataset_module

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(dataset_module, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(training, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "platform.db")
    output = io.StringIO(); writer = csv.writer(output); writer.writerow(["date", "product_id", "sales"])
    start = date(2025, 1, 1)
    for i in range(70):
        for product in ("A", "B"):
            writer.writerow([(start + timedelta(days=i)).isoformat(), product, 12 + i % 9])
    dataset_id = "a" * 32
    (uploads / f"{dataset_id}_sales.csv").write_text(output.getvalue(), encoding="utf-8")
    import json
    (uploads / f"{dataset_id}.json").write_text(json.dumps({"dataset_id": dataset_id, "filename": f"{dataset_id}_sales.csv", "column_mapping": {"date": "date", "product_id": "product_id", "sales": "sales"}}), encoding="utf-8")
    def save_in_test_dir(model, company_id="company_1", version="version"):
        model_dir = tmp_path / "models" / company_id / version
        model_dir.mkdir(parents=True)
        return str(model_dir / "model.pkl")
    monkeypatch.setattr(training, "save_model", save_in_test_dir)
    from fastapi import BackgroundTasks
    tasks = BackgroundTasks()
    result = training.train_dataset(dataset_id, tasks, user={"role": "analyst"})
    assert result["status"] == "queued"
    asyncio.run(tasks())
    result = training.training_status(result["run_id"])
    assert result["status"] == "completed"
    assert set(result["metrics"]) == {"Linear Regression", "Random Forest", "XGBoost"}
    assert training.training_status(result["run_id"])["version"] == result["version"]
    assert training.model_versions()["versions"][0]["version"] == result["version"]
    from backend.routes.analytics import ForecastInput, forecast
    features = {"day": 2, "month": 1, "quarter": 1, "day_of_week": 1, "week_of_year": 1,
                "is_weekend": 0, "lag_1": 10, "lag_7": 10, "lag_14": 10, "lag_30": 10,
                "rolling_mean_7": 10, "rolling_mean_14": 10, "rolling_mean_30": 10,
                "price": 0, "discount": 0, "promotion": 0, "inventory": 0}
    forecast_result = forecast(ForecastInput(features=[features]))
    assert forecast_result["periods"] == 1
    assert isinstance(forecast_result["forecasts"][0], float)


