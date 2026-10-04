# Demand Forecasting Platform

A practical demand forecasting portfolio app with a FastAPI backend, time-aware model comparison, model versioning, a React dashboard, and optional PostgreSQL storage.

## Features

- Upload and validate CSV data; upload limit 25 MB. Required columns are `date`, `product_id`, and `sales`.
- Map source columns using `PUT /api/v1/datasets/{dataset_id}/mapping` with JSON such as `{"date":"order_day","product_id":"sku","sales":"units"}`.
- Train Linear Regression, Random Forest, and XGBoost and compare MAE, RMSE, and R² on a chronological 80/20 holdout. Lag and rolling features use only prior observations. Missing optional inputs `price`, `discount`, `promotion`, and `inventory` default to zero.
- Persist model bundles with their feature order, runs, and comparison metrics. SQLite is the zero-configuration local database; set `DATABASE_URL` to use PostgreSQL through SQLAlchemy.
- Forecast summaries, reorder point/safety stock recommendation, a stated unit-elasticity what-if scenario, and explainability. SHAP is used when available; supported tree models fall back to global feature importance.
- JWT bearer login with `admin`, `analyst`, and `viewer` roles. Dataset upload/mapping requires admin or analyst; analysis and results are readable to authenticated roles; training requires admin or analyst. The legacy `/api/v1/predict` remains available for compatibility.

## Run locally on Windows PowerShell (Python 3.12)

```powershell
cd C:\Users\Divya\demand-forecasting-platform
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: use a long random JWT_SECRET and set all three role passwords.
$env:JWT_SECRET = "replace-with-a-long-random-secret"
$env:ADMIN_PASSWORD = "your-admin-password"
$env:ANALYST_PASSWORD = "your-analyst-password"
$env:VIEWER_PASSWORD = "your-viewer-password"
uvicorn backend.app.main:app --reload
```

Open `http://localhost:8000/docs` for the API and `http://localhost:8000/health` for health status. The dashboard runs separately:

```powershell
cd frontend
npm install
npm run dev
```

Set matching role environment variables in the backend shell before starting it. The frontend defaults to `http://localhost:8000`; override it with `VITE_API_URL` if needed.

## Docker

Copy `.env.example` to `.env`, replace the secret and role passwords, then run `docker compose up --build`. The app is at `http://localhost:5173`; docs at `http://localhost:8000/docs`. Compose starts PostgreSQL and sets its database URL for the API.

## Tests

```powershell
python -m pytest -q
```

CI runs backend tests on Python 3.12 and builds the frontend. Add model/data tests as the project evolves. Uploaded data lives under `data/uploads`; saved models under `models/company_1`; local DB under `data/platform.db`.
