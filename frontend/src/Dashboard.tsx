import { useEffect, useMemo, useState, type FormEvent } from "react";

type Metrics = {
  mae?: number;
  rmse?: number;
  r2?: number;
};

type Version = {
  version: string;
  model_name: string;
  model_path?: string;
  metrics?: Record<string, Metrics> | Metrics;
};

type Field = "date" | "product_id" | "sales";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const navItems = [
  ["overview", "⌂", "Overview"],
  ["datasets", "▣", "Datasets"],
  ["models", "◈", "Models"],
  ["forecast", "↗", "Forecast"],
  ["inventory", "◇", "Inventory"],
  ["whatif", "✦", "What-if"],
] as const;

function metricEntries(version?: Version) {
  if (!version?.metrics) return [];

  const metrics = version.metrics as Record<string, Metrics>;

  if (
    typeof metrics.mae === "number" ||
    typeof metrics.rmse === "number" ||
    typeof metrics.r2 === "number"
  ) {
    return [[version.model_name || "Model", metrics as Metrics]] as [
      string,
      Metrics
    ][];
  }

  return Object.entries(metrics);
}

function formatNumber(value?: number, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

export default function Dashboard() {
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [username, setUsername] = useState(
    localStorage.getItem("username") || "analyst"
  );
  const [password, setPassword] = useState("");

  const [active, setActive] =
    useState<(typeof navItems)[number][0]>("overview");

  const [file, setFile] = useState<File | null>(null);
  const [dataset, setDataset] = useState("");
  const [columns, setColumns] = useState<string[]>([]);
  const [needsMapping, setNeedsMapping] = useState(false);
  const [mapping, setMapping] = useState<Record<Field, string>>({
    date: "",
    product_id: "",
    sales: "",
  });

  const [versions, setVersions] = useState<Version[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const [forecast, setForecast] = useState<number | null>(null);

  const [dailyForecasts, setDailyForecasts] = useState<
    { date: string; predicted_sales: number }[]
  >([]);
  const [totalDemand, setTotalDemand] = useState<number | null>(null);

  const [currentStock, setCurrentStock] = useState("450");
  const [leadTime, setLeadTime] = useState("7");
  const [demandStd, setDemandStd] = useState("10");
  const [inventoryResult, setInventoryResult] = useState<{
    safety_stock?: number;
    reorder_point?: number;
    recommended_order_quantity?: number;
    stockout_risk?: boolean;
  } | null>(null);

  const [priceChange, setPriceChange] = useState("0");
  const [promotion, setPromotion] = useState("0");
  const [whatIfResult, setWhatIfResult] = useState<{
    baseline_prediction?: number;
    scenario_prediction?: number;
    assumption?: string;
  } | null>(null);

  const canEdit = username === "admin" || username === "analyst";

  const headers = () => ({
    Authorization: `Bearer ${token}`,
  });

  const latestVersion = versions.find((v) =>
    v.version.startsWith("model_")
  );

  const latestMetrics = useMemo(() => {
    return metricEntries(latestVersion);
  }, [latestVersion]);

  const bestModel = useMemo(() => {
    if (!latestMetrics.length) return null;

    return [...latestMetrics].sort(
      (a, b) => (a[1].rmse ?? Infinity) - (b[1].rmse ?? Infinity)
    )[0];
  }, [latestMetrics]);

  async function refresh() {
    try {
      const response = await fetch(`${API}/api/v1/models`, {
        headers: headers(),
      });

      if (!response.ok) return;

      const data = await response.json();
      setVersions(data.versions || []);
    } catch {
      setMessage("Backend unavailable. Start FastAPI on port 8000.");
    }
  }

  useEffect(() => {
    if (token) void refresh();
  }, [token]);

  async function login(event: FormEvent) {
    event.preventDefault();
    setBusy(true);

    try {
      const response = await fetch(`${API}/api/v1/auth/token`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setMessage(data.detail || "Sign in failed.");
        return;
      }

      localStorage.setItem("token", data.access_token);
      localStorage.setItem("username", username);

      setToken(data.access_token);
      setMessage("Welcome back.");
      setPassword("");
    } catch {
      setMessage("Cannot connect to the FastAPI backend.");
    } finally {
      setBusy(false);
    }
  }

  async function upload() {
    if (!file) return;

    setBusy(true);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/api/v1/upload-dataset`, {
        method: "POST",
        headers: headers(),
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        setMessage(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail)
        );
        return;
      }

      const cols: string[] = data.columns || [];

      setDataset(data.dataset_id);
      setColumns(cols);
      setNeedsMapping(Boolean(data.needs_mapping));

      setMapping({
        date: cols.includes("date") ? "date" : cols[0] || "",
        product_id: cols.includes("product_id")
          ? "product_id"
          : cols[1] || "",
        sales: cols.includes("sales") ? "sales" : cols[2] || "",
      });

      setMessage(
        `Uploaded ${data.rows} rows. ${
          data.needs_mapping
            ? "Map the required columns."
            : "Dataset is ready for training."
        }`
      );

      setActive("datasets");
    } catch {
      setMessage("Dataset upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function mapColumns() {
    if (!dataset) return;

    setBusy(true);

    try {
      const response = await fetch(
        `${API}/api/v1/datasets/${dataset}/mapping`,
        {
          method: "PUT",
          headers: {
            ...headers(),
            "Content-Type": "application/json",
          },
          body: JSON.stringify(mapping),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        setMessage(JSON.stringify(data.detail));
        return;
      }

      setNeedsMapping(false);
      setMessage("Columns mapped successfully. Dataset is ready.");
    } catch {
      setMessage("Column mapping failed.");
    } finally {
      setBusy(false);
    }
  }

  async function train() {
    if (!dataset || needsMapping) return;

    setBusy(true);

    try {
      const response = await fetch(
        `${API}/api/v1/datasets/${dataset}/train`,
        {
          method: "POST",
          headers: headers(),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        setMessage(JSON.stringify(data.detail));
        return;
      }

      setMessage(`Training run ${data.run_id} started...`);

      for (let i = 0; i < 180; i++) {
        await new Promise((resolve) => setTimeout(resolve, 2000));

        const statusResponse = await fetch(
          `${API}/api/v1/training/${data.run_id}`,
          {
            headers: headers(),
          }
        );

        const run = await statusResponse.json();

        if (run.status === "completed") {
          setMessage(
            `Training complete. Best model: ${run.best_model}.`
          );
          await refresh();
          setActive("models");
          return;
        }

        if (run.status === "failed") {
          setMessage(`Training failed: ${run.error}`);
          return;
        }
      }

      setMessage("Training is still running. Check the API docs.");
    } catch {
      setMessage("Training request failed.");
    } finally {
      setBusy(false);
    }
  }

  async function runForecast() {
    if (!latestVersion) {
      setMessage("Train a model first.");
      return;
    }

    setBusy(true);

    try {
      const response = await fetch(`${API}/api/v1/forecast`, {
        method: "POST",
        headers: {
          ...headers(),
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          version: latestVersion.version,
          horizon_days: 7,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setMessage(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? data)
        );
        return;
      }

      const daily = Array.isArray(data.daily_forecasts)
        ? data.daily_forecasts.filter(
            (item: { date?: unknown; predicted_sales?: unknown }) =>
              typeof item.date === "string" &&
              typeof item.predicted_sales === "number"
          )
        : [];

      if (daily.length === 0) {
        setForecast(null);
        setDailyForecasts([]);
        setTotalDemand(null);
        setMessage("No daily predictions were returned by the API.");
        return;
      }

      setDailyForecasts(daily);
      setTotalDemand(
        typeof data.total_demand === "number"
          ? data.total_demand
          : daily.reduce(
              (sum: number, item: { predicted_sales: number }) =>
                sum + item.predicted_sales,
              0
            )
      );
      setForecast(daily[0].predicted_sales);
      setMessage("7-day forecast generated successfully.");
      setActive("forecast");
    } catch {
      setMessage("Forecast request failed.");
    } finally {
      setBusy(false);
    }
  }

  async function optimizeInventory() {
    if (forecast === null) {
      setMessage("Generate a forecast first.");
      return;
    }

    setBusy(true);

    try {
      const response = await fetch(
        `${API}/api/v1/inventory/optimize`,
        {
          method: "POST",
          headers: {
            ...headers(),
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            daily_forecasts: [forecast],
            lead_time_days: Number(leadTime),
            current_stock: Number(currentStock),
            daily_demand_std: Number(demandStd),
            service_level_z: 1.65,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        setMessage(JSON.stringify(data.detail));
        return;
      }

      setInventoryResult(data);
      setMessage("Inventory recommendation calculated.");
      setActive("inventory");
    } catch {
      setMessage("Inventory optimization failed.");
    } finally {
      setBusy(false);
    }
  }

  async function runWhatIf() {
    if (forecast === null) {
      setMessage("Generate a baseline forecast first.");
      return;
    }

    setBusy(true);

    try {
      const response = await fetch(`${API}/api/v1/what-if`, {
        method: "POST",
        headers: {
          ...headers(),
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          daily_forecasts: [forecast],
          price_change_pct: Number(priceChange),
          promotion_pct: Number(promotion),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setMessage(JSON.stringify(data.detail));
        return;
      }

      setWhatIfResult(data);
      setMessage("What-if scenario calculated.");
      setActive("whatif");
    } catch {
      setMessage("What-if analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  function signOut() {
    localStorage.removeItem("token");
    localStorage.removeItem("username");

    setToken("");
    setVersions([]);
    setDataset("");
    setForecast(null);
  }

  if (!token) {
    return (
      <main className="auth-page">
        <div className="auth-orb orb-one" />
        <div className="auth-orb orb-two" />

        <section className="login-card">
          <div className="brand-mark">DF</div>

          <p className="eyebrow">AI DEMAND INTELLIGENCE</p>
          <h1>Forecast smarter.</h1>
          <p className="login-sub">
            Predict demand, optimize inventory and turn historical
            sales data into actionable decisions.
          </p>

          <form onSubmit={login}>
            <label>
              Account
              <select
                value={username}
                onChange={(event) =>
                  setUsername(event.target.value)
                }
              >
                <option value="analyst">Analyst</option>
                <option value="admin">Admin</option>
                <option value="viewer">Viewer</option>
              </select>
            </label>

            <label>
              Password
              <input
                type="password"
                value={password}
                onChange={(event) =>
                  setPassword(event.target.value)
                }
                placeholder="Enter password"
                required
              />
            </label>

            <button className="primary-btn full-btn" disabled={busy}>
              {busy ? "Signing in..." : "Sign in to workspace →"}
            </button>
          </form>

          {message && <div className="notice">{message}</div>}

          <div className="login-footer">
            <span>Secure JWT authentication</span>
            <span>•</span>
            <span>FastAPI powered</span>
          </div>
        </section>
      </main>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark small">DF</div>
          <div>
            <strong>DemandFlow</strong>
            <span>AI Intelligence</span>
          </div>
        </div>

        <nav>
          <p className="nav-label">WORKSPACE</p>

          {navItems.map(([id, icon, label]) => (
            <button
              key={id}
              className={`nav-item ${active === id ? "active" : ""}`}
              onClick={() => setActive(id)}
            >
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="api-status">
            <span className="status-dot" />
            <div>
              <strong>API Online</strong>
              <small>localhost:8000</small>
            </div>
          </div>

          <button className="logout-btn" onClick={signOut}>
            ↪ Sign out
          </button>
        </div>
      </aside>

      <main className="dashboard">
        <header className="topbar">
          <div>
            <p className="eyebrow">DEMAND INTELLIGENCE</p>
            <h1>
              {active === "overview"
                ? "Good to see you."
                : navItems.find((item) => item[0] === active)?.[2]}
            </h1>
          </div>

          <div className="user-chip">
            <div className="avatar">
              {username.charAt(0).toUpperCase()}
            </div>
            <div>
              <strong>{username}</strong>
              <small>AI Analyst</small>
            </div>
          </div>
        </header>

        {active === "overview" && (
          <>
            <section className="hero-card">
              <div>
                <span className="hero-tag">AI-POWERED FORECASTING</span>
                <h2>
                  Turn sales history into
                  <br />
                  <span>better inventory decisions.</span>
                </h2>
                <p>
                  Upload your historical sales data, train multiple
                  models and use the best-performing model for
                  demand forecasting.
                </p>

                <button
                  className="hero-btn"
                  onClick={() => setActive("datasets")}
                >
                  Start with your dataset →
                </button>
              </div>

              <div className="hero-visual">
                <div className="chart-card">
                  <div className="chart-header">
                    <span>Demand trend</span>
                    <strong>+18.4%</strong>
                  </div>

                  <div className="mini-bars">
                    {[42, 58, 48, 67, 55, 76, 68, 88, 79, 96].map(
                      (height, index) => (
                        <div
                          key={index}
                          className="mini-bar"
                          style={{ height: `${height}%` }}
                        />
                      )
                    )}
                  </div>

                  <div className="chart-axis">
                    <span>W1</span>
                    <span>W2</span>
                    <span>W3</span>
                    <span>W4</span>
                  </div>
                </div>
              </div>
            </section>

            <section className="kpi-grid">
              <article className="kpi-card purple">
                <div className="kpi-icon">◈</div>
                <span>Best Model</span>
                <strong>{bestModel?.[0] || "—"}</strong>
                <small>Lowest validation RMSE</small>
              </article>

              <article className="kpi-card teal">
                <div className="kpi-icon">≈</div>
                <span>Best RMSE</span>
                <strong>
                  {formatNumber(bestModel?.[1]?.rmse)}
                </strong>
                <small>Temporal holdout evaluation</small>
              </article>

              <article className="kpi-card orange">
                <div className="kpi-icon">◇</div>
                <span>Forecast</span>
                <strong>
                  {forecast !== null
                    ? formatNumber(forecast, 1)
                    : "—"}
                </strong>
                <small>Predicted daily demand</small>
              </article>

              <article className="kpi-card blue">
                <div className="kpi-icon">✓</div>
                <span>Model Versions</span>
                <strong>
                  {versions.filter((v) =>
                    v.version.startsWith("model_")
                  ).length}
                </strong>
                <small>Production-ready versions</small>
              </article>
            </section>

            <section className="two-column">
              <article className="panel-card">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">MODELS</span>
                    <h2>Model comparison</h2>
                  </div>

                  <button
                    className="text-btn"
                    onClick={() => setActive("models")}
                  >
                    View all →
                  </button>
                </div>

                {latestMetrics.length ? (
                  <div className="model-list">
                    {latestMetrics.map(([name, metrics]) => (
                      <div className="model-row" key={name}>
                        <div className="model-name">
                          <div className="model-dot" />
                          <strong>{name}</strong>
                        </div>

                        <div className="model-stat">
                          <span>MAE</span>
                          <strong>{formatNumber(metrics.mae)}</strong>
                        </div>

                        <div className="model-stat">
                          <span>RMSE</span>
                          <strong>{formatNumber(metrics.rmse)}</strong>
                        </div>

                        <div className="model-stat">
                          <span>R²</span>
                          <strong>{formatNumber(metrics.r2, 3)}</strong>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    <span>◈</span>
                    <strong>No trained models yet</strong>
                    <p>Upload a dataset and start training.</p>
                  </div>
                )}
              </article>

              <article className="panel-card insight-card">
                <span className="section-kicker">AI INSIGHT</span>
                <h2>Ready for your next forecast</h2>

                <div className="insight-icon">✦</div>

                <p>
                  {bestModel
                    ? `${bestModel[0]} currently has the lowest validation RMSE at ${formatNumber(
                        bestModel[1].rmse
                      )}.`
                    : "Train multiple models to automatically compare their performance."}
                </p>

                <button
                  className="primary-btn"
                  onClick={() => setActive("forecast")}
                >
                  Open forecasting →
                </button>
              </article>
            </section>
          </>
        )}

        {active === "datasets" && (
          <section className="workspace-grid">
            <article className="panel-card large-panel">
              <span className="section-kicker">DATA PIPELINE</span>
              <h2>Upload historical sales</h2>
              <p className="panel-description">
                Upload a CSV containing at minimum date, product_id
                and sales. Optional business features can improve
                forecasting performance.
              </p>

              <div className="upload-zone">
                <div className="upload-icon">↑</div>
                <strong>
                  {file ? file.name : "Choose your sales CSV"}
                </strong>
                <span>
                  CSV only · Maximum supported file size: 25 MB
                </span>

                <input
                  type="file"
                  accept=".csv"
                  onChange={(event) =>
                    setFile(event.target.files?.[0] || null)
                  }
                />

                <button
                  className="primary-btn"
                  onClick={() => void upload()}
                  disabled={!file || busy}
                >
                  {busy ? "Uploading..." : "Validate & upload"}
                </button>
              </div>

              {dataset && (
                <div className="dataset-meta">
                  <div>
                    <span>Dataset ID</span>
                    <strong>{dataset}</strong>
                  </div>
                  <div>
                    <span>Status</span>
                    <strong>
                      {needsMapping
                        ? "Mapping required"
                        : "Ready for training"}
                    </strong>
                  </div>
                </div>
              )}
            </article>

            {dataset && needsMapping && (
              <article className="panel-card large-panel">
                <span className="section-kicker">SCHEMA MAPPING</span>
                <h2>Map required columns</h2>
                <p className="panel-description">
                  Connect your CSV headers to the standardized
                  forecasting schema.
                </p>

                {(["date", "product_id", "sales"] as Field[]).map(
                  (field) => (
                    <label key={field}>
                      {field === "product_id"
                        ? "Product ID"
                        : field === "date"
                        ? "Date"
                        : "Sales"}
                      <select
                        value={mapping[field]}
                        onChange={(event) =>
                          setMapping({
                            ...mapping,
                            [field]: event.target.value,
                          })
                        }
                      >
                        {columns.map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </label>
                  )
                )}

                <button
                  className="primary-btn"
                  onClick={() => void mapColumns()}
                  disabled={busy}
                >
                  Validate mapping →
                </button>
              </article>
            )}

            {canEdit && dataset && !needsMapping && (
              <article className="panel-card training-card">
                <span className="section-kicker">MACHINE LEARNING</span>
                <h2>Train model pipeline</h2>
                <p>
                  Chronological validation with Linear Regression,
                  Random Forest and XGBoost.
                </p>

                <div className="training-steps">
                  <span>01 Data validation</span>
                  <span>02 Feature engineering</span>
                  <span>03 Model comparison</span>
                  <span>04 Versioning</span>
                </div>

                <button
                  className="primary-btn"
                  onClick={() => void train()}
                  disabled={busy}
                >
                  {busy ? "Training..." : "Train & compare models →"}
                </button>
              </article>
            )}
          </section>
        )}

        {active === "models" && (
          <section className="panel-card large-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">MODEL REGISTRY</span>
                <h2>Model performance</h2>
              </div>

              <button
                className="secondary-btn"
                onClick={() => void refresh()}
              >
                ↻ Refresh
              </button>
            </div>

            {versions.filter((v) =>
              v.version.startsWith("model_")
            ).length ? (
              <div className="model-table">
                <div className="table-header">
                  <span>Version</span>
                  <span>Model</span>
                  <span>MAE</span>
                  <span>RMSE</span>
                  <span>R²</span>
                </div>

                {versions
                  .filter((v) => v.version.startsWith("model_"))
                  .map((version) => {
                    const metrics = metricEntries(version);
                    const metric = metrics[0]?.[1];

                    return (
                      <div className="table-row" key={version.version}>
                        <span className="version-pill">
                          {version.version}
                        </span>
                        <strong>{version.model_name}</strong>
                        <span>{formatNumber(metric?.mae)}</span>
                        <span>{formatNumber(metric?.rmse)}</span>
                        <span>{formatNumber(metric?.r2, 3)}</span>
                      </div>
                    );
                  })}
              </div>
            ) : (
              <div className="empty-state">
                <span>◈</span>
                <strong>No production model versions</strong>
                <p>Train your first model from the Datasets section.</p>
              </div>
            )}
          </section>
        )}

        {active === "forecast" && (
          <section className="workspace-grid">
            <article className="panel-card forecast-main">
              <span className="section-kicker">DEMAND FORECAST</span>
              <h2>Generate prediction</h2>
              <p className="panel-description">
                Use the latest saved model to generate a daily
                demand prediction.
              </p>

              <div className="forecast-value">
                <span>Predicted daily demand</span>
                <strong>
                  {forecast !== null
                    ? formatNumber(forecast, 1)
                    : "—"}
                </strong>
                <small>units / day</small>
              </div>

              <button
                className="primary-btn"
                onClick={() => void runForecast()}
                disabled={!latestVersion || busy}
              >
                {busy ? "Generating..." : "Generate forecast →"}
              </button>

              {dailyForecasts.length > 0 && (
                <div className="forecast-results">
                  <h3>7-Day Forecast</h3>
                  <p className="panel-description">
                    Total predicted demand:{" "}
                    <strong>{formatNumber(totalDemand ?? undefined, 1)}</strong>{" "}
                    units
                  </p>

                  <div className="table-container">
                    <div className="table-row">
                      <strong>Date</strong>
                      <strong>Predicted sales</strong>
                    </div>
                    {dailyForecasts.map((item) => (
                      <div className="table-row" key={item.date}>
                        <span>{item.date}</span>
                        <strong>{formatNumber(item.predicted_sales, 1)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </article>

            <article className="panel-card">
              <span className="section-kicker">MODEL CONTEXT</span>
              <h2>Current production model</h2>

              <div className="detail-list">
                <div>
                  <span>Model</span>
                  <strong>{bestModel?.[0] || "—"}</strong>
                </div>

                <div>
                  <span>Version</span>
                  <strong>
                    {latestVersion?.version || "No version"}
                  </strong>
                </div>

                <div>
                  <span>RMSE</span>
                  <strong>
                    {formatNumber(bestModel?.[1]?.rmse)}
                  </strong>
                </div>

                <div>
                  <span>R²</span>
                  <strong>
                    {formatNumber(bestModel?.[1]?.r2, 3)}
                  </strong>
                </div>
              </div>
            </article>
          </section>
        )}

        {active === "inventory" && (
          <section className="workspace-grid">
            <article className="panel-card">
              <span className="section-kicker">INVENTORY INTELLIGENCE</span>
              <h2>Optimize reorder quantity</h2>
              <p className="panel-description">
                Calculate safety stock and reorder recommendations
                from predicted demand and lead time.
              </p>

              <div className="form-grid">
                <label>
                  Current stock
                  <input
                    type="number"
                    value={currentStock}
                    onChange={(event) =>
                      setCurrentStock(event.target.value)
                    }
                  />
                </label>

                <label>
                  Lead time (days)
                  <input
                    type="number"
                    value={leadTime}
                    onChange={(event) =>
                      setLeadTime(event.target.value)
                    }
                  />
                </label>

                <label>
                  Daily demand std.
                  <input
                    type="number"
                    value={demandStd}
                    onChange={(event) =>
                      setDemandStd(event.target.value)
                    }
                  />
                </label>
              </div>

              <button
                className="primary-btn"
                onClick={() => void optimizeInventory()}
                disabled={forecast === null || busy}
              >
                Calculate recommendation →
              </button>
            </article>

            <article className="panel-card">
              <span className="section-kicker">RECOMMENDATION</span>
              <h2>Inventory decision</h2>

              {inventoryResult ? (
                <div className="inventory-result">
                  <div className="big-number">
                    {formatNumber(
                      inventoryResult.recommended_order_quantity,
                      0
                    )}
                    <small>units to reorder</small>
                  </div>

                  <div className="result-grid">
                    <div>
                      <span>Safety stock</span>
                      <strong>
                        {formatNumber(
                          inventoryResult.safety_stock,
                          1
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Reorder point</span>
                      <strong>
                        {formatNumber(
                          inventoryResult.reorder_point,
                          1
                        )}
                      </strong>
                    </div>
                  </div>

                  <div
                    className={`risk-badge ${
                      inventoryResult.stockout_risk
                        ? "risk"
                        : "safe"
                    }`}
                  >
                    {inventoryResult.stockout_risk
                      ? "⚠ Stockout risk detected"
                      : "✓ Stock level looks healthy"}
                  </div>
                </div>
              ) : (
                <div className="empty-state compact">
                  <span>◇</span>
                  <p>Generate a forecast to calculate inventory.</p>
                </div>
              )}
            </article>
          </section>
        )}

        {active === "whatif" && (
          <section className="workspace-grid">
            <article className="panel-card">
              <span className="section-kicker">SCENARIO ANALYSIS</span>
              <h2>What-if simulation</h2>
              <p className="panel-description">
                Test how the configured price and promotion
                assumptions change the forecast.
              </p>

              <div className="form-grid">
                <label>
                  Price change %
                  <input
                    type="number"
                    value={priceChange}
                    onChange={(event) =>
                      setPriceChange(event.target.value)
                    }
                  />
                </label>

                <label>
                  Promotion %
                  <input
                    type="number"
                    value={promotion}
                    onChange={(event) =>
                      setPromotion(event.target.value)
                    }
                  />
                </label>
              </div>

              <button
                className="primary-btn"
                onClick={() => void runWhatIf()}
                disabled={forecast === null || busy}
              >
                Run scenario →
              </button>
            </article>

            <article className="panel-card scenario-card">
              <span className="section-kicker">SCENARIO RESULT</span>
              <h2>Baseline vs scenario</h2>

              {whatIfResult ? (
                <>
                  <div className="scenario-values">
                    <div>
                      <span>Baseline</span>
                      <strong>
                        {formatNumber(
                          whatIfResult.baseline_prediction,
                          1
                        )}
                      </strong>
                    </div>

                    <div className="scenario-arrow">→</div>

                    <div>
                      <span>Scenario</span>
                      <strong>
                        {formatNumber(
                          whatIfResult.scenario_prediction,
                          1
                        )}
                      </strong>
                    </div>
                  </div>

                  <p className="assumption">
                    Assumption:{" "}
                    {whatIfResult.assumption ||
                      "Configured business elasticity assumptions."}
                  </p>
                </>
              ) : (
                <div className="empty-state compact">
                  <span>✦</span>
                  <p>Run a scenario after generating a forecast.</p>
                </div>
              )}
            </article>
          </section>
        )}

        {message && (
          <div className="toast" role="status">
            <span>●</span>
            {message}
          </div>
        )}

        <footer>
          DemandFlow AI · Demand Forecasting Platform ·{" "}
          <a href={`${API}/docs`} target="_blank">
            API Documentation
          </a>
        </footer>
      </main>
    </div>
  );
}




