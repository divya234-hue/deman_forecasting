from __future__ import annotations
import json, os, sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH=Path(__file__).resolve().parents[1]/"data"/"platform.db"
DB_PATH.parent.mkdir(parents=True,exist_ok=True)

def _connect():
    url=os.getenv("DATABASE_URL")
    if url:
        try:
            from sqlalchemy import create_engine
            engine=create_engine(url,pool_pre_ping=True)
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE IF NOT EXISTS training_runs (run_id VARCHAR(64) PRIMARY KEY, dataset_id VARCHAR(64), status VARCHAR(32), payload TEXT NOT NULL, created_at VARCHAR(64) NOT NULL)")
                connection.exec_driver_sql("CREATE TABLE IF NOT EXISTS model_versions (version VARCHAR(128) PRIMARY KEY, model_name VARCHAR(128), model_path TEXT, metrics TEXT, created_at VARCHAR(64) NOT NULL)")
            return engine
        except ImportError as exc:
            raise RuntimeError("Install SQLAlchemy and psycopg[binary] to use DATABASE_URL.") from exc
    connection=sqlite3.connect(DB_PATH)
    connection.row_factory=sqlite3.Row
    connection.execute("CREATE TABLE IF NOT EXISTS training_runs (run_id TEXT PRIMARY KEY, dataset_id TEXT, status TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS model_versions (version TEXT PRIMARY KEY, model_name TEXT, model_path TEXT, metrics TEXT, created_at TEXT NOT NULL)")
    return connection

def _is_engine(connection): return hasattr(connection,"begin") and not isinstance(connection,sqlite3.Connection)

def save_run(run:dict):
    conn=_connect(); now=datetime.now(timezone.utc).isoformat()
    values=(run["run_id"],run.get("dataset_id"),run["status"],json.dumps(run),now)
    if _is_engine(conn):
        from sqlalchemy import text
        with conn.begin() as connection:
            connection.execute(text("INSERT INTO training_runs VALUES (:id,:dataset,:status,:payload,:created) ON CONFLICT (run_id) DO UPDATE SET status=EXCLUDED.status,payload=EXCLUDED.payload"),dict(zip(("id","dataset","status","payload","created"),values)))
        conn.dispose()
    else:
        try:
            with conn: conn.execute("INSERT OR REPLACE INTO training_runs VALUES (?,?,?,?,?)",values)
        finally: conn.close()

def get_run(run_id:str):
    conn=_connect()
    if _is_engine(conn):
        from sqlalchemy import text
        with conn.connect() as connection: row=connection.execute(text("SELECT payload FROM training_runs WHERE run_id=:id"),{"id":run_id}).first()
        conn.dispose(); return json.loads(row[0]) if row else None
    try:
        row=conn.execute("SELECT payload FROM training_runs WHERE run_id=?",(run_id,)).fetchone()
        return json.loads(row[0]) if row else None
    finally: conn.close()

def save_model_version(version:str,model_name:str,model_path:str,metrics:dict):
    conn=_connect()
    values=(version,model_name,model_path,json.dumps(metrics),datetime.now(timezone.utc).isoformat())
    if _is_engine(conn):
        from sqlalchemy import text
        with conn.begin() as connection:
            connection.execute(text("INSERT INTO model_versions VALUES (:v,:n,:p,:m,:d) ON CONFLICT (version) DO UPDATE SET model_name=EXCLUDED.model_name,model_path=EXCLUDED.model_path,metrics=EXCLUDED.metrics"),dict(zip(("v","n","p","m","d"),values)))
        conn.dispose()
    else:
        try:
            with conn: conn.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?)",values)
        finally: conn.close()

def list_model_versions():
    conn=_connect()
    if _is_engine(conn):
        from sqlalchemy import text
        with conn.connect() as connection: rows=connection.execute(text("SELECT * FROM model_versions ORDER BY created_at DESC")).mappings().all()
        conn.dispose(); result=[dict(row) for row in rows]
    else:
        try: result=[dict(row) for row in conn.execute("SELECT * FROM model_versions ORDER BY created_at DESC")]
        finally: conn.close()
    for row in result: row["metrics"]=json.loads(row["metrics"])
    return result
