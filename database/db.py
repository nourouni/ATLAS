from __future__ import annotations

import os
from pathlib import Path

import psycopg

REQUIRED_ENV_VARS = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def connect() -> psycopg.Connection:
    load_dotenv()
    load_dotenv(Path(__file__).with_name(".env"))
    missing = [key for key in REQUIRED_ENV_VARS if not os.environ.get(key)]
    if missing:
        env_paths = [Path(".env"), Path(__file__).with_name(".env")]
        raise RuntimeError(
            "Missing PostgreSQL environment variables: "
            + ", ".join(missing)
            + ". Create .env or database/.env from database/.env.example and set your Docker PostgreSQL values. "
            + "Checked: "
            + ", ".join(str(path) for path in env_paths)
        )
    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        sslmode=os.environ.get("POSTGRES_SSLMODE", "prefer"),
    )
