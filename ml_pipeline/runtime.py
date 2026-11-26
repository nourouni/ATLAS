from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from .paths import PROJECT_ROOT


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline input is missing: {path.relative_to(PROJECT_ROOT)}")


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists() and source.is_file():
        if source.resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_tree_contents(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            copy_if_exists(path, target / path.relative_to(source))


def promote_product_ownership_artifacts() -> None:
    from .paths import ARTIFACTS_ROOT, PRODUCT_SIM_OUTPUTS

    target_dir = ARTIFACTS_ROOT / "lookup_data" / "ownership"
    for name in [
        "product_ownership_pairs.parquet",
        "product_ownership_features.parquet",
        "product_ownership_long.parquet",
    ]:
        copy_if_exists(PRODUCT_SIM_OUTPUTS / name, target_dir / name)


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_from_dir(name: str, path: Path, module_dir: Path):
    sys.path.insert(0, str(module_dir))
    try:
        return import_module_from_path(name, path)
    finally:
        if sys.path and sys.path[0] == str(module_dir):
            sys.path.pop(0)


def database_loader():
    database_dir = PROJECT_ROOT / "database"
    return import_from_dir("pipeline_db_loader", database_dir / "load_operational_data.py", database_dir)


def with_db_transaction(work: Callable[[Any, Any], dict[str, Any]], skip_schema: bool) -> dict[str, Any]:
    db = database_loader()
    with db.connect() as conn:
        with conn.transaction():
            if not skip_schema:
                db.apply_schema(conn)
            counts = work(db, conn)
        counts["table_counts"] = db.verify_counts(conn)
        return counts
