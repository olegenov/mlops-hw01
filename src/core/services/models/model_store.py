from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
from joblib import dump, load

INDEX_PATH = Path("artifacts/models/index.json")
MODELS_DIR = Path("artifacts/models")


class ModelNotFoundError(KeyError):
    """Model not found in registry/store."""


def _ensure():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(
            json.dumps({}, ensure_ascii=False, indent=2)
        )


def _read_index() -> Dict[str, Dict[str, Any]]:
    _ensure()
    return json.loads(
        INDEX_PATH.read_text() or "{}"
    )


def _write_index(index: Dict[str, Dict[str, Any]]):
    INDEX_PATH.write_text(
        json.dumps(
            index,
            ensure_ascii=False,
            indent=2
        )
    )


def model_path(model_id: str) -> Path:
    """
    Model path for id.
    """
    return MODELS_DIR / f"{model_id}.joblib"


def save_model(pipeline: Any, meta: Dict[str, Any], model_id: Optional[str] = None) -> str:
    """
    Save model on local store.
    """
    _ensure()
    index = _read_index()
    mid = model_id or uuid4().hex
    path = model_path(mid)
    dump(pipeline, path)
    payload = {
        **meta,
        "model_id": mid,
        "path": str(path),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    index[mid] = payload
    _write_index(index)

    return mid


def load_model(model_id: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model from local store.

    Raises:
        KeyError
    """
    _ensure()
    index = _read_index()

    if model_id not in index:
        raise  ModelNotFoundError(f"model_id not found: {model_id}")

    meta = index[model_id]
    pipe = load(model_path(model_id))
    return pipe, meta


def list_models() -> List[Dict[str, Any]]:
    """
    Get list of local models.
    """
    _ensure()
    idx = _read_index()
    return list(idx.values())


def delete_model(model_id: str) -> bool:
    """
    Delete model from local store.
    """
    _ensure()
    index = _read_index()
    existed = False
    if model_id in index:
        existed = True
        index.pop(model_id, None)
        _write_index(index)
    p = model_path(model_id)
    if p.exists():
        existed = True
        p.unlink()
    return existed
