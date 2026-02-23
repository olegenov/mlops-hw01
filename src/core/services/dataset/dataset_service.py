from __future__ import annotations

import io
import json
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd

from src.core.services.dataset.storage import (
    dataset_path,
    list_datasets as storage_list_datasets,
    delete_dataset as storage_delete_dataset,
)
from src.core.services.dataset.dvc_integration import (
    add_and_push as dvc_add_and_push,
    remove_output as dvc_remove_output,
)

class DatasetNotFoundError(FileNotFoundError):
    """Dataset file is missing."""

class DatasetReadError(ValueError):
    """Failed to load dataset file or parse its contents."""

class DatasetParseError(ValueError):
    """Failed to parse uploaded dataset (CSV/JSON)."""


class EmptyDatasetError(ValueError):
    """Parsed dataset is empty."""


def upload(file_bytes: bytes, filename: Optional[str], content_type: Optional[str]) -> Dict[str, Any]:
    """
    Parse CSV/JSON, persist to data/datasets and perform DVC add+push.

    Returns:
        {"dataset_id": str, "rows": int, "cols": int}

    Raises:
        DatasetParseError, EmptyDatasetError
    """
    suffix = (filename or "").lower()
    try:
        if (content_type and "json" in content_type) or suffix.endswith(".json"):
            data = json.loads(file_bytes.decode("utf-8"))
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise DatasetParseError(str(e))

    if df.empty:
        raise EmptyDatasetError("Empty dataset")

    ds_id = uuid.uuid4().hex
    p = dataset_path(ds_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

    try:
        meta_path = Path(str(p) + ".meta.json")
        meta = {"original_filename": filename or None}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception:
        pass

    dvc_add_and_push(p)

    return {
        "dataset_id": ds_id,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "original_filename": filename or None,
    }


def list_datasets() -> List[Dict[str, Any]]:
    """Return datasets metadata from storage layer."""
    return storage_list_datasets()


def delete_dataset(dataset_id: str) -> bool:
    """
    Untrack dataset in DVC (if present) and remove local file.

    Returns:
        True if removed, False if not found.
    """
    p = dataset_path(dataset_id)
    dvc_remove_output(p)
    return storage_delete_dataset(dataset_id)


def load_df(dataset_id: str) -> pd.DataFrame:
   """
   Load dataset as DataFrame; raise typed errors on failure.

   Raises:
       DatasetNotFoundError, DatasetReadError
   """
   p = dataset_path(dataset_id)
   if not p.exists():
       raise DatasetNotFoundError()
   try:
       return pd.read_csv(p)
   except Exception as e:
       raise DatasetReadError(f"{e}")


__all__ = [
   "DatasetParseError",
   "EmptyDatasetError",
   "upload",
   "list_datasets",
   "delete_dataset",
   "load_df",
]
