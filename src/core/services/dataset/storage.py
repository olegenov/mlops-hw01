from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

DATASETS_DIR = Path("data/datasets")
MODELS_DIR = Path("artifacts/models")

def ensure_dirs():
    """
    Make dirs if needed.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def dataset_path(dataset_id: str) -> Path:
    """
    Dataset path for id.
    """
    return DATASETS_DIR / f"{dataset_id}.csv"

def list_datasets() -> List[Dict]:
    """
    List datasets.
    """
    ensure_dirs()
    items: List[Dict] = []
    for p in sorted(DATASETS_DIR.glob("*.csv")):
        stat = p.stat()
        orig_name = None
        meta_path = Path(str(p) + ".meta.json")

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text() or "{}")
                orig_name = meta.get("original_filename") or None
            except Exception:
                orig_name = None

        items.append({
            "id": p.stem,
            "filename": p.name,
            "size_bytes": stat.st_size,
            "created_at": int(stat.st_mtime),
            "original_filename": orig_name,
        })
    return items

def delete_dataset(dataset_id: str) -> bool:
    """
    Delete dataset.
    """
    p = dataset_path(dataset_id)
    existed = False

    if p.exists():
        p.unlink()
        existed = True
        meta_path = Path(str(p) + ".meta.json")
        if meta_path.exists():
            meta_path.unlink()

    return existed
