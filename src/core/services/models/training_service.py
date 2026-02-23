from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.core.services.dataset.dataset_service import load_df
from src.core.services.models.model_store import (
    save_model as store_save_model,
    load_model as store_load_model
)
from src.core.services.models.clearml_integration import (
    start_task as cl_start_task,
    log_metrics as cl_log_metrics,
    register_model as cl_register_model,
)
from src.models.registry import build_model
from src.core.services.dataset.dataset_service import (
    DatasetNotFoundError,
    DatasetReadError,
)

class TargetColumnNotFoundError(ValueError):
    """Target column is not present in the dataset."""
    def __init__(self, column: str):
        super().__init__(f"Target column {column} not found")
        self.column = column


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return preprocessor, list(X.columns)


def train_model(
    *,
    dataset_id: str,
    target_column: str,
    model_key: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
    model_id: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Train a model and return (model_id, metrics).

    Raises:
        DatasetNotFoundError: dataset file not found
        DatasetReadError: failed to read dataset
        TargetColumnNotFoundError: target column missing
    """
    try:
        df = load_df(dataset_id)
    except (DatasetNotFoundError, DatasetReadError):
        raise

    if target_column not in df.columns:
        raise TargetColumnNotFoundError(target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessor, feature_order = _build_preprocessor(X)
    model = build_model(model_key, hyperparams or {})
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, y_pred))}

    meta = {
        "model_key": model_key,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "features": feature_order,
        "metrics": metrics,
        "hyperparams": (hyperparams or {}),
        "test_size": test_size,
        "shuffle": shuffle,
        "random_state": random_state,
    }

    mid = store_save_model(pipe, meta, model_id=model_id)

    # ClearML best-effort; do not fail training if ClearML is misconfigured
    try:
        task = cl_start_task(
            project="MLOps-HW01",
            name=f"train/{model_key}",
            params={
                "model_key": model_key,
                "dataset_id": dataset_id,
                "target_column": target_column,
                "hyperparams": hyperparams or {},
                "test_size": test_size,
                "shuffle": shuffle,
                "random_state": random_state,
            },
        )
        cl_log_metrics(task, metrics)
        _, saved_meta = store_load_model(mid)
        local_path = saved_meta.get("path") if isinstance(saved_meta, dict) else None
        if local_path:
            cl_register_model(task, local_path=local_path, model_name=f"{model_key}-{mid[:8]}")
        if task is not None:
            try:
                task.close()
            except Exception:
                pass
    except Exception:
        pass

    return mid, metrics
