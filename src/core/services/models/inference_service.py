from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.services.models.model_store import load_model as store_load_model

class NoInstancesError(ValueError):
    """Inference called without instances."""

def predict(
    *,
    model_id: str,
    instances: List[Dict[str, Any]]
) -> Tuple[List[Any], Optional[List[List[float]]]]:
    """
    Run inference for the given model.

    Returns:
        (predictions, probabilities|None)

    Raises:
        ModelNotFoundError: model not found
        NoInstancesError: empty instances
    """
    if not instances:
        raise NoInstancesError("No instances provided")

    pipe, meta = store_load_model(model_id)
    feats: List[str] = meta.get("features", [])

    X = pd.DataFrame(instances)
    if feats:
        for c in feats:
            if c not in X.columns:
                X[c] = None
        X = X[feats]

    preds = pipe.predict(X)
    probabilities: Optional[List[List[float]]] = None

    try:
        estimator = getattr(pipe, "named_steps", {}).get("model")

        if estimator is not None and hasattr(estimator, "predict_proba"):
            Xt = pipe.named_steps["prep"].transform(X)
            proba = estimator.predict_proba(Xt)
            probabilities = proba.tolist()
    except Exception:
        probabilities = None

    return preds.tolist(), probabilities
