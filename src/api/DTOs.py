from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class DatasetInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: str
    filename: str
    size_bytes: int
    created_at: int
    original_filename: Optional[str] = None

class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_key: str = Field(..., description="Ключ модели: logistic_regression | random_forest")
    dataset_id: str
    target_column: str = Field(default="target")
    hyperparams: Optional[Dict[str, Any]] = None
    test_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42
    model_id: Optional[str] = Field(default=None, description="Если задан — переобучение существующей модели (overwrite)")

class TrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    model_key: str
    metrics: Dict[str, float]

class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    instances: List[Dict[str, Any]] = Field(..., description="Список объектов: {feature: value}")

class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None

class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    model_key: str
    dataset_id: str
    target_column: str
    created_at: str
    metrics: Optional[Dict[str, float]] = None
