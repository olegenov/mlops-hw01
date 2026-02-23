from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import requests


class ApiError(RuntimeError):
    """API request failed."""


class ApiClient:
    """Thin HTTP client for REST API."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

    # System
    def health(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        self._raise_for_status(r)
        return r.json()

    # Datasets
    def list_datasets(self) -> List[Dict[str, Any]]:
        r = self.session.get(f"{self.base_url}/datasets", timeout=self.timeout)
        self._raise_for_status(r)
        return r.json()

    def upload_dataset(self, filename: str, content: bytes) -> Dict[str, Any]:
        files = {"file": (filename, content, "application/octet-stream")}
        r = self.session.post(f"{self.base_url}/datasets", files=files, timeout=max(self.timeout, 120.0))
        self._raise_for_status(r)
        return r.json()

    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        r = self.session.delete(f"{self.base_url}/datasets/{dataset_id}", timeout=self.timeout)
        self._raise_for_status(r)
        return r.json()

    # Models
    def list_model_classes(self) -> List[str]:
        r = self.session.get(f"{self.base_url}/models/classes", timeout=self.timeout)
        self._raise_for_status(r)
        data = r.json()
        return data.get("classes", [])

    def list_models(self) -> List[Dict[str, Any]]:
        r = self.session.get(f"{self.base_url}/models", timeout=self.timeout)
        self._raise_for_status(r)
        return r.json()

    def train(
        self,
        *,
        model_key: str,
        dataset_id: str,
        target_column: str = "target",
        hyperparams: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model_key": model_key,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "hyperparams": hyperparams or {},
            "test_size": float(test_size),
            "shuffle": bool(shuffle),
            "random_state": int(random_state),
            "model_id": model_id,
        }
        r = self.session.post(f"{self.base_url}/train", json=payload, timeout=max(self.timeout, 600.0))
        self._raise_for_status(r)
        return r.json()

    def predict(self, *, model_id: str, instances: List[Dict[str, Any]]) -> Tuple[List[Any], Optional[List[List[float]]]]:
        r = self.session.post(
            f"{self.base_url}/predict",
            json={"model_id": model_id, "instances": instances},
            timeout=max(self.timeout, 60.0),
        )
        self._raise_for_status(r)
        data = r.json()
        return data.get("predictions", []), data.get("probabilities")

    # Internals
    @staticmethod
    def _raise_for_status(r: requests.Response):
        if 200 <= r.status_code < 300:
            return
        try:
            payload = r.json()
        except Exception:
            payload = {"detail": r.text}
        raise ApiError(f"HTTP {r.status_code}: {payload}")