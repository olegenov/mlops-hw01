from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Type, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ModelSpec:
    """Model description and deafult params"""

    key: str
    cls: Type[Any]
    defaults: Dict[str, Any]

    def build(self, params: Optional[Dict[str, Any]] = None) -> Any:
        params = params or {}
        allowed = set(self.defaults.keys())
        clean = {k: v for k, v in params.items() if k in allowed}
        merged = {**self.defaults, **clean}

        return self.cls(**merged)


class ModelRegistry:
    """Supported models registry"""

    def __init__(self):
        self._specs: Dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec):
        self._specs[spec.key] = spec

    def list_keys(self) -> List[str]:
        return sorted(self._specs.keys())

    def get(self, key: str) -> ModelSpec:
        if key not in self._specs:
            raise KeyError(f"Unknown model_key: {key}")

        return self._specs[key]

    def defaults(self, key: str) -> Dict[str, Any]:
        return dict(self.get(key).defaults)

    def build(
        self,
        key: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.get(key).build(params)


# Registry initialization

registry = ModelRegistry()
registry.register(
    ModelSpec(
        key="logistic_regression",
        cls=LogisticRegression,
        defaults={"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
    )
)
registry.register(
    ModelSpec(
        key="random_forest",
        cls=RandomForestClassifier,
        defaults={"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": -1},
    )
)

# Registry API

def list_model_classes() -> List[str]:
    return registry.list_keys()


def get_default_hyperparams(model_key: str) -> Dict[str, Any]:
    return registry.defaults(model_key)


def build_model(model_key: str, params: Dict[str, Any]) -> Any:
    return registry.build(model_key, params)
