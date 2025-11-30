from __future__ import annotations
import logging
import os
from typing import Any, Dict, Optional, List

from src.core.settings import get_settings

log = logging.getLogger("clearml")

try:
    from clearml import Task, Model
except Exception:
    Task = None
    Model = None


def _export_clearml_env():
    s = get_settings()
    os.environ.setdefault(
        "CLEARML_API_HOST",
        s.clearml_api_host
    )
    if s.clearml_api_access_key:
        os.environ.setdefault(
            "CLEARML_API_ACCESS_KEY",
            s.clearml_api_access_key
        )


def start_task(
    project: str,
    name: str,
    params: Optional[Dict[str, Any]] = None
):
    """
    Start ClearML task

    Returns:
        New task or None
    """
    if Task is None:
        log.warning("ClearML SDK is not available; skipping Task.init()")
        return None
    try:
        _export_clearml_env()
        task = Task.init(
            project_name=project,
            task_name=name,
            task_type=Task.TaskTypes.training
        )

        if params:
            task.connect(params)
        
        log.info(
            "ClearML task started: %s / %s (id=%s)",
            project,
            name,
            getattr(task, "id", "n/a")
        )
        return task
    except Exception as e:
        log.warning("Failed to start ClearML task: %s", e)
        return None


def log_metrics(task, metrics: Dict[str, float]):
    """
    Log metrics to ClearML
    """
    if task is None:
        return
    try:
        logger = task.get_logger()

        for k, v in metrics.items():
            logger.report_single_value(k, float(v))
    except Exception as e:
        log.warning("Failed to log metrics to ClearML: %s", e)


def register_model(task, local_path: str, model_name: str) -> Optional[str]:
    """
    Register model in ClearML

    Returns:
        Model ID or None
    """
    if task is None or Model is None:
        return None
    try:
        M = Model

        model = M(name=model_name, framework="scikit-learn")  # type: ignore[call-arg]

        attach = getattr(model, "connect", None)
        if callable(attach):
            try:
                attach(task)
            except Exception:
                pass

        upd = getattr(model, "update_weights", None)
        if callable(upd):
            upd(local_path)
        
        upl = getattr(model, "upload", None)
        if callable(upl):
            upl()
        
        pub = getattr(model, "publish", None)
        if callable(pub):
            pub()
        
        model_id = getattr(model, "id", None)
        log.info(
            "ClearML model published: name=%s id=%s",
            model_name,
            model_id
        )
        return str(model_id) if model_id else None
    except Exception as e:
        log.warning("Failed to register model in ClearML: %s", e)
        return None


def list_models(project: Optional[str] = None, name_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get models from ClearML

    Returns:
        List of models or empty list
    """
    if Model is None:
        log.warning("ClearML SDK is not available; list_models() returns empty list")
        return []
    try:
        _export_clearml_env()
        items = Model.query_models(
            project_name=project,
            model_name=name_prefix,
            only_published=True
        )
        out: List[Dict[str, Any]] = []
    
        for m in items:
            try:
                out.append({
                    "id": m.id,
                    "name": m.name,
                    "framework": getattr(m, "framework", None),
                    "project": getattr(m, "project", None),
                    "published": True,
                })
            except Exception:
                continue
    
        return out
    except Exception as e:
        log.warning("Failed to query ClearML models: %s", e)
        return []


def download_model(model_id: str) -> Optional[str]:
    """
    Download ClearML model weights locally and return file path (best-effort).

    Returns:
        Local file path or None
    """
    if Model is None:
        log.warning("ClearML SDK is not available; cannot download model")
        return None
    try:
        _export_clearml_env()
        m = Model(model_id=model_id)
        path = m.get_local_copy()

        if not path:
            log.warning(
                "ClearML returned empty local path for model_id=%s",
                model_id
            )
            return None

        return path
    except Exception as e:
        log.warning("Failed to download ClearML model %s: %s", model_id, e)
        return None
