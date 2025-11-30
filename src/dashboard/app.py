from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st

from src.dashboard.client import ApiClient, ApiError

API_BASE_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="MLOps Dashboard", layout="wide")
st.title("MLOps Dashboard")


def _get_api_base() -> str:
    return st.session_state.get("api_base", API_BASE_DEFAULT)


def _set_api_base(value: str):
    st.session_state["api_base"] = (value or "").strip() or API_BASE_DEFAULT


def _client() -> ApiClient:
    return ApiClient(_get_api_base(), timeout=10.0)


def _json_text_area(label: str, value: Any, height: int = 140) -> Optional[Any]:
    text = st.text_area(label, value=json.dumps(value, ensure_ascii=False, indent=2), height=height)
    try:
        return json.loads(text) if text.strip() else None
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return None


with st.sidebar:
    st.header("Settings")
    init_value = _get_api_base()
    api_base = st.text_input("REST API base URL", value=init_value)
    if api_base != init_value:
        _set_api_base(api_base)

    if st.button("Check health"):
        try:
            info = _client().health()
            st.success(f"Health OK: {info}")
        except Exception as e:
            st.error(f"Health error: {e}")


def render_datasets_tab():
    st.subheader("Datasets")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Refresh", key="ds_refresh"):
            st.session_state["_ds_ts"] = True
        try:
            items = _client().list_datasets()
            st.dataframe(items, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to list datasets: {e}")

    with c2:
        up = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if up is not None and st.button("Upload", key="ds_upload"):
            try:
                res = _client().upload_dataset(up.name, up.getvalue())
                st.success(res)
            except Exception as e:
                st.error(f"Upload failed: {e}")

        st.divider()
        try:
            ids = [d["id"] for d in _client().list_datasets()]
        except Exception:
            ids = []
        selected = st.selectbox(
            "dataset_id",
            options=ids,
            key="ds_delete_dataset_id"
        )
        if st.button("Delete dataset") and selected:
            try:
                res = _client().delete_dataset(selected)
                st.success(res)
            except Exception as e:
                st.error(f"Delete failed: {e}")


def render_train_tab():
    st.subheader("Training")
    try:
        classes = _client().list_model_classes()
    except Exception as e:
        classes = []
        st.error(f"Failed to fetch model classes: {e}")

    try:
        ds_ids = [d["id"] for d in _client().list_datasets()]
    except Exception as e:
        ds_ids = []
        st.error(f"Failed to fetch datasets: {e}")

    c1, c2, c3 = st.columns(3)
    with c1:
        model_key = st.selectbox(
            "model_key",
            options=classes or ["logistic_regression", "random_forest"],
            key="train_model_key"
        )
    with c2:
        dataset_id = st.selectbox(
            "dataset_id",
            options=ds_ids,
            key="train_dataset_id"
        )
    with c3:
        target_col = st.text_input("target_column", value="target")

    default_hp = {
        "logistic_regression": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
        "random_forest": {"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": -1},
    }
    hp_obj = _json_text_area("hyperparams", default_hp.get(model_key, {}), height=160)

    c4, c5, c6 = st.columns(3)
    with c4:
        test_size = st.number_input("test_size", value=0.2, min_value=0.05, max_value=0.5, step=0.05)
    with c5:
        shuffle = st.checkbox("shuffle", value=True)
    with c6:
        random_state = st.number_input("random_state", value=42, step=1)

    model_id_override = st.text_input("model_id (for retrain)", value="")

    if st.button("Train"):
        if hp_obj is None:
            st.error("Invalid hyperparams JSON")
        else:
            try:
                res = _client().train(
                    model_key=model_key,
                    dataset_id=dataset_id,
                    target_column=target_col,
                    hyperparams=hp_obj or {},
                    test_size=float(test_size),
                    shuffle=bool(shuffle),
                    random_state=int(random_state),
                    model_id=model_id_override or None,
                )
                st.success(res)
            except Exception as e:
                st.error(f"Train failed: {e}")

    st.divider()
    st.write("Models")
    try:
        models = _client().list_models()
        st.dataframe(models, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to list models: {e}")


def render_inference_tab():
    st.subheader("Inference")
    try:
        models = _client().list_models()
    except Exception as e:
        models = []
        st.error(f"Failed to list models: {e}")

    model_id = st.selectbox(
        "model_id",
        options=[m["model_id"] for m in models] if models else [],
        key="infer_model_id"
    )
    st.write("Instances (JSON list of dicts)")
    st.code('[{"feature1": 1, "feature2": "A"}, {"feature1": 2, "feature2": "B"}]', language="json")
    text = st.text_area("instances_json", value="", height=160)

    if st.button("Predict"):
        try:
            instances = json.loads(text or "[]")
            preds, probs = _client().predict(model_id=model_id, instances=instances)
            st.success("OK")
            st.write("predictions:", preds)
            if probs is not None:
                st.write("probabilities:", probs)
        except Exception as e:
            st.error(f"Inference failed: {e}")


tabs = st.tabs(["Datasets", "Train", "Inference"])
with tabs[0]:
    render_datasets_tab()
with tabs[1]:
    render_train_tab()
with tabs[2]:
    render_inference_tab()