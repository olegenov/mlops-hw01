from __future__ import annotations
import logging
import uvicorn
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File

from src.core.settings import get_settings
from src.core.logging_config import configure_logging
from src.core.services.dataset.storage import ensure_dirs
from src.core.services.models.model_store import (
    list_models as store_list_models,
    delete_model as store_delete_model,
)
from src.models.registry import list_model_classes
from src.api.DTOs import DatasetInfo, TrainRequest, TrainResponse, PredictRequest, PredictResponse, ModelInfo

from src.api.errors import (
    ERR_FAILED_PARSE_DATASET,
    ERR_EMPTY_DATASET,
    ERR_DATASET_NOT_FOUND,
    ERR_FAILED_READ_DATASET,
    ERR_MODEL_NOT_FOUND,
    ERR_NO_INSTANCES,
    ERR_PREDICTION_FAILED,
)
from src.core.services.models.training_service import (
    train_model as svc_train_model,
    TargetColumnNotFoundError
)
from src.core.services.models.inference_service import (
    predict as svc_predict,
    NoInstancesError
)

from src.core.services.dataset.dataset_service import (
    upload as ds_upload,
    list_datasets as ds_list_datasets,
    delete_dataset as ds_delete_dataset,
    EmptyDatasetError,
    DatasetParseError,
    DatasetNotFoundError,
    DatasetReadError
)
from src.core.services.dataset.dvc_integration import setup_dvc

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("api")

ensure_dirs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_dvc()
    yield

app = FastAPI(
    title="MLOps Service",
    version="0.1.0",
    description="REST API for model training and inference.",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="Health check",
    description="Service liveness probe",
    tags=["system"]
)
def health():
    logger.info("health")
    return {"status": "ok"}


@app.get(
    "/models/classes",
    summary="List available model classes",
    description="Return list of model identifiers that can be trained",
    tags=["models"]
)
def get_models_classes():
    return {"classes": list_model_classes()}


@app.get(
    "/datasets",
    response_model=List[DatasetInfo],
    summary="List datasets",
    description="Return list of uploaded datasets with basic metadata",
    tags=["datasets"]
)
def list_datasets():
    """
    Get a list of available datasets.
    Thin controller delegating to DatasetService.
    """
    return ds_list_datasets()


@app.post(
    "/datasets",
    summary="Upload dataset",
    description="Upload CSV or JSON dataset; provide file in 'file' form field",
    tags=["datasets"],
    responses={400: {"description": "Parse error or empty dataset"}}
)
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()

    try:
        result = ds_upload(content, file.filename, file.content_type)
        logger.info(
            "dataset uploaded id=%s rows=%d cols=%d",
            result["dataset_id"],
            result["rows"],
            result["cols"]
        )
        return result
    except DatasetParseError as e:
        raise HTTPException(
            status_code=400,
            detail=ERR_FAILED_PARSE_DATASET.format(err=e)
        )
    except EmptyDatasetError:
        raise HTTPException(
            status_code=400,
            detail=ERR_EMPTY_DATASET
        )


@app.delete(
    "/datasets/{dataset_id}",
    summary="Delete dataset",
    description="Delete dataset file and untrack in DVC",
    tags=["datasets"],
    responses={404: {"description": "Dataset not found"}}
)
def delete_dataset(dataset_id: str):
    ok = ds_delete_dataset(dataset_id)

    if not ok:
        raise HTTPException(
            status_code=404,
            detail=ERR_DATASET_NOT_FOUND
        )

    logger.info("dataset deleted id=%s", dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id}


@app.post(
    "/train",
    response_model=TrainResponse,
    summary="Train model",
    description="Train selected model on an uploaded dataset; logs experiment to ClearML",
    tags=["train"],
    responses={
        400: {"description": "Validation error"},
        404: {"description": "Dataset not found"}
    }
)
def train(req: TrainRequest):
    try:
        mid, metrics = svc_train_model(
            dataset_id=req.dataset_id,
            target_column=req.target_column,
            model_key=req.model_key,
            hyperparams=(req.hyperparams or {}),
            test_size=req.test_size,
            shuffle=req.shuffle,
            random_state=req.random_state,
            model_id=req.model_id,
        )
        logger.info(
            "model trained id=%s key=%s metrics=%s",
            mid,
            req.model_key, 
            metrics
        )
        return TrainResponse(
            model_id=mid,
            model_key=req.model_key,
            metrics=metrics
        )
    except DatasetNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=ERR_DATASET_NOT_FOUND
        )
    except DatasetReadError as e:
        raise HTTPException(
            status_code=400,
            detail=ERR_FAILED_READ_DATASET.format(err=e)
        )
    except TargetColumnNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get(
    "/models", 
    response_model=List[ModelInfo],
    summary="List models",
    description="List trained models metadata",
    tags=["models"]
)
def list_models():
    items = store_list_models()
    out: List[ModelInfo] = []

    for m in items:
        out.append(
            ModelInfo(
                model_id=m["model_id"],
                model_key=m["model_key"],
                dataset_id=m["dataset_id"],
                target_column=m["target_column"],
                created_at=m["created_at"],
                metrics=m.get("metrics"),
            )
        )

    return out


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict",
    description="Run inference with a trained model",
    tags=["inference"],
    responses={
        400: {"description": "Validation error"},
        404: {"description": "Model not found"}
    }
)
def predict(req: PredictRequest):
    try:
        preds, probs = svc_predict(
            model_id=req.model_id,
            instances=req.instances
        )
        return PredictResponse(
            predictions=preds,
            probabilities=probs
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=ERR_MODEL_NOT_FOUND
        )
    except NoInstancesError:
        raise HTTPException(
            status_code=400,
            detail=ERR_NO_INSTANCES
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=ERR_PREDICTION_FAILED.format(err=e)
        )


@app.delete(
    "/models/{model_id}",
    summary="Delete model",
    description="Delete trained model artifact and remove from index",
    tags=["models"],
    responses={404: {"description": "Model not found"}}
)
def delete_model(model_id: str):
    """
    Delete a model.
    """
    ok = store_delete_model(model_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=ERR_MODEL_NOT_FOUND
        )

    logger.info("model deleted id=%s", model_id)

    return {"status": "deleted", "model_id": model_id}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
