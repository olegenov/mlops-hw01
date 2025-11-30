from __future__ import annotations
# pyright: reportAttributeAccessIssue=false

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, cast

import grpc
from google.protobuf import empty_pb2  # noqa: F401

from src.core.logging_config import configure_logging
from src.core.settings import get_settings
from src.core.services.dataset.storage import ensure_dirs
from src.core.services.dataset.dvc_integration import setup_dvc
from src.core.services.models.model_store import (
    list_models as store_list_models,
    delete_model as store_delete_model,
)
from src.core.services.models.training_service import train_model as svc_train_model
from src.core.services.models.inference_service import predict as svc_predict

from . import ml_service_pb2 as pb
from . import ml_service_pb2_grpc as pbg
pb = cast(Any, pb)


class MLService(pbg.MLServiceServicer):
    def __init__(self):
        self.settings = get_settings()
        ensure_dirs()
        try:
            setup_dvc()
        except Exception:
            pass
        self.logger = logging.getLogger("grpc")

    async def Health(self, request, context):
        self.logger.info("health")
        return pb.HealthReply(status="ok")

    async def Train(self, request, context) -> Any:
        try:
            hyperparams: Dict[str, Any] = json.loads(request.hyperparams_json) if request.hyperparams_json else {}
        except Exception as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid hyperparams_json: {e}")
            raise

        try:
            mid, metrics = svc_train_model(
                dataset_id=request.dataset_id,
                target_column=request.target_column,
                model_key=request.model_key,
                hyperparams=hyperparams,
                test_size=request.test_size if request.test_size else 0.2,
                shuffle=bool(request.shuffle),
                random_state=int(request.random_state) if request.random_state else 42,
                model_id=(request.model_id or None),
            )
            return pb.TrainReply(model_id=mid, model_key=request.model_key, metrics=metrics)
        except FileNotFoundError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Dataset not found")
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    async def Predict(self, request, context) -> Any:
        try:
            instances: List[Dict[str, Any]] = json.loads(request.instances_json) if request.instances_json else []
        except Exception as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid instances_json: {e}")
            raise
        try:
            preds, probs = svc_predict(model_id=request.model_id, instances=instances)
            predictions = [str(v) for v in preds]
            probabilities = [pb.DoubleList(values=[float(x) for x in row]) for row in (probs or [])]
            return pb.PredictReply(predictions=predictions, probabilities=probabilities)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

    async def ListModels(self, request, context) -> Any:
        items = store_list_models()
        out = []
        for m in items:
            out.append(pb.ModelInfo(
                model_id=m["model_id"],
                model_key=m["model_key"],
                dataset_id=m["dataset_id"],
                target_column=m["target_column"],
                created_at=m["created_at"],
            ))
        return pb.ListModelsReply(items=out)

    async def DeleteModel(self, request, context) -> Any:
        ok = store_delete_model(request.model_id)
        if not ok:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")
            raise
        return pb.DeleteModelReply(deleted=True)


async def serve(host: str = "0.0.0.0", port: int = 50051):
    settings = get_settings()
    configure_logging(settings.log_level)
    server = grpc.aio.server()
    pbg.add_MLServiceServicer_to_server(MLService(), server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logging.getLogger("grpc").info("gRPC server listening on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
