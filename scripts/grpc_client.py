from __future__ import annotations
import argparse
import asyncio
import json
from typing import Any, Dict, List

import grpc
from google.protobuf import empty_pb2

from src.grpc_svc import ml_service_pb2 as pb
from src.grpc_svc import ml_service_pb2_grpc as pbg


async def run(args):
    async with grpc.aio.insecure_channel(args.addr) as channel:
        stub = pbg.MLServiceStub(channel)

        if args.cmd == "health":
            resp = await stub.Health(empty_pb2.Empty())
            print(resp)

        elif args.cmd == "list-models":
            resp = await stub.ListModels(empty_pb2.Empty())
            for m in resp.items:
                print(m)

        elif args.cmd == "delete-model":
            resp = await stub.DeleteModel(pb.DeleteModelRequest(model_id=args.model_id))
            print(resp)

        elif args.cmd == "train":
            hyperparams_json = args.hyperparams_json or "{}"
            req = pb.TrainRequest(
                model_key=args.model_key,
                dataset_id=args.dataset_id,
                target_column=args.target_column,
                hyperparams_json=hyperparams_json,
                test_size=args.test_size,
                shuffle=not args.no_shuffle,
                random_state=args.random_state,
                model_id=args.model_id or "",
            )
            resp = await stub.Train(req)
            print(resp)

        elif args.cmd == "predict":
            instances: List[Dict[str, Any]] = json.loads(args.instances_json)
            req = pb.PredictRequest(model_id=args.model_id, instances_json=json.dumps(instances))
            resp = await stub.Predict(req)
            print(resp)

        else:
            raise SystemExit("Unknown command")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="localhost:50051", help="host:port")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")

    sub.add_parser("list-models")

    sp_del = sub.add_parser("delete-model")
    sp_del.add_argument("--model-id", required=True)

    sp_train = sub.add_parser("train")
    sp_train.add_argument("--model-key", required=True, choices=["logistic_regression", "random_forest"])
    sp_train.add_argument("--dataset-id", required=True)
    sp_train.add_argument("--target-column", default="target")
    sp_train.add_argument("--hyperparams-json", default="{}")
    sp_train.add_argument("--test-size", type=float, default=0.2)
    sp_train.add_argument("--no-shuffle", action="store_true")
    sp_train.add_argument("--random-state", type=int, default=42)
    sp_train.add_argument("--model-id", default="")

    sp_predict = sub.add_parser("predict")
    sp_predict.add_argument("--model-id", required=True)
    sp_predict.add_argument("--instances-json", required=True, help="JSON list of dicts")

    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
