SHELL := /bin/bash

.PHONY: help install proto run-api run-grpc run-ui minikube-start minikube-build k8s-apply k8s-delete minikube-up minikube-open-ui minikube-open-api run-all stop-all

help:
	@echo "Targets:"
	@echo "  install          - pip install -r requirements.txt"
	@echo "  proto            - generate gRPC python stubs"
	@echo "  run-api          - run FastAPI (uvicorn)"
	@echo "  run-grpc         - run gRPC server"
	@echo "  run-ui           - run Streamlit dashboard"
	@echo "  run-all          - run API, gRPC and UI concurrently (ctrl+c to stop)"
	@echo "  stop-all         - kill API/gRPC/UI processes started separately"
	@echo "  minikube-start   - start Minikube (driver=docker)"
	@echo "  minikube-build   - build images into Minikube registry"
	@echo "  k8s-apply        - apply K8s manifests"
	@echo "  k8s-delete       - delete K8s resources"
	@echo "  minikube-up      - start cluster, build images, apply manifests"
	@echo "  minikube-open-ui - open Streamlit UI service"
	@echo "  minikube-open-api- open FastAPI service (Swagger)"

install:
	python -m pip install -r requirements.txt

proto:
	python -m grpc_tools.protoc -I=protos --python_out=src/grpc_svc --grpc_python_out=src/grpc_svc protos/ml_service.proto

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-grpc:
	python -m src.grpc_svc.server

run-ui:
	PYTHONPATH=. streamlit run src/dashboard/app.py --server.address=0.0.0.0 --server.port=8501

minikube-start:
	minikube start --driver=docker

minikube-build:
	minikube image build -t mlops-hw/api:latest -f Dockerfile.api .
	minikube image build -t mlops-hw/grpc:latest -f Dockerfile.grpc .
	minikube image build -t mlops-hw/ui:latest -f Dockerfile.ui .

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/pvc.yaml
	kubectl apply -f k8s/minio.yaml
	kubectl apply -f k8s/api.yaml
	kubectl apply -f k8s/grpc.yaml
	kubectl apply -f k8s/ui.yaml

k8s-delete:
	kubectl delete -f k8s/ui.yaml --ignore-not-found
	kubectl delete -f k8s/grpc.yaml --ignore-not-found
	kubectl delete -f k8s/api.yaml --ignore-not-found
	kubectl delete -f k8s/minio.yaml --ignore-not-found
	kubectl delete -f k8s/pvc.yaml --ignore-not-found
	kubectl delete -f k8s/secret.yaml --ignore-not-found
	kubectl delete -f k8s/configmap.yaml --ignore-not-found
	kubectl delete -f k8s/namespace.yaml --ignore-not-found

minikube-up: minikube-start minikube-build k8s-apply
	@echo "Use: make minikube-open-ui OR make minikube-open-api"

minikube-open-ui:
	minikube service -n mlops-hw mlops-ui

minikube-open-api:
	minikube service -n mlops-hw mlops-api

# Run all local services concurrently in one terminal
run-all:
	$(MAKE) -j 3 run-api run-grpc run-ui

# Stop local services (best-effort, safe if not running)
stop-all:
	- pkill -f "uvicorn src.api.main:app" || true
	- pkill -f "python -m src.grpc_svc.server" || true
	- pkill -f "streamlit run src/dashboard/app.py" || true
