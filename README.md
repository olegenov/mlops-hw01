# ML MLOps Service

FastAPI REST + gRPC сервис для обучения и инференса ML-моделей (LogisticRegression, RandomForest), с версионированием датасетов через DVC (MinIO S3), эксперимент-трекингом и публикацией моделей в ClearML, и дашбордом Streamlit.

Ключевые файлы:
- API: src/api/main.py, схемы: src/api/DTOs.py
- Модели/реестр: src/models/registry.py
- Хранилища: src/core/services/dataset/storage.py, src/core/services/models/model_store.py
- Интеграции: src/core/services/dataset/dvc_integration.py, src/core/services/models/clearml_integration.py
- gRPC: .proto protos/ml_service.proto, сервер src/grpc_svc/server.py, клиент scripts/grpc_client.py
- Дашборд: src/dashboard/app.py
- Kubernetes (Minikube): манифесты в каталоге k8s/
- Makefile: Makefile
- Пример переменных: .env.example
- Зависимости: requirements.txt

Стек: FastAPI, gRPC (grpc.aio), Swagger, Streamlit, scikit-learn, DVC[s3], MinIO, ClearML, Kubernetes/Minikube, Docker.

## Что умеет сервис

REST API:
- GET /health — проверка статуса
- GET /models/classes — список доступных классов моделей
- Datasets:
  - GET /datasets — список загруженных датасетов
  - POST /datasets — загрузка CSV/JSON
  - DELETE /datasets/{dataset_id} — удаление датасета
- Модели:
  - POST /train — обучение с гиперпараметрами (поддержка переобучения по model_id)
  - GET /models — список обученных моделей
  - POST /predict — инференс по model_id
  - DELETE /models/{model_id} — удаление модели

gRPC сервис:
- Процедуры Health, Train, Predict, ListModels, DeleteModel

DVC + S3 (MinIO):
- При загрузке датасета выполняется dvc add + dvc push на S3 MinIO
- При удалении — dvc remove и удаление локального файла

ClearML:
- Каждое обучение регистрируется как эксперимент
- Модель публикуется в ClearML (best-effort; требуется корректная конфигурация ClearML/MinIO)

Streamlit:
- Три вкладки: Datasets, Train, Inference, работает через REST API

## Быстрый старт локально

Требуется Python 3.11+ и активированное виртуальное окружение (venv).

0) Установка зависимостей
- make install

Вариант 1 — одной командой:
- make run-all
  - Запустит REST API (http://localhost:8000), gRPC (localhost:50051) и UI (http://localhost:8501) параллельно
  - Остановить: Ctrl+C или make stop-all

Вариант 2 — по отдельности:

1) Запуск REST API (Swagger на http://localhost:8000/docs)
- make run-api

2) Запуск gRPC сервера
- make run-grpc

3) Запуск Streamlit UI (http://localhost:8501)
- make run-ui
- Можно указать API адрес через переменную окружения API_BASE_URL (по умолчанию http://localhost:8000)

Переменные окружения берутся из [.env.example](.env.example). Необязательные: ClearML (если не настроен — интеграция пропускается).

## Примеры REST запросов

Список классов:
- curl -s http://localhost:8000/models/classes

Загрузка датасета:
- curl -s -F "file=@data.csv" http://localhost:8000/datasets

Обучение:
- curl -s -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{
  "model_key": "logistic_regression",
  "dataset_id": "<dataset_id>",
  "target_column": "target",
  "hyperparams": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
  "test_size": 0.2, "shuffle": true, "random_state": 42
}'

Инференс:
- curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "model_id": "<model_id>",
  "instances": [{"f_num":1.0, "f_cat":"A"}, {"f_num":0.5, "f_cat":"B"}]
}'

Удаление модели:
- curl -s -X DELETE http://localhost:8000/models/<model_id>

## gRPC

Сгенерировать Python-код из .proto (если меняли протокол):
- make proto

Запуск сервера:
- make run-grpc

Клиент-скрипт (примеры):
- python scripts/grpc_client.py health
- python scripts/grpc_client.py list-models
- python scripts/grpc_client.py train --model-key logistic_regression --dataset-id <ds_id> --target-column target --hyperparams-json "{}"
- python scripts/grpc_client.py predict --model-id <model_id> --instances-json '[{"f_num":0.9,"f_cat":"B"}]'

## DVC + MinIO (S3)

Конфигурация задаётся через переменные:
- S3_ENDPOINT_URL=http://minio:9000
- S3_ACCESS_KEY_ID=minioadmin
- S3_SECRET_ACCESS_KEY=minioadmin
- DVC_REMOTE=s3://mlops

Приложение:
- Инициализирует DVC на старте
- Добавляет и пушит датасеты при загрузке
- Выполняет dvc remove при удалении датасета

## ClearML

Локальный ClearML (docker compose):
- docker compose -f docker-compose.clearml.yml up -d
- В UI: http://localhost:8080

## Kubernetes / Minikube (driver=docker)

Требуется установленный minikube и kubectl.

1) Старт кластера:
- make minikube-start

2) Сборка образов внутри minikube:
- make minikube-build

3) Применение манифестов:
- make k8s-apply
- Создаст namespace mlops-hw, MinIO, PVC (dvc-cache, datasets), Deployments и Services для API/gRPC/UI, ConfigMap и Secret.

4) Открытие сервисов:
- Swagger: make minikube-open-api (NodePort 30080)
- UI: make minikube-open-ui (NodePort 30851)

Где что лежит:
- Namespace: k8s/namespace.yaml
- MinIO: k8s/minio.yaml
- API: k8s/api.yaml
- gRPC: k8s/grpc.yaml
- UI: k8s/ui.yaml
- PVC: k8s/pvc.yaml
- Config/Secrets: k8s/configmap.yaml, k8s/secret.yaml