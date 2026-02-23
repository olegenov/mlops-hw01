from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    s3_endpoint_url: str = Field(default="http://minio:9000", alias="S3_ENDPOINT_URL")
    s3_access_key_id: str = Field(default="", alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field(default="", alias="S3_SECRET_ACCESS_KEY")
    s3_bucket: str = Field(default="mlops", alias="S3_BUCKET")

    clearml_api_host: str = Field(default="http://clearml-webserver:8080", alias="CLEARML_API_HOST")
    clearml_api_access_key: str = Field(default="", alias="CLEARML_API_ACCESS_KEY")
    clearml_api_secret_key: str = Field(default="", alias="CLEARML_API_SECRET_KEY")

    dvc_remote: str = Field(default="s3://mlops", alias="DVC_REMOTE")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
