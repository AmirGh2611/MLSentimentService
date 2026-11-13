from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: Path = Path("models/mlsentimentservice.joblib")
    model_version: str = "0.1.0"

    model_config = {"env_prefix": "SENTIMENT_"}


settings = Settings()