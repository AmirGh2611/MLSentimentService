from functools import lru_cache

import joblib

from mlsentimentservice.config import settings


@lru_cache(maxsize=1)
def load_model():
    """Load the model artifact once and cache it in memory."""
    artifact = joblib.load(settings.model_path)
    return artifact["pipeline"], artifact["version"]


def predict(text: str) -> dict:
    pipeline, version = load_model()
    label = int(pipeline.predict([text])[0])
    score = float(pipeline.predict_proba([text])[0][label])
    return {
        "label": "positive" if label == 1 else "negative",
        "score": round(score, 4),
        "model_version": version,
    }