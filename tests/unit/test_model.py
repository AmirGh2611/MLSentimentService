from unittest.mock import patch

import pytest

from mlsentimentservice.model import predict


class FakePipeline:
    """A minimal stand-in for a trained sklearn Pipeline."""

    def __init__(self, label: int, proba: float):
        self._label = label
        self._proba = proba

    def predict(self, X):
        return [self._label]

    def predict_proba(self, X):
        if self._label == 1:
            return [[1 - self._proba, self._proba]]
        return [[self._proba, 1 - self._proba]]


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Ensure each test starts with a clean lru_cache."""
    from mlsentimentservice import model as model_module

    model_module.load_model.cache_clear()
    yield
    model_module.load_model.cache_clear()


def test_predict_positive():
    with patch(
        "mlsentimentservice.model.load_model",
        return_value=(FakePipeline(label=1, proba=0.92), "0.1.0"),
    ):
        result = predict("Absolutely wonderful film.")

    assert result["label"] == "positive"
    assert result["score"] == 0.92
    assert result["model_version"] == "0.1.0"


def test_predict_negative():
    with patch(
        "mlsentimentservice.model.load_model",
        return_value=(FakePipeline(label=0, proba=0.88), "0.1.0"),
    ):
        result = predict("Terrible, awful, boring.")

    assert result["label"] == "negative"
    assert result["score"] == 0.88


def test_predict_returns_expected_keys():
    with patch(
        "mlsentimentservice.model.load_model",
        return_value=(FakePipeline(label=1, proba=0.5), "0.1.0"),
    ):
        result = predict("text")

    assert set(result.keys()) == {"label", "score", "model_version"}