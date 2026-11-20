from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.schema import example_payload

client = TestClient(app)


def test_health():
    result = client.get("/health")
    assert result.status_code == 200
    assert isinstance(result.json()["model_ready"], bool)


def test_invalid_input():
    payload = example_payload()
    payload["age"] = 10
    payload["failures"] = 9
    result = client.post("/predict", json=payload)
    assert result.status_code == 422


def test_predict():
    if not Path("models/model.joblib").is_file():
        pytest.skip("Сначала обучите модель: python -m src.train")
    result = client.post("/predict", json=example_payload())
    assert result.status_code == 200
    body = result.json()
    assert body["target"] == "G3"
    assert body["unit"]
    assert isinstance(body["prediction"], float)
