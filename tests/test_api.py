# tests/test_api.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.app import app
from src import train
from src.ml_pipeline import ARTIFACTS_DIR


def ensure_model_trained():
    """
    Ensure that the trained model artifacts exist before API tests run.
    This avoids relying on the order of other tests.
    """
    model_path = ARTIFACTS_DIR / "model_best.joblib"
    meta_path = ARTIFACTS_DIR / "model_best_meta.json"
    if not (model_path.exists() and meta_path.exists()):
        train.main()


def test_health_route():
    ensure_model_trained()
    # Use context manager so startup events run
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["model_label"] == "best"
        assert body["feature_count"] > 0


def test_predict_route():
    ensure_model_trained()

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "features_scaled.csv")
    row = df.iloc[0].to_dict()

    payload = {
        "features": {k: float(v) for k, v in row.items()}
    }

    # Use context manager so startup events load the model
    with TestClient(app) as client:
        resp = client.post("/predict", json=payload)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "predicted_success" in body
    assert "probability_success" in body
