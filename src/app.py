# src/app.py
from __future__ import annotations

from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .ml_pipeline import load_model_and_metadata

app = FastAPI(
    title="CI/CD Run Success Predictor",
    description="Predicts whether a CI/CD pipeline run will succeed based on preprocessed features.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "message": "COM774 CW2 API is running. See /health and /docs."
    }

# ---- Global model state ----
MODEL_LABEL = "best"
model = None           # will be loaded on startup
FEATURE_NAMES = []     # will be loaded on startup


class RunFeatures(BaseModel):
    """
    Request body schema for /predict

    {
      "features": {
        "feature_name_1": 0.123,
        "feature_name_2": 1.0,
        ...
      }
    }

    The keys in `features` must match the columns of features_scaled.csv.
    """
    features: Dict[str, float]


@app.on_event("startup")
def load_artifacts():
    """
    Load the trained model and metadata when the FastAPI app starts.
    """
    global model, FEATURE_NAMES
    try:
        model, FEATURE_NAMES = load_model_and_metadata(label=MODEL_LABEL)
    except FileNotFoundError:
        # Model not trained yet – API will report this via /health and /predict
        model = None
        FEATURE_NAMES = []


@app.get("/health")
def health_check():
    """
    Health check endpoint.

    - status: "ok" if model is loaded, otherwise "model_not_loaded"
    - model_label: which model label is loaded ("best")
    - feature_count: how many input features the model expects
    """
    status = "ok" if model is not None else "model_not_loaded"
    return {
        "status": status,
        "model_label": MODEL_LABEL,
        "feature_count": len(FEATURE_NAMES),
    }


def build_feature_frame(payload: RunFeatures) -> pd.DataFrame:
    """
    Convert incoming JSON into a single-row DataFrame with the correct
    feature order and columns.
    """
    if model is None or not FEATURE_NAMES:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first and restart the service.",
        )

    data = payload.features
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature(s): {missing}",
        )

    # Ensure correct column order
    row = {f: data[f] for f in FEATURE_NAMES}
    return pd.DataFrame([row])


@app.post("/predict")
def predict(run: RunFeatures):
    """
    Predict whether a CI/CD run will succeed (1) or fail (0).

    Response:
    {
      "predicted_success": 1,
      "probability_success": 0.84
    }
    """
    X = build_feature_frame(run)

    if hasattr(model, "predict_proba"):
        prob_success = float(model.predict_proba(X)[0, 1])
    else:
        prob_success = None

    pred = int(model.predict(X)[0])

    return {
        "predicted_success": pred,
        "probability_success": prob_success,
    }
