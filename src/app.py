# src/app.py
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .ml_pipeline import load_model_and_metadata

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("CI_CD_API")

logger.info("Application module imported. Initializing FastAPI...")

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI(
    title="CI/CD Run Success Predictor",
    description="Predicts whether a CI/CD pipeline run will succeed based on preprocessed features.",
    version="1.0.0",
)

@app.get("/")
def root():
    logger.info("Root endpoint called.")
    return {
        "message": "COM774 CW2 API is running. See /health, /metrics and /docs."
    }

# ---- Global model state ----
MODEL_LABEL = "best"
model = None
FEATURE_NAMES = []

# ---- Monitoring counters ----
PREDICTION_COUNT = 0
LAST_PREDICTION = None


class RunFeatures(BaseModel):
    features: Dict[str, float]


# ------------------------------
# Startup: Load Model
# ------------------------------
@app.on_event("startup")
def load_artifacts():
    global model, FEATURE_NAMES
    logger.info("Startup event: Loading trained model and metadata...")

    try:
        model, FEATURE_NAMES = load_model_and_metadata(label=MODEL_LABEL)
        logger.info(f"Model loaded successfully with {len(FEATURE_NAMES)} features.")
    except FileNotFoundError:
        logger.warning("Model artifacts not found. API will run in degraded mode.")
        model = None
        FEATURE_NAMES = []


# ------------------------------
# Health Check
# ------------------------------
@app.get("/health")
def health_check():
    status = "ok" if model is not None else "model_not_loaded"
    logger.info(f"Health checked. Status={status}, Features={len(FEATURE_NAMES)}")

    return {
        "status": status,
        "model_label": MODEL_LABEL,
        "feature_count": len(FEATURE_NAMES),
    }


# ------------------------------
# Metrics Endpoint
# ------------------------------
@app.get("/metrics")
def metrics():
    """
    Lightweight monitoring endpoint for MLOps governance.
    Provides basic runtime metrics without needing Prometheus.
    """
    logger.info("Metrics endpoint called.")
    return {
        "model_loaded": model is not None,
        "model_label": MODEL_LABEL,
        "feature_count": len(FEATURE_NAMES),
        "prediction_count": PREDICTION_COUNT,
        "last_prediction": LAST_PREDICTION,
    }


# ------------------------------
# Helper: Build Feature DataFrame
# ------------------------------
def build_feature_frame(payload: RunFeatures) -> pd.DataFrame:
    if model is None or not FEATURE_NAMES:
        logger.error("Predict called but model is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first and restart the service.",
        )

    data = payload.features
    missing = [f for f in FEATURE_NAMES if f not in data]

    if missing:
        logger.warning(f"Prediction request missing features: {missing}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature(s): {missing}",
        )

    logger.info("All required features present. Building DataFrame.")
    row = {f: data[f] for f in FEATURE_NAMES}
    return pd.DataFrame([row])


# ------------------------------
# Prediction Endpoint
# ------------------------------
@app.post("/predict")
def predict(run: RunFeatures):
    global PREDICTION_COUNT, LAST_PREDICTION

    logger.info("Received /predict request.")
    PREDICTION_COUNT += 1  # increment monitoring counter

    X = build_feature_frame(run)

    if hasattr(model, "predict_proba"):
        prob_success = float(model.predict_proba(X)[0, 1])
    else:
        prob_success = None

    pred = int(model.predict(X)[0])

    LAST_PREDICTION = {
        "predicted_success": pred,
        "probability_success": prob_success,
    }

    logger.info(
        f"Prediction made: predicted_success={pred}, probability_success={prob_success}"
    )

    return LAST_PREDICTION
