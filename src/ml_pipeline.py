# src/ml_pipeline.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | PIPELINE | %(message)s"
)
logger = logging.getLogger("ML_PIPELINE")

logger.info("ml_pipeline module imported.")

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURE_SPEC_PATH = PROJECT_ROOT / "feature_spec.json"


def load_feature_spec() -> dict:
    logger.info(f"Loading feature spec from {FEATURE_SPEC_PATH}")

    with FEATURE_SPEC_PATH.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    logger.info("Feature spec loaded successfully.")
    return spec


def load_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:

    logger.info("Loading dataset for training/testing...")

    features_path = DATA_DIR / "features_scaled.csv"
    cleaned_path = DATA_DIR / "ci_cd_runs_cleaned.csv"

    logger.info(f"Reading features from: {features_path}")
    logger.info(f"Reading cleaned labels from: {cleaned_path}")

    if not features_path.exists():
        logger.error("features_scaled.csv missing — preprocessing not run.")
        raise FileNotFoundError(
            f"{features_path} not found – run CW1 preprocessing first."
        )

    if not cleaned_path.exists():
        logger.error("ci_cd_runs_cleaned.csv missing — preprocessing not run.")
        raise FileNotFoundError(
            f"{cleaned_path} not found – run CW1 preprocessing first."
        )

    X = pd.read_csv(features_path)
    cleaned = pd.read_csv(cleaned_path)

    if "success" not in cleaned.columns:
        logger.error("'success' column missing from cleaned dataset.")
        raise ValueError(
            "Expected 'success' column in ci_cd_runs_cleaned.csv "
            "(target from CW1/feature_spec.json)."
        )

    y = cleaned["success"]

    if len(X) != len(y):
        logger.error(
            f"Feature/label length mismatch: X={len(X)}, y={len(y)}"
        )
        raise ValueError(
            f"Mismatch between features ({len(X)}) and labels ({len(y)}). "
            "Ensure CW1 outputs are consistent."
        )

    logger.info("Splitting dataset into train/test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_names = list(X.columns)

    logger.info(
        f"Dataset loaded: Train={X_train.shape}, Test={X_test.shape}, Features={len(feature_names)}"
    )

    return X_train, X_test, y_train, y_test, feature_names


def evaluate_model(model, X_test, y_test) -> dict:
    logger.info("Evaluating model performance...")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    metrics["report"] = classification_report(y_test, y_pred, digits=3)

    logger.info(f"Model evaluation complete: {metrics}")
    return metrics


def save_model_and_metadata(model, feature_names: List[str], label: str = "best") -> None:
    logger.info(f"Saving model + metadata with label '{label}'...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"model_{label}.joblib"
    meta_path = ARTIFACTS_DIR / f"model_{label}_meta.json"

    dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    logger.info(f"Metadata saved to: {meta_path}")


def load_model_and_metadata(label: str = "best"):
    logger.info(f"Loading model + metadata with label '{label}'...")

    model_path = ARTIFACTS_DIR / f"model_{label}.joblib"
    meta_path = ARTIFACTS_DIR / f"model_{label}_meta.json"

    if not model_path.exists() or not meta_path.exists():
        logger.error("Model or metadata missing — training has not been run.")
        raise FileNotFoundError("Model or metadata not found – train the model first.")

    model = load(model_path)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info("Model & metadata loaded successfully.")
    return model, meta["feature_names"]
