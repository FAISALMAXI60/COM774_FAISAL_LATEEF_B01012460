# tests/test_model.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from pathlib import Path

from src.ml_pipeline import load_dataset, ARTIFACTS_DIR
from src import train

def test_dataset_loads_correctly():
    X_train, X_test, y_train, y_test, feature_names = load_dataset()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(feature_names) == X_train.shape[1]

def test_training_creates_artifacts():
    # Run training
    train.main()

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    assert metrics_path.exists(), "metrics.json was not created."

    with metrics_path.open() as f:
        metrics = json.load(f)

    assert metrics["selected"] in ("baseline", "improved")
    assert "f1" in metrics[metrics["selected"]]
