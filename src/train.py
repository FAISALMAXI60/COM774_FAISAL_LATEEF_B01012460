# src/train.py
from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .ml_pipeline import (
    load_dataset,
    evaluate_model,
    save_model_and_metadata,
    ARTIFACTS_DIR,
)


def train_baseline(X_train, y_train):
    """
    Iteration 1: Baseline model (Logistic Regression).
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_improved(X_train, y_train):
    """
    Iteration 2: Improved model (RandomForest).
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    X_train, X_test, y_train, y_test, feature_names = load_dataset()

    # --- Iteration 1: baseline ---
    baseline_model = train_baseline(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)

    # --- Iteration 2: improved ---
    improved_model = train_improved(X_train, y_train)
    improved_metrics = evaluate_model(improved_model, X_test, y_test)

    print("=== Baseline (Logistic Regression) ===")
    print(json.dumps(baseline_metrics, indent=2))
    print("=== Improved (Random Forest) ===")
    print(json.dumps(improved_metrics, indent=2))

    # Select best model by F1 (primary), then accuracy
    def score(m):
        return (m.get("f1", 0.0), m.get("accuracy", 0.0))

    best_label = "baseline"
    best_metrics = baseline_metrics
    best_model = baseline_model

    if score(improved_metrics) > score(baseline_metrics):
        best_label = "improved"
        best_metrics = improved_metrics
        best_model = improved_model

    print(f"Selected best model: {best_label}")

    # Save best model + metadata
    save_model_and_metadata(best_model, feature_names, label="best")

    # Save metrics for experiment tracking & testing
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": baseline_metrics,
                "improved": improved_metrics,
                "selected": best_label,
            },
            f,
            indent=2,
        )

    # Regression guard for CI (Testing Approaches requirement)
    # Adjust threshold based on what you see when you run it.
    if best_metrics["f1"] < 0.6:
        raise SystemExit(f"F1 too low ({best_metrics['f1']:.3f}) – potential regression.")


if __name__ == "__main__":
    main()
