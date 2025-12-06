# tests/test_data.py
from pathlib import Path
import pandas as pd

def test_cleaned_data_has_no_nans():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "ci_cd_runs_cleaned.csv")
    assert not df.isna().any().any(), "Cleaned data still has NaNs."

def test_scaled_features_match_cleaned_rows():
    root = Path(__file__).resolve().parents[1]
    cleaned = pd.read_csv(root / "data" / "ci_cd_runs_cleaned.csv")
    scaled = pd.read_csv(root / "data" / "features_scaled.csv")
    assert len(cleaned) == len(scaled), "Mismatch between row counts."
