# test_predict_local.py
import json
from pathlib import Path

import pandas as pd
import requests


root = Path(__file__).resolve().parent
features_path = root / "data" / "features_scaled.csv"

# 1. Load one example row
df = pd.read_csv(features_path)
row = df.iloc[0].to_dict()

payload = {
    "features": {k: float(v) for k, v in row.items()}
}

# 2. Call local FastAPI service
url = "http://127.0.0.1:8000/predict"

print("Sending request to:", url)
resp = requests.post(url, json=payload)

print("Status:", resp.status_code)
print("Response:")
print(json.dumps(resp.json(), indent=2))