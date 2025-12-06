#!/usr/bin/env python3
"""
COM774 CW1 Preprocessing Pipeline – v2 (Faisal Lateef, B01012460)
Adds:
- One-hot encoding of categorical features
- Min–Max scaling of numeric features

Outputs (always):
- features_onehot.csv
- features_scaled.csv

If 'pyarrow' is available, also writes Parquet:
- features_onehot.parquet
- features_scaled.parquet
"""
import argparse, os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_raw(n_rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    repos = ["frontend-web","backend-api","mobile-app","ml-pipeline","infra-terraform"]
    triggers = ["push","PullRequest","schedule","manual","Push"]
    ros = ["ubuntu-latest","windows-latest","macos-latest"]
    authors = [f"dev{i}" for i in range(1,31)]
    start = datetime(2025, 6, 1)
    created = [start + timedelta(hours=i) for i in range(n_rows)]
    df = pd.DataFrame({
        "run_id": [f"RUN-{100000+i}" for i in range(n_rows)],
        "repo": rng.choice(repos, size=n_rows),
        "commit_sha": [f"{rng.integers(16**10):010x}" for _ in range(n_rows)],
        "author": rng.choice(authors, size=n_rows),
        "author_experience_commits": np.where(rng.random(n_rows) < 0.04, np.nan, np.exp(rng.normal(3.3,0.9,size=n_rows)).astype(int)),
        "files_changed": np.maximum(1, np.exp(rng.normal(1.2,0.8,size=n_rows)).astype(int)),
        "lines_added": np.exp(rng.normal(5.0,1.1,size=n_rows)).astype(int),
        "lines_deleted": np.exp(rng.normal(4.7,1.1,size=n_rows)).astype(int),
        "test_count": np.maximum(1, np.exp(rng.normal(3.8,0.8,size=n_rows)).astype(int)),
        "test_failures": rng.binomial(10, 0.05, size=n_rows),
        "coverage": np.where(rng.random(n_rows)<0.05, np.nan, np.clip(rng.normal(75,12,size=n_rows), 5, 100)),
        "pipeline_duration_s": np.maximum(30, np.exp(rng.normal(5.3,0.7,size=n_rows)).astype(int)),
        "jobs_total": np.maximum(1, np.exp(rng.normal(2.2,0.7,size=n_rows)).astype(int)),
        "jobs_failed": rng.binomial(5, 0.1, size=n_rows),
        "trigger": rng.choice(triggers, size=n_rows),
        "created_at": [dt.isoformat() for dt in created],
        "day_of_week": [dt.strftime("%A") for dt in created],
        "hour": [dt.hour for dt in created],
        "message_length": rng.integers(5,200,size=n_rows),
        "had_hotfix_keyword": (rng.random(n_rows)<0.1).astype(int),
        "prev_7d_failure_rate": rng.beta(2,8,size=n_rows),
        "prev_30d_failure_rate": rng.beta(2,8,size=n_rows),
        "flaky_tests_count": rng.binomial(5,0.05,size=n_rows),
        "infra_alerts_count": rng.poisson(0.2,size=n_rows),
        "artifact_size_mb": np.clip(rng.normal(120,40,size=n_rows),5,800),
        "runner_os": rng.choice(ros, size=n_rows),
        "cache_hit_rate": np.where(rng.random(n_rows)<0.05, np.nan, rng.beta(5,3,size=n_rows)),
        "dependency_updates": (rng.random(n_rows)<0.15).astype(int),
        "security_alerts_count": rng.poisson(0.05,size=n_rows),
        "success": (rng.random(n_rows)>0.25).astype(int),
        "failure_reason": rng.choice([None,"tests_failed","lint_error","infra_timeout","artifact_upload_error","docker_build_failed"], size=n_rows)
    })
    df["churn"] = df["lines_added"] + df["lines_deleted"]
    dups = df.sample(25, random_state=3)
    return pd.concat([df, dups], ignore_index=True)

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    if "trigger" in df.columns:
        df["trigger"] = df["trigger"].astype(str).str.strip().str.lower()
    if {"commit_sha","created_at"}.issubset(df.columns):
        df.sort_values("created_at", inplace=True)
        df = df.drop_duplicates(subset=["commit_sha","created_at"], keep="last")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            med = pd.to_numeric(df[c], errors="coerce").median()
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").fillna("Unknown")
    if "coverage" in df.columns:
        df.loc[(df["coverage"]<0)|(df["coverage"]>100), "coverage"] = np.nan
        df["coverage"] = df["coverage"].fillna(df["coverage"].median())
    if "pipeline_duration_s" in df.columns:
        df["pipeline_duration_s"] = df["pipeline_duration_s"].clip(lower=0)
    for c in ["lines_added","lines_deleted","churn","pipeline_duration_s","artifact_size_mb"]:
        if c in df.columns:
            lo, hi = np.nanpercentile(df[c], [1,99])
            df[c] = df[c].clip(lower=lo, upper=hi)
    if {"churn","files_changed"}.issubset(df.columns):
        df["churn_per_file"] = df["churn"] / df["files_changed"].replace(0,1)
    if {"test_failures","test_count"}.issubset(df.columns):
        df["test_fail_rate"] = df["test_failures"] / df["test_count"].replace(0,1)
    if "author_experience_commits" in df.columns:
        df["author_experience_log"] = np.log1p(df["author_experience_commits"])
    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin(["Saturday","Sunday"]).astype(int)
    return df

def main():
    parser = argparse.ArgumentParser(description="COM774 CW1 – Preprocessing v2: encoding + scaling (CSV-first)")
    parser.add_argument("--input", default="ci_cd_runs_raw.csv")
    parser.add_argument("--output", default="ci_cd_runs_cleaned.csv")
    parser.add_argument("--report", default="preprocessing_report.md")
    parser.add_argument("--feature-spec", default="feature_spec.json")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--no-encode", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    # Load or generate RAW
    if os.path.exists(args.input):
        raw = pd.read_csv(args.input)
    else:
        raw = generate_synthetic_raw(args.rows)
        raw.to_csv(args.input, index=False)

    clean = clean_pipeline(raw)
    clean.to_csv(args.output, index=False)

    # Spec
    if os.path.exists(args.feature_spec):
        with open(args.feature_spec, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = {
            "target": "success",
            "features": {
                "numeric": ["files_changed","lines_added","lines_deleted","churn","test_count","test_failures","coverage","pipeline_duration_s","jobs_total","message_length","prev_7d_failure_rate","prev_30d_failure_rate","flaky_tests_count","infra_alerts_count","artifact_size_mb","cache_hit_rate","security_alerts_count","hour","churn_per_file","test_fail_rate","author_experience_log"],
                "categorical": ["repo","trigger","runner_os","day_of_week","author"],
                "binary": ["had_hotfix_keyword","dependency_updates","is_weekend"]
            },
            "leakage_notice": "Avoid post-run fields (e.g., jobs_failed) for pre-run prediction in CW2."
        }
        with open(args.feature_spec, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)

    num_cols = [c for c in spec["features"]["numeric"] if c in clean.columns]
    cat_cols = [c for c in spec["features"]["categorical"] if c in clean.columns]
    bin_cols = [c for c in spec["features"]["binary"] if c in clean.columns]

    if not args.no_encode:
        X_num = clean[num_cols].copy()
        X_cat_oh = pd.get_dummies(clean[cat_cols], columns=cat_cols, drop_first=False, dtype=np.int8) if cat_cols else pd.DataFrame(index=clean.index)
        X_bin = clean[bin_cols].copy() if bin_cols else pd.DataFrame(index=clean.index)
        X_onehot = pd.concat([X_num, X_bin, X_cat_oh], axis=1)
        X_onehot.to_csv("features_onehot.csv", index=False)
        # Try parquet if available
        try:
            X_onehot.to_parquet("features_onehot.parquet", index=False)
        except Exception:
            pass

    if not args.no_scale:
        # Prefer already-encoded features if present
        if os.path.exists("../features_onehot.csv"):
            X_base = pd.read_csv("../features_onehot.csv")
        else:
            X_base = clean[num_cols + bin_cols + cat_cols].copy()
            if cat_cols:
                X_base = pd.get_dummies(X_base, columns=cat_cols, drop_first=False, dtype=np.int8)
        mins = X_base[num_cols].min()
        maxs = X_base[num_cols].max()
        den = (maxs - mins).replace(0, 1)
        X_base[num_cols] = (X_base[num_cols] - mins) / den
        X_base.to_csv("features_scaled.csv", index=False)
        try:
            X_base.to_parquet("features_scaled.parquet", index=False)
        except Exception:
            pass

    # Append to report
    info = f"""
## Categorical Encoding & Numeric Scaling (v2 CSV-first)

- Encoded columns: {', '.join(cat_cols) if cat_cols else '(none)'}
- One-hot file(s): features_onehot.csv (+ .parquet if supported)
- Min–Max scaled numeric columns: {len(num_cols)} column(s)
- Scaled file(s): features_scaled.csv (+ .parquet if supported)
"""
    if os.path.exists(args.report):
        with open(args.report, "a", encoding="utf-8") as f:
            f.write(info)
    else:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("# Preprocessing Report (v2)\n" + info)

if __name__ == "__main__":
    main()
