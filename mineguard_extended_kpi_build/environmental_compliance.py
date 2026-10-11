from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

ENV_FEATURES = [
    "TP1721", "RH1722", "TP1711", "RH1712", "AN422", "AN423",
    "MM263", "MM264", "MM256"
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MineGuard methane_data.csv")
    ap.add_argument("--out", default="environmental_compliance_model.joblib")
    ap.add_argument("--metrics", default="environmental_metrics.json")
    ap.add_argument("--horizon", type=int, default=900, help="seconds ahead")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=ENV_FEATURES)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Site-control limits are loaded from percentiles of training data.
    # These are operational control limits, not regulatory/legal limits.
    n = len(df); split = int(n * 0.80)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    limits = {
        "TP1721_hi": float(train["TP1721"].quantile(0.99)),
        "TP1711_hi": float(train["TP1711"].quantile(0.99)),
        "RH1722_hi": float(train["RH1722"].quantile(0.99)),
        "RH1712_hi": float(train["RH1712"].quantile(0.99)),
        "AN422_lo": float(train["AN422"].quantile(0.01)),
        "AN423_lo": float(train["AN423"].quantile(0.01)),
    }

    future = df.shift(-args.horizon)

    # Build future-breach labels from the complete series, then fit ONLY on the
    # chronological training period and score only on the held-out tail.
    future_breach = (
        (future["TP1721"] > limits["TP1721_hi"]) |
        (future["TP1711"] > limits["TP1711_hi"]) |
        (future["RH1722"] > limits["RH1722_hi"]) |
        (future["RH1712"] > limits["RH1712_hi"]) |
        (future["AN422"] < limits["AN422_lo"]) |
        (future["AN423"] < limits["AN423_lo"])
    ).astype(int)

    X_all = df[ENV_FEATURES].copy()
    y_all = future_breach

    X_train = X_all.iloc[:split]
    y_train = y_all.iloc[:split]
    X_test = X_all.iloc[split:]
    y_test = y_all.iloc[split:]

    valid_train = X_train.notna().all(axis=1) & y_train.notna()
    valid_test = X_test.notna().all(axis=1) & y_test.notna()
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    X_test = X_test.loc[valid_test]
    y_test = y_test.loc[valid_test]

    if LGBMClassifier is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.50).astype(int)

    metrics = {
        "task": "environmental_threshold_forecast",
        "horizon_seconds": args.horizon,
        "operational_limits": limits,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "note": "Control limits are data-derived operational limits, not legal/regulatory thresholds. Replace with site-approved limits for compliance use.",
    }
    joblib.dump({"model": model, "features": ENV_FEATURES, "limits": limits, "horizon_seconds": args.horizon}, args.out)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
