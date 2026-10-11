from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

EAR_URL = "https://raw.githubusercontent.com/muratgok/drowsiness/main/dataset_EAR_300.csv"
HEAD_URL = "https://raw.githubusercontent.com/muratgok/drowsiness/main/dataset_HeadPose_300.csv"


def fetch(url: str, path: Path) -> None:
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)


def label_binary(v: str) -> int:
    return 0 if str(v).strip().lower() == "awake" else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="fatigue_data")
    ap.add_argument("--out", default="fatigue_model.joblib")
    ap.add_argument("--metrics", default="fatigue_metrics.json")
    args = ap.parse_args()

    d = Path(args.dir); d.mkdir(parents=True, exist_ok=True)
    earp = d / "dataset_EAR_300.csv"
    headp = d / "dataset_HeadPose_300.csv"
    fetch(EAR_URL, earp); fetch(HEAD_URL, headp)

    ear = pd.read_csv(earp)
    head = pd.read_csv(headp)
    df = pd.merge(ear, head, on=["Timestamp_s", "Label"], how="inner")
    df = df.sort_values("Timestamp_s").reset_index(drop=True)
    df["fatigue"] = df["Label"].map(label_binary)

    features = ["EAR", "Blinks_min", "Yaw_deg", "Pitch_deg", "Roll_deg"]
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = df.loc[X.index, "fatigue"].astype(int)

    n = len(X); a = int(n * 0.70); b = int(n * 0.85)
    Xtr, Xv, Xte = X.iloc[:a], X.iloc[a:b], X.iloc[b:]
    ytr, yv, yte = y.iloc[:a], y.iloc[a:b], y.iloc[b:]

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(Xtr, ytr)
    pv = model.predict_proba(Xv)[:, 1]
    # Keep the operating point validation-derived rather than test-tuned.
    best_thr = 0.50; best_f1 = -1
    for thr in np.linspace(0.10, 0.90, 81):
        pp = (pv >= thr).astype(int)
        f = f1_score(yv, pp, zero_division=0)
        if f > best_f1:
            best_f1 = f; best_thr = float(thr)

    pt = model.predict_proba(Xte)[:, 1]
    pred = (pt >= best_thr).astype(int)
    cm = confusion_matrix(yte, pred).tolist()
    metrics = {
        "task": "worker_fatigue_binary_classification",
        "label_rule": "Awake=0; Slight fatigue/Fatigue/Drowsy/Micro-sleep=1",
        "features": features,
        "threshold": best_thr,
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, pt)) if yte.nunique() > 1 else None,
        "confusion_matrix": cm,
        "test_rows": int(len(Xte)),
        "data_note": "Open fatigue-feature benchmark; not a mine-specific field study.",
    }
    joblib.dump({"model": model, "features": features, "threshold": best_thr}, args.out)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
