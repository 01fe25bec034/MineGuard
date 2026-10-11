from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False


def find_time_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    if candidates:
        return candidates[0]
    raise ValueError("Could not find a timestamp/time column in slope CSV.")


def pick_columns(df: pd.DataFrame) -> list[str]:
    keys = [
        "displ", "extens", "pore", "pressure", "rain", "precip",
        "soil", "moist", "water", "settle", "tilt", "inclin"
    ]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = [c for c in numeric if any(k in c.lower() for k in keys)]
    if len(selected) < 2:
        selected = numeric[: min(8, len(numeric))]
    if len(selected) < 2:
        raise ValueError("Slope file needs at least two numeric monitoring columns.")
    return selected


def make_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        out[f"{c}_now"] = s
        out[f"{c}_d1"] = s.diff(1)
        out[f"{c}_d4"] = s.diff(4)
        out[f"{c}_d16"] = s.diff(16)
        out[f"{c}_mean16"] = s.rolling(16, min_periods=8).mean()
        out[f"{c}_std16"] = s.rolling(16, min_periods=8).std()
        out[f"{c}_max16"] = s.rolling(16, min_periods=8).max()
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="USGS slope/landslide monitoring CSV")
    ap.add_argument("--out", default="slope_model.joblib")
    ap.add_argument("--metrics", default="slope_metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tcol = find_time_column(df)
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    cols = pick_columns(df)

    # Derived benchmark target: "movement escalation".
    # If a displacement-like signal exists, use its future change.
    disp = next((c for c in cols if any(k in c.lower() for k in ["displ", "extens", "settle", "inclin", "tilt"])), None)
    if disp is None:
        # Fall back to the first sensor as a movement proxy.
        disp = cols[0]
    s = pd.to_numeric(df[disp], errors="coerce")
    future_delta = s.shift(-8) - s
    baseline = (s.diff().abs().rolling(64, min_periods=32).median()).replace(0, np.nan)

    # Derived proxy target: future displacement escalation.
    # Keep periods without a valid baseline/future value as NaN,
    # rather than incorrectly labeling them as normal (0).
    target = (future_delta.abs() > (3.0 * baseline)).astype(float)
    target[future_delta.isna() | baseline.isna()] = np.nan

    X = make_features(df, cols)

    # LightGBM can handle missing values natively.
    # Drop only features that are completely empty and rows that
    # have no usable feature at all.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]

    valid = target.notna() & X.notna().any(axis=1)

    X = X.loc[valid].reset_index(drop=True)
    y = target.loc[valid].astype(int).reset_index(drop=True)

    n = len(X)
    a = int(n * 0.70)
    b = int(n * 0.85)
    Xtr, Xv, Xte = X.iloc[:a], X.iloc[a:b], X.iloc[b:]
    ytr, yv, yte = y.iloc[:a], y.iloc[a:b], y.iloc[b:]

    if HAVE_LGBM:
        pos = float(ytr.sum())
        neg = float(len(ytr) - ytr.sum())
        scale_pos_weight = neg / max(pos, 1.0)

        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(Xtr, ytr)
    pv = model.predict_proba(Xv)[:, 1]
    pt = model.predict_proba(Xte)[:, 1]
    vthr = float(np.quantile(pv, 0.75)) if np.isfinite(pv).all() else 0.5
    pred = (pt >= vthr).astype(int)

    metrics = {
        "task": "slope_movement_escalation_benchmark",
        "target_note": "Derived proxy target: future displacement-like change exceeds 3x local baseline volatility.",
        "feature_columns": cols,
        "threshold": vthr,
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, pt)) if yte.nunique() > 1 else None,
        "test_rows": int(len(Xte)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "threshold": vthr}, args.out)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
