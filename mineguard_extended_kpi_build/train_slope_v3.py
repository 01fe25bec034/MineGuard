import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def make_features(df, sensor_cols):
    out = pd.DataFrame(index=df.index)

    for c in sensor_cols:
        s = pd.to_numeric(df[c], errors="coerce")

        out[f"{c}_now"] = s
        out[f"{c}_d1"] = s.diff(1)
        out[f"{c}_d4"] = s.diff(4)
        out[f"{c}_d8"] = s.diff(8)

        out[f"{c}_mean4"] = s.rolling(4, min_periods=2).mean()
        out[f"{c}_std4"] = s.rolling(4, min_periods=2).std()
        out[f"{c}_mean8"] = s.rolling(8, min_periods=4).mean()
        out[f"{c}_std8"] = s.rolling(8, min_periods=4).std()

        out[f"{c}_max8"] = s.rolling(8, min_periods=4).max()
        out[f"{c}_min8"] = s.rolling(8, min_periods=4).min()

    # Explicit rainfall features
    rain_cols = [
        c for c in sensor_cols
        if "precipitation" in c.lower()
        and "cumprecipitation" not in c.lower()
    ]

    for c in rain_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        out[f"{c}_sum4"] = s.rolling(4, min_periods=2).sum()
        out[f"{c}_sum8"] = s.rolling(8, min_periods=4).sum()
        out[f"{c}_sum16"] = s.rolling(16, min_periods=8).sum()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slope", default="slope_middle.csv")
    ap.add_argument("--gps", default="gps_ground_truth.csv")
    ap.add_argument("--out", default="outputs/slope_v3_model.joblib")
    ap.add_argument("--metrics", default="outputs/slope_v3_metrics.json")
    args = ap.parse_args()

    # ------------------------------------------------------------
    # 1. Load middle-station data
    # ------------------------------------------------------------
    slope = pd.read_csv(args.slope)

    slope["date_time_PST"] = pd.to_datetime(
        slope["date_time_PST"],
        errors="coerce"
    )

    slope = slope.dropna(subset=["date_time_PST"])
    slope = slope.sort_values("date_time_PST")
    slope = slope.drop_duplicates("date_time_PST")

    # GPS ground truth covers Oct 2016 through Sep 2017.
    # Restrict the sensor data before timezone conversion so older
    # water-year records do not participate in the DST conversion.
    slope = slope[
        (slope["date_time_PST"] >= pd.Timestamp("2016-10-01"))
        & (slope["date_time_PST"] <= pd.Timestamp("2017-10-01"))
    ].copy()

    # The middle-station timestamps are local California time.
    # During the DST fall-back hour some local timestamps occur twice.
    # Those ambiguous timestamps are dropped because there is no safe
    # way to infer which UTC instance they represent from the CSV alone.
    localized = slope["date_time_PST"].dt.tz_localize(
        "America/Los_Angeles",
        ambiguous="NaT",
        nonexistent="shift_forward"
    )

    valid_tz = localized.notna()

    slope = slope.loc[valid_tz].copy()

    slope["timestamp_utc"] = (
        localized.loc[valid_tz]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    # A DST conversion or source overlap can create duplicate UTC
    # timestamps. Keep one observation per timestamp.
    slope = (
        slope
        .drop_duplicates(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # 2. Load GPS ground truth
    # ------------------------------------------------------------
    gps = pd.read_csv(
        args.gps,
        header=3
    )

    gps["date_time_UTC"] = pd.to_datetime(
        gps["date_time_UTC"],
        errors="coerce"
    )

    gps["horizontal_displacement_m"] = pd.to_numeric(
        gps["horizontal_displacement_m"],
        errors="coerce"
    )

    gps = gps.dropna(
        subset=["date_time_UTC", "horizontal_displacement_m"]
    )

    gps = gps.sort_values("date_time_UTC")
    gps = gps.drop_duplicates("date_time_UTC")

    # ------------------------------------------------------------
    # 3. Interpolate GPS displacement onto sensor timestamps
    #    for t + 60 minutes.
    # ------------------------------------------------------------
    gps_series = (
        gps.set_index("date_time_UTC")[
            "horizontal_displacement_m"
        ]
        .sort_index()
    )

    # Ensure the GPS index is unique before reindex/interpolation.
    gps_series = gps_series[
        ~gps_series.index.duplicated(keep="first")
    ]

    current_gps = gps_series.reindex(
        gps_series.index.union(slope["timestamp_utc"])
    ).interpolate(method="time")

    current_gps = current_gps.reindex(slope["timestamp_utc"]).to_numpy()

    future_times = slope["timestamp_utc"] + pd.Timedelta(minutes=60)

    future_gps = gps_series.reindex(
        gps_series.index.union(future_times)
    ).interpolate(method="time")

    future_gps = future_gps.reindex(future_times).to_numpy()

    slope["gps_now"] = current_gps
    slope["gps_future_60m"] = future_gps

    slope["future_movement_60m_m"] = (
        slope["gps_future_60m"] - slope["gps_now"]
    )

    # ------------------------------------------------------------
    # 4. Select useful sensor columns
    # ------------------------------------------------------------
    exclude = {
        "date_time_PST",
        "timestamp_utc",
        "gps_now",
        "gps_future_60m",
        "future_movement_60m_m",
    }

    sensor_cols = []

    for c in slope.columns:
        if c in exclude:
            continue

        if pd.api.types.is_numeric_dtype(slope[c]):
            if slope[c].notna().sum() >= 500:
                sensor_cols.append(c)

    print("Sensor columns:", len(sensor_cols))

    # ------------------------------------------------------------
    # 5. Feature engineering
    # ------------------------------------------------------------
    X = make_features(slope, sensor_cols)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]

    # ------------------------------------------------------------
    # 6. Chronological split
    # ------------------------------------------------------------
    valid = (
        slope["future_movement_60m_m"].notna()
        & X.notna().any(axis=1)
    )

    X = X.loc[valid].reset_index(drop=True)
    movement = slope.loc[
        valid, "future_movement_60m_m"
    ].reset_index(drop=True)

    n = len(X)

    a = int(n * 0.70)
    b = int(n * 0.85)

    Xtr, Xv, Xte = (
        X.iloc[:a],
        X.iloc[a:b],
        X.iloc[b:]
    )

    mtr, mv, mte = (
        movement.iloc[:a],
        movement.iloc[a:b],
        movement.iloc[b:]
    )

    # ------------------------------------------------------------
    # 7. Lock movement threshold using TRAIN only
    #    Top 10% = unusually high future movement
    # ------------------------------------------------------------
    movement_threshold = float(
        mtr.quantile(0.90)
    )

    ytr = (mtr >= movement_threshold).astype(int)
    yv = (mv >= movement_threshold).astype(int)
    yte = (mte >= movement_threshold).astype(int)

    print("\nRows:")
    print("train:", len(Xtr))
    print("validation:", len(Xv))
    print("test:", len(Xte))

    print("\nMovement threshold (90th percentile of TRAIN):",
          movement_threshold, "m")

    print("\nPositive rates:")
    print("train:", ytr.mean())
    print("validation:", yv.mean())
    print("test:", yte.mean())

    # ------------------------------------------------------------
    # 8. Train LightGBM
    # ------------------------------------------------------------
    pos = float(ytr.sum())
    neg = float(len(ytr) - ytr.sum())
    scale_pos_weight = neg / max(pos, 1.0)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )

    model.fit(Xtr, ytr)

    # ------------------------------------------------------------
    # 9. Validation threshold selection
    #    Use F1 on validation only.
    # ------------------------------------------------------------
    pv = model.predict_proba(Xv)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 91)

    best_thr = 0.5
    best_f1 = -1

    for t in thresholds:
        pred = (pv >= t).astype(int)
        f1 = f1_score(yv, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    # ------------------------------------------------------------
    # 10. Final held-out TEST
    # ------------------------------------------------------------
    pt = model.predict_proba(Xte)[:, 1]
    pred = (pt >= best_thr).astype(int)

    metrics = {
        "task": "slope_high_movement_60min_benchmark",
        "target_note": (
            "Positive = future 60-minute GPS horizontal displacement "
            "increase at or above the 90th percentile of the training period."
        ),
        "movement_threshold_m": movement_threshold,
        "decision_threshold": best_thr,
        "train_rows": len(Xtr),
        "validation_rows": len(Xv),
        "test_rows": len(Xte),
        "train_positive_rate": float(ytr.mean()),
        "validation_positive_rate": float(yv.mean()),
        "test_positive_rate": float(yte.mean()),
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(
            precision_score(yte, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(yte, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(yte, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(yte, pt)
        ),
        "pr_auc": float(
            average_precision_score(yte, pt)
        ),
        "feature_count": X.shape[1],
    }

    print("\nFINAL TEST METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "decision_threshold": best_thr,
            "movement_threshold_m": movement_threshold,
        },
        args.out,
    )

    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2)
    )

    print("\nSaved:", args.out)
    print("Saved:", args.metrics)


if __name__ == "__main__":
    main()
