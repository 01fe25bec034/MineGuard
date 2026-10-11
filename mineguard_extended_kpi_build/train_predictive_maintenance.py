from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    raise SystemExit("LightGBM is required. Install with: pip install lightgbm")

NASA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def load_subset(root: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trainp = root / f"train_{subset}.txt"
    testp = root / f"test_{subset}.txt"
    rul_p = root / f"RUL_{subset}.txt"
    if not trainp.exists():
        raise FileNotFoundError(f"Missing {trainp}. Extract NASA CMAPSSData.zip into {root}.")
    train = pd.read_csv(trainp, sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(testp, sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(rul_p, header=None, names=["rul"])
    return train, test.assign(test_rul=np.repeat(rul["rul"].values, [0]*0)) if False else rul


def add_rul(train: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    maxc = train.groupby("unit")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = np.minimum(maxc - train["cycle"], cap)
    return train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cmapss_data")
    ap.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--out", default="maintenance_model.joblib")
    ap.add_argument("--metrics", default="maintenance_metrics.json")
    args = ap.parse_args()

    root = Path(args.data); root.mkdir(parents=True, exist_ok=True)
    z = root / "CMAPSSData.zip"
    if not z.exists():
        print(f"Downloading NASA CMAPSS from {NASA_URL}")
        urllib.request.urlretrieve(NASA_URL, z)
        with zipfile.ZipFile(z, "r") as zz:
            zz.extractall(root)

    train = pd.read_csv(root / f"train_{args.subset}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(root / f"test_{args.subset}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(root / f"RUL_{args.subset}.txt", header=None, names=["RUL"])
    train = add_rul(train)

    features = [c for c in COLS if c not in DROP + ["unit"]]
    X_train = train[features]
    y_train = train["RUL"]
    X_test = test[features]

    model = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    test_with_pred = test.copy()
    test_with_pred["pred_rul"] = np.clip(model.predict(X_test), 0, 125)
    final_pred = (
        test_with_pred.sort_values(["unit", "cycle"])
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")["pred_rul"]
        .to_numpy()
    )
    true = rul["RUL"].to_numpy()

    mae = mean_absolute_error(true, final_pred)
    rmse = mean_squared_error(true, final_pred) ** 0.5
    availability_proxy = float(np.mean(final_pred > 30.0))

    metrics = {
        "task": "predictive_maintenance_rul_benchmark",
        "subset": args.subset,
        "mae_cycles": float(mae),
        "rmse_cycles": float(rmse),
        "predicted_availability_proxy": availability_proxy,
        "availability_rule": ">30 predicted cycles remaining",
        "note": "NASA C-MAPSS is an equipment prognostics benchmark, not mine-specific machinery; do not present as mine-site availability validation.",
    }
    joblib.dump({"model": model, "features": features}, args.out)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
