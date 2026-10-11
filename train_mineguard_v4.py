import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

RAW_FILE = "methane_data.csv"
TARGET_FILE = "mineguard_target.csv"

CHUNK_SIZE = 100000
HISTORY = 300

FEATURE_COLS = [
    "MM263",
    "MM264",
    "MM256",
    "AN422",
    "AN423",
    "TP1721",
    "RH1722",
    "TP1711",
    "RH1712",
    "V"
]

FEATURE_NAMES = []

for col in FEATURE_COLS:
    FEATURE_NAMES += [
        f"{col}_current",
        f"{col}_mean30",
        f"{col}_mean60",
        f"{col}_mean180",
        f"{col}_mean300",
        f"{col}_change30",
        f"{col}_change60",
        f"{col}_change180",
        f"{col}_change300",
        f"{col}_std60",
        f"{col}_max60",
    ]

FEATURE_NAMES += [
    "CH4_max_current",
    "CH4_mean30",
    "CH4_mean60",
    "CH4_mean180",
    "CH4_change30",
    "CH4_change60",
    "CH4_change180",
    "CH4_acceleration",
    "CH4_trend_ratio",
    "CH4_std30",
    "CH4_std60",
    "CH4_std180",
    "CH4_spread",
    "CH4_spread_mean60",
    "VENT_mean60",
    "VENT_change30",
    "VENT_change60",
    "VENT_change180",
]

N_FEATURES = len(FEATURE_NAMES)

print("========================================")
print("       MINEGUARD V4 LIGHTGBM")
print("========================================")
print(f"Features: {N_FEATURES}")
print(f"Chunk size: {CHUNK_SIZE:,}")
print()

# ---------------------------------------------------------
# DATA SIZE
# ---------------------------------------------------------

raw_rows = sum(
    len(chunk)
    for chunk in pd.read_csv(
        RAW_FILE,
        usecols=FEATURE_COLS,
        chunksize=CHUNK_SIZE
    )
)

target_rows = sum(
    len(chunk)
    for chunk in pd.read_csv(
        TARGET_FILE,
        usecols=["hazard_3_6min"],
        chunksize=CHUNK_SIZE
    )
)

usable_rows = min(
    target_rows,
    raw_rows - HISTORY
)

test_start = int(usable_rows * 0.80)

# Use first 90% of the original training period for training.
# Last 10% of training period is validation.
validation_start = int(test_start * 0.90)

print(f"Raw rows:            {raw_rows:,}")
print(f"Target rows:         {target_rows:,}")
print(f"Usable rows:         {usable_rows:,}")
print(f"Training end:        {test_start:,}")
print(f"Validation start:    {validation_start:,}")
print(f"Test start:          {test_start:,}")
print(f"Test rows:           {usable_rows - test_start:,}")
print()

# ---------------------------------------------------------
# FEATURE GENERATOR
# ---------------------------------------------------------

def generate_features():

    previous = pd.DataFrame(
        columns=FEATURE_COLS
    )

    global_position = 0

    for chunk in pd.read_csv(
        RAW_FILE,
        usecols=FEATURE_COLS,
        chunksize=CHUNK_SIZE
    ):

        combined = pd.concat(
            [previous, chunk],
            ignore_index=True
        )

        feature_data = {}

        # -----------------------------
        # Per-sensor temporal features
        # -----------------------------

        for col in FEATURE_COLS:

            s = combined[col]

            feature_data[f"{col}_current"] = s

            feature_data[f"{col}_mean30"] = (
                s.rolling(30, min_periods=30).mean()
            )

            feature_data[f"{col}_mean60"] = (
                s.rolling(60, min_periods=60).mean()
            )

            feature_data[f"{col}_mean180"] = (
                s.rolling(180, min_periods=180).mean()
            )

            feature_data[f"{col}_mean300"] = (
                s.rolling(300, min_periods=300).mean()
            )

            feature_data[f"{col}_change30"] = (
                s - s.shift(30)
            )

            feature_data[f"{col}_change60"] = (
                s - s.shift(60)
            )

            feature_data[f"{col}_change180"] = (
                s - s.shift(180)
            )

            feature_data[f"{col}_change300"] = (
                s - s.shift(300)
            )

            feature_data[f"{col}_std60"] = (
                s.rolling(60, min_periods=60).std()
            )

            feature_data[f"{col}_max60"] = (
                s.rolling(60, min_periods=60).max()
            )

        # -----------------------------
        # Combined methane signal
        # -----------------------------

        ch4 = combined[
            ["MM263", "MM264", "MM256"]
        ]

        ch4_max = ch4.max(axis=1)

        feature_data["CH4_max_current"] = ch4_max

        feature_data["CH4_mean30"] = (
            ch4_max.rolling(30, min_periods=30).mean()
        )

        feature_data["CH4_mean60"] = (
            ch4_max.rolling(60, min_periods=60).mean()
        )

        feature_data["CH4_mean180"] = (
            ch4_max.rolling(180, min_periods=180).mean()
        )

        change30 = (
            ch4_max - ch4_max.shift(30)
        )

        change60 = (
            ch4_max - ch4_max.shift(60)
        )

        change180 = (
            ch4_max - ch4_max.shift(180)
        )

        feature_data["CH4_change30"] = change30
        feature_data["CH4_change60"] = change60
        feature_data["CH4_change180"] = change180

        feature_data["CH4_acceleration"] = (
            change30 - (change60 - change30)
        )

        feature_data["CH4_trend_ratio"] = (
            feature_data["CH4_mean30"]
            /
            (feature_data["CH4_mean180"].abs() + 0.01)
        )

        feature_data["CH4_std30"] = (
            ch4_max.rolling(30, min_periods=30).std()
        )

        feature_data["CH4_std60"] = (
            ch4_max.rolling(60, min_periods=60).std()
        )

        feature_data["CH4_std180"] = (
            ch4_max.rolling(180, min_periods=180).std()
        )

        feature_data["CH4_spread"] = (
            ch4.max(axis=1) -
            ch4.min(axis=1)
        )

        feature_data["CH4_spread_mean60"] = (
            feature_data["CH4_spread"]
            .rolling(60, min_periods=60)
            .mean()
        )

        # -----------------------------
        # Ventilation context
        # -----------------------------

        vent = combined[
            ["AN422", "AN423"]
        ].mean(axis=1)

        feature_data["VENT_mean60"] = (
            vent.rolling(60, min_periods=60).mean()
        )

        feature_data["VENT_change30"] = (
            vent - vent.shift(30)
        )

        feature_data["VENT_change60"] = (
            vent - vent.shift(60)
        )

        feature_data["VENT_change180"] = (
            vent - vent.shift(180)
        )

        features = pd.DataFrame(
            feature_data
        )

        start = len(previous)

        features = features.iloc[
            start:start + len(chunk)
        ].copy()

        previous = combined.iloc[
            -HISTORY:
        ].copy()

        positions = np.arange(
            global_position,
            global_position + len(chunk)
        )

        global_position += len(chunk)

        valid = positions >= HISTORY
        valid &= positions < target_rows

        if valid.any():

            yield (
                positions[valid],
                features.loc[valid].to_numpy(
                    dtype=np.float32
                )
            )


# ---------------------------------------------------------
# PASS 1 — BUILD TRAINING SAMPLE
# ---------------------------------------------------------

print("PASS 1/3: building LightGBM training sample...")
print()

train_X = []
train_y = []

validation_X = []
validation_y = []

target_reader = pd.read_csv(
    TARGET_FILE,
    usecols=["hazard_3_6min"],
    chunksize=CHUNK_SIZE
)

processed = 0

for positions, X in generate_features():

    try:
        target_chunk = next(target_reader)

    except StopIteration:
        break

    y_all = target_chunk[
        "hazard_3_6min"
    ].to_numpy(
        dtype=np.int8
    )

    y_indices = positions - processed

    valid = (
        (y_indices >= 0) &
        (y_indices < len(y_all))
    )

    if not valid.any():

        processed += len(y_all)
        continue

    X_valid = X[valid]

    y_valid = y_all[
        y_indices[valid]
    ]

    pos_valid = positions[valid]

    # Training sample:
    # use every third row in first 90% of training.
    train_mask = (
        (pos_valid < validation_start) &
        ((pos_valid % 3) == 0)
    )

    # Chronological validation:
    validation_mask = (
        (pos_valid >= validation_start) &
        (pos_valid < test_start)
    )

    if train_mask.any():

        train_X.append(
            X_valid[train_mask]
        )

        train_y.append(
            y_valid[train_mask]
        )

    if validation_mask.any():

        validation_X.append(
            X_valid[validation_mask]
        )

        validation_y.append(
            y_valid[validation_mask]
        )

    processed += len(y_all)


X_train = np.vstack(train_X)
y_train = np.concatenate(train_y)

X_val = np.vstack(validation_X)
y_val = np.concatenate(validation_y)

del train_X
del train_y
del validation_X
del validation_y

print(
    f"Training rows loaded:   {len(y_train):,}"
)

print(
    f"Validation rows loaded: {len(y_val):,}"
)

print(
    f"Training hazard rate:    {y_train.mean() * 100:.2f}%"
)

print(
    f"Validation hazard rate:  {y_val.mean() * 100:.2f}%"
)

print()


# ---------------------------------------------------------
# PASS 2 — LIGHTGBM TRAINING
# ---------------------------------------------------------

print("PASS 2/3: training LightGBM V4...")
print()

model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    max_bin=63,
    scale_pos_weight=5.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=40,
            verbose=True
        )
    ],
)

best_iteration = model.best_iteration_

print()
print(
    f"Best iteration: {best_iteration}"
)

print()


# ---------------------------------------------------------
# PASS 3 — FULL CHRONOLOGICAL TEST
# ---------------------------------------------------------

print("PASS 3/3: evaluating full chronological test set...")
print()

target_reader = pd.read_csv(
    TARGET_FILE,
    usecols=["hazard_3_6min"],
    chunksize=CHUNK_SIZE
)

all_probabilities = []
all_targets = []

processed = 0

for positions, X in generate_features():

    try:
        target_chunk = next(target_reader)

    except StopIteration:
        break

    y_all = target_chunk[
        "hazard_3_6min"
    ].to_numpy(
        dtype=np.int8
    )

    y_indices = positions - processed

    valid = (
        (y_indices >= 0) &
        (y_indices < len(y_all))
    )

    if not valid.any():

        processed += len(y_all)
        continue

    X_valid = X[valid]

    y_valid = y_all[
        y_indices[valid]
    ]

    pos_valid = positions[valid]

    test_mask = (
        pos_valid >= test_start
    )

    if test_mask.any():

        X_test = X_valid[test_mask]

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        all_probabilities.extend(
            probabilities
        )

        all_targets.extend(
            y_valid[test_mask]
        )

    processed += len(y_all)


y_test = np.array(
    all_targets,
    dtype=np.int8
)

probabilities = np.array(
    all_probabilities
)

predictions = (
    probabilities >= 0.50
).astype(np.int8)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("========================================")
print("          MINEGUARD V4 RESULTS")
print("========================================")

print()
print(
    "Test samples:",
    f"{len(y_test):,}"
)

print(
    "Test hazard rate:",
    f"{y_test.mean() * 100:.2f}%"
)

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)

print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

print()
print(
    "ROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "PR-AUC :",
    f"{pr_auc:.4f}"
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

joblib.dump(
    model,
    "mineguard_v4_model.joblib"
)

joblib.dump(
    FEATURE_NAMES,
    "mineguard_v4_features.joblib"
)

np.save(
    "mineguard_v4_test_probabilities.npy",
    probabilities
)

np.save(
    "mineguard_v4_test_targets.npy",
    y_test
)

print()
print("Saved V4 files:")
print("  mineguard_v4_model.joblib")
print("  mineguard_v4_features.joblib")
print("  mineguard_v4_test_probabilities.npy")
print("  mineguard_v4_test_targets.npy")

print()
print("========================================")
print("       MINEGUARD V4 COMPLETE")
print("========================================")
