import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

RAW_FILE = "methane_data.csv"
TARGET_FILE = "mineguard_target.csv"

CHUNK_SIZE = 100000
HISTORY = 300

# V2 focuses heavily on methane + ventilation context
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

# Additional methane-specific features
FEATURE_NAMES += [
    "CH4_max_current",
    "CH4_mean30",
    "CH4_mean60",
    "CH4_mean180",
    "CH4_change30",
    "CH4_change60",
    "CH4_change180",
    "CH4_spread",
    "CH4_spread_mean60",
    "VENT_mean60",
    "VENT_change60",
]

N_FEATURES = len(FEATURE_NAMES)

print("========================================")
print("       MINEGUARD V2 TRAINING")
print("========================================")
print(f"Features: {N_FEATURES}")
print(f"Chunk size: {CHUNK_SIZE:,}")
print()

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

split_row = int(usable_rows * 0.80)

print(f"Raw rows:       {raw_rows:,}")
print(f"Target rows:    {target_rows:,}")
print(f"Usable rows:    {usable_rows:,}")
print(f"Training rows:  {split_row:,}")
print(f"Testing rows:   {usable_rows - split_row:,}")
print()


def generate_features():

    previous = pd.DataFrame(columns=FEATURE_COLS)

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

        # Combined methane signal
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

        feature_data["CH4_change30"] = (
            ch4_max - ch4_max.shift(30)
        )

        feature_data["CH4_change60"] = (
            ch4_max - ch4_max.shift(60)
        )

        feature_data["CH4_change180"] = (
            ch4_max - ch4_max.shift(180)
        )

        feature_data["CH4_spread"] = (
            ch4.max(axis=1) - ch4.min(axis=1)
        )

        feature_data["CH4_spread_mean60"] = (
            feature_data["CH4_spread"]
            .rolling(60, min_periods=60)
            .mean()
        )

        # Ventilation context
        vent = combined[
            ["AN422", "AN423"]
        ].mean(axis=1)

        feature_data["VENT_mean60"] = (
            vent.rolling(60, min_periods=60).mean()
        )

        feature_data["VENT_change60"] = (
            vent - vent.shift(60)
        )

        features = pd.DataFrame(feature_data)

        start = len(previous)

        features = features.iloc[
            start:start + len(chunk)
        ].copy()

        previous = combined.iloc[-HISTORY:].copy()

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


print("PASS 1/3: fitting V2 feature scaler...")
print()

scaler = StandardScaler()

processed = 0

for positions, X in generate_features():

    mask = positions < split_row

    if mask.any():
        scaler.partial_fit(X[mask])
        processed += mask.sum()

print()
print(
    f"Scaler fitted on {processed:,} training rows."
)

print()
print("PASS 2/3: training V2 classifier...")
print()

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    max_iter=1,
    learning_rate="optimal",
    random_state=42
)

classes = np.array([0, 1])

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
    ].to_numpy(dtype=np.int8)

    y_indices = positions - processed

    valid = (
        (y_indices >= 0) &
        (y_indices < len(y_all))
    )

    if not valid.any():
        processed += len(y_all)
        continue

    X_valid = X[valid]
    y_valid = y_all[y_indices[valid]]

    train_mask = positions[valid] < split_row

    if train_mask.any():

        X_train = scaler.transform(
            X_valid[train_mask]
        )

        y_train = y_valid[train_mask]

        weights = np.where(
            y_train == 1,
            5.0,
            1.0
        )

        model.partial_fit(
            X_train,
            y_train,
            classes=classes,
            sample_weight=weights
        )

    processed += len(y_all)

print()
print("V2 classifier training complete.")

print()
print("PASS 3/3: evaluating V2 chronological test set...")
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
    ].to_numpy(dtype=np.int8)

    y_indices = positions - processed

    valid = (
        (y_indices >= 0) &
        (y_indices < len(y_all))
    )

    if not valid.any():
        processed += len(y_all)
        continue

    X_valid = X[valid]
    y_valid = y_all[y_indices[valid]]
    pos_valid = positions[valid]

    test_mask = pos_valid >= split_row

    if test_mask.any():

        X_test = scaler.transform(
            X_valid[test_mask]
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        all_probabilities.extend(probabilities)
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

print()
print("========================================")
print("             MINEGUARD V2 RESULTS")
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

joblib.dump(
    model,
    "mineguard_v2_model.joblib"
)

joblib.dump(
    scaler,
    "mineguard_v2_scaler.joblib"
)

joblib.dump(
    FEATURE_NAMES,
    "mineguard_v2_features.joblib"
)

np.save(
    "mineguard_v2_test_probabilities.npy",
    probabilities
)

np.save(
    "mineguard_v2_test_targets.npy",
    y_test
)

print()
print("Saved V2 files:")
print("  mineguard_v2_model.joblib")
print("  mineguard_v2_scaler.joblib")
print("  mineguard_v2_features.joblib")
print("  mineguard_v2_test_probabilities.npy")
print("  mineguard_v2_test_targets.npy")

print()
print("========================================")
print("       MINEGUARD V2 COMPLETE")
print("========================================")
