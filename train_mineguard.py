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

FEATURE_COLS = [
    "AN422",
    "AN423",
    "TP1721",
    "RH1722",
    "TP1711",
    "RH1712",
    "MM263",
    "MM264",
    "MM256",
    "V"
]

FEATURE_NAMES = []

for col in FEATURE_COLS:
    FEATURE_NAMES += [
        f"{col}_current",
        f"{col}_mean60",
        f"{col}_mean180",
        f"{col}_mean300",
        f"{col}_change60",
        f"{col}_change180",
        f"{col}_change300",
        f"{col}_max60"
    ]

N_FEATURES = len(FEATURE_NAMES)

print("========================================")
print("       MINEGUARD STREAMING TRAINING")
print("========================================")
print(f"Features: {N_FEATURES}")
print(f"Chunk size: {CHUNK_SIZE:,}")
print()

# ---------------------------------------------------------
# Determine number of usable rows
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

split_row = int(usable_rows * 0.80)

print(f"Raw rows:       {raw_rows:,}")
print(f"Target rows:    {target_rows:,}")
print(f"Usable rows:    {usable_rows:,}")
print(f"Training rows:  {split_row:,}")
print(f"Testing rows:   {usable_rows - split_row:,}")
print()

# ---------------------------------------------------------
# Feature generator
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

        for col in FEATURE_COLS:

            s = combined[col]

            feature_data[f"{col}_current"] = s

            feature_data[f"{col}_mean60"] = (
                s.rolling(
                    60,
                    min_periods=60
                ).mean()
            )

            feature_data[f"{col}_mean180"] = (
                s.rolling(
                    180,
                    min_periods=180
                ).mean()
            )

            feature_data[f"{col}_mean300"] = (
                s.rolling(
                    300,
                    min_periods=300
                ).mean()
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

            feature_data[f"{col}_max60"] = (
                s.rolling(
                    60,
                    min_periods=60
                ).max()
            )

        features = pd.DataFrame(
            feature_data
        )

        # Keep only rows belonging to current chunk
        start = len(previous)

        features = features.iloc[
            start:start + len(chunk)
        ].copy()

        # Update history for next chunk
        previous = combined.iloc[
            -HISTORY:
        ].copy()

        # Global row numbers
        positions = np.arange(
            global_position,
            global_position + len(chunk)
        )

        global_position += len(chunk)

        # Remove rows before 300 seconds
        valid = positions >= HISTORY

        # Target file ends 360 rows before raw file
        valid &= positions < target_rows

        if valid.any():

            yield (
                positions[valid],
                features.loc[valid].to_numpy(
                    dtype=np.float32
                )
            )


# ---------------------------------------------------------
# PASS 1 — Fit scaler using TRAINING DATA ONLY
# ---------------------------------------------------------

print("PASS 1/3: fitting feature scaler...")
print()

scaler = StandardScaler()

processed = 0

for positions, X in generate_features():

    mask = positions < split_row

    if mask.any():

        scaler.partial_fit(
            X[mask]
        )

        processed += mask.sum()

print()
print(
    f"Scaler fitted on {processed:,} training rows."
)

# ---------------------------------------------------------
# PASS 2 — Train streaming classifier
# ---------------------------------------------------------

print()
print("PASS 2/3: training MineGuard classifier...")
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
first_model_call = True

for positions, X in generate_features():

    # Read matching target chunk
    try:
        target_chunk = next(target_reader)
    except StopIteration:
        break

    y_all = target_chunk[
        "hazard_3_6min"
    ].to_numpy(dtype=np.int8)

    # Keep only positions represented by target chunk
    local_positions = positions[
        positions < len(y_all) + processed
    ]

    # Easier alignment using global positions
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

        X_train = X_valid[train_mask]
        y_train = y_valid[train_mask]

        X_train = scaler.transform(
            X_train
        )

        # Give hazard events higher importance
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

print("\nClassifier training complete.")

# ---------------------------------------------------------
# PASS 3 — Evaluate on chronological test data
# ---------------------------------------------------------

print()
print("PASS 3/3: evaluating chronological test set...")
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

# Save V1 test predictions for threshold analysis
np.save(
    "mineguard_test_probabilities.npy",
    probabilities
)

np.save(
    "mineguard_test_targets.npy",
    y_test
)

print()
print("Saved threshold-analysis files:")
print("  mineguard_test_probabilities.npy")
print("  mineguard_test_targets.npy")

predictions = (
    probabilities >= 0.50
).astype(np.int8)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("========================================")
print("             MINEGUARD RESULTS")
print("========================================")

print()
print("Test samples:", f"{len(y_test):,}")

print(
    "\nTest hazard rate:",
    f"{y_test.mean() * 100:.2f}%"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)

print("\nConfusion matrix:")
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

print(
    "\nROC-AUC:",
    f"{roc_auc:.4f}"
)

print(
    "PR-AUC :",
    f"{pr_auc:.4f}"
)

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(
    model,
    "mineguard_model.joblib"
)

joblib.dump(
    scaler,
    "mineguard_scaler.joblib"
)

joblib.dump(
    FEATURE_NAMES,
    "mineguard_features.joblib"
)

print()
print("Saved:")
print("  mineguard_model.joblib")
print("  mineguard_scaler.joblib")
print("  mineguard_features.joblib")

print()
print("========================================")
print("       MINEGUARD TRAINING COMPLETE")
print("========================================")