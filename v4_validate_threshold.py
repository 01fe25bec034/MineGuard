import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

RAW_FILE = "methane_data.csv"
TARGET_FILE = "mineguard_target.csv"

MODEL_FILE = "mineguard_v4_model.joblib"
FEATURE_FILE = "mineguard_v4_features.joblib"

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

# Same split used by V4
TARGET_ROWS = sum(
    len(chunk)
    for chunk in pd.read_csv(
        TARGET_FILE,
        usecols=["hazard_3_6min"],
        chunksize=CHUNK_SIZE
    )
)

RAW_ROWS = sum(
    len(chunk)
    for chunk in pd.read_csv(
        RAW_FILE,
        usecols=FEATURE_COLS,
        chunksize=CHUNK_SIZE
    )
)

usable_rows = min(
    TARGET_ROWS,
    RAW_ROWS - HISTORY
)

test_start = int(usable_rows * 0.80)
validation_start = int(test_start * 0.90)

print("========================================")
print("   MINEGUARD V4 VALIDATION ANALYSIS")
print("========================================")
print(f"Usable rows:       {usable_rows:,}")
print(f"Validation start:  {validation_start:,}")
print(f"Test start:        {test_start:,}")
print()

model = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)

print("V4 model loaded.")
print()

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

        change30 = ch4_max - ch4_max.shift(30)
        change60 = ch4_max - ch4_max.shift(60)
        change180 = ch4_max - ch4_max.shift(180)

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

        features = pd.DataFrame(feature_data)

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
        valid &= positions < TARGET_ROWS

        if valid.any():

            yield (
                positions[valid],
                features.loc[valid].to_numpy(
                    dtype=np.float32
                )
            )


# ---------------------------------------------------------
# EVALUATE VALIDATION + TEST
# ---------------------------------------------------------

target_reader = pd.read_csv(
    TARGET_FILE,
    usecols=["hazard_3_6min"],
    chunksize=CHUNK_SIZE
)

validation_probabilities = []
validation_targets = []

test_probabilities = []
test_targets = []

processed = 0

print("Running V4 validation/test inference...")
print()

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

    y_valid = y_all[
        y_indices[valid]
    ]

    pos_valid = positions[valid]

    validation_mask = (
        (pos_valid >= validation_start) &
        (pos_valid < test_start)
    )

    test_mask = (
        pos_valid >= test_start
    )

    if validation_mask.any():

        p = model.predict_proba(
            X_valid[validation_mask]
        )[:, 1]

        validation_probabilities.extend(p)
        validation_targets.extend(
            y_valid[validation_mask]
        )

    if test_mask.any():

        p = model.predict_proba(
            X_valid[test_mask]
        )[:, 1]

        test_probabilities.extend(p)
        test_targets.extend(
            y_valid[test_mask]
        )

    processed += len(y_all)


val_p = np.array(validation_probabilities)
val_y = np.array(validation_targets, dtype=np.int8)

test_p = np.array(test_probabilities)
test_y = np.array(test_targets, dtype=np.int8)

print(
    f"Validation samples: {len(val_y):,}"
)

print(
    f"Validation hazard rate: {val_y.mean() * 100:.2f}%"
)

print(
    f"Test samples: {len(test_y):,}"
)

print(
    f"Test hazard rate: {test_y.mean() * 100:.2f}%"
)

print()

# ---------------------------------------------------------
# THRESHOLD SEARCH ON VALIDATION ONLY
# ---------------------------------------------------------

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

best_f1 = -1
best_f1_threshold = None

best_f2 = -1
best_f2_threshold = None

best_recall = -1
best_recall_threshold = None
best_recall_precision = None

for threshold in thresholds:

    pred = (
        val_p >= threshold
    ).astype(np.int8)

    precision = precision_score(
        val_y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        val_y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        val_y,
        pred,
        zero_division=0
    )

    f2 = fbeta_score(
        val_y,
        pred,
        beta=2,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1
        best_f1_threshold = threshold

    if f2 > best_f2:

        best_f2 = f2
        best_f2_threshold = threshold

    # Safety-oriented constraint:
    # retain at least 70% precision
    if precision >= 0.70 and recall > best_recall:

        best_recall = recall
        best_recall_threshold = threshold
        best_recall_precision = precision


print("========================================")
print(" VALIDATION THRESHOLD SELECTION")
print("========================================")

print(
    f"Best F1 threshold: "
    f"{best_f1_threshold:.2f}"
)

print(
    f"Validation F1: "
    f"{best_f1:.4f}"
)

print()

print(
    f"Best F2 threshold: "
    f"{best_f2_threshold:.2f}"
)

print(
    f"Validation F2: "
    f"{best_f2:.4f}"
)

print()
print("========================================")
print(" VALIDATION SAFETY THRESHOLD ANALYSIS")
print("========================================")

for required_precision in [0.70, 0.60, 0.50, 0.40]:

    best_recall = -1
    best_threshold = None
    best_precision = None

    for threshold in thresholds:

        pred = (
            val_p >= threshold
        ).astype(np.int8)

        precision = precision_score(
            val_y,
            pred,
            zero_division=0
        )

        recall = recall_score(
            val_y,
            pred,
            zero_division=0
        )

        if (
            precision >= required_precision
            and recall > best_recall
        ):

            best_recall = recall
            best_threshold = threshold
            best_precision = precision

    print()
    print(
        f"Precision >= {required_precision * 100:.0f}%"
    )

    if best_threshold is None:

        print("No validation threshold satisfies this constraint.")

    else:

        print(
            f"Threshold: {best_threshold:.2f}"
        )

        print(
            f"Precision: {best_precision:.4f}"
        )

        print(
            f"Recall:    {best_recall:.4f}"
        )


# ---------------------------------------------------------
# LOCK THRESHOLD USING VALIDATION ONLY
# ---------------------------------------------------------

# F2 is chosen because MineGuard is safety-oriented
# and recall should receive greater weight than precision.

LOCKED_THRESHOLD = best_f2_threshold

print()
print("========================================")
print("       LOCKED OPERATING THRESHOLD")
print("========================================")

print(
    f"MineGuard threshold: {LOCKED_THRESHOLD:.2f}"
)

# ---------------------------------------------------------
# CHOOSE SAFETY-ORIENTED THRESHOLD
# ---------------------------------------------------------

LOCKED_THRESHOLD = best_recall_threshold

print()
# ============================================================
# LOCKED OPERATING THRESHOLD
# ============================================================

if best_f2_threshold is None:
    raise RuntimeError("Best F2 threshold was not found.")

LOCKED_THRESHOLD = float(best_f2_threshold)

print("\n========================================")
print("       LOCKED OPERATING THRESHOLD")
print("========================================")
print(f"MineGuard threshold: {LOCKED_THRESHOLD:.2f}")

# ---------------------------------------------------------
# FINAL TEST — NEVER USED FOR SELECTION
# ---------------------------------------------------------

final_pred = (
    test_p >= LOCKED_THRESHOLD
).astype(np.int8)

final_precision = precision_score(
    test_y,
    final_pred,
    zero_division=0
)

final_recall = recall_score(
    test_y,
    final_pred,
    zero_division=0
)

final_f1 = f1_score(
    test_y,
    final_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    test_y,
    test_p
)

pr_auc = average_precision_score(
    test_y,
    test_p
)

print()
print("========================================")
print("       FINAL LOCKED V4 TEST")
print("========================================")

print(
    f"Threshold: {LOCKED_THRESHOLD:.2f}"
)

print(
    f"Precision: {final_precision:.4f}"
)

print(
    f"Recall:    {final_recall:.4f}"
)

print(
    f"F1:        {final_f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)

print()

np.save(
    "mineguard_v4_locked_test_probabilities.npy",
    test_p
)

np.save(
    "mineguard_v4_locked_test_targets.npy",
    test_y
)

with open(
    "mineguard_v4_threshold.txt",
    "w"
) as f:

    f.write(
        f"{LOCKED_THRESHOLD:.2f}\n"
    )

print("Saved:")
print("  mineguard_v4_threshold.txt")
print("  mineguard_v4_locked_test_probabilities.npy")
print("  mineguard_v4_locked_test_targets.npy")
