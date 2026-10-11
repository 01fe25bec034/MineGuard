import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

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

print("========================================")
print("     MINEGUARD V1 THRESHOLD TEST")
print("========================================")

# ---------------------------------------------------------
# LOAD SAVED MODEL
# ---------------------------------------------------------

model = joblib.load("mineguard_model.joblib")
scaler = joblib.load("mineguard_scaler.joblib")

target = pd.read_csv(
    TARGET_FILE,
    usecols=["hazard_3_6min"]
)

target_values = target[
    "hazard_3_6min"
].to_numpy(dtype=np.int8)

target_rows = len(target_values)

usable_rows = target_rows

split_row = int(
    usable_rows * 0.80
)

print()
print("Target rows:", f"{target_rows:,}")
print("Test starts at:", f"{split_row:,}")
print(
    "Expected test rows:",
    f"{usable_rows - split_row:,}"
)

# ---------------------------------------------------------
# EXACT SAME FEATURE GENERATOR AS TRAINING
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
# EVALUATION
# ---------------------------------------------------------

all_probabilities = []
all_targets = []

processed = 0

print()
print("Evaluating saved V1 model...")
print()

for positions, X in generate_features():

    # Exact target alignment used by the training script
    y_indices = positions - processed

    valid = (
        (y_indices >= 0) &
        (y_indices < target_rows)
    )

    if not valid.any():
        processed += len(positions)
        continue

    X_valid = X[valid]

    y_valid = target_values[
        y_indices[valid]
    ]

    pos_valid = positions[valid]

    test_mask = (
        pos_valid >= split_row
    )

    if test_mask.any():

        X_test = X_valid[test_mask]

        y_test_chunk = y_valid[
            test_mask
        ]

        # IMPORTANT:
        # Convert to NumPy before scaler.
        X_scaled = scaler.transform(
            X_test
        )

        probabilities = (
            model.predict_proba(
                X_scaled
            )[:, 1]
        )

        all_probabilities.extend(
            probabilities
        )

        all_targets.extend(
            y_test_chunk
        )

    processed += len(positions)

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

y_true = np.array(
    all_targets,
    dtype=np.int8
)

y_prob = np.array(
    all_probabilities
)

print()
print("========================================")
print("       MINEGUARD THRESHOLD RESULTS")
print("========================================")

print()
print(
    "Evaluation samples:",
    f"{len(y_true):,}"
)

print(
    "Expected samples:  ",
    f"{usable_rows - split_row:,}"
)

print(
    "Hazard cases:",
    f"{y_true.sum():,}"
)

print(
    "Hazard rate:",
    f"{y_true.mean() * 100:.2f}%"
)

print()
print(
    "Threshold | Precision | Recall | F1"
)
print("----------------------------------------")

thresholds = [
    0.50,
    0.40,
    0.30,
    0.20,
    0.10,
    0.05
]

for threshold in thresholds:

    predictions = (
        y_prob >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

print()
print("========================================")
print("       THRESHOLD TEST COMPLETE")
print("========================================")