import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

probabilities = np.load(
    "mineguard_v4_test_probabilities.npy"
)

targets = np.load(
    "mineguard_v4_test_targets.npy"
)

print("========================================")
print("   MINEGUARD V4 THRESHOLD ANALYSIS")
print("========================================")

print(
    f"Evaluation samples: {len(targets):,}"
)

print(
    f"Hazard cases:       {targets.sum():,}"
)

print(
    f"Hazard rate:        {targets.mean() * 100:.2f}%"
)

print()

thresholds = [
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05
]

print(
    "Threshold | Precision | Recall | F1"
)

print(
    "-------------------------------------"
)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        targets,
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
print("Best F1 among tested thresholds:")

best_threshold = None
best_f1 = -1

for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"F1:        {best_f1:.4f}"
)

print()
print(
    "Highest recall with precision >= 70%:"
)

best_recall = -1
best_constraint_threshold = None
best_constraint_precision = 0

for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )

    if precision >= 0.70 and recall > best_recall:

        best_recall = recall
        best_constraint_threshold = threshold
        best_constraint_precision = precision

print(
    f"Threshold: {best_constraint_threshold:.2f}"
)

print(
    f"Precision:  {best_constraint_precision:.4f}"
)

print(
    f"Recall:     {best_recall:.4f}"
)
