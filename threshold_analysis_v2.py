import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

probabilities = np.load("mineguard_v2_test_probabilities.npy")
targets = np.load("mineguard_v2_test_targets.npy")

print("========================================")
print("   MINEGUARD V2 THRESHOLD ANALYSIS")
print("========================================")

print(f"Evaluation samples: {len(targets):,}")
print(f"Hazard cases:       {targets.sum():,}")
print(f"Hazard rate:        {targets.mean() * 100:.2f}%")
print()

thresholds = [
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

print("Threshold | Precision | Recall | F1")
print("-------------------------------------")

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
