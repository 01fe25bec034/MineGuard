import numpy as np
import pandas as pd

DATA_FILE = "methane_data.csv"
PROB_FILE = "mineguard_v4_locked_test_probabilities.npy"
OUTPUT_FILE = "dashboard_v4_replay.csv"

USABLE_ROWS = 9_199_570
TEST_START = 7_359_656

# Keep a compact but useful replay window.
# 100,000 seconds ≈ 27.8 hours.
REPLAY_ROWS = 100_000

usecols = [
    "year", "month", "day", "hour", "minute", "second",
    "AN422", "AN423",
    "TP1721", "RH1722",
    "TP1711", "RH1712",
    "MM263", "MM264", "MM256",
    "V"
]

print("=" * 60)
print("      CREATE MINEGUARD V4 DASHBOARD REPLAY")
print("=" * 60)

probs = np.load(PROB_FILE)

expected = USABLE_ROWS - TEST_START

if len(probs) != expected:
    raise ValueError(
        f"Prediction count mismatch: {len(probs)} vs {expected}"
    )

rows = []
total_seen = 0

print("Reading test-period data...")

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=usecols,
    chunksize=250_000
):
    start = total_seen
    end = total_seen + len(chunk)

    overlap_start = max(start, TEST_START)
    overlap_end = min(end, USABLE_ROWS)

    if overlap_start < overlap_end:
        a = overlap_start - start
        b = overlap_end - start
        rows.append(chunk.iloc[a:b].copy())

    total_seen = end

    if total_seen >= USABLE_ROWS:
        break

test = pd.concat(rows, ignore_index=True)

# Keep the last REPLAY_ROWS for the dashboard.
test = test.tail(REPLAY_ROWS).reset_index(drop=True)

# Corresponding probability positions in the complete test array.
prob_start = len(probs) - len(test)
test["risk_probability"] = probs[prob_start:]

test["timestamp"] = pd.to_datetime(
    {
        "year": test["year"],
        "month": test["month"],
        "day": test["day"],
        "hour": test["hour"],
        "minute": test["minute"],
        "second": test["second"],
    }
)

test["max_ch4"] = test[
    ["MM263", "MM264", "MM256"]
].max(axis=1)

test.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(f"Rows saved: {len(test):,}")
print(f"Output:     {OUTPUT_FILE}")
print("=" * 60)