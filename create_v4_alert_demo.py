import numpy as np
import pandas as pd

DATA_FILE = "methane_data.csv"
PROB_FILE = "mineguard_v4_locked_test_probabilities.npy"
OUTPUT_FILE = "dashboard_v4_replay.csv"

USABLE_ROWS = 9_199_570
TEST_START = 7_359_656
WINDOW = 600

usecols = [
    "year", "month", "day", "hour", "minute", "second",
    "AN422", "AN423",
    "TP1721", "RH1722",
    "TP1711", "RH1712",
    "MM263", "MM264", "MM256",
    "V"
]

print("=" * 60)
print("     CREATE V4 HIGH-RISK DASHBOARD DEMO")
print("=" * 60)

probs = np.load(PROB_FILE)

expected = USABLE_ROWS - TEST_START

if len(probs) != expected:
    raise ValueError(
        f"Probability count mismatch: {len(probs)} != {expected}"
    )

# Find the highest-risk prediction.
best_local_idx = int(np.argmax(probs))
best_probability = float(probs[best_local_idx])

best_global_idx = TEST_START + best_local_idx

print(f"Highest V4 probability: {best_probability:.4f}")
print(f"Global test-row index:  {best_global_idx:,}")

# We want a window around the highest-risk point.
start_idx = max(TEST_START, best_global_idx - WINDOW)
end_idx = min(USABLE_ROWS, best_global_idx + WINDOW + 1)

print(f"Replay rows: {start_idx:,} -> {end_idx:,}")

rows = []
total_seen = 0

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=usecols,
    chunksize=250_000
):
    chunk_start = total_seen
    chunk_end = total_seen + len(chunk)

    overlap_start = max(chunk_start, start_idx)
    overlap_end = min(chunk_end, end_idx)

    if overlap_start < overlap_end:
        a = overlap_start - chunk_start
        b = overlap_end - chunk_start
        rows.append(chunk.iloc[a:b].copy())

    total_seen = chunk_end

    if total_seen >= end_idx:
        break

demo = pd.concat(rows, ignore_index=True)

# Match probabilities to the selected global-row window.
prob_start = start_idx - TEST_START
prob_end = prob_start + len(demo)

demo["risk_probability"] = probs[prob_start:prob_end]

demo["timestamp"] = pd.to_datetime(
    {
        "year": demo["year"],
        "month": demo["month"],
        "day": demo["day"],
        "hour": demo["hour"],
        "minute": demo["minute"],
        "second": demo["second"],
    }
)

demo["max_ch4"] = demo[
    ["MM263", "MM264", "MM256"]
].max(axis=1)

demo.to_csv(
    OUTPUT_FILE,
    index=False
)

best_position = int(np.argmax(
    demo["risk_probability"].to_numpy()
))

print()
print(f"Saved rows:       {len(demo):,}")
print(f"Demo position:    {best_position}")
print(f"Demo timestamp:   {demo.iloc[best_position]['timestamp']}")
print(f"Demo AI risk:     {demo.iloc[best_position]['risk_probability'] * 100:.1f}%")
print(f"Demo max CH4:     {demo.iloc[best_position]['max_ch4']:.2f}%")
print()
print(f"Output: {OUTPUT_FILE}")
print("=" * 60)