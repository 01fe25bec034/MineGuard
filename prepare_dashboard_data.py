import pandas as pd
import numpy as np

RAW_FILE = "methane_data.csv"
TARGET_FILE = "mineguard_target.csv"

WINDOW = 5000
HISTORY = 300

print("Finding a hazard-rich section of the real dataset...")

targets = pd.read_csv(
    TARGET_FILE,
    usecols=["hazard_3_6min"],
    dtype={"hazard_3_6min": np.int8}
)["hazard_3_6min"].to_numpy()

# Prefer the chronological test period
test_start = int(len(targets) * 0.80)

test_targets = targets[test_start:]

# Find a 5000-row section with many hazards
window_counts = np.convolve(
    test_targets.astype(np.int32),
    np.ones(WINDOW, dtype=np.int32),
    mode="valid"
)

best_local = int(np.argmax(window_counts))
best_target_start = test_start + best_local
best_count = int(window_counts[best_local])

# Target row i corresponds to raw row i+300.
# Include 300 rows of history before the displayed region.
raw_start = best_target_start
raw_rows_needed = WINDOW + HISTORY

print(f"Selected target start: {best_target_start:,}")
print(f"Hazards in selected window: {best_count:,}")
print(f"Reading raw rows {raw_start:,} to "
      f"{raw_start + raw_rows_needed - 1:,}")

FEATURE_COLS = [
    "year", "month", "day", "hour", "minute", "second",
    "AN422", "AN423",
    "TP1721", "RH1722",
    "TP1711", "RH1712",
    "MM263", "MM264", "MM256",
    "V"
]

print()
print("Extracting replay segment...")

pieces = []
remaining_start = raw_start
remaining = raw_rows_needed

reader = pd.read_csv(
    RAW_FILE,
    usecols=FEATURE_COLS,
    chunksize=100000
)

current_position = 0

for chunk in reader:

    chunk_start = current_position
    chunk_end = current_position + len(chunk)

    overlap_start = max(chunk_start, raw_start)
    overlap_end = min(
        chunk_end,
        raw_start + raw_rows_needed
    )

    if overlap_start < overlap_end:

        local_start = overlap_start - chunk_start
        local_end = overlap_end - chunk_start

        pieces.append(
            chunk.iloc[local_start:local_end].copy()
        )

    current_position = chunk_end

    if current_position >= raw_start + raw_rows_needed:
        break

replay = pd.concat(
    pieces,
    ignore_index=True
)

replay.to_csv(
    "dashboard_replay.csv",
    index=False
)

print()
print("Dashboard replay file created:")
print("  dashboard_replay.csv")
print(f"Rows: {len(replay):,}")
print()
print("You can now run the dashboard without repeatedly")
print("reading the 1 GB raw dataset.")
