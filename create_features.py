import pandas as pd
import numpy as np

FILE = "methane_data.csv"
TARGET_FILE = "mineguard_target.csv"
OUTPUT_FILE = "mineguard_features.csv"

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

CHUNK_SIZE = 100000

print("Creating MineGuard AI features...")
print("Using past/current sensor information only.\n")

# -------------------------------------------------
# First pass: determine total rows
# -------------------------------------------------

total_rows = 0

for chunk in pd.read_csv(
    FILE,
    usecols=FEATURE_COLS,
    chunksize=CHUNK_SIZE
):
    total_rows += len(chunk)

print(f"Total raw rows: {total_rows:,}")

# -------------------------------------------------
# Feature generation
# -------------------------------------------------

first_chunk = True
processed = 0

for chunk in pd.read_csv(
    FILE,
    usecols=FEATURE_COLS,
    chunksize=CHUNK_SIZE
):

    processed += len(chunk)

    # Historical rolling features
    for col in FEATURE_COLS:

        # Current value
        chunk[f"{col}_current"] = chunk[col]

        # 1-minute average
        chunk[f"{col}_mean60"] = (
            chunk[col]
            .rolling(60, min_periods=1)
            .mean()
        )

        # 3-minute average
        chunk[f"{col}_mean180"] = (
            chunk[col]
            .rolling(180, min_periods=1)
            .mean()
        )

        # 5-minute average
        chunk[f"{col}_mean300"] = (
            chunk[col]
            .rolling(300, min_periods=1)
            .mean()
        )

        # Change over 1 minute
        chunk[f"{col}_change60"] = (
            chunk[col] - chunk[col].shift(60)
        )

        # Change over 3 minutes
        chunk[f"{col}_change180"] = (
            chunk[col] - chunk[col].shift(180)
        )

        # Change over 5 minutes
        chunk[f"{col}_change300"] = (
            chunk[col] - chunk[col].shift(300)
        )

        # Maximum during previous 1 minute
        chunk[f"{col}_max60"] = (
            chunk[col]
            .rolling(60, min_periods=1)
            .max()
        )

    # Remove original raw columns.
    chunk = chunk.drop(columns=FEATURE_COLS)

    # -------------------------------------------------
    # Save chunk
    # -------------------------------------------------

    if first_chunk:
        chunk.to_csv(
            OUTPUT_FILE,
            index=False
        )
        first_chunk = False
    else:
        chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )

    print(f"Processed {processed:,} / {total_rows:,} rows")

print("\nFeature generation complete.")
print(f"Saved: {OUTPUT_FILE}")