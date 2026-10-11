import pandas as pd
import numpy as np

FILE = "methane_data.csv"

TARGET_COLS = ["MM263", "MM264", "MM256"]

print("Creating MineGuard future hazard target...")
print("Target: CH4 >= 1.0% in the future 3–6 minutes\n")

# We only need the 3 target methane sensors
df = pd.read_csv(
    FILE,
    usecols=TARGET_COLS
)

# Maximum methane among the three critical sensors
df["max_ch4"] = df[TARGET_COLS].max(axis=1)

# Future window:
# 181 seconds to 360 seconds ahead
future_max = (
    df["max_ch4"]
    .shift(-181)
    .rolling(window=180, min_periods=180)
    .max()
    .shift(-179)
)

df["hazard_3_6min"] = (future_max >= 1.0).astype("Int8")

# Remove rows where the future window doesn't exist
df = df.iloc[:-360].copy()

print("Total usable rows:", len(df))

print("\nTarget distribution:")
print(df["hazard_3_6min"].value_counts())

print("\nTarget percentage:")
print(
    df["hazard_3_6min"]
    .value_counts(normalize=True)
    .mul(100)
)

df.to_csv("mineguard_target.csv", index=False)

print("\nSaved: mineguard_target.csv")