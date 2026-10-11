import pandas as pd

FILE = "methane_data.csv"

methane_columns = [
    "MM263",
    "MM264",
    "MM256"
]

print("Analyzing MineGuard methane sensors...")
print("Scanning the complete dataset...\n")

total_rows = 0

for chunk in pd.read_csv(FILE, chunksize=100000):

    total_rows += len(chunk)

    if total_rows % 1_000_000 == 0:
        print(f"Processed {total_rows:,} rows...")

print("\nDataset rows:", f"{total_rows:,}")

print("\n=== WARNING THRESHOLD ANALYSIS ===")
print("Threshold = 1.0% CH4\n")

for column in methane_columns:

    count = 0
    maximum = float("-inf")

    for chunk in pd.read_csv(
        FILE,
        usecols=[column],
        chunksize=100000
    ):

        values = chunk[column]

        count += (values >= 1.0).sum()

        chunk_max = values.max()

        if chunk_max > maximum:
            maximum = chunk_max

    percentage = (count / total_rows) * 100

    print(
        f"{column}: "
        f"{count:,} readings >= 1.0% "
        f"({percentage:.4f}%), "
        f"maximum = {maximum}%"
    )