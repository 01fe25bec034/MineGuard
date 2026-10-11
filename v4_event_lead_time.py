import numpy as np
import pandas as pd

DATA_FILE = "methane_data.csv"
PROB_FILE = "mineguard_v4_locked_test_probabilities.npy"

THRESHOLD = 0.24
CH4_THRESHOLD = 1.0

USABLE_ROWS = 9_199_570
TEST_START = 7_359_656
TEST_SAMPLES = USABLE_ROWS - TEST_START

MAX_WARNING_WINDOW = 600  # 10 minutes


def build_timestamp(df):
    return pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
            "minute": df["minute"],
            "second": df["second"],
        }
    )


print("=" * 60)
print("     MINEGUARD V4 CORRECTED EVENT LEAD TIME")
print("=" * 60)

probs = np.load(PROB_FILE)

if len(probs) != TEST_SAMPLES:
    raise ValueError(
        f"Probability length mismatch: got {len(probs)}, "
        f"expected {TEST_SAMPLES}"
    )

print(f"Test predictions: {len(probs):,}")
print(f"Locked threshold: {THRESHOLD:.2f}")

usecols = [
    "year", "month", "day", "hour", "minute", "second",
    "MM263", "MM264", "MM256"
]

rows = []
total_seen = 0

print("Reading test-period methane data...")

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

print(f"Test rows loaded: {len(test):,}")

test["timestamp"] = build_timestamp(test)

test["max_ch4"] = test[
    ["MM263", "MM264", "MM256"]
].max(axis=1)

test["ai_probability"] = probs

test["ai_alert"] = (
    test["ai_probability"] >= THRESHOLD
)

test["methane_hazard"] = (
    test["max_ch4"] >= CH4_THRESHOLD
)

hazard = test["methane_hazard"].to_numpy()
ai_alert = test["ai_alert"].to_numpy()
timestamps = test["timestamp"].to_numpy()

# ------------------------------------------------------------
# Detect methane event starts
# Event ends after 60 continuous seconds below threshold.
# ------------------------------------------------------------

event_starts = []

in_event = False
below_count = 0

for i, is_hazard in enumerate(hazard):

    if is_hazard:

        if not in_event:
            event_starts.append(i)
            in_event = True

        below_count = 0

    elif in_event:

        below_count += 1

        if below_count >= 60:
            in_event = False
            below_count = 0

print(f"Methane events detected: {len(event_starts)}")

# ------------------------------------------------------------
# Find AI WARNING ONSETS.
#
# An onset is a transition:
#
#     False -> True
#
# This represents when MineGuard actually started warning.
# ------------------------------------------------------------

alert_onsets = []

for i in range(1, len(ai_alert)):
    if ai_alert[i] and not ai_alert[i - 1]:
        alert_onsets.append(i)

alert_onsets = np.array(alert_onsets, dtype=np.int64)

print(f"AI warning episodes:      {len(alert_onsets)}")

# ------------------------------------------------------------
# Event-level lead time
#
# For each methane event, find the LAST warning episode onset
# before the event, within 10 minutes.
# ------------------------------------------------------------

lead_times = []
detected_events = 0

for event_idx in event_starts:

    window_start = max(
        0,
        event_idx - MAX_WARNING_WINDOW
    )

    candidates = alert_onsets[
        (alert_onsets >= window_start) &
        (alert_onsets < event_idx)
    ]

    if len(candidates) == 0:
        continue

    warning_start = candidates[-1]

    lead_seconds = (
        timestamps[event_idx]
        - timestamps[warning_start]
    ) / np.timedelta64(1, "s")

    if 0 < lead_seconds <= MAX_WARNING_WINDOW:

        lead_times.append(float(lead_seconds))
        detected_events += 1

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 60)
print("          CORRECTED EVENT-LEVEL RESULTS")
print("=" * 60)

total_events = len(event_starts)

print(f"Total methane events:      {total_events}")
print(f"Events with prior alert:   {detected_events}")

if total_events > 0:

    detection_rate = (
        detected_events / total_events
    )

    print(
        f"Event detection rate:      "
        f"{detection_rate * 100:.2f}%"
    )

if lead_times:

    lead = np.array(lead_times)

    print()
    print("AI WARNING LEAD TIME")
    print(
        f"Mean:                       "
        f"{lead.mean():.1f} sec"
    )
    print(
        f"Median:                     "
        f"{np.median(lead):.1f} sec"
    )
    print(
        f"25th percentile:            "
        f"{np.percentile(lead, 25):.1f} sec"
    )
    print(
        f"75th percentile:            "
        f"{np.percentile(lead, 75):.1f} sec"
    )
    print(
        f"Minimum:                    "
        f"{lead.min():.1f} sec"
    )
    print(
        f"Maximum:                    "
        f"{lead.max():.1f} sec"
    )

    print()
    print("LEAD-TIME BUCKETS")

    print(
        f"< 30 sec:                   "
        f"{(lead < 30).sum()}"
    )

    print(
        f"30–60 sec:                  "
        f"{((lead >= 30) & (lead < 60)).sum()}"
    )

    print(
        f"60–120 sec:                 "
        f"{((lead >= 60) & (lead < 120)).sum()}"
    )

    print(
        f"120–180 sec:                "
        f"{((lead >= 120) & (lead < 180)).sum()}"
    )

    print(
        f"180–300 sec:                "
        f"{((lead >= 180) & (lead < 300)).sum()}"
    )

    print(
        f">= 300 sec:                 "
        f"{(lead >= 300).sum()}"
    )

    np.savetxt(
        "mineguard_v4_corrected_event_lead_times_sec.csv",
        lead,
        delimiter=",",
        header="lead_time_seconds",
        comments=""
    )

    print()
    print("Saved:")
    print(
        "  mineguard_v4_corrected_event_lead_times_sec.csv"
    )

else:

    print()
    print("No valid pre-event warnings found.")