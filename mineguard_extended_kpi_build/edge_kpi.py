from __future__ import annotations

import json
import time
from pathlib import Path


def uptime_metrics(heartbeat_times: list[float], expected_interval_sec: float = 1.0, tolerance: float = 1.5):
    if len(heartbeat_times) < 2:
        return {"uptime_percent": None, "missed_heartbeats": None}
    gaps = [b - a for a, b in zip(heartbeat_times[:-1], heartbeat_times[1:])]
    missed = sum(1 for g in gaps if g > expected_interval_sec * tolerance)
    total = len(gaps)
    uptime = 100.0 * (1.0 - missed / max(total, 1))
    return {
        "uptime_percent": uptime,
        "missed_heartbeats": missed,
        "heartbeat_count": len(heartbeat_times),
    }


def exposure_scenario_kpi(cases: list[dict]) -> dict:
    # Each case must contain expected_critical and actual_critical.
    if not cases:
        return {"coverage_percent": None}
    correct = sum(int(c["expected_critical"] == c["actual_critical"]) for c in cases)
    return {
        "scenario_cases": len(cases),
        "decision_coverage_percent": 100.0 * correct / len(cases),
    }


if __name__ == "__main__":
    hb = [0, 1, 2, 3, 4, 6]
    print(json.dumps(uptime_metrics(hb), indent=2))
    cases = [
        {"expected_critical": True, "actual_critical": True},
        {"expected_critical": False, "actual_critical": False},
    ]
    print(json.dumps(exposure_scenario_kpi(cases), indent=2))
