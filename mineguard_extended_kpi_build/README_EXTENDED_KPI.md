# MineGuard Extended KPI Build

This build adds the remaining mandatory KPI tracks around the already-validated MineGuard V4 methane model.

## KPI tracks
1. Slope / tailings: slope-movement escalation benchmark using USGS landslide monitoring data.
2. Fatigue: fatigue/drowsiness classifier using open EAR, blink-rate and head-pose features.
3. Predictive maintenance: RUL model using NASA C-MAPSS as a prognostics benchmark.
4. Environmental compliance: site-configurable environmental-threshold forecasting on the MineGuard mine dataset.
5. Infrastructure KPI support: edge heartbeat/uptime and worker-exposure scenario coverage.

## Important honesty rule
Some public datasets are proxies rather than mine-specific datasets. Do not present their metrics as mine-site deployment accuracy. Present them as algorithm validation benchmarks. The physical MineGuard system still needs site-specific calibration and validation.

## Current validated core (do not change)
- V4 LightGBM methane model
- locked threshold 0.24
- ROC-AUC 0.9334
- PR-AUC 0.6510
- test recall 77.21%
- 64.19% event-level detection
- 120 s median event-level warning lead time

## Recommended run order
1. Train slope module.
2. Train fatigue module.
3. Train predictive-maintenance module.
4. Configure and train environmental module.
5. Run the KPI aggregator.
6. Integrate outputs into dashboard_v4.py.

## Data sources
- USGS landslide monitoring datasets: public monitoring streams include displacement/extensometer, pore pressure, soil moisture and rainfall.
- Fatigue dataset: public repository with EAR, blink rate and head pose labels.
- NASA C-MAPSS: public run-to-failure multivariate time series for prognostics/RUL benchmarking.
- MineGuard environmental module: uses the existing methane_data.csv and user-configured site control limits.
