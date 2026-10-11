#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m pip install -U pandas numpy scikit-learn joblib lightgbm
mkdir -p outputs fatigue_data cmapss_data

echo "MineGuard Extended KPI build pack is ready."
echo "1) Run fatigue: python3 train_fatigue.py"
echo "2) Download a USGS slope CSV, then: python3 train_slope_stability.py --csv /path/to/slope.csv --out outputs/slope_model.joblib --metrics outputs/slope_metrics.json"
echo "3) Run maintenance: python3 train_predictive_maintenance.py --data cmapss_data --out outputs/maintenance_model.joblib --metrics outputs/maintenance_metrics.json"
echo "4) Run environmental: python3 environmental_compliance.py --csv /path/to/methane_data.csv --out outputs/environmental_model.joblib --metrics outputs/environmental_metrics.json"
echo "5) Run: python3 edge_kpi.py"
