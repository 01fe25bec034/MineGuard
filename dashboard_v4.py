import streamlit as st
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from rfid_engine import check_authorization
from ppe_engine import check_ppe
from mineguard_fusion import fuse_risk, explain_risk

# =========================================================
# MineGuard V5 — polished command-center dashboard
# =========================================================

DATA_FILE = "dashboard_v4_replay.csv"
MODEL_THRESHOLD = 0.24
SAFE_CH4 = 0.60
RISKY_CH4 = 1.00
MEDIAN_LEAD_SECONDS = 120

st.set_page_config(
    page_title="MineGuard | AI Mining Safety Command Center",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1550px; padding-top: 1.0rem; padding-bottom: 2rem;}
    .mg-title {font-size: 2.5rem; font-weight: 850; letter-spacing: -0.03em; margin-bottom: 0;}
    .mg-subtitle {opacity: .64; font-size: .98rem; margin-bottom: .25rem;}
    .eyebrow {font-size: .72rem; font-weight: 700; letter-spacing: .10em; text-transform: uppercase; opacity: .58;}
    .section-title {font-size: 1.22rem; font-weight: 800; margin-top: 1.35rem; margin-bottom: .65rem;}
    .muted {opacity: .62; font-size: .82rem;}
    .metric-card {border: 1px solid rgba(120,120,120,.18); border-radius: 14px; padding: 14px 16px; min-height: 108px;}
    .metric-label {font-size: .74rem; font-weight: 700; letter-spacing: .05em; opacity: .62; text-transform: uppercase;}
    .metric-value {font-size: 1.75rem; font-weight: 820; margin-top: .25rem;}
    .metric-note {font-size: .74rem; opacity: .58; margin-top: .1rem;}
    .command-box {border: 1px solid rgba(120,120,120,.18); border-radius: 15px; padding: 16px;}
    .footer {margin-top: 28px; padding-top: 14px; border-top: 1px solid rgba(120,120,120,.16); opacity: .55; font-size: .74rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["max_ch4"] = df[["MM263", "MM264", "MM256"]].max(axis=1)
    return df


data = load_data()

# =========================================================
# SIDEBAR / DEMO CONTROL
# =========================================================

st.sidebar.title("MineGuard Control")
st.sidebar.caption("V4 held-out test replay • command-center mode")
st.sidebar.divider()

control_mode = st.sidebar.radio(
    "Input mode",
    ["Scenario presets", "Manual demo"],
    index=0,
)

if control_mode == "Scenario presets":
    scenario = st.sidebar.selectbox(
        "Demo scenario",
        [
            "High-risk command scenario",
            "AI early warning",
            "Unauthorized worker",
            "PPE failure",
            "Collision warning",
            "Safe baseline",
        ],
    )

    # Presets alter only the digital-twin context shown on the dashboard.
    # The validated V4 prediction remains untouched.
    preset_map = {
        "High-risk command scenario": ("MG001", "ZONE 3", "VEST MISSING", "WARNING", 600),
        "AI early warning": ("MG003", "ZONE 2", "PPE OK", "SAFE", 600),
        "Unauthorized worker": ("MG001", "ZONE 3", "PPE OK", "SAFE", 600),
        "PPE failure": ("MG003", "ZONE 2", "VEST MISSING", "SAFE", 600),
        "Collision warning": ("MG003", "ZONE 2", "PPE OK", "WARNING", 600),
        "Safe baseline": ("MG003", "ZONE 1", "PPE OK", "SAFE", len(data) - 1),
    }

    rfid_card, zone_choice, ppe_choice, collision_choice, default_position = preset_map[scenario]

    position = st.sidebar.slider(
        "Replay position",
        min_value=60,
        max_value=len(data) - 1,
        value=min(default_position, len(data) - 1),
        step=1,
    )
else:
    position = st.sidebar.slider(
        "Replay position",
        min_value=60,
        max_value=len(data) - 1,
        value=min(600, len(data) - 1),
        step=1,
    )

    rfid_card = st.sidebar.selectbox(
        "RFID worker",
        ["MG001", "MG002", "MG003", "UNKNOWN"],
        index=0,
    )
    zone_choice = st.sidebar.selectbox(
        "Active zone",
        ["ZONE 1", "ZONE 2", "ZONE 3"],
        index=2,
    )
    ppe_choice = st.sidebar.selectbox(
        "PPE status",
        ["PPE OK", "HELMET MISSING", "VEST MISSING"],
        index=0,
    )
    collision_choice = st.sidebar.selectbox(
        "Collision status",
        ["SAFE", "WARNING", "CRITICAL"],
        index=0,
    )

st.sidebar.divider()
st.sidebar.caption("Prototype display thresholds")
st.sidebar.caption("SAFE < 0.60% CH₄ • MODERATE 0.60–1.00% • VERY RISKY ≥ 1.00%")

# =========================================================
# CURRENT DATA
# =========================================================

current = data.iloc[position]
window = data.iloc[max(0, position - 299):position + 1].copy()

max_ch4 = float(current["max_ch4"])
ai_risk = float(current["risk_probability"])

# =========================================================
# SAFETY INPUTS
# =========================================================

auth_result = check_authorization(rfid_card, zone_choice)
authorized = bool(auth_result["authorized"])

if ppe_choice == "PPE OK":
    ppe_result = check_ppe(True, True)
elif ppe_choice == "HELMET MISSING":
    ppe_result = check_ppe(False, True)
else:
    ppe_result = check_ppe(True, False)

ppe_ok = bool(ppe_result["ok"])
collision_status = collision_choice

fusion_score, fusion_level = fuse_risk(
    max_ch4,
    ai_risk,
    collision_status,
    ppe_ok,
    authorized,
)

reasons, actions = explain_risk(
    max_ch4,
    ai_risk,
    collision_status,
    ppe_ok,
    authorized,
)

# =========================================================
# DERIVED STATUS
# =========================================================

if max_ch4 >= RISKY_CH4:
    methane_status = "VERY RISKY"
elif max_ch4 >= SAFE_CH4:
    methane_status = "MODERATE"
else:
    methane_status = "SAFE"

ai_status = "EARLY WARNING" if ai_risk >= MODEL_THRESHOLD else "LOW RISK"

# A clearer command response for severe fused states.
if fusion_level in {"CRITICAL", "HIGH"}:
    command_actions = []
    if fusion_level == "CRITICAL":
        command_actions.append("Restrict / stop entry into the affected zone.")
    else:
        command_actions.append("Escalate to the supervisor and investigate the affected zone.")
    if not authorized:
        command_actions.append("Deny access until worker authorization is verified.")
    if not ppe_ok:
        command_actions.append("Do not allow entry until required PPE is verified.")
    if collision_status == "WARNING":
        command_actions.append("Increase vehicle / worker separation.")
    elif collision_status == "CRITICAL":
        command_actions.append("Stop movement and clear the collision hazard.")
    if ai_risk >= MODEL_THRESHOLD:
        command_actions.append("Prepare the early-warning response for the predicted methane hazard.")
    if max_ch4 >= RISKY_CH4:
        command_actions.append("Follow the site's physical methane safety procedure.")
else:
    command_actions = list(actions)

# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="mg-title">⛏️ MineGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mg-subtitle">Edge-to-Core AI Mining Safety Command Center</div>',
    unsafe_allow_html=True,
)

h1, h2, h3, h4 = st.columns([1.55, 1.0, 1.0, 1.0])
with h1:
    st.caption(
        f"Replay: {current['timestamp'].strftime('%d %b %Y • %H:%M:%S')} • prediction horizon: 3–6 min"
    )
with h2:
    st.caption("● V4 AI ONLINE")
with h3:
    st.caption("● SAFETY FUSION ONLINE")
with h4:
    st.caption(f"● FINAL: {fusion_level}")

# =========================================================
# COMMAND STATUS
# =========================================================

if fusion_level == "CRITICAL":
    st.error("🚨 CRITICAL COMMAND STATE — immediate intervention required.")
elif fusion_level == "HIGH":
    st.error("🔴 HIGH COMMAND STATE — immediate attention required.")
elif fusion_level == "WATCH":
    st.warning("🟡 WATCH COMMAND STATE — continue monitoring and investigate.")
else:
    st.success("🟢 SAFE COMMAND STATE — no elevated fused safety condition.")

# =========================================================
# TOP METRICS
# =========================================================

st.markdown('<div class="section-title">Live Safety Snapshot</div>', unsafe_allow_html=True)

metric_specs = [
    ("MAXIMUM CH₄", f"{max_ch4:.2f}%", methane_status, "Current maximum across MM263 / MM264 / MM256"),
    ("AI FUTURE RISK", f"{ai_risk * 100:.1f}%", ai_status, "Locked alert threshold = 24%"),
    ("FUSED RISK", f"{fusion_score * 100:.0f}%", fusion_level, "Deterministic safety decision layer"),
    ("WARNING LEAD", "120 sec", "median", "Held-out event-level benchmark"),
]

mc = st.columns(4)
for col, (label, value, note, subnote) in zip(mc, metric_specs):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note} • {subnote}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# ZONE MAP
# =========================================================

st.markdown('<div class="section-title">Mine Zone Risk Map</div>', unsafe_allow_html=True)

zones = [
    ("ZONE 1", "MM263", float(current["MM263"])),
    ("ZONE 2", "MM264", float(current["MM264"])),
    ("ZONE 3", "MM256", float(current["MM256"])),
]

zone_cols = st.columns(3)
for col, (zone_name, sensor_name, value) in zip(zone_cols, zones):
    if value >= RISKY_CH4:
        zone_status = "🔴 VERY RISKY"
        zone_msg = "Red LED / immediate attention"
        box = "error"
    elif value >= SAFE_CH4:
        zone_status = "⚪ MODERATE"
        zone_msg = "White LED / monitor"
        box = "warning"
    else:
        zone_status = "🟢 SAFE"
        zone_msg = "Green LED / normal"
        box = "success"

    with col:
        with st.container(border=True):
            title = f"{zone_name}  ← ACTIVE" if zone_name == zone_choice else zone_name
            st.subheader(title)
            st.metric(sensor_name + " methane", f"{value:.2f}%")
            st.write(f"**Status:** {zone_status}")
            getattr(st, box)(zone_msg)

# =========================================================
# COMMAND DECISION
# =========================================================

st.markdown('<div class="section-title">MineGuard Command Decision</div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 1.45])

with left:
    st.markdown('<div class="command-box">', unsafe_allow_html=True)
    st.write("**Final safety level**")
    st.markdown(f"### {fusion_level}")
    st.write(f"**Fusion risk:** {fusion_score * 100:.0f}%")
    st.write(f"**AI risk:** {ai_risk * 100:.1f}%")
    st.write(f"**Active zone:** {zone_choice}")
    st.write(f"**Worker:** {auth_result['worker']}")
    st.write(f"**Zone access:** {'AUTHORIZED' if authorized else 'DENIED'}")
    st.write(f"**PPE:** {'OK' if ppe_ok else 'MISSING'}")
    st.write(f"**Collision:** {collision_status}")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.write("**Why is MineGuard warning?**")

    prev_idx = max(0, len(window) - 61)
    previous_ch4 = float(window.iloc[prev_idx]["max_ch4"])
    ch4_change_60 = max_ch4 - previous_ch4

    previous_airflow = float(window.iloc[prev_idx]["AN422"])
    airflow_change = float(current["AN422"]) - previous_airflow

    context = []
    if max_ch4 >= RISKY_CH4:
        context.append(f"Current methane is above the prototype risky threshold ({max_ch4:.2f}%).")
    elif max_ch4 >= SAFE_CH4:
        context.append(f"Current methane is elevated ({max_ch4:.2f}%).")
    if ch4_change_60 > 0.02:
        context.append(f"Methane is rising over the recent window (+{ch4_change_60:.3f}).")
    if airflow_change < -0.02:
        context.append(f"Airflow indicator AN422 is weakening ({airflow_change:.3f}).")
    if ai_risk >= MODEL_THRESHOLD:
        context.append(f"V4 predicts elevated future hazard probability ({ai_risk * 100:.1f}%).")
    if not authorized:
        context.append(f"RFID authorization denied for {zone_choice}.")
    if not ppe_ok:
        context.append("Required PPE is missing.")
    if collision_status == "WARNING":
        context.append("Vehicle / worker proximity warning active.")
    elif collision_status == "CRITICAL":
        context.append("Critical collision proximity detected.")
    if not context:
        context.append("No strong contextual warning signal is currently visible.")

    for item in context:
        st.write(f"• {item}")

    st.write("**Recommended response**")
    for item in command_actions[:6]:
        st.write(f"→ {item}")

# =========================================================
# WORKER / SAFETY CONTEXT
# =========================================================

st.markdown('<div class="section-title">Worker Safety Context</div>', unsafe_allow_html=True)

wcols = st.columns(4)
worker_metrics = [
    ("RFID WORKER", auth_result["worker"]),
    ("ZONE ACCESS", "AUTHORIZED" if authorized else "DENIED"),
    ("PPE STATUS", ppe_result["status"]),
    ("COLLISION", collision_status),
]
for col, (label, value) in zip(wcols, worker_metrics):
    with col:
        st.metric(label, value)

# =========================================================
# VALIDATED AI PERFORMANCE
# =========================================================

with st.expander("Validated MineGuard V4 performance", expanded=False):
    pcols = st.columns(6)
    perf = [
        ("ROC-AUC", "0.9334"),
        ("PR-AUC", "0.6510"),
        ("Test recall", "77.21%"),
        ("F1", "0.5795"),
        ("Event detection", "64.19%"),
        ("Median lead", "120 s"),
    ]
    for col, (label, value) in zip(pcols, perf):
        with col:
            st.metric(label, value)
    st.caption(
        "Held-out test evaluation. Event benchmark: 430 methane events; 276 received an early warning."
    )

# =========================================================
# TRENDS
# =========================================================

st.markdown('<div class="section-title">Predictive Trends</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Methane", "AI Risk", "Mine Conditions"])

with tab1:
    fig = go.Figure()
    for sensor in ["MM263", "MM264", "MM256"]:
        fig.add_trace(go.Scatter(
            x=window["timestamp"],
            y=window[sensor],
            mode="lines",
            name=sensor,
        ))
    fig.add_hline(y=SAFE_CH4, line_dash="dot", annotation_text="0.60% prototype moderate threshold")
    fig.add_hline(y=RISKY_CH4, line_dash="dash", annotation_text="1.00% prototype risky threshold")
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="Time",
        yaxis_title="CH₄ (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=window["timestamp"],
        y=window["risk_probability"] * 100,
        mode="lines",
        name="V4 AI Risk",
    ))
    fig2.add_hline(
        y=MODEL_THRESHOLD * 100,
        line_dash="dash",
        annotation_text="Locked V4 alert threshold (24%)",
    )
    fig2.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="Time",
        yaxis_title="AI risk (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, width="stretch")

with tab3:
    cond = pd.DataFrame({
        "Metric": ["AN422", "AN423", "TP1721", "RH1722", "V"],
        "Value": [
            float(current["AN422"]),
            float(current["AN423"]),
            float(current["TP1721"]),
            float(current["RH1722"]),
            float(current["V"]),
        ],
    })
    st.dataframe(cond, use_container_width=True, hide_index=True)

# =========================================================
# RECENT ALERT TIMELINE
# =========================================================

st.markdown('<div class="section-title">Recent Alert Timeline</div>', unsafe_allow_html=True)

recent = window.copy()
recent["ai_alert"] = recent["risk_probability"] >= MODEL_THRESHOLD
recent["methane_alert"] = recent["max_ch4"] >= RISKY_CH4

alerts = recent[recent["ai_alert"] | recent["methane_alert"]].tail(12).copy()

if alerts.empty:
    st.info("No AI or methane alerts in the current five-minute replay window.")
else:
    display = alerts[["timestamp", "max_ch4", "risk_probability", "ai_alert", "methane_alert"]].copy()
    display["Time"] = display["timestamp"].dt.strftime("%H:%M:%S")
    display["CH₄"] = display["max_ch4"].round(2).astype(str) + "%"
    display["AI Risk"] = (display["risk_probability"] * 100).round(1).astype(str) + "%"
    display["AI"] = np.where(display["ai_alert"], "EARLY WARNING", "—")
    display["Methane"] = np.where(display["methane_alert"], "THRESHOLD", "—")
    display = display[["Time", "CH₄", "AI Risk", "AI", "Methane"]]
    st.dataframe(display, use_container_width=True, hide_index=True)

# =========================================================
# SYSTEM STATUS
# =========================================================


# ============================================================
# MineGuard Extended KPI Command Center
# ============================================================

kpi_path = Path("mineguard_extended_kpi_build/outputs/mineguard_kpi_summary.json")

if kpi_path.exists():
    kpi = json.loads(kpi_path.read_text())

    st.markdown(
        '<div class="section-title">MineGuard KPI Command Center</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Extended Hackfest KPI coverage • validated results are separated "
        "from external benchmarks, monitoring modules, and prototype tests."
    )

    methane = kpi["methane"]
    env = kpi["environmental"]
    fatigue = kpi["fatigue"]
    maint = kpi["predictive_maintenance"]
    slope = kpi["slope"]
    edge = kpi["edge"]

    cols = st.columns(3)

    with cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🔥 Methane prediction</div>
                <div class="metric-value">F1 {methane["f1"]*100:.1f}%</div>
                <div class="metric-note">
                    ROC-AUC {methane["roc_auc"]:.3f} • PR-AUC {methane["pr_auc"]:.3f}
                    <br><b>VALIDATED ML</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🌿 Environmental forecast</div>
                <div class="metric-value">F1 {env["f1"]*100:.1f}%</div>
                <div class="metric-note">
                    Precision {env["precision"]*100:.1f}% • Recall {env["recall"]*100:.1f}%
                    <br><b>BENCHMARK • 15 MIN HORIZON</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🧠 FatigueGuard</div>
                <div class="metric-value">F1 {fatigue["f1"]*100:.1f}%</div>
                <div class="metric-note">
                    Recall {fatigue["recall"]*100:.1f}% • ROC-AUC {fatigue["roc_auc"]:.2f}
                    <br><b>OPEN BENCHMARK • SMALL TEST SET</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">⚙️ Predictive maintenance</div>
                <div class="metric-value">RMSE {maint["rmse_cycles"]:.1f} cycles</div>
                <div class="metric-note">
                    MAE {maint["mae_cycles"]:.1f} cycles • Availability proxy {maint["predicted_availability_proxy"]*100:.0f}%
                    <br><b>NASA RUL BENCHMARK</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">⛰️ SlopeGuard</div>
                <div class="metric-value">MONITORING</div>
                <div class="metric-note">
                    USGS sensor + GPS movement dataset
                    <br><b>ML BENCHMARK NOT CLAIMED</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📡 EdgeGuard</div>
                <div class="metric-value">{edge["uptime_percent"]:.0f}% test uptime</div>
                <div class="metric-note">
                    {edge["missed_heartbeats"]} missed / {edge["heartbeat_count"]} heartbeats
                    • Decision coverage {edge["decision_coverage_percent"]:.0f}%
                    <br><b>PROTOTYPE HEARTBEAT TEST</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.warning(
        "Extended KPI summary not found. Run the KPI build before using the command center."
    )


st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
s1, s2, s3, s4, s5 = st.columns(5)
for col, label in zip(
    [s1, s2, s3, s4, s5],
    ["V4 AI", "Methane sensors", "Safety fusion", "RFID", "PPE / collision"],
):
    col.success("ONLINE")
    col.caption(label)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        MineGuard V5 • V4 held-out methane replay • locked AI threshold 0.24 • 3–6 minute prediction target
        <div style="height: 10px;"></div>
        Prototype demonstration only — not a certified mine safety system.
    </div>
    """,
    unsafe_allow_html=True,
)
