import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from rfid_engine import check_authorization
from ppe_engine import check_ppe
from mineguard_fusion import fuse_risk, explain_risk


# =========================================================
# CONFIGURATION
# =========================================================

DATA_FILE = "dashboard_v4_replay.csv"

MODEL_THRESHOLD = 0.24

# Prototype display thresholds.
# These are demonstration thresholds, not universal legal limits.
SAFE_CH4 = 0.60
RISKY_CH4 = 1.00


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MineGuard V4 Safety Command Center",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .mg-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .mg-subtitle {
        opacity: 0.65;
        font-size: 1rem;
        margin-bottom: 4px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 24px;
        margin-bottom: 10px;
    }

    .small-note {
        opacity: 0.58;
        font-size: 0.78rem;
    }

    .fusion-box {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 12px;
    }

    .footer {
        margin-top: 30px;
        padding-top: 14px;
        border-top: 1px solid rgba(255,255,255,0.10);
        opacity: 0.55;
        font-size: 0.75rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD REPLAY DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["max_ch4"] = df[
        ["MM263", "MM264", "MM256"]
    ].max(axis=1)

    return df


data = load_data()


# =========================================================
# SIDEBAR — REPLAY CONTROL
# =========================================================

st.sidebar.title("MineGuard Control")

st.sidebar.caption(
    "V4 held-out test replay"
)

st.sidebar.divider()

position = st.sidebar.slider(
    "Replay position",
    min_value=60,
    max_value=len(data) - 1,
    value=min(600, len(data) - 1),
    step=1,
)

st.sidebar.info(
    "This demonstration replays held-out test-period "
    "data using the locked MineGuard V4 predictions."
)


# =========================================================
# CURRENT SAMPLE
# =========================================================

current = data.iloc[position]

window = data.iloc[
    max(0, position - 299):position + 1
].copy()


# =========================================================
# CURRENT SENSOR / AI VALUES
# =========================================================

max_ch4 = float(
    current["max_ch4"]
)

ai_risk = float(
    current["risk_probability"]
)


# =========================================================
# WORKER SAFETY INPUTS
# =========================================================

st.sidebar.divider()

st.sidebar.subheader(
    "Worker Safety Inputs"
)

rfid_card = st.sidebar.selectbox(
    "RFID worker",
    [
        "MG001",
        "MG002",
        "MG003",
        "UNKNOWN",
    ],
)

zone_choice = st.sidebar.selectbox(
    "Active zone",
    [
        "ZONE 1",
        "ZONE 2",
        "ZONE 3",
    ],
)

ppe_choice = st.sidebar.selectbox(
    "PPE status",
    [
        "PPE OK",
        "HELMET MISSING",
        "VEST MISSING",
    ],
)

collision_choice = st.sidebar.selectbox(
    "Collision status",
    [
        "SAFE",
        "WARNING",
        "CRITICAL",
    ],
)


# =========================================================
# RFID DECISION
# =========================================================

auth_result = check_authorization(
    rfid_card,
    zone_choice,
)

authorized = auth_result["authorized"]


# =========================================================
# PPE DECISION
# =========================================================

if ppe_choice == "PPE OK":

    ppe_result = check_ppe(
        True,
        True,
    )

elif ppe_choice == "HELMET MISSING":

    ppe_result = check_ppe(
        False,
        True,
    )

else:

    ppe_result = check_ppe(
        True,
        False,
    )

ppe_ok = ppe_result["ok"]


# =========================================================
# COLLISION
# =========================================================

collision_status = collision_choice


# =========================================================
# SAFETY FUSION
# =========================================================

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
# CURRENT SENSOR STATUS
# =========================================================

if max_ch4 >= RISKY_CH4:

    sensor_status = "VERY RISKY"

elif max_ch4 >= SAFE_CH4:

    sensor_status = "MODERATE"

else:

    sensor_status = "SAFE"


# =========================================================
# AI STATUS
# =========================================================

if ai_risk >= MODEL_THRESHOLD:

    ai_status = "EARLY WARNING"

else:

    ai_status = "LOW RISK"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="mg-title">⛏️ MineGuard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="mg-subtitle">'
    'Edge-to-Core AI Mining Safety Command Center'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Replay timestamp: "
    f"{current['timestamp'].strftime('%d %b %Y • %H:%M:%S')}"
    "   |   Prediction horizon: 3–6 minutes"
)

st.info(
    "MineGuard V4 ONLINE  •  "
    f"Locked AI threshold: {MODEL_THRESHOLD:.2f}  •  "
    f"FINAL SAFETY: {fusion_level}"
)


# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "MAXIMUM CH₄",
        f"{max_ch4:.2f}%"
    )

with c2:

    st.metric(
        "AI FUTURE RISK",
        f"{ai_risk * 100:.1f}%"
    )

with c3:

    st.metric(
        "AI ASSESSMENT",
        ai_status
    )

with c4:

    st.metric(
        "FINAL SAFETY",
        fusion_level
    )


# =========================================================
# MAIN ALERT
# =========================================================

if fusion_level == "CRITICAL":

    st.error(
        "🚨 CRITICAL SAFETY STATE: MineGuard fusion has "
        "identified a high-priority safety condition."
    )

elif fusion_level == "HIGH":

    st.error(
        "🔴 HIGH SAFETY RISK: immediate attention required."
    )

elif fusion_level == "WATCH":

    st.warning(
        "🟡 WATCH: MineGuard recommends continued monitoring."
    )

else:

    st.success(
        "🟢 SAFE: no elevated fused safety condition."
    )


# =========================================================
# THREE-ZONE RISK MAP
# =========================================================

st.markdown(
    '<div class="section-title">Mine Zone Risk Map</div>',
    unsafe_allow_html=True,
)

zones = [
    (
        "ZONE 1",
        "MM263",
        float(current["MM263"]),
    ),
    (
        "ZONE 2",
        "MM264",
        float(current["MM264"]),
    ),
    (
        "ZONE 3",
        "MM256",
        float(current["MM256"]),
    ),
]

zone_cols = st.columns(3)

for col, (zone_name, sensor_name, value) in zip(
    zone_cols,
    zones,
):

    if value >= RISKY_CH4:

        status = "🔴 VERY RISKY"

        message = (
            "Red LED / immediate attention"
        )

    elif value >= SAFE_CH4:

        status = "⚪ MODERATE"

        message = (
            "White LED / monitor"
        )

    else:

        status = "🟢 SAFE"

        message = (
            "Green LED / normal"
        )

    with col:

        with st.container(border=True):

            st.subheader(zone_name)

            st.metric(
                f"{sensor_name} methane",
                f"{value:.2f}%"
            )

            st.write(
                f"**Status:** {status}"
            )

            if value >= RISKY_CH4:

                st.error(message)

            elif value >= SAFE_CH4:

                st.warning(message)

            else:

                st.success(message)


# =========================================================
# WORKER SAFETY STATUS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Worker Safety Context'
    '</div>',
    unsafe_allow_html=True,
)

w1, w2, w3, w4 = st.columns(4)

with w1:

    st.metric(
        "RFID WORKER",
        auth_result["worker"]
    )

with w2:

    st.metric(
        "ZONE ACCESS",
        "AUTHORIZED"
        if authorized
        else "DENIED"
    )

with w3:

    st.metric(
        "PPE STATUS",
        ppe_result["status"]
    )

with w4:

    st.metric(
        "COLLISION",
        collision_status
    )


# =========================================================
# WHY IS MINEGUARD WARNING?
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Why is MineGuard warning?'
    '</div>',
    unsafe_allow_html=True,
)

# Recent methane movement
reference_index = max(
    0,
    len(window) - 61
)

previous_ch4 = float(
    window.iloc[reference_index]["max_ch4"]
)

ch4_change_60 = (
    max_ch4 - previous_ch4
)

# Recent airflow movement
previous_airflow = float(
    window.iloc[reference_index]["AN422"]
)

airflow_change = (
    float(current["AN422"])
    - previous_airflow
)

context_reasons = []

if max_ch4 >= RISKY_CH4:

    context_reasons.append(
        f"Current methane is above the "
        f"prototype danger threshold "
        f"({max_ch4:.2f}%)."
    )

elif max_ch4 >= SAFE_CH4:

    context_reasons.append(
        f"Current methane is elevated "
        f"({max_ch4:.2f}%)."
    )

if ch4_change_60 > 0.02:

    context_reasons.append(
        f"Methane is rising over the recent "
        f"window (+{ch4_change_60:.3f})."
    )

if airflow_change < -0.02:

    context_reasons.append(
        f"Airflow indicator AN422 is weakening "
        f"({airflow_change:.3f})."
    )

if ai_risk >= MODEL_THRESHOLD:

    context_reasons.append(
        f"V4 predicts elevated future hazard "
        f"probability ({ai_risk * 100:.1f}%)."
    )

if not authorized:

    context_reasons.append(
        f"RFID authorization denied for "
        f"{zone_choice}."
    )

if not ppe_ok:

    context_reasons.append(
        "Required PPE is missing."
    )

if collision_status == "WARNING":

    context_reasons.append(
        "Vehicle/worker proximity warning active."
    )

elif collision_status == "CRITICAL":

    context_reasons.append(
        "Critical collision proximity detected."
    )

if not context_reasons:

    context_reasons.append(
        "No strong contextual warning signal "
        "is currently visible."
    )

for reason in context_reasons:

    st.write(
        f"• {reason}"
    )


# =========================================================
# RECOMMENDED RESPONSE
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Recommended Response'
    '</div>',
    unsafe_allow_html=True,
)

if actions:

    for action in actions:

        st.write(
            f"→ {action}"
        )

else:

    st.write(
        "→ Continue normal monitoring."
    )


# =========================================================
# FUSION DETAILS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'MineGuard Safety Fusion'
    '</div>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns(4)

with f1:

    st.metric(
        "FUSION RISK",
        f"{fusion_score * 100:.0f}%"
    )

with f2:

    st.metric(
        "FINAL LEVEL",
        fusion_level
    )

with f3:

    st.metric(
        "AUTHORIZATION",
        "YES" if authorized else "NO"
    )

with f4:

    st.metric(
        "PPE",
        "OK" if ppe_ok else "MISSING"
    )

st.caption(
    "Safety fusion is a deterministic decision-support "
    "layer. Physical safety thresholds remain independent "
    "safety overrides."
)


# =========================================================
# VALIDATED V4 PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Validated MineGuard V4 Performance'
    '</div>',
    unsafe_allow_html=True,
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "ROC-AUC",
        "0.9334"
    )

with p2:

    st.metric(
        "PR-AUC",
        "0.6510"
    )

with p3:

    st.metric(
        "TEST RECALL",
        "77.21%"
    )

with p4:

    st.metric(
        "EVENT DETECTION",
        "64.19%"
    )

st.caption(
    "Held-out test evaluation • 430 methane events • "
    "276 events received an early warning • "
    "median warning lead time: 120 seconds."
)


# =========================================================
# METHANE TREND
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Methane Concentration — Recent 5 Minutes'
    '</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

for sensor in [
    "MM263",
    "MM264",
    "MM256",
]:

    fig.add_trace(
        go.Scatter(
            x=window["timestamp"],
            y=window[sensor],
            mode="lines",
            name=sensor,
        )
    )

fig.add_hline(
    y=SAFE_CH4,
    line_dash="dot",
    annotation_text="0.60% prototype moderate threshold",
)

fig.add_hline(
    y=RISKY_CH4,
    line_dash="dash",
    annotation_text="1.00% prototype risky threshold",
)

fig.update_layout(
    height=420,
    margin=dict(
        l=10,
        r=10,
        t=20,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="CH₄ (%)",
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    width="stretch",
)


# =========================================================
# AI RISK TREND
# =========================================================

st.markdown(
    '<div class="section-title">'
    'MineGuard V4 Predictive Risk — Recent 5 Minutes'
    '</div>',
    unsafe_allow_html=True,
)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=window["timestamp"],
        y=window["risk_probability"] * 100,
        mode="lines",
        name="V4 AI Risk",
    )
)

fig2.add_hline(
    y=MODEL_THRESHOLD * 100,
    line_dash="dash",
    annotation_text="Locked V4 threshold",
)

fig2.update_layout(
    height=330,
    margin=dict(
        l=10,
        r=10,
        t=20,
        b=20,
    ),
    xaxis_title="Time",
    yaxis_title="AI risk (%)",
    hovermode="x unified",
)

st.plotly_chart(
    fig2,
    width="stretch",
)


# =========================================================
# MINE CONDITIONS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Mine Conditions'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:

    st.metric(
        "Airflow AN422",
        f"{current['AN422']:.2f}",
    )

with m2:

    st.metric(
        "Airflow AN423",
        f"{current['AN423']:.2f}",
    )

with m3:

    st.metric(
        "Temperature",
        f"{current['TP1721']:.1f} °C",
    )

with m4:

    st.metric(
        "Humidity",
        f"{current['RH1722']:.1f}%",
    )

with m5:

    st.metric(
        "Machine Speed",
        f"{current['V']:.2f}",
    )


# =========================================================
# RECENT ALERTS
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Recent V4 Alert Events'
    '</div>',
    unsafe_allow_html=True,
)

recent = window.copy()

recent["AI Alert"] = (
    recent["risk_probability"]
    >= MODEL_THRESHOLD
)

recent["Methane Alert"] = (
    recent["max_ch4"]
    >= RISKY_CH4
)

alerts = recent[
    recent["AI Alert"]
    | recent["Methane Alert"]
].tail(10).copy()

if alerts.empty:

    st.info(
        "No alerts in the current replay window."
    )

else:

    display = alerts[
        [
            "timestamp",
            "max_ch4",
            "risk_probability",
            "AI Alert",
            "Methane Alert",
        ]
    ].copy()

    display["AI Risk"] = (
        display["risk_probability"]
        * 100
    ).round(1).astype(str) + "%"

    display["CH₄"] = (
        display["max_ch4"]
        .round(2)
        .astype(str)
        + "%"
    )

    display["AI Alert"] = (
        display["AI Alert"]
        .map(
            {
                True: "HIGH",
                False: "—",
            }
        )
    )

    display["Methane Alert"] = (
        display["Methane Alert"]
        .map(
            {
                True: "WARNING",
                False: "—",
            }
        )
    )

    display = display[
        [
            "timestamp",
            "CH₄",
            "AI Risk",
            "AI Alert",
            "Methane Alert",
        ]
    ].rename(
        columns={
            "timestamp": "Time"
        }
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        MineGuard V4 • Held-out methane test replay •
        Locked threshold 0.24 • 3–6 minute prediction target
        <div style="height: 12px;"></div>
        Prototype demonstration only — not a certified mine safety system.
    </div>
    """,
    unsafe_allow_html=True,
)