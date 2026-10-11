import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_FILE = "dashboard_v4_replay.csv"

MODEL_THRESHOLD = 0.24

# Prototype methane zones
SAFE_CH4 = 0.60
RISKY_CH4 = 1.00

st.set_page_config(
    page_title="MineGuard V4 Safety Command Center",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
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
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 22px;
        margin-bottom: 10px;
    }

    .small-note {
        opacity: 0.58;
        font-size: 0.78rem;
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


# ---------------------------------------------------------
# LOAD V4 REPLAY
# ---------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["max_ch4"] = df[
        ["MM263", "MM264", "MM256"]
    ].max(axis=1)

    return df


data = load_data()


# ---------------------------------------------------------
# REPLAY CONTROL
# ---------------------------------------------------------

st.sidebar.title("MineGuard Control")
st.sidebar.caption("V4 held-out test replay")
st.sidebar.divider()

position = st.sidebar.slider(
    "Replay position",
    min_value=60,
    max_value=len(data) - 1,
    value=len(data) - 1,
    step=1,
)

st.sidebar.info(
    "This demonstration replays held-out test-period data "
    "through the locked MineGuard V4 predictions."
)

current = data.iloc[position]

window = data.iloc[
    max(0, position - 299): position + 1
].copy()


# ---------------------------------------------------------
# CURRENT VALUES
# ---------------------------------------------------------

max_ch4 = float(current["max_ch4"])
ai_risk = float(current["risk_probability"])

if max_ch4 >= RISKY_CH4:
    sensor_status = "VERY RISKY"
elif max_ch4 >= SAFE_CH4:
    sensor_status = "MODERATE"
else:
    sensor_status = "SAFE"

if ai_risk >= MODEL_THRESHOLD:
    ai_status = "EARLY WARNING"
else:
    ai_status = "LOW RISK"

overall = (
    "CRITICAL"
    if max_ch4 >= RISKY_CH4
    else "ATTENTION"
    if ai_risk >= MODEL_THRESHOLD
    else "NORMAL"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

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
    f"Replay timestamp: "
    f"{current['timestamp'].strftime('%d %b %Y • %H:%M:%S')}"
    f"   |   Prediction horizon: 3–6 minutes"
)

st.info(
    f"MineGuard V4 ONLINE  •  Locked AI threshold: "
    f"{MODEL_THRESHOLD:.2f}  •  Overall status: {overall}"
)


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

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
        "CURRENT STATUS",
        sensor_status
    )

with c4:
    st.metric(
        "AI ASSESSMENT",
        ai_status
    )


# ---------------------------------------------------------
# ALERT
# ---------------------------------------------------------

if max_ch4 >= RISKY_CH4 and ai_risk >= MODEL_THRESHOLD:

    st.error(
        "🚨 CRITICAL: current methane is at the prototype "
        "very-risky threshold AND V4 predicts elevated "
        "future hazard probability."
    )

elif max_ch4 >= RISKY_CH4:

    st.error(
        "🚨 METHANE ALERT: current methane has reached the "
        "prototype very-risky threshold."
    )

elif ai_risk >= MODEL_THRESHOLD:

    st.warning(
        "⚠️ AI EARLY WARNING: MineGuard V4 predicts elevated "
        "probability of a methane hazard within the 3–6 minute "
        "prediction window."
    )

elif max_ch4 >= SAFE_CH4:

    st.warning(
        "🟡 MODERATE ZONE CONDITION: methane is elevated. "
        "Continue monitoring."
    )

else:

    st.success(
        "🟢 NORMAL: no current methane threshold breach "
        "and no V4 early-warning alert."
    )


# ---------------------------------------------------------
# THREE ZONES
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Mine Zone Risk Map</div>',
    unsafe_allow_html=True,
)

zones = [
    ("ZONE 1", "MM263", float(current["MM263"])),
    ("ZONE 2", "MM264", float(current["MM264"])),
    ("ZONE 3", "MM256", float(current["MM256"])),
]

zone_cols = st.columns(3)

for col, (zone_name, sensor_name, value) in zip(
    zone_cols, zones
):

    if value >= RISKY_CH4:
        status = "🔴 VERY RISKY"
    elif value >= SAFE_CH4:
        status = "⚪ MODERATE"
    else:
        status = "🟢 SAFE"

    with col:
        with st.container(border=True):
            st.subheader(zone_name)

            st.metric(
                f"{sensor_name} methane",
                f"{value:.2f}%"
            )

            st.write(f"**Status:** {status}")

            if value >= RISKY_CH4:
                st.error("Red LED / immediate attention")
            elif value >= SAFE_CH4:
                st.warning("White LED / monitor")
            else:
                st.success("Green LED / normal")


# ---------------------------------------------------------
# WHY IS THIS ZONE RISKY?
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Why is MineGuard warning?'
    '</div>',
    unsafe_allow_html=True,
)

drivers = []

ch4_change_60 = (
    float(current["max_ch4"])
    - float(window.iloc[max(0, len(window) - 61)]["max_ch4"])
)

airflow_change = (
    float(current["AN422"])
    - float(window.iloc[max(0, len(window) - 61)]["AN422"])
)

if max_ch4 >= RISKY_CH4:
    drivers.append(
        f"Current methane is elevated at {max_ch4:.2f}%."
    )

if ch4_change_60 > 0.02:
    drivers.append(
        f"Methane is rising over the recent window "
        f"(+{ch4_change_60:.3f})."
    )

elif ch4_change_60 < -0.02:
    drivers.append(
        f"Methane is decreasing over the recent window "
        f"({ch4_change_60:.3f})."
    )

if airflow_change < -0.02:
    drivers.append(
        f"Airflow indicator AN422 is weakening "
        f"({airflow_change:.3f})."
    )

if ai_risk >= MODEL_THRESHOLD:
    drivers.append(
        f"V4 predicts elevated future hazard probability "
        f"({ai_risk * 100:.1f}%)."
    )

if not drivers:
    drivers.append(
        "No strong contextual warning signal is currently "
        "visible; MineGuard remains in normal monitoring mode."
    )

for driver in drivers:
    st.write(f"• {driver}")


# ---------------------------------------------------------
# VALIDATED AI PERFORMANCE
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Validated MineGuard V4 Performance'
    '</div>',
    unsafe_allow_html=True,
)

p1, p2, p3, p4 = st.columns(4)

p1.metric("ROC-AUC", "0.9334")
p2.metric("PR-AUC", "0.6510")
p3.metric("Test Recall", "77.21%")
p4.metric("Event Detection", "64.19%")

st.caption(
    "Held-out test evaluation. Event-level benchmark: "
    "median AI warning lead time = 120 seconds."
)


# ---------------------------------------------------------
# METHANE TREND
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Methane Concentration — Recent 5 Minutes'
    '</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()

for sensor in ["MM263", "MM264", "MM256"]:

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
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="CH₄ (%)",
    hovermode="x unified",
)

st.plotly_chart(
    fig,
    width="stretch",
)


# ---------------------------------------------------------
# AI RISK TREND
# ---------------------------------------------------------

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
        name="V4 AI risk",
    )
)

fig2.add_hline(
    y=MODEL_THRESHOLD * 100,
    line_dash="dash",
    annotation_text="Locked V4 alert threshold",
)

fig2.update_layout(
    height=330,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="AI risk (%)",
    hovermode="x unified",
)

st.plotly_chart(
    fig2,
    width="stretch",
)


# ---------------------------------------------------------
# MINE CONDITIONS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Mine Conditions</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Airflow AN422", f"{current['AN422']:.2f}")
m2.metric("Airflow AN423", f"{current['AN423']:.2f}")
m3.metric("Temperature", f"{current['TP1721']:.1f} °C")
m4.metric("Humidity", f"{current['RH1722']:.1f}%")
m5.metric("Machine Speed", f"{current['V']:.2f}")


# ---------------------------------------------------------
# RECENT ALERTS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Recent V4 Alert Events</div>',
    unsafe_allow_html=True,
)

recent = window.copy()

recent["AI Alert"] = (
    recent["risk_probability"] >= MODEL_THRESHOLD
)

recent["Methane Alert"] = (
    recent["max_ch4"] >= RISKY_CH4
)

alerts = recent[
    recent["AI Alert"] | recent["Methane Alert"]
].tail(10).copy()

if alerts.empty:

    st.info("No alerts in the current replay window.")

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
        display["risk_probability"] * 100
    ).round(1).astype(str) + "%"

    display["CH₄"] = (
        display["max_ch4"]
    ).round(2).astype(str) + "%"

    display["AI Alert"] = display[
        "AI Alert"
    ].map({True: "HIGH", False: "—"})

    display["Methane Alert"] = display[
        "Methane Alert"
    ].map({True: "WARNING", False: "—"})

    display = display[
        [
            "timestamp",
            "CH₄",
            "AI Risk",
            "AI Alert",
            "Methane Alert",
        ]
    ].rename(
        columns={"timestamp": "Time"}
    )

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        MineGuard V4 • Held-out methane test replay •
        Locked threshold 0.24 • 3–6 minute prediction target
        <br><br>
        Prototype demonstration only — not a certified mine safety system.
    </div>
    """,
    unsafe_allow_html=True,
)