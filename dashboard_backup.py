import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from collision_engine import collision_risk

MODEL_FILE = "mineguard_model.joblib"
SCALER_FILE = "mineguard_scaler.joblib"
FEATURE_FILE = "mineguard_features.joblib"
DATA_FILE = "dashboard_replay.csv"

MODEL_THRESHOLD = 0.50
CH4_WARNING = 1.0

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

st.set_page_config(
    page_title="MineGuard Safety Command Center",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .mg-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .mg-subtitle {
        opacity: 0.7;
        font-size: 1rem;
    }

    .status-strip {
        display: flex;
        gap: 9px;
        flex-wrap: wrap;
        margin: 14px 0 22px 0;
    }

    .status-pill {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 7px 13px;
        font-size: 0.80rem;
        background: rgba(255,255,255,0.03);
    }

    .info-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 16px;
        min-height: 110px;
        background: rgba(255,255,255,0.025);
    }

    .label {
        font-size: 0.78rem;
        opacity: 0.62;
    }

    .big-value {
        font-size: 1.8rem;
        font-weight: 750;
        margin-top: 5px;
    }

    .small-note {
        font-size: 0.75rem;
        opacity: 0.55;
        margin-top: 4px;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .footer {
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px solid rgba(255,255,255,0.10);
        opacity: 0.55;
        font-size: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_FILE)
    return model, scaler, feature_names


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
            "minute": df["minute"],
            "second": df["second"],
        }
    )

    return df


@st.cache_data
def generate_predictions(data, feature_names):

    base = data[FEATURE_COLS].copy()
    feature_data = {}

    for col in FEATURE_COLS:

        s = base[col]

        feature_data[f"{col}_current"] = s
        feature_data[f"{col}_mean60"] = (
            s.rolling(60, min_periods=60).mean()
        )
        feature_data[f"{col}_mean180"] = (
            s.rolling(180, min_periods=180).mean()
        )
        feature_data[f"{col}_mean300"] = (
            s.rolling(300, min_periods=300).mean()
        )
        feature_data[f"{col}_change60"] = (
            s - s.shift(60)
        )
        feature_data[f"{col}_change180"] = (
            s - s.shift(180)
        )
        feature_data[f"{col}_change300"] = (
            s - s.shift(300)
        )
        feature_data[f"{col}_max60"] = (
            s.rolling(60, min_periods=60).max()
        )

    features = pd.DataFrame(feature_data)

    X = features[feature_names]

    valid = X.notna().all(axis=1)

    probabilities = np.full(
        len(X),
        np.nan,
        dtype=np.float32
    )

    if valid.any():

        X_valid = scaler.transform(X.loc[valid])

        probabilities[valid] = model.predict_proba(
            X_valid
        )[:, 1]

    result = data.copy()
    result["risk_probability"] = probabilities

    return result


model, scaler, feature_names = load_model()
data = load_data()
result = generate_predictions(data, feature_names)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("MineGuard Control")
st.sidebar.caption("Historical Replay Mode")
st.sidebar.divider()

position = st.sidebar.slider(
    "Replay position",
    min_value=300,
    max_value=len(result) - 1,
    value=min(3128, len(result) - 1),
    step=1
)

collision_distance = st.sidebar.slider(
    "Simulated collision distance (cm)",
    min_value=5,
    max_value=200,
    value=120,
    step=1
)

st.sidebar.info(
    "Historical mine data is being replayed through the "
    "trained MineGuard V1 model."
)

# ---------------------------------------------------------
# CURRENT SAMPLE
# ---------------------------------------------------------

current = result.iloc[position]

window = result.iloc[
    max(300, position - 299):position + 1
].copy()

timestamp = current["timestamp"]

max_ch4 = max(
    float(current["MM263"]),
    float(current["MM264"]),
    float(current["MM256"])
)

risk = current["risk_probability"]

if pd.isna(risk):
    risk = 0.0

if max_ch4 >= CH4_WARNING:
    sensor_status = "WARNING"
else:
    sensor_status = "NORMAL"

if risk >= MODEL_THRESHOLD:
    ai_status = "HIGH"
elif risk >= 0.30:
    ai_status = "WATCH"
else:
    ai_status = "LOW"

collision_status, collision_score, collision_message = collision_risk(
    collision_distance
)

if (
    collision_status == "CRITICAL"
    or sensor_status == "WARNING"
    or ai_status == "HIGH"
):
    overall = "ATTENTION"

elif (
    collision_status == "WARNING"
    or ai_status == "WATCH"
):
    overall = "WATCH"

else:
    overall = "NORMAL"


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="mg-title">⛏️ MineGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="mg-subtitle">'
    'Edge-to-Core AI Mining Safety Command Center'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Replay timestamp: {timestamp.strftime('%d %b %Y • %H:%M:%S')}"
    f"   |   Prediction horizon: 3–6 minutes"
)

st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-pill">● SENSOR INPUT: REPLAY</div>
        <div class="status-pill">● FEATURE ENGINE: ONLINE</div>
        <div class="status-pill">● AI MODEL: V1 ONLINE</div>
        <div class="status-pill">● COLLISION: {collision_status}</div>
        <div class="status-pill">● DASHBOARD: ONLINE</div>
        <div class="status-pill">● OVERALL: {overall}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TOP CARDS
# ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">MAXIMUM CH₄</div>
            <div class="big-value">{max_ch4:.2f}%</div>
            <div class="small-note">
                Sensor warning: {CH4_WARNING:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">AI RISK PROBABILITY</div>
            <div class="big-value">{risk * 100:.1f}%</div>
            <div class="small-note">
                Model alert: {MODEL_THRESHOLD * 100:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">SENSOR STATUS</div>
            <div class="big-value">{sensor_status}</div>
            <div class="small-note">
                Critical methane monitoring
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="label">AI ASSESSMENT</div>
            <div class="big-value">{ai_status}</div>
            <div class="small-note">
                MineGuard V1
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# ALERT MESSAGE
# ---------------------------------------------------------

if collision_status == "CRITICAL":

    st.error(
        f"🚨 COLLISION CRITICAL: object detected at "
        f"{collision_distance} cm. {collision_message}"
    )

elif collision_status == "WARNING":

    st.warning(
        f"⚠️ PROXIMITY WARNING: object detected at "
        f"{collision_distance} cm. {collision_message}"
    )

elif sensor_status == "WARNING" and ai_status == "HIGH":

    st.error(
        "🚨 CRITICAL: methane threshold reached and "
        "MineGuard AI indicates elevated future hazard risk."
    )

elif sensor_status == "WARNING":

    st.warning(
        f"⚠️ SENSOR WARNING: current CH₄ is at or above "
        f"the {CH4_WARNING:.1f}% threshold. "
        "AI risk is assessed separately."
    )

elif ai_status == "HIGH":

    st.error(
        "🚨 AI PREDICTION ALERT: elevated probability of "
        "a future methane hazard within 3–6 minutes."
    )

elif ai_status == "WATCH":

    st.warning(
        "🟡 AI WATCH: elevated risk detected. Continue monitoring."
    )

else:

    st.success(
        "🟢 NORMAL: no current methane threshold breach, "
        "elevated AI risk, or collision warning."
    )


# ---------------------------------------------------------
# CRITICAL METHANE SENSORS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Critical Methane Sensors</div>',
    unsafe_allow_html=True
)

s1, s2, s3 = st.columns(3)

s1.metric("MM263", f"{current['MM263']:.2f}% CH₄")
s2.metric("MM264", f"{current['MM264']:.2f}% CH₄")
s3.metric("MM256", f"{current['MM256']:.2f}% CH₄")

# ---------------------------------------------------------
# COLLISION MONITORING
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Collision Monitoring</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Proximity Distance",
    f"{collision_distance} cm"
)

col2.metric(
    "Collision Risk",
    f"{collision_score * 100:.0f}%"
)

col3.metric(
    "Collision Status",
    collision_status
)

# ---------------------------------------------------------
# METHANE CHART
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'Methane Concentration — Last 5 Minutes'
    '</div>',
    unsafe_allow_html=True
)

fig = go.Figure()

for sensor in ["MM263", "MM264", "MM256"]:
    fig.add_trace(
        go.Scatter(
            x=window["timestamp"],
            y=window[sensor],
            mode="lines",
            name=sensor
        )
    )

fig.add_hline(
    y=CH4_WARNING,
    line_dash="dash",
    annotation_text="1.0% warning threshold"
)

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="CH₄ (%)",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ---------------------------------------------------------
# AI RISK CHART
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'MineGuard AI Risk — Last 5 Minutes'
    '</div>',
    unsafe_allow_html=True
)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=window["timestamp"],
        y=window["risk_probability"] * 100,
        mode="lines",
        name="AI Risk"
    )
)

fig2.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="50% AI alert threshold"
)

fig2.update_layout(
    height=330,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="Risk probability (%)",
    hovermode="x unified"
)

st.plotly_chart(
    fig2,
    width="stretch"
)


# ---------------------------------------------------------
# MINE CONDITIONS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Mine Conditions</div>',
    unsafe_allow_html=True
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
    '<div class="section-title">Recent Alert Events</div>',
    unsafe_allow_html=True
)

recent = window.copy()

recent["sensor_alert"] = (
    recent[["MM263", "MM264", "MM256"]].max(axis=1)
    >= CH4_WARNING
)

recent["ai_alert"] = (
    recent["risk_probability"] >= MODEL_THRESHOLD
)

alerts = recent[
    recent["sensor_alert"] | recent["ai_alert"]
][
    [
        "timestamp",
        "MM263",
        "MM264",
        "MM256",
        "risk_probability",
        "sensor_alert",
        "ai_alert"
    ]
].tail(8).copy()

if alerts.empty:

    st.info(
        "No alert events in the current five-minute window."
    )

else:

    alerts["AI Risk"] = (
        alerts["risk_probability"] * 100
    ).round(1).astype(str) + "%"

    alerts["Sensor Alert"] = alerts[
        "sensor_alert"
    ].map(
        {True: "WARNING", False: "—"}
    )

    alerts["AI Alert"] = alerts[
        "ai_alert"
    ].map(
        {True: "HIGH", False: "—"}
    )

    alerts = alerts[
        [
            "timestamp",
            "MM263",
            "MM264",
            "MM256",
            "AI Risk",
            "Sensor Alert",
            "AI Alert"
        ]
    ]

    alerts = alerts.rename(
        columns={"timestamp": "Time"}
    )

    st.dataframe(
        alerts,
        width="stretch",
        hide_index=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        MineGuard prototype • Historical mine-data replay •
        V1 trained model • 3–6 minute prediction target
        <br><br>
        Prototype demonstration only — not a certified mine safety system.
    </div>
    """,
    unsafe_allow_html=True
)
