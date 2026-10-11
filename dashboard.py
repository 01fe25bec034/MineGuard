import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_FILE = "mineguard_model.joblib"
SCALER_FILE = "mineguard_scaler.joblib"
FEATURE_FILE = "mineguard_features.joblib"
DATA_FILE = "dashboard_replay.csv"

MODEL_THRESHOLD = 0.50

SAFE_LIMIT = 0.60
MODERATE_LIMIT = 1.00

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

ZONES = {
    "Zone 1": "MM263",
    "Zone 2": "MM264",
    "Zone 3": "MM256",
}


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MineGuard Safety Command Center",
    page_icon="⛏️",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .mg-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .mg-subtitle {
        font-size: 1rem;
        opacity: 0.68;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-top: 24px;
        margin-bottom: 10px;
    }

    .zone-card {
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 16px;
        padding: 18px;
        min-height: 205px;
        background: rgba(255,255,255,0.025);
    }

    .zone-name {
        font-size: 1.25rem;
        font-weight: 800;
    }

    .zone-sensor {
        opacity: 0.6;
        font-size: 0.8rem;
        margin-top: 2px;
    }

    .zone-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 13px;
    }

    .zone-risk {
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 5px;
    }

    .status-strip {
        display: flex;
        gap: 9px;
        flex-wrap: wrap;
        margin: 14px 0 20px 0;
    }

    .status-pill {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 999px;
        padding: 7px 13px;
        font-size: 0.8rem;
        background: rgba(255,255,255,0.03);
    }

    .module-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 16px;
        min-height: 150px;
        background: rgba(255,255,255,0.025);
    }

    .module-title {
        font-weight: 750;
        font-size: 1.05rem;
    }

    .module-status {
        margin-top: 8px;
        font-size: 0.95rem;
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
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_FILE)

    return model, scaler, feature_names


# =========================================================
# LOAD DATA
# =========================================================

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


# =========================================================
# GENERATE V1 AI PREDICTIONS
# =========================================================

@st.cache_data
def generate_predictions(data, feature_names):

    base = data[FEATURE_COLS].copy()

    feature_data = {}

    for col in FEATURE_COLS:

        s = base[col]

        feature_data[f"{col}_current"] = s

        feature_data[f"{col}_mean60"] = (
            s.rolling(
                60,
                min_periods=60
            ).mean()
        )

        feature_data[f"{col}_mean180"] = (
            s.rolling(
                180,
                min_periods=180
            ).mean()
        )

        feature_data[f"{col}_mean300"] = (
            s.rolling(
                300,
                min_periods=300
            ).mean()
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
            s.rolling(
                60,
                min_periods=60
            ).max()
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

        X_valid = scaler.transform(
            X.loc[valid]
        )

        probabilities[valid] = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

    result = data.copy()

    result["risk_probability"] = probabilities

    return result


# =========================================================
# MODEL + DATA
# =========================================================

model, scaler, feature_names = load_model()

data = load_data()

result = generate_predictions(
    data,
    feature_names
)


# =========================================================
# USE LATEST DEMO RECORD
# =========================================================

current = result.iloc[-1]

window = result.tail(300).copy()

timestamp = current["timestamp"]

risk = current["risk_probability"]

if pd.isna(risk):
    risk = 0.0


# =========================================================
# ZONE RISK
# =========================================================

def zone_risk(ch4):

    if ch4 >= MODERATE_LIMIT:

        return (
            "VERY RISKY",
            "🔴",
            "RED"
        )

    elif ch4 >= SAFE_LIMIT:

        return (
            "MODERATE",
            "⚪",
            "WHITE"
        )

    else:

        return (
            "SAFE",
            "🟢",
            "GREEN"
        )


zone_results = {}

for zone, sensor in ZONES.items():

    value = float(current[sensor])

    status, led, led_name = zone_risk(value)

    zone_results[zone] = {
        "sensor": sensor,
        "value": value,
        "status": status,
        "led": led,
        "led_name": led_name,
    }


# =========================================================
# OVERALL AI STATUS
# =========================================================

if risk >= MODEL_THRESHOLD:

    ai_status = "HIGH"

elif risk >= 0.30:

    ai_status = "WATCH"

else:

    ai_status = "LOW"


very_risky_zones = [
    zone
    for zone, info in zone_results.items()
    if info["status"] == "VERY RISKY"
]

moderate_zones = [
    zone
    for zone, info in zone_results.items()
    if info["status"] == "MODERATE"
]


if very_risky_zones or ai_status == "HIGH":

    overall_status = "ATTENTION"

elif moderate_zones or ai_status == "WATCH":

    overall_status = "WATCH"

else:

    overall_status = "NORMAL"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="mg-title">⛏️ MineGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="mg-subtitle">'
    'Multi-Zone Edge-to-Core Mining Safety Command Center'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Prototype data snapshot: "
    f"{timestamp.strftime('%d %b %Y • %H:%M:%S')}"
    f"   |   AI prediction horizon: 3–6 minutes"
)


# =========================================================
# SYSTEM STATUS
# =========================================================

st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-pill">● METHANE AI: ONLINE</div>
        <div class="status-pill">● DASHBOARD: ONLINE</div>
        <div class="status-pill">● RFID: HARDWARE PENDING</div>
        <div class="status-pill">● CAMERA: HARDWARE PENDING</div>
        <div class="status-pill">● COLLISION: SENSOR PENDING</div>
        <div class="status-pill">● OVERALL: {overall_status}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP SUMMARY
# =========================================================

c1, c2, c3, c4 = st.columns(4)

max_ch4 = max(
    float(current["MM263"]),
    float(current["MM264"]),
    float(current["MM256"])
)

c1.metric(
    "Maximum CH₄",
    f"{max_ch4:.2f}%"
)

c2.metric(
    "Overall AI Risk",
    f"{risk * 100:.1f}%"
)

c3.metric(
    "AI Assessment",
    ai_status
)

c4.metric(
    "Prediction Horizon",
    "3–6 min"
)


# =========================================================
# ZONE MONITORING
# =========================================================

# =========================================================
# ZONE MONITORING
# =========================================================

st.markdown(
    '<div class="section-title">Multi-Zone Methane Detection</div>',
    unsafe_allow_html=True
)

zone_columns = st.columns(3)

for column, (zone, info) in zip(
    zone_columns,
    zone_results.items()
):

    with column:

        with st.container(border=True):

            st.markdown(f"### {zone}")

            st.caption(
                f"Methane sensor: {info['sensor']}"
            )

            st.metric(
                "Methane Level",
                f"{info['value']:.2f}% CH₄"
            )

            st.markdown(
                f"### {info['led']} {info['status']}"
            )

            st.caption(
                f"Physical LED: {info['led_name']}"
            )


# =========================================================
# AI ALERT
# =========================================================

if very_risky_zones:

    st.error(
        "🚨 VERY RISKY ZONE: "
        + ", ".join(very_risky_zones)
        + " has reached the prototype methane danger threshold."
    )

elif ai_status == "HIGH":

    st.error(
        "🚨 MINEGUARD AI ALERT: elevated probability "
        "of a future methane hazard in the 3–6 minute window."
    )

elif moderate_zones:

    st.warning(
        "🟡 MODERATE ZONE: "
        + ", ".join(moderate_zones)
        + " requires continued monitoring."
    )

else:

    st.success(
        "🟢 ALL ZONES CURRENTLY SAFE."
    )


# =========================================================
# AI RISK CHART
# =========================================================

st.markdown(
    '<div class="section-title">MineGuard AI Risk</div>',
    unsafe_allow_html=True
)

fig_ai = go.Figure()

fig_ai.add_trace(
    go.Scatter(
        x=window["timestamp"],
        y=window["risk_probability"] * 100,
        mode="lines",
        name="AI Risk"
    )
)

fig_ai.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="50% model alert threshold"
)

fig_ai.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="Risk Probability (%)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_ai,
    width="stretch"
)


# =========================================================
# METHANE TREND
# =========================================================

st.markdown(
    '<div class="section-title">Methane Trend by Zone</div>',
    unsafe_allow_html=True
)

fig_ch4 = go.Figure()

for zone, sensor in ZONES.items():

    fig_ch4.add_trace(
        go.Scatter(
            x=window["timestamp"],
            y=window[sensor],
            mode="lines",
            name=f"{zone} ({sensor})"
        )
    )

fig_ch4.add_hline(
    y=MODERATE_LIMIT,
    line_dash="dash",
    annotation_text="1.0% danger threshold"
)

fig_ch4.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="CH₄ (%)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_ch4,
    width="stretch"
)


# =========================================================
# WORKER + RFID
# =========================================================

st.markdown(
    '<div class="section-title">Worker Authorization</div>',
    unsafe_allow_html=True
)

r1, r2, r3 = st.columns(3)

with r1:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                RFID Worker Access
            </div>
            <div class="module-status">
                🟡 Hardware pending
            </div>
            <br>
            Worker scans will be checked against
            zone authorization rules.
        </div>
        """,
        unsafe_allow_html=True
    )

with r2:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Authorized Access
            </div>
            <div class="module-status">
                🟢 READY
            </div>
            <br>
            Zone-specific access control logic
            prepared for ESP32 + RFID.
        </div>
        """,
        unsafe_allow_html=True
    )

with r3:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Unauthorized Access
            </div>
            <div class="module-status">
                🔴 ALERT READY
            </div>
            <br>
            Unauthorized scans will trigger
            dashboard and local alerts.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CAMERA
# =========================================================

st.markdown(
    '<div class="section-title">AI Camera Surveillance</div>',
    unsafe_allow_html=True
)

cam1, cam2 = st.columns(2)

with cam1:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Camera Feed
            </div>
            <div class="module-status">
                🟡 ESP32-CAM hardware pending
            </div>
            <br>
            The final module will monitor workers
            and detect required PPE such as helmets
            and safety vests.
        </div>
        """,
        unsafe_allow_html=True
    )

with cam2:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                PPE Monitoring
            </div>
            <div class="module-status">
                🟡 Vision AI pending
            </div>
            <br>
            Missing PPE will generate a dashboard
            warning and local alert.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# COLLISION
# =========================================================

st.markdown(
    '<div class="section-title">Vehicle Collision Monitoring</div>',
    unsafe_allow_html=True
)

co1, co2, co3 = st.columns(3)

with co1:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Proximity Sensor
            </div>
            <div class="module-status">
                🟡 HC-SR04 pending
            </div>
            <br>
            Ultrasonic distance sensing will
            monitor nearby vehicles or obstacles.
        </div>
        """,
        unsafe_allow_html=True
    )

with co2:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Collision Warning
            </div>
            <div class="module-status">
                🟢 Logic ready
            </div>
            <br>
            SAFE → WARNING → CRITICAL
            proximity states.
        </div>
        """,
        unsafe_allow_html=True
    )

with co3:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Local Response
            </div>
            <div class="module-status">
                🔴 Buzzer / LED ready
            </div>
            <br>
            Critical proximity will trigger
            LCD, buzzer and LED alerts.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# OUTPUT CHANNELS
# =========================================================

st.markdown(
    '<div class="section-title">Three Output Channels</div>',
    unsafe_allow_html=True
)

o1, o2, o3 = st.columns(3)

with o1:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                LCD
            </div>
            <div class="module-status">
                🟡 ESP32 hardware pending
            </div>
            <br>
            Local methane level, zone,
            authorization and collision status.
        </div>
        """,
        unsafe_allow_html=True
    )

with o2:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                LED Indicators
            </div>
            <div class="module-status">
                🟢 SAFE &nbsp; ⚪ MODERATE &nbsp; 🔴 RISKY
            </div>
            <br>
            One simple physical status
            indicator for each zone.
        </div>
        """,
        unsafe_allow_html=True
    )

with o3:

    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Buzzer
            </div>
            <div class="module-status">
                🟡 Hardware pending
            </div>
            <br>
            Local audible warning for
            critical safety events.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# ACTIVE ALERTS
# =========================================================

st.markdown(
    '<div class="section-title">Active Alerts</div>',
    unsafe_allow_html=True
)

alerts = []

for zone, info in zone_results.items():

    if info["status"] == "VERY RISKY":

        alerts.append(
            f"🔴 {zone}: methane level {info['value']:.2f}%"
        )

    elif info["status"] == "MODERATE":

        alerts.append(
            f"⚪ {zone}: methane level {info['value']:.2f}%"
        )

if ai_status == "HIGH":

    alerts.append(
        "🚨 MineGuard AI: elevated future methane risk"
    )

if alerts:

    for alert in alerts:
        st.warning(alert)

else:

    st.success(
        "No active methane alerts in the current prototype snapshot."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        MineGuard prototype • Multi-zone methane monitoring •
        V1 trained AI • 3–6 minute prediction horizon
        <br><br>
        RFID, ESP32-CAM and HC-SR04 modules are prepared for
        hardware integration in the next phase.
        <br><br>
        Prototype demonstration only — not a certified mine safety system.
    </div>
    """,
    unsafe_allow_html=True
)
