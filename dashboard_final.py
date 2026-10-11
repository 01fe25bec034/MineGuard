import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from mineguard_fusion import fuse_risk, explain_risk


# =========================================================
# CONFIG
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
    "V",
]

ZONES = {
    "Zone 1": "MM263",
    "Zone 2": "MM264",
    "Zone 3": "MM256",
}

# Fixed historical demo snapshot.
# This disappears when live ESP32 input is connected.
DEMO_ROW = 3128


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MineGuard Safety Command Center",
    page_icon="⛏️",
    layout="wide",
)


# =========================================================
# STYLE
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
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 18px;
        min-height: 215px;
        background: rgba(255,255,255,0.025);
    }

    .zone-name {
        font-size: 1.25rem;
        font-weight: 800;
    }

    .zone-sensor {
        opacity: 0.62;
        font-size: 0.8rem;
    }

    .zone-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 12px;
    }

    .zone-risk {
        font-size: 1.1rem;
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
        min-height: 145px;
        background: rgba(255,255,255,0.025);
    }

    .module-title {
        font-weight: 750;
        font-size: 1.05rem;
    }

    .module-status {
        margin-top: 8px;
        font-size: 0.93rem;
    }

    .risk-box {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 18px;
        margin-top: 10px;
        background: rgba(255,255,255,0.025);
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
        dtype=np.float32,
    )

    if valid.any():

        X_valid = scaler.transform(
            X.loc[valid]
        )

        probabilities[valid] = (
            model.predict_proba(X_valid)[:, 1]
        )

    result = data.copy()

    result["risk_probability"] = probabilities

    return result


model, scaler, feature_names = load_model()

data = load_data()

result = generate_predictions(
    data,
    feature_names,
)


# =========================================================
# DEMO SNAPSHOT
# =========================================================

demo_position = min(
    DEMO_ROW,
    len(result) - 1,
)

current = result.iloc[demo_position]

window = result.iloc[
    max(300, demo_position - 299):
    demo_position + 1
].copy()

timestamp = current["timestamp"]

mine_ai_risk = current["risk_probability"]

if pd.isna(mine_ai_risk):
    mine_ai_risk = 0.0


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("MineGuard")

st.sidebar.caption(
    "Final prototype • Demo snapshot mode"
)

st.sidebar.divider()

focus_zone = st.sidebar.selectbox(
    "Focus zone",
    list(ZONES.keys()),
)

offline_demo = st.sidebar.toggle(
    "Simulate network disconnect",
    value=False,
)

st.sidebar.info(
    "Historical data is used only for the software demo. "
    "The ESP32/sensor layer will provide live values later."
)


# =========================================================
# ZONE RISK HELPERS
# =========================================================

def zone_class(ch4):

    if ch4 >= MODERATE_LIMIT:
        return "VERY RISKY", "🔴", "RED"

    if ch4 >= SAFE_LIMIT:
        return "MODERATE", "⚪", "WHITE"

    return "SAFE", "🟢", "GREEN"


zone_results = {}

for zone, sensor in ZONES.items():

    ch4 = float(current[sensor])

    status, symbol, led = zone_class(ch4)

    fusion_score, fusion_level = fuse_risk(
        ch4=ch4,
        ai_probability=mine_ai_risk,
        collision_status="PENDING",
        ppe_ok=None,
        authorized=None,
    )

    reasons, actions = explain_risk(
        ch4=ch4,
        ai_probability=mine_ai_risk,
        collision_status="PENDING",
        ppe_ok=None,
        authorized=None,
    )

    zone_results[zone] = {
        "sensor": sensor,
        "ch4": ch4,
        "status": status,
        "symbol": symbol,
        "led": led,
        "fusion_score": fusion_score,
        "fusion_level": fusion_level,
        "reasons": reasons,
        "actions": actions,
    }


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="mg-title">⛏️ MineGuard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="mg-subtitle">'
    'Multi-Zone Edge-to-Core Mining Safety Command Center'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Demo snapshot: "
    f"{timestamp.strftime('%d %b %Y • %H:%M:%S')}"
    f"   |   Prediction horizon: 3–6 minutes"
)


# =========================================================
# SYSTEM STATUS
# =========================================================

network_text = (
    "EDGE-ONLY DEMO"
    if offline_demo
    else "CORE CONNECTED"
)

network_note = (
    "Local safety path active"
    if offline_demo
    else "Dashboard connection available"
)

st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-pill">● METHANE AI: ONLINE</div>
        <div class="status-pill">● RISK FUSION: ONLINE</div>
        <div class="status-pill">● NETWORK: {network_text}</div>
        <div class="status-pill">● RFID: HARDWARE PENDING</div>
        <div class="status-pill">● CAMERA: HARDWARE PENDING</div>
        <div class="status-pill">● COLLISION: SENSOR PENDING</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if offline_demo:

    st.warning(
        "EDGE-ONLY MODE: the dashboard connection is considered "
        "unavailable for this demonstration. Local safety actions "
        "remain conceptually active."
    )

else:

    st.success(
        f"CORE CONNECTED: {network_note}"
    )


# =========================================================
# SUMMARY
# =========================================================

max_ch4 = max(
    float(current["MM263"]),
    float(current["MM264"]),
    float(current["MM256"]),
)

if mine_ai_risk >= MODEL_THRESHOLD:
    ai_status = "HIGH"
elif mine_ai_risk >= 0.30:
    ai_status = "WATCH"
else:
    ai_status = "LOW"


summary1, summary2, summary3, summary4 = st.columns(4)

summary1.metric(
    "Maximum CH₄",
    f"{max_ch4:.2f}%",
)

summary2.metric(
    "Mine-wide AI Risk",
    f"{mine_ai_risk * 100:.1f}%",
)

summary3.metric(
    "AI Assessment",
    ai_status,
)

summary4.metric(
    "Prediction Horizon",
    "3–6 min",
)


# =========================================================
# MINE MAP
# =========================================================

st.markdown(
    '<div class="section-title">Mine Zone Map</div>',
    unsafe_allow_html=True,
)

map1, map2, map3 = st.columns(3)

for column, zone in zip(
    [map1, map2, map3],
    ["Zone 1", "Zone 2", "Zone 3"],
):

    info = zone_results[zone]

    with column:

        with st.container(border=True):

            st.markdown(
                f"### {info['symbol']} {zone}"
            )

            st.metric(
                "CH₄",
                f"{info['ch4']:.2f}%",
            )

            st.markdown(
                f"**{info['fusion_level']}**"
            )

            st.caption(
                f"{info['sensor']} • LED {info['led']}"
            )


# =========================================================
# FOCUS ZONE
# =========================================================

focus = zone_results[focus_zone]

st.markdown(
    f'<div class="section-title">'
    f'{focus_zone} Risk Intelligence'
    f'</div>',
    unsafe_allow_html=True,
)

fc1, fc2, fc3 = st.columns(3)

fc1.metric(
    "Current CH₄",
    f"{focus['ch4']:.2f}%",
)

fc2.metric(
    "Fusion Risk",
    f"{focus['fusion_score'] * 100:.0f}%",
)

fc3.metric(
    "Zone State",
    focus["fusion_level"],
)


# =========================================================
# WHY IS THIS ZONE RISKY?
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Why Is This Zone Risky?'
    '</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):

    for reason in focus["reasons"]:
        st.write(f"• {reason}")

    st.markdown("**Recommended response**")

    for action in focus["actions"]:
        st.write(f"→ {action}")


# =========================================================
# METHANE TREND
# =========================================================

st.markdown(
    '<div class="section-title">Methane Trend by Zone</div>',
    unsafe_allow_html=True,
)

fig_ch4 = go.Figure()

for zone, sensor in ZONES.items():

    fig_ch4.add_trace(
        go.Scatter(
            x=window["timestamp"],
            y=window[sensor],
            mode="lines",
            name=f"{zone} ({sensor})",
        )
    )

fig_ch4.add_hline(
    y=MODERATE_LIMIT,
    line_dash="dash",
    annotation_text="1.0% prototype threshold",
)

fig_ch4.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="CH₄ (%)",
    hovermode="x unified",
)

st.plotly_chart(
    fig_ch4,
    width="stretch",
)


# =========================================================
# AI RISK TREND
# =========================================================

st.markdown(
    '<div class="section-title">MineGuard AI Risk Trend</div>',
    unsafe_allow_html=True,
)

fig_ai = go.Figure()

fig_ai.add_trace(
    go.Scatter(
        x=window["timestamp"],
        y=window["risk_probability"] * 100,
        mode="lines",
        name="AI Risk",
    )
)

fig_ai.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="50% model threshold",
)

fig_ai.update_layout(
    height=300,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="Risk Probability (%)",
    hovermode="x unified",
)

st.plotly_chart(
    fig_ai,
    width="stretch",
)


# =========================================================
# WORKER SAFETY
# =========================================================

st.markdown(
    '<div class="section-title">Worker Safety</div>',
    unsafe_allow_html=True,
)

w1, w2, w3 = st.columns(3)

with w1:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                RFID Authorization
            </div>
            <div class="module-status">
                🟡 Hardware pending
            </div>
            <br>
            Zone-specific worker authorization
            will be checked when RFID readers
            are connected.
        </div>
        """,
        unsafe_allow_html=True,
    )

with w2:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                PPE Camera
            </div>
            <div class="module-status">
                🟡 Vision AI pending
            </div>
            <br>
            Helmet and safety-vest detection
            will generate safety alerts.
        </div>
        """,
        unsafe_allow_html=True,
    )

with w3:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Worker Safety State
            </div>
            <div class="module-status">
                🟡 Live hardware pending
            </div>
            <br>
            Authorization + PPE + zone risk
            will form the worker safety state.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# COLLISION
# =========================================================

st.markdown(
    '<div class="section-title">Vehicle Collision Monitoring</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                HC-SR04 Proximity
            </div>
            <div class="module-status">
                🟡 Sensor pending
            </div>
            <br>
            Measures distance to nearby
            vehicles or obstacles.
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Collision Logic
            </div>
            <div class="module-status">
                🟢 Logic ready
            </div>
            <br>
            SAFE → WARNING → CRITICAL
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">
                Local Alert
            </div>
            <div class="module-status">
                🟡 ESP32 pending
            </div>
            <br>
            LCD + buzzer + LED response
            for dangerous proximity.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# EDGE OUTPUTS
# =========================================================

st.markdown(
    '<div class="section-title">Local Safety Outputs</div>',
    unsafe_allow_html=True,
)

o1, o2, o3 = st.columns(3)

with o1:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">LCD</div>
            <div class="module-status">
                🟡 ESP32 pending
            </div>
            <br>
            Zone, methane, worker authorization
            and collision state.
        </div>
        """,
        unsafe_allow_html=True,
    )

with o2:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">LEDs</div>
            <div class="module-status">
                🟢 Safe • ⚪ Moderate • 🔴 Risky
            </div>
            <br>
            Physical zone status indicator.
        </div>
        """,
        unsafe_allow_html=True,
    )

with o3:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-title">Buzzer</div>
            <div class="module-status">
                🟡 ESP32 pending
            </div>
            <br>
            Local audible warning for
            critical events.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ACTIVE ALERTS
# =========================================================

st.markdown(
    '<div class="section-title">Active Alerts</div>',
    unsafe_allow_html=True,
)

active_alerts = []

for zone, info in zone_results.items():

    if info["fusion_level"] in ["HIGH", "CRITICAL"]:

        active_alerts.append(
            f"🚨 {zone}: {info['fusion_level']} "
            f"risk • CH₄ {info['ch4']:.2f}%"
        )

    elif info["status"] == "MODERATE":

        active_alerts.append(
            f"⚪ {zone}: MODERATE methane "
            f"• CH₄ {info['ch4']:.2f}%"
        )

if mine_ai_risk >= MODEL_THRESHOLD:

    active_alerts.append(
        "🚨 Mine-wide AI prediction indicates elevated "
        "future methane risk."
    )

if active_alerts:

    for alert in active_alerts:
        st.warning(alert)

else:

    st.success(
        "🟢 No active hazard alerts in this demo snapshot."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        MineGuard prototype • 3-zone methane intelligence •
        V1 trained model • risk fusion • edge-to-core architecture
        <br><br>
        RFID, PPE vision, HC-SR04 and physical outputs are
        prepared for the ESP32 integration phase.
        <br><br>
        Prototype demonstration only — not a certified mine safety system.
    </div>
    """,
    unsafe_allow_html=True,
)
