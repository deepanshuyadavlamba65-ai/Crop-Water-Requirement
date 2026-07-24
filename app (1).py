import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Water Ledger",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------------------
# Theme — "Field Ledger": a dusk irrigation-log palette, not the usual cream/terracotta
# --------------------------------------------------------------------------------------
INK = "#F2ECD8"        # cream text
PAPER = "#16211A"       # deep pine-soil background
PAPER_2 = "#1E2C22"      # panel background
LINE = "#3A4B3C"        # hairline rule
GOLD = "#D8AA4D"        # wheat / signature accent
WATER = "#5FA6C7"        # irrigation blue
LEAF = "#7FA65A"        # crop green
RUST = "#C9714B"        # warning / high-demand
MUTED = "#9FAE9A"        # secondary text

FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
"""

CUSTOM_CSS = f"""
<style>
{FONT_CSS}

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

.stApp {{
    background: {PAPER};
    color: {INK};
}}

/* Hide default streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {PAPER_2};
    border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {{
    color: {INK} !important;
}}

/* Headings */
h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    color: {INK};
    letter-spacing: 0.2px;
}}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 12px;
    color: {GOLD};
    margin-bottom: 4px;
}}

.ledger-rule {{
    border: none;
    border-top: 1px solid {LINE};
    margin: 18px 0;
}}

/* Metric card */
.field-card {{
    background: {PAPER_2};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 22px 26px;
}}

.field-card .label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: {MUTED};
}}

.field-card .value {{
    font-family: 'Fraunces', serif;
    font-size: 40px;
    font-weight: 600;
    color: {INK};
    margin-top: 2px;
}}

.field-card .sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: {WATER};
    margin-top: 4px;
}}

/* Buttons */
.stButton > button {{
    background: {GOLD};
    color: {PAPER};
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 13px;
    padding: 10px 18px;
    width: 100%;
}}
.stButton > button:hover {{
    background: #E8BE6C;
    color: {PAPER};
}}

/* Sliders */
div[data-testid="stSlider"] > div > div > div > div {{
    background: {GOLD};
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    color: {MUTED};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {GOLD} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {LINE};
}}

/* Expander */
details {{
    background: {PAPER_2};
    border: 1px solid {LINE} !important;
    border-radius: 4px;
}}

/* Selectbox / number input */
div[data-baseweb="select"] > div, input {{
    background-color: {PAPER} !important;
    color: {INK} !important;
    border-color: {LINE} !important;
}}

.stCaption, .css-1offfwp {{
    color: {MUTED} !important;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Load model artifacts (cached)
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "crop_water_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    crop_enc = joblib.load(os.path.join(BASE_DIR, "crop_encoder.pkl"))
    soil_enc = joblib.load(os.path.join(BASE_DIR, "soil_encoder.pkl"))
    return model, scaler, crop_enc, soil_enc


@st.cache_data
def load_training_data():
    path = os.path.join(BASE_DIR, "training_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


model, scaler, crop_enc, soil_enc = load_artifacts()
train_df = load_training_data()

FEATURE_COLS = [
    "Crop_Type_enc", "Soil_Type_enc", "Temperature_C", "Humidity_%",
    "Wind_Speed_kmph", "Solar_Radiation_MJm2", "Rainfall_mm",
    "Soil_Moisture_%", "Growth_Stage_Days",
]

CROP_OPTIONS = list(crop_enc.classes_)
SOIL_OPTIONS = list(soil_enc.classes_)


def predict_one(crop_type, soil_type, temperature, humidity, wind_speed,
                 solar_radiation, rainfall, soil_moisture, growth_stage_days):
    crop_code = crop_enc.transform([crop_type])[0]
    soil_code = soil_enc.transform([soil_type])[0]
    row = pd.DataFrame([{
        "Crop_Type_enc": crop_code,
        "Soil_Type_enc": soil_code,
        "Temperature_C": temperature,
        "Humidity_%": humidity,
        "Wind_Speed_kmph": wind_speed,
        "Solar_Radiation_MJm2": solar_radiation,
        "Rainfall_mm": rainfall,
        "Soil_Moisture_%": soil_moisture,
        "Growth_Stage_Days": growth_stage_days,
    }])[FEATURE_COLS]
    scaled = scaler.transform(row)
    return float(model.predict(scaled)[0])


# --------------------------------------------------------------------------------------
# Sidebar — the field entry form
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Field Entry</div>', unsafe_allow_html=True)
    st.markdown("### Today's conditions")
    st.caption("Log the field, then read the recommended irrigation depth.")

    crop_type = st.selectbox("Crop", CROP_OPTIONS, index=CROP_OPTIONS.index("Wheat") if "Wheat" in CROP_OPTIONS else 0)
    soil_type = st.selectbox("Soil type", SOIL_OPTIONS, index=SOIL_OPTIONS.index("Loamy") if "Loamy" in SOIL_OPTIONS else 0)

    st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Weather</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)", 5.0, 48.0, 28.0, 0.5)
    humidity = st.slider("Humidity (%)", 5.0, 100.0, 55.0, 1.0)
    wind_speed = st.slider("Wind speed (km/h)", 0.0, 40.0, 8.0, 0.5)
    solar_radiation = st.slider("Solar radiation (MJ/m²)", 2.0, 32.0, 20.0, 0.5)
    rainfall = st.slider("Rainfall — last 24h (mm)", 0.0, 60.0, 2.0, 0.5)

    st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Field &amp; crop</div>', unsafe_allow_html=True)
    soil_moisture = st.slider("Soil moisture (%)", 0.0, 100.0, 30.0, 1.0)
    growth_stage_days = st.slider("Days since sowing", 0, 200, 45, 1)

    st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
    field_area = st.number_input("Field area (acres)", min_value=0.0, value=1.0, step=0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Predict water requirement")


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Data Science Internship · Precision Irrigation</div>', unsafe_allow_html=True)
st.markdown("# Crop Water Requirement Ledger")
st.markdown(
    f'<span style="color:{MUTED}">A Random Forest model reads today\'s weather and field state, '
    f'and estimates how many millimetres of water the crop needs per day.</span>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Gauge helper
# --------------------------------------------------------------------------------------
def water_gauge(value, max_value=12):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " mm/day", "font": {"family": "Fraunces, serif", "size": 46, "color": INK}},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": MUTED, "tickfont": {"color": MUTED, "family": "IBM Plex Mono"}},
            "bar": {"color": WATER, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_value * 0.35], "color": "#233327"},
                {"range": [max_value * 0.35, max_value * 0.7], "color": "#2B3C2E"},
                {"range": [max_value * 0.7, max_value], "color": "#3C2A22"},
            ],
            "threshold": {
                "line": {"color": GOLD, "width": 3},
                "thickness": 0.85,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK},
        height=280,
        margin=dict(t=30, b=10, l=30, r=30),
    )
    return fig


def demand_label(mm):
    if mm < 2:
        return "Low demand", LEAF
    elif mm < 5:
        return "Moderate demand", GOLD
    else:
        return "High demand", RUST


# --------------------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------------------
if run or "last_result" in st.session_state:
    if run:
        pred = predict_one(crop_type, soil_type, temperature, humidity, wind_speed,
                            solar_radiation, rainfall, soil_moisture, growth_stage_days)
        st.session_state["last_result"] = dict(
            pred=pred, crop=crop_type, soil=soil_type, temperature=temperature,
            humidity=humidity, wind_speed=wind_speed, solar_radiation=solar_radiation,
            rainfall=rainfall, soil_moisture=soil_moisture,
            growth_stage_days=growth_stage_days, field_area=field_area,
        )
    r = st.session_state["last_result"]
    pred = r["pred"]
    label, label_color = demand_label(pred)

    liters_per_acre = pred * 4046.86  # 1 mm over 1 m^2 = 1 L; 1 acre = 4046.86 m^2
    total_liters = liters_per_acre * r["field_area"]

    col_gauge, col_readout = st.columns([1.1, 1])

    with col_gauge:
        st.plotly_chart(water_gauge(pred), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div style="text-align:center; font-family:\'IBM Plex Mono\', monospace; '
            f'color:{label_color}; letter-spacing:2px; text-transform:uppercase; font-size:13px;">{label}</div>',
            unsafe_allow_html=True,
        )

    with col_readout:
        st.markdown(f"""
        <div class="field-card">
            <div class="label">Irrigation depth per acre</div>
            <div class="value">{liters_per_acre:,.0f} L</div>
            <div class="sub">per day, at {pred:.2f} mm/day</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="field-card">
            <div class="label">Total for {r['field_area']:.1f} acre field</div>
            <div class="value">{total_liters:,.0f} L</div>
            <div class="sub">≈ {total_liters/1000:,.1f} m³ / day</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

    # ---------------- Crop comparison at same conditions ----------------
    st.markdown('<div class="eyebrow">Cross-check</div>', unsafe_allow_html=True)
    st.markdown("### How other crops would fare in this same field")
    st.caption("Same weather and soil, every crop swapped in — a sanity check for the prediction above.")

    compare_rows = []
    for c in CROP_OPTIONS:
        p = predict_one(c, r["soil"], r["temperature"], r["humidity"], r["wind_speed"],
                         r["solar_radiation"], r["rainfall"], r["soil_moisture"], r["growth_stage_days"])
        compare_rows.append({"Crop": c, "Water_Requirement_mm": p})
    comp_df = pd.DataFrame(compare_rows).sort_values("Water_Requirement_mm", ascending=True)

    bar_colors = [GOLD if c == r["crop"] else "#3E5442" for c in comp_df["Crop"]]
    fig_bar = go.Figure(go.Bar(
        x=comp_df["Water_Requirement_mm"],
        y=comp_df["Crop"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in comp_df["Water_Requirement_mm"]],
        textposition="outside",
        textfont={"color": INK, "family": "IBM Plex Mono"},
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "family": "IBM Plex Sans"},
        height=320,
        margin=dict(t=10, b=10, l=10, r=40),
        xaxis=dict(title="mm/day", gridcolor=LINE, zerolinecolor=LINE),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("Fill in the field entry on the left and press **Predict water requirement** to read today's ledger.")

    if train_df is not None:
        st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Reference</div>', unsafe_allow_html=True)
        st.markdown("### Water demand across the training records")
        fig_hist = go.Figure(go.Histogram(
            x=train_df["Water_Requirement_mm"],
            marker_color=WATER,
            nbinsx=40,
        ))
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": INK, "family": "IBM Plex Sans"},
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title="Water requirement (mm/day)", gridcolor=LINE),
            yaxis=dict(title="Records", gridcolor=LINE),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------------------------------
# Model notes — feature importance + metrics, always visible at the bottom
# --------------------------------------------------------------------------------------
st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Under the hood</div>', unsafe_allow_html=True)
st.markdown("### About this model")

tab1, tab2, tab3 = st.tabs(["Feature importance", "Validation scores", "Data & method"])

with tab1:
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=imp.values,
            y=imp.index,
            orientation="h",
            marker_color=LEAF,
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": INK, "family": "IBM Plex Sans"},
            height=340,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title="Relative importance", gridcolor=LINE),
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.write("This model type doesn't expose feature importances directly.")

with tab2:
    st.markdown(f"""
    Three models were trained on an 80/20 split of 1,500 synthetic field records
    and compared on held-out data:

    | Model | MAE | RMSE | R² |
    |---|---|---|---|
    | **Random Forest (deployed)** | 0.150 | 0.264 | **0.761** |
    | Gradient Boosting | 0.182 | 0.281 | 0.729 |
    | Linear Regression | 0.303 | 0.410 | 0.423 |

    Random Forest had the strongest R² and was saved as `crop_water_model.pkl`.
    """)

with tab3:
    st.markdown("""
    **Target.** `Water_Requirement_mm` was engineered from a simplified reference
    evapotranspiration (ETo) estimate, scaled by a crop coefficient (Kc) and a soil
    water-holding factor, then adjusted down for rainfall and soil moisture and up
    for crop growth stage.

    **Inputs.** Crop type, soil type, temperature, humidity, wind speed, solar
    radiation, rainfall, soil moisture, and days since sowing — label-encoded and
    standard-scaled before entering the model.

    **A note on the data.** This build uses a synthetically generated dataset
    (1,500 rows) rather than field-sensor readings, so it's best read as a working
    proof of concept — the relationships are plausible but not agronomically
    validated. A natural next step would be retraining on real irrigation and
    weather-station records for a specific region.
    """)

st.markdown(
    f'<div style="text-align:center; color:{MUTED}; font-family:\'IBM Plex Mono\', monospace; '
    f'font-size:11px; margin-top:24px;">Crop Water Requirement Prediction — Data Science Internship Project</div>',
    unsafe_allow_html=True,
)
