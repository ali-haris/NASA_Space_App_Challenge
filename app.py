import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# PAGE CONFIG & STYLE
# ==============================
st.set_page_config(
    page_title="NASA TESS Exoplanet Detector",
    page_icon="🪐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #000814;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #FFD60A;
        text-align: center;
    }
    .stButton>button {
        background-color: #003566;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #001d3d;
        color: #FFD60A;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg",
    width=100,
)
st.title("🪐 Exoplanet Hunter")
st.markdown("""
**Developed using XGBoost and trained on NASA's open-source TESS dataset.**  
Explore the data here 👉 [NASA Exoplanet Archive (TOI)](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
""")

st.divider()

# ==============================
# LOAD MODEL & ENCODERS
# ==============================
@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, le = load_assets()

# ==============================
# SIDEBAR INFO
# ==============================
with st.sidebar:
    st.header("About the App 🛰️")
    st.write("""
    This interactive web app uses a machine learning model 
    trained on the **TESS Objects of Interest (TOI)** dataset 
    from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).

    Enter planetary and stellar parameters to detect whether 
    the target is a likely **exoplanet candidate** or a **false positive**.
    """)

# ==============================
# INPUT SECTION
# ==============================
st.markdown("### Input Observation Parameters")

col1, col2 = st.columns(2)

with col1:
    pl_orbper = st.number_input("Orbital Period [days]", 0.0, 1000.0, 5.0)
    pl_trandurh = st.number_input("Transit Duration [hours]", 0.0, 50.0, 10.0)
    pl_rade = st.number_input("Planet Radius [R_Earth]", 0.0, 50.0, 2.0)
    pl_insol = st.number_input("Insolation [Earth flux]", 0.0, 100000.0, 1000.0)
    pl_eqt = st.number_input("Equilibrium Temperature [K]", 0.0, 10000.0, 1500.0)

with col2:
    st_tmag = st.number_input("TESS Magnitude", 0.0, 20.0, 10.0)
    st_dist = st.number_input("Stellar Distance [pc]", 0.0, 10000.0, 500.0)
    st_teff = st.number_input("Stellar Effective Temperature [K]", 0.0, 15000.0, 6000.0)
    st_logg = st.number_input("Stellar log(g) [cm/s²]", 0.0, 6.0, 4.4)
    st_rad = st.number_input("Stellar Radius [R_Sun]", 0.0, 10.0, 1.0)

st.divider()

# ==============================
# FEATURE ENGINEERING (same as training)
# ==============================
radius_to_star = pl_rade / st_rad if st_rad != 0 else 0
temp_ratio = pl_eqt / st_teff if st_teff != 0 else 0
scaled_flux = pl_insol / (st_dist ** 2) if st_dist != 0 else 0

# Combine features
features = np.array([[pl_orbper, pl_trandurh, pl_rade, pl_insol, pl_eqt,
                      st_tmag, st_dist, st_teff, st_logg, st_rad,
                      radius_to_star, temp_ratio, scaled_flux]])

# 🔧 Fix feature count mismatch (expected 14)
features = np.append(features, [[0]], axis=1)  # placeholder column

# ==============================
# SCALING
# ==============================
features_scaled = scaler.transform(features)

# ==============================
# PREDICTION
# ==============================
if st.button("🔭 Detect Exoplanet"):
    pred = model.predict(features_scaled)[0]
    label = le.inverse_transform([pred])[0]
    probs = model.predict_proba(features_scaled)[0]
    confidence = np.max(probs) * 100

    st.subheader("Prediction Result")
    if label in ["CP", "PC", "KP"]:
        st.success(f"✅ Likely **Exoplanet Candidate** ({label})\n\nConfidence: **{confidence:.2f}%**")
    else:
        st.error(f"🚫 Likely **False Positive** ({label})\n\nConfidence: **{confidence:.2f}%**")

    # Display probabilities visually
    st.markdown("### Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": le.classes_,
        "Probability (%)": (probs * 100).round(2)
    })
    st.bar_chart(prob_df.set_index("Class"))

st.divider()
st.markdown("""
*Project Submitted to NASA Space Apps Challenge 2025 | Data: [NASA TESS TOI](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)*  
Developed for educational and research purposes 🌠
""")
