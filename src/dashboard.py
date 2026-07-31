import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Mental Fatigue Predictor", page_icon="⚽", layout="wide")

# --- Load model artifacts ---
model = joblib.load("models/deployment_model.joblib")
features = joblib.load("models/deployment_features.joblib")
label_map = joblib.load("models/label_map.joblib")
inv_label_map = {v: k for k, v in label_map.items()}
feature_stats = pd.read_csv("models/feature_stats.csv", index_col=0)

COLOR = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}
EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# --- Global styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 20%, #1a1f2e 0%, #0b0e14 55%, #05070a 100%);
}

section[data-testid="stSidebar"] {
    background: #11151c;
    border-right: 1px solid #232935;
}

.hero {
    padding: 8px 0 24px 0;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60A5FA, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero p {
    color: #8b95a5;
    font-size: 1.05rem;
    margin-top: 4px;
}

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 28px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 22px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 6px;
}

.confidence-text {
    color: #8b95a5;
    font-size: 0.95rem;
    margin-top: 6px;
}

.prob-row { margin: 14px 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: #c9d1d9;
    margin-bottom: 6px;
    font-weight: 600;
}
.prob-track {
    background: #1c2230;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

.placeholder-box {
    text-align: center;
    padding: 60px 20px;
    color: #5b6472;
}
.placeholder-box .big-icon { font-size: 3rem; margin-bottom: 12px; }

.footer-note {
    color: #4b5563;
    font-size: 0.8rem;
    text-align: center;
    padding-top: 20px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 14px 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: player input ---
with st.sidebar:
    st.markdown("### ⚽ Player Input")
    st.caption("Enter today's values")

    st.markdown("**Wellness**")
    wellness_features = ["mood", "stress", "readiness", "sleep_duration", "sleep_quality", "soreness"]
    input_data = {}
    for feat in wellness_features:
        row = feature_stats.loc[feat]
        label = feat.replace("_", " ").title()
        if feat == "sleep_duration":
            input_data[feat] = st.slider(f"😴 {label} (hrs)", 0.0, 12.0, float(round(row["mean"], 1)), 0.5)
        else:
            input_data[feat] = st.slider(f"• {label}", 1, 5, int(round(row["mean"])))

    st.markdown("---")
    st.markdown("**Training Load**")
    load_features = ["daily_load", "weekly_load", "acwr", "atl", "ctl28", "ctl42", "monotony", "strain"]
    for feat in load_features:
        row = feature_stats.loc[feat]
        input_data[feat] = st.number_input(
            feat.replace("_", " ").upper(),
            min_value=0.0,
            max_value=float(row["max"] * 1.2),
            value=float(round(row["mean"], 2)),
            step=1.0
        )

    st.markdown("---")
    predict_btn = st.button("🔍  Predict Fatigue Status", use_container_width=True, type="primary")

# --- Main area ---
st.markdown("""
<div class="hero">
    <h1>Mental Fatigue Prediction Dashboard</h1>
    <p>Machine learning-based fatigue classification using wellness and training load metrics</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.1], gap="large")

if predict_btn:
    X_input = pd.DataFrame([input_data])[features]
    pred_class = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    pred_label = inv_label_map[pred_class]
    confidence = pred_proba[pred_class] * 100

    with col_left:
        st.markdown(f"""
        <div class="glass-card">
            <div class="result-badge" style="background:{COLOR[pred_label]}22; color:{COLOR[pred_label]}; border:1px solid {COLOR[pred_label]}55;">
                {EMOJI[pred_label]} {pred_label} Fatigue
            </div>
            <p class="confidence-text">Model confidence: <b style="color:#e6edf3">{confidence:.1f}%</b></p>
        """, unsafe_allow_html=True)

        for i in range(3):
            label = inv_label_map[i]
            pct = pred_proba[i] * 100
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label"><span>{EMOJI[label]} {label}</span><span>{pct:.1f}%</span></div>
                <div class="prob-track"><div class="prob-fill" style="width:{pct}%; background:{COLOR[label]};"></div></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mood", input_data["mood"])
        m2.metric("Sleep Quality", input_data["sleep_quality"])
        m3.metric("Soreness", input_data["soreness"])

    with col_right:
        risk_score = pred_proba[1] * 50 + pred_proba[2] * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "", "font": {"color": "#e6edf3", "size": 40}},
            title={"text": "Fatigue Risk Index", "font": {"color": "#8b95a5", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4b5563"},
                "bar": {"color": COLOR[pred_label]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "#16241c"},
                    {"range": [33, 66], "color": "#2d2415"},
                    {"range": [66, 100], "color": "#2d1515"},
                ],
            }
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=320,
            margin=dict(t=60, b=10, l=30, r=30)
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown(f"""
        <div class="glass-card" style="margin-top:-10px;">
            <p style="color:#8b95a5; font-size:0.9rem; margin:0;">
                💡 <b style="color:#e6edf3">Interpretation:</b><br>
                A <b style="color:{COLOR[pred_label]}">{pred_label.lower()}</b> fatigue prediction suggests
                {"the player is well-recovered and ready for normal training/match load." if pred_label=="Low" else
                 "monitoring is advised — consider moderate load management." if pred_label=="Medium" else
                 "elevated fatigue risk — consider reduced load, extra recovery, or rotation."}
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    with col_left:
        st.markdown("""
        <div class="glass-card placeholder-box">
            <div class="big-icon">⚽</div>
            <b style="color:#c9d1d9;">No prediction yet</b><br>
            Fill in player values in the sidebar and click Predict.
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.markdown("""
        <div class="glass-card placeholder-box">
            <div class="big-icon">📊</div>
            Risk gauge and interpretation will appear here.
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p class="footer-note">Model: Random Forest · Trained on SoccerMon dataset (elite women's football, 2020–2021) · For research/demonstration purposes only.</p>
""", unsafe_allow_html=True)