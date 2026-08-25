"""
Streamlit frontend for Zameen Vivaad — Land Acquisition Delay Predictor.
Loads the ML models directly and provides a simple form UI.
"""

import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zameen Vivaad — Delay Predictor",
    page_icon="🏗️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load models (cached so they load only once)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(BASE_DIR, "risk_classifier.joblib"))
    reg = joblib.load(os.path.join(BASE_DIR, "delay_regressor.joblib"))
    
    # Fix for XGBoost device mismatch warning
    if hasattr(reg, "steps"):
        reg.steps[-1][1].set_params(device="cpu")
        
    le = joblib.load(os.path.join(BASE_DIR, "risk_label_encoder.joblib"))
    return clf, reg, le


risk_classifier, delay_regressor, label_encoder = load_models()

# ---------------------------------------------------------------------------
# Valid options for categorical fields
# ---------------------------------------------------------------------------
STATES = ["Bihar", "Gujarat", "Karnataka", "Madhya Pradesh", "Maharashtra",
          "Odisha", "Rajasthan", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]

STATE_DISTRICTS = {
    "Bihar": ["Gaya", "Muzaffarpur", "Patna"],
    "Gujarat": ["Ahmedabad", "Rajkot", "Surat", "Vadodara"],
    "Karnataka": ["Bengaluru", "Hubballi", "Mysuru"],
    "Madhya Pradesh": ["Bhopal", "Gwalior", "Indore", "Jabalpur"],
    "Maharashtra": ["Aurangabad", "Nagpur", "Nashik", "Pune", "Thane"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Kota", "Udaipur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Uttar Pradesh": ["Ghaziabad", "Kanpur", "Lucknow", "Meerut", "Noida", "Varanasi"],
    "West Bengal": ["Howrah", "Kolkata", "Siliguri"],
}

PROJECT_TYPES = ["Airport Expansion", "Dam/Reservoir", "Industrial Corridor",
                 "Irrigation Canal", "National Highway", "Power Transmission Line",
                 "Railway Line", "SEZ Development", "State Highway", "Urban Metro"]

COMPENSATION_STATUSES = ["Fully Disbursed", "Not Disbursed", "Partially Disbursed"]

APPROVAL_STAGES = ["Award Declared", "Notification (Sec 11)", "Possession Complete",
                   "Possession Initiated", "Rehabilitation Ongoing", "SIA Completed"]

LEGAL_DISPUTE_STATUSES = ["None", "Ongoing - High Court", "Ongoing - Lower Court",
                          "Ongoing - Supreme Court", "Resolved Against", "Resolved in Favor"]

POSSESSION_STATUSES = ["Fully Complete", "Not Started", "Partially Complete"]

COORDINATION_LEVELS = ["Low", "Medium", "High"]

# Column order expected by models
NUMERIC_COLS = [
    "land_area_hectares", "affected_families", "compensation_disbursed_pct",
    "days_since_notification", "legal_disputes_count",
    "rehabilitation_progress_pct", "stakeholder_responsiveness_score",
    "historical_district_delay_rate", "planned_duration_days", "project_age_days",
]
CATEGORICAL_COLS = [
    "state", "district", "project_type", "compensation_status",
    "approval_stage", "legal_dispute_status", "possession_status",
    "inter_department_coordination_issues",
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏗️ Zameen Vivaad AI")
st.caption("AI-Powered Predictive Analytics for Land Acquisition Delays")

st.divider()

# --- Project Info ---
st.subheader("📋 Project Information")
col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State", STATES)
    project_type = st.selectbox("Project Type", PROJECT_TYPES)
    land_area = st.number_input("Land Area (Hectares)", min_value=0.0, value=50.0, step=1.0)
    planned_duration = st.number_input("Planned Duration (Days)", min_value=1, value=730, step=1)
with col2:
    district = st.selectbox("District", STATE_DISTRICTS.get(state, []))
    affected_families = st.number_input("Affected Families", min_value=0, value=100, step=1)
    project_age = st.number_input("Project Age (Days)", min_value=0, value=900, step=1)

st.divider()

# --- Compensation & Rehabilitation ---
st.subheader("💰 Compensation & Rehabilitation")
col1, col2 = st.columns(2)
with col1:
    comp_status = st.selectbox("Compensation Status", COMPENSATION_STATUSES)
    rehab_pct = st.number_input("Rehabilitation Progress (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
with col2:
    comp_pct = st.number_input("Compensation Disbursed (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

st.divider()

# --- Legal & Approval ---
st.subheader("⚖️ Legal & Approval")
col1, col2 = st.columns(2)
with col1:
    approval_stage = st.selectbox("Approval Stage", APPROVAL_STAGES)
    legal_count = st.number_input("Legal Disputes Count", min_value=0, value=2, step=1)
    days_notif = st.number_input("Days Since Notification", min_value=0, value=500, step=1)
with col2:
    legal_status = st.selectbox("Legal Dispute Status", LEGAL_DISPUTE_STATUSES)
    possession = st.selectbox("Possession Status", POSSESSION_STATUSES)

st.divider()

# --- Coordination & History ---
st.subheader("🤝 Coordination & History")
col1, col2 = st.columns(2)
with col1:
    coordination = st.selectbox("Inter-Dept Coordination Issues", COORDINATION_LEVELS)
    stakeholder_score = st.number_input("Stakeholder Responsiveness (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
with col2:
    hist_delay = st.number_input("Historical District Delay Rate (0-1)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

st.divider()

# --- Predict ---
if st.button("🔮 Predict Risk & Delay", type="primary", use_container_width=True):
    # Build input dataframe
    legal_val = legal_status

    data = {
        "land_area_hectares": [land_area],
        "affected_families": [affected_families],
        "compensation_disbursed_pct": [comp_pct],
        "days_since_notification": [days_notif],
        "legal_disputes_count": [legal_count],
        "rehabilitation_progress_pct": [rehab_pct],
        "stakeholder_responsiveness_score": [stakeholder_score],
        "historical_district_delay_rate": [hist_delay],
        "planned_duration_days": [planned_duration],
        "project_age_days": [project_age],
        "state": [state],
        "district": [district],
        "project_type": [project_type],
        "compensation_status": [comp_status],
        "approval_stage": [approval_stage],
        "legal_dispute_status": [legal_val],
        "possession_status": [possession],
        "inter_department_coordination_issues": [coordination],
    }

    df = pd.DataFrame(data)

    try:
        # Classification
        risk_enc = risk_classifier.predict(df)[0]
        risk_proba = risk_classifier.predict_proba(df)[0]
        risk_label = label_encoder.inverse_transform([risk_enc])[0]

        # Regression
        delay_prob = float(delay_regressor.predict(df)[0])
        delay_prob = max(0.0, min(1.0, delay_prob))

        # Display results
        st.divider()
        st.subheader("📊 Prediction Results")

        res1, res2 = st.columns(2)
        with res1:
            color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_label, "⚪")
            st.metric("Risk Category", f"{color} {risk_label}")
        with res2:
            st.metric("Delay Probability", f"{delay_prob * 100:.1f}%")

        st.progress(delay_prob)

        # Probability breakdown
        st.write("**Risk Probability Breakdown:**")
        prob_cols = st.columns(len(label_encoder.classes_))
        for i, cls in enumerate(label_encoder.classes_):
            with prob_cols[i]:
                st.metric(cls, f"{risk_proba[i] * 100:.1f}%")

        # Store in session state for AI recommendations
        st.session_state["last_prediction"] = {
            "risk_label": risk_label,
            "delay_prob": delay_prob,
            "project_data": {
                "state": state, "district": district, "project_type": project_type,
                "land_area_hectares": land_area, "affected_families": int(affected_families),
                "compensation_status": comp_status, "compensation_disbursed_pct": comp_pct,
                "approval_stage": approval_stage, "days_since_notification": int(days_notif),
                "legal_disputes_count": int(legal_count),
                "legal_dispute_status": legal_val if legal_val != "None" else None,
                "possession_status": possession, "rehabilitation_progress_pct": rehab_pct,
                "stakeholder_responsiveness_score": stakeholder_score,
                "historical_district_delay_rate": hist_delay,
                "inter_department_coordination_issues": coordination,
                "planned_duration_days": int(planned_duration), "project_age_days": int(project_age),
            },
        }

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# AI Recommendations (shown after prediction is stored in session state)
# ---------------------------------------------------------------------------
if "last_prediction" in st.session_state:
    pred = st.session_state["last_prediction"]
    st.divider()
    st.subheader("🤖 AI Recommendations")
    st.caption(f"Tailored strategies to reduce **{pred['risk_label']} Risk** for this project")

    if st.button("✨ Generate AI Recommendations", use_container_width=True):
        with st.spinner("Consulting AI expert..."):
            try:
                payload = {
                    "project": pred["project_data"],
                    "risk_category": pred["risk_label"],
                    "delay_probability": pred["delay_prob"],
                }
                resp = requests.post(
                    "http://localhost:8000/api/recommend",
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    recs = resp.json()["recommendations"]
                    color_map = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#21c354"}
                    border_color = color_map.get(pred["risk_label"], "#4f8bf9")
                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid {border_color}; padding: 1rem 1.2rem; border-radius: 8px; background: #1e1e2e; margin-top: 0.5rem;">
                        """ + "".join([
                            f'<p style="margin: 0.5rem 0; color: #e0e0e0;">'
                            f'<span style="color: {border_color}; font-weight: bold;">{i+1}.</span> {rec}</p>'
                            for i, rec in enumerate(recs)
                        ]) + "</div>",
                        unsafe_allow_html=True,
                    )
                elif resp.status_code == 503:
                    st.warning("⚠️ Groq API key not configured. Add your key to the `.env` file.")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the FastAPI backend. Make sure it is running on port 8000.")
            except Exception as e:
                st.error(f"Recommendation failed: {e}")
