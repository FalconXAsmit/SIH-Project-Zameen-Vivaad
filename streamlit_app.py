"""
Streamlit frontend for LandGuard — Land Acquisition Delay Predictor.
Loads the ML models directly and provides a polished light-mode form UI.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LandGuard — Delay Predictor",
    page_icon="🏗️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — bright light-mode polish
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Root variables ---------- */
:root {
    --primary: #2563EB;
    --primary-light: #3B82F6;
    --primary-bg: #EFF6FF;
    --accent-green: #10B981;
    --accent-amber: #F59E0B;
    --accent-red: #EF4444;
    --text-dark: #1E293B;
    --text-mid: #475569;
    --text-light: #94A3B8;
    --bg-white: #FFFFFF;
    --bg-off: #F8FAFC;
    --border: #E2E8F0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 24px rgba(0,0,0,0.08), 0 4px 8px rgba(0,0,0,0.04);
    --radius: 12px;
}

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background: linear-gradient(180deg, #F0F4FF 0%, #FFFFFF 35%, #F8FAFC 100%) !important;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }



/* ---------- Section cards ---------- */
.section-card {
    background: var(--bg-white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.5rem 1rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
.section-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-dark);
    margin: 0 0 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}
.section-subtitle {
    font-size: 0.82rem;
    color: var(--text-light);
    margin: 0 0 1rem;
    font-weight: 400;
}

/* ---------- Result cards ---------- */
.result-card {
    background: linear-gradient(135deg, #F0F4FF 0%, #FFFFFF 100%);
    border: 1px solid #DBEAFE;
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-md);
}

/* ---------- Risk badges ---------- */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.01em;
}
.risk-low {
    background: #D1FAE5;
    color: #065F46;
    border: 1px solid #6EE7B7;
}
.risk-medium {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}
.risk-high {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}

/* ---------- Delay gauge ---------- */
.delay-gauge {
    text-align: center;
    padding: 1rem 0;
}
.delay-value {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.delay-label {
    font-size: 0.85rem;
    color: var(--text-light);
    font-weight: 500;
    margin-top: 0.1rem;
}

/* ---------- Progress bar override ---------- */
.stProgress > div > div > div > div {
    border-radius: 8px;
    background: linear-gradient(90deg, var(--accent-green), var(--primary-light), var(--accent-red)) !important;
}

/* ---------- Probability chips ---------- */
.prob-chip {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: var(--bg-white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    min-width: 80px;
    box-shadow: var(--shadow-sm);
}
.prob-chip .prob-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.prob-chip .prob-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-dark);
}

/* ---------- AI Recommendation card ---------- */
.rec-card {
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-top: 0.6rem;
}
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin: 0.65rem 0;
    padding: 0.6rem 0.8rem;
    background: var(--bg-off);
    border-radius: 8px;
    border-left: 3px solid var(--primary);
    font-size: 0.92rem;
    color: var(--text-dark);
    line-height: 1.5;
    transition: background 0.2s;
}
.rec-item:hover {
    background: var(--primary-bg);
}
.rec-num {
    background: var(--primary);
    color: white;
    font-weight: 700;
    font-size: 0.72rem;
    min-width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 2px;
}

/* ---------- Button overrides ---------- */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
}

/* Secondary buttons */
.stButton > button:not([kind="primary"]) {
    background: linear-gradient(135deg, var(--primary-bg) 0%, #E0E7FF 100%) !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 10px !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: linear-gradient(135deg, #DBEAFE 0%, #C7D2FE 100%) !important;
    transform: translateY(-1px) !important;
}

/* ---------- Input overrides ---------- */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary-light) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ---------- Metric overrides ---------- */
[data-testid="stMetric"] {
    background: var(--bg-white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    color: var(--text-mid) !important;
}

/* ---------- Divider ---------- */
hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}

/* ---------- Hide Streamlit header/footer ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
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
# Groq AI client (for recommendations)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def build_recommendation_prompt(project_data: dict, risk_category: str, delay_probability: float) -> str:
    """Build a structured prompt for the LLM."""
    return f"""You are an expert consultant in land acquisition, infrastructure project management, and conflict resolution in India.

A land acquisition project has been assessed by an AI system and classified as **{risk_category} Risk** with a delay probability of **{delay_probability * 100:.1f}%**.

Here are the project details:
- State: {project_data['state']}, District: {project_data['district']}
- Project Type: {project_data['project_type']}
- Land Area: {project_data['land_area_hectares']} hectares, Affected Families: {project_data['affected_families']}
- Compensation Status: {project_data['compensation_status']} ({project_data['compensation_disbursed_pct']:.1f}% disbursed)
- Approval Stage: {project_data['approval_stage']}
- Days Since Notification: {project_data['days_since_notification']}
- Legal Disputes: {project_data['legal_disputes_count']} ({project_data['legal_dispute_status'] or 'None'})
- Possession Status: {project_data['possession_status']}
- Rehabilitation Progress: {project_data['rehabilitation_progress_pct']:.1f}%
- Stakeholder Responsiveness Score: {project_data['stakeholder_responsiveness_score']}/10
- Historical District Delay Rate: {project_data['historical_district_delay_rate'] * 100:.1f}%
- Inter-Department Coordination Issues: {project_data['inter_department_coordination_issues']}
- Planned Duration: {project_data['planned_duration_days']} days, Project Age: {project_data['project_age_days']} days

Based on the **{risk_category} Risk** classification and the specific project parameters above, provide exactly 5 concise, actionable recommendations to reduce the land acquisition dispute and delay risk.

Format your response as a numbered list (1. 2. 3. 4. 5.). Each recommendation should be specific to this project's data, not generic advice. Keep each point to 1-2 sentences."""

# ---------------------------------------------------------------------------
# Valid options for categorical fields
# ---------------------------------------------------------------------------
STATES = [
    "Bihar", "Gujarat", "Karnataka", "Madhya Pradesh", "Maharashtra",
    "Odisha", "Rajasthan", "Tamil Nadu", "Uttar Pradesh", "West Bengal",
]

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

PROJECT_TYPES = [
    "Airport Expansion", "Dam/Reservoir", "Industrial Corridor",
    "Irrigation Canal", "National Highway", "Power Transmission Line",
    "Railway Line", "SEZ Development", "State Highway", "Urban Metro",
]

COMPENSATION_STATUSES = ["Fully Disbursed", "Not Disbursed", "Partially Disbursed"]

APPROVAL_STAGES = [
    "Award Declared", "Notification (Sec 11)", "Possession Complete",
    "Possession Initiated", "Rehabilitation Ongoing", "SIA Completed",
]

LEGAL_DISPUTE_STATUSES = [
    "None", "Ongoing - High Court", "Ongoing - Lower Court",
    "Ongoing - Supreme Court", "Resolved Against", "Resolved in Favor",
]

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
# Hero header
# ---------------------------------------------------------------------------
st.title("🏗️ LandGuard AI")
st.caption("AI-Powered Predictive Analytics for Land Acquisition Delays")

# ---------------------------------------------------------------------------
# Section 1 — Project Information
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card"><div class="section-title">📋 Project Information</div><div class="section-subtitle">Basic details about the land acquisition project</div></div>', unsafe_allow_html=True)

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

# ---------------------------------------------------------------------------
# Section 2 — Compensation & Rehabilitation
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card"><div class="section-title">💰 Compensation & Rehabilitation</div><div class="section-subtitle">Financial disbursement and rehabilitation progress</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    comp_status = st.selectbox("Compensation Status", COMPENSATION_STATUSES)
    rehab_pct = st.number_input("Rehabilitation Progress (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
with col2:
    comp_pct = st.number_input("Compensation Disbursed (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

# ---------------------------------------------------------------------------
# Section 3 — Legal & Approval
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card"><div class="section-title">⚖️ Legal & Approval</div><div class="section-subtitle">Legal disputes and approval pipeline status</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    approval_stage = st.selectbox("Approval Stage", APPROVAL_STAGES)
    legal_count = st.number_input("Legal Disputes Count", min_value=0, value=2, step=1)
    days_notif = st.number_input("Days Since Notification", min_value=0, value=500, step=1)
with col2:
    legal_status = st.selectbox("Legal Dispute Status", LEGAL_DISPUTE_STATUSES)
    possession = st.selectbox("Possession Status", POSSESSION_STATUSES)

# ---------------------------------------------------------------------------
# Section 4 — Coordination & History
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card"><div class="section-title">🤝 Coordination & History</div><div class="section-subtitle">Inter-department coordination and historical metrics</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    coordination = st.selectbox("Inter-Dept Coordination Issues", COORDINATION_LEVELS)
    stakeholder_score = st.number_input(
        "Stakeholder Responsiveness (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1
    )
with col2:
    hist_delay = st.number_input(
        "Historical District Delay Rate (0-1)", min_value=0.0, max_value=1.0, value=0.35, step=0.01
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("🔮  Predict Risk & Delay", type="primary", use_container_width=True):
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

        # ----- Display results -----
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="border-left: 4px solid #2563EB;"><div class="section-title">📊 Prediction Results</div><div class="section-subtitle">Model output based on your inputs</div></div>', unsafe_allow_html=True)

        # Risk + Delay side-by-side
        res1, res2 = st.columns(2)

        risk_css = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(risk_label, "risk-medium")
        risk_icon = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(risk_label, "⚪")

        with res1:
            st.markdown("##### Risk Category")
            st.markdown(
                f'<span class="risk-badge {risk_css}">{risk_icon} {risk_label} Risk</span>',
                unsafe_allow_html=True,
            )

        # Color for delay value
        if delay_prob < 0.35:
            delay_color = "#10B981"
        elif delay_prob < 0.65:
            delay_color = "#F59E0B"
        else:
            delay_color = "#EF4444"

        with res2:
            st.markdown(f'<div class="delay-gauge"><div class="delay-value" style="color: {delay_color};">{delay_prob * 100:.1f}%</div><div class="delay-label">Delay Probability</div></div>', unsafe_allow_html=True)

        st.progress(delay_prob)

        # Probability breakdown
        st.markdown("**Risk Probability Breakdown**")
        chip_html = '<div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-top:0.3rem;">'
        for i, cls in enumerate(label_encoder.classes_):
            chip_html += (
                f'<div class="prob-chip">'
                f'<span class="prob-label">{cls}</span>'
                f'<span class="prob-value">{risk_proba[i] * 100:.1f}%</span>'
                f'</div>'
            )
        chip_html += "</div>"
        st.markdown(chip_html, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Charts section
        # ------------------------------------------------------------------
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-card" style="border-left: 4px solid #7C3AED;"><div class="section-title">📈 Visual Analysis</div><div class="section-subtitle">Charts showing input factors and risk breakdown</div></div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        # --- Donut chart: Risk probability breakdown ---
        with chart_col1:
            risk_colors = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(label_encoder.classes_),
                values=[round(p * 100, 1) for p in risk_proba],
                hole=0.55,
                marker=dict(colors=[risk_colors.get(c, "#94A3B8") for c in label_encoder.classes_]),
                textinfo="label+percent",
                textfont=dict(size=13, family="Inter"),
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            )])
            fig_donut.update_layout(
                title=dict(text="Risk Probability", font=dict(size=15, family="Inter", color="#1E293B"), x=0.5),
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{risk_label}</b>",
                    x=0.5, y=0.5, font=dict(size=18, color=risk_colors.get(risk_label, "#475569"), family="Inter"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True, key="donut_risk")

        # --- Radar chart: Key project factors (normalized 0-100) ---
        with chart_col2:
            radar_labels = [
                "Compensation\nDisbursed",
                "Rehabilitation\nProgress",
                "Stakeholder\nResponsiveness",
                "Historical\nDelay Rate",
                "Legal\nDisputes",
            ]
            radar_values = [
                comp_pct,                          # already 0-100
                rehab_pct,                         # already 0-100
                stakeholder_score * 10,             # 0-10 → 0-100
                hist_delay * 100,                   # 0-1 → 0-100
                min(legal_count * 20, 100),          # rough scale: 5 disputes = 100
            ]
            # Close the polygon
            radar_values_closed = radar_values + [radar_values[0]]
            radar_labels_closed = radar_labels + [radar_labels[0]]

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_values_closed,
                theta=radar_labels_closed,
                fill="toself",
                fillcolor="rgba(37,99,235,0.15)",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=6, color="#2563EB"),
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            ))
            fig_radar.update_layout(
                title=dict(text="Key Factor Profile", font=dict(size=15, family="Inter", color="#1E293B"), x=0.5),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#E2E8F0"),
                    angularaxis=dict(tickfont=dict(size=10, family="Inter", color="#475569"), gridcolor="#E2E8F0"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(t=40, b=30, l=40, r=40),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_factors")

        # --- Horizontal bar chart: All numeric inputs ---
        bar_labels = [
            "Land Area (ha)", "Affected Families", "Compensation %",
            "Days Since Notif.", "Legal Disputes", "Rehab Progress %",
            "Stakeholder Score", "Hist. Delay Rate", "Planned Duration (d)", "Project Age (d)",
        ]
        bar_values = [
            land_area, affected_families, comp_pct,
            days_notif, legal_count, rehab_pct,
            stakeholder_score, hist_delay * 100, planned_duration, project_age,
        ]
        # Normalize to 0-100 for visual comparison
        max_vals = [500, 1000, 100, 2000, 20, 100, 10, 100, 2000, 3000]
        bar_normalized = [min(v / m * 100, 100) for v, m in zip(bar_values, max_vals)]

        bar_colors = ["#3B82F6" if n < 50 else "#F59E0B" if n < 75 else "#EF4444" for n in bar_normalized]

        fig_bar = go.Figure(data=go.Bar(
            y=bar_labels,
            x=bar_normalized,
            orientation="h",
            marker=dict(color=bar_colors, cornerradius=4),
            text=[f"{v:g}" for v in bar_values],
            textposition="auto",
            textfont=dict(size=12, family="Inter", color="#1E293B"),
            hovertemplate="%{y}: %{text}<extra></extra>",
        ))
        fig_bar.update_layout(
            title=dict(text="Input Parameters Overview", font=dict(size=15, family="Inter", color="#1E293B"), x=0.5),
            xaxis=dict(title="Relative Scale (%)", range=[0, 105], gridcolor="#E2E8F0", tickfont=dict(family="Inter", color="#94A3B8")),
            yaxis=dict(tickfont=dict(size=12, family="Inter", color="#475569"), autorange="reversed"),
            margin=dict(t=40, b=40, l=10, r=20),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_inputs")

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
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown(f'<div class="section-card" style="border-left: 4px solid #7C3AED;"><div class="section-title">🤖 AI Recommendations</div><div class="section-subtitle">Tailored strategies to reduce <strong>{pred["risk_label"]} Risk</strong> for this project</div></div>', unsafe_allow_html=True)

    if st.button("✨  Generate AI Recommendations", use_container_width=True):
        if not groq_client:
            st.warning("⚠️ Groq API key not configured. Add `GROQ_API_KEY` to your `.env` file or Streamlit secrets.")
        else:
            with st.spinner("Consulting AI expert…"):
                try:
                    prompt = build_recommendation_prompt(
                        pred["project_data"], pred["risk_label"], pred["delay_prob"]
                    )
                    chat = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a land acquisition and conflict resolution expert for Indian infrastructure projects."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=600,
                    )
                    raw = chat.choices[0].message.content.strip()

                    # Parse numbered list
                    lines = [line.strip() for line in raw.split("\n") if line.strip()]
                    recs = [
                        line.lstrip("0123456789.-) ").strip()
                        for line in lines
                        if line and line[0].isdigit()
                    ]
                    if not recs:
                        recs = [raw]

                    color_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
                    border_color = color_map.get(pred["risk_label"], "#2563EB")

                    rec_html = '<div class="rec-card" style="background:var(--bg-off); border:1px solid var(--border);">'
                    for i, rec in enumerate(recs):
                        rec_html += (
                            f'<div class="rec-item" style="border-left-color:{border_color};">'
                            f'<span class="rec-num" style="background:{border_color};">{i + 1}</span>'
                            f'<span>{rec}</span>'
                            f'</div>'
                        )
                    rec_html += "</div>"
                    st.markdown(rec_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Recommendation failed: {e}")
