"""
LandGuard — FastAPI backend for Land Acquisition Delay Prediction.

Endpoints:
  GET  /docs           → Swagger UI (interactive API docs)
  GET  /redoc          → ReDoc (API docs)
  GET  /health         → Health check
  POST /api/predict    → Predict risk & delay
  GET  /api/features   → List input features & valid options
  POST /api/recommend  → Get AI-generated recommendations to reduce land dispute
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from groq import Groq   

load_dotenv()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LandGuard — Land Acquisition Delay Predictor",
    description=(
        "AI-powered predictive analytics system for land acquisition delays. "
        "Submit project parameters and receive a risk category (Low / Medium / High) "
        "along with a delay probability score."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    risk_classifier = joblib.load(os.path.join(BASE_DIR, "risk_classifier.joblib"))
    delay_regressor = joblib.load(os.path.join(BASE_DIR, "delay_regressor.joblib"))
    
    # Fix for XGBoost device mismatch warning
    if hasattr(delay_regressor, "steps"):
        delay_regressor.steps[-1][1].set_params(device="cpu")
        
    label_encoder = joblib.load(os.path.join(BASE_DIR, "risk_label_encoder.joblib"))
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    risk_classifier = delay_regressor = label_encoder = None
    MODELS_LOADED = False

# ---------------------------------------------------------------------------
# Valid categorical values
# ---------------------------------------------------------------------------
VALID_VALUES = {
    "state": ["Bihar", "Gujarat", "Karnataka", "Madhya Pradesh", "Maharashtra",
              "Odisha", "Rajasthan", "Tamil Nadu", "Uttar Pradesh", "West Bengal"],
    "district": ["Ahmedabad", "Aurangabad", "Bengaluru", "Bhopal", "Bhubaneswar",
                 "Chennai", "Coimbatore", "Cuttack", "Gaya", "Ghaziabad", "Gwalior",
                 "Howrah", "Hubballi", "Indore", "Jabalpur", "Jaipur", "Jodhpur",
                 "Kanpur", "Kolkata", "Kota", "Lucknow", "Madurai", "Meerut",
                 "Muzaffarpur", "Mysuru", "Nagpur", "Nashik", "Noida", "Patna",
                 "Pune", "Rajkot", "Rourkela", "Siliguri", "Surat", "Thane",
                 "Udaipur", "Vadodara", "Varanasi"],
    "project_type": ["Airport Expansion", "Dam/Reservoir", "Industrial Corridor",
                     "Irrigation Canal", "National Highway", "Power Transmission Line",
                     "Railway Line", "SEZ Development", "State Highway", "Urban Metro"],
    "compensation_status": ["Fully Disbursed", "Not Disbursed", "Partially Disbursed"],
    "approval_stage": ["Award Declared", "Notification (Sec 11)", "Possession Complete",
                       "Possession Initiated", "Rehabilitation Ongoing", "SIA Completed"],
    "legal_dispute_status": ["Ongoing - High Court", "Ongoing - Lower Court",
                             "Ongoing - Supreme Court", "Resolved Against", "Resolved in Favor"],
    "possession_status": ["Fully Complete", "Not Started", "Partially Complete"],
    "inter_department_coordination_issues": ["High", "Low", "Medium"],
}

CATEGORICAL_COLS = list(VALID_VALUES.keys())
NUMERIC_COLS = [
    "land_area_hectares", "affected_families", "compensation_disbursed_pct",
    "days_since_notification", "legal_disputes_count",
    "rehabilitation_progress_pct", "stakeholder_responsiveness_score",
    "historical_district_delay_rate", "planned_duration_days", "project_age_days",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ProjectInput(BaseModel):
    """Input data for a single land-acquisition project."""
    state: str = Field(..., description="State where the project is located")
    district: str = Field(..., description="District of the project")
    project_type: str = Field(..., description="Type of infrastructure project")
    land_area_hectares: float = Field(..., ge=0)
    affected_families: int = Field(..., ge=0)
    compensation_status: str
    compensation_disbursed_pct: float = Field(..., ge=0, le=100)
    approval_stage: str
    days_since_notification: int = Field(..., ge=0)
    legal_disputes_count: int = Field(..., ge=0)
    legal_dispute_status: Optional[str] = None
    possession_status: str
    rehabilitation_progress_pct: float = Field(..., ge=0, le=100)
    stakeholder_responsiveness_score: float = Field(..., ge=0, le=10)
    historical_district_delay_rate: float = Field(..., ge=0, le=1)
    inter_department_coordination_issues: str
    planned_duration_days: int = Field(..., ge=1)
    project_age_days: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "state": "Maharashtra", "district": "Pune",
                "project_type": "National Highway", "land_area_hectares": 50.0,
                "affected_families": 100, "compensation_status": "Partially Disbursed",
                "compensation_disbursed_pct": 30.0, "approval_stage": "SIA Completed",
                "days_since_notification": 500, "legal_disputes_count": 2,
                "legal_dispute_status": "Ongoing - High Court",
                "possession_status": "Partially Complete",
                "rehabilitation_progress_pct": 40.0,
                "stakeholder_responsiveness_score": 7.5,
                "historical_district_delay_rate": 0.35,
                "inter_department_coordination_issues": "Medium",
                "planned_duration_days": 730, "project_age_days": 900,
            }]
        }
    }


class PredictionResponse(BaseModel):
    risk_category: str
    delay_probability: float
    risk_probabilities: dict


class RecommendRequest(BaseModel):
    """Project input data + predicted risk label for generating AI recommendations."""
    project: ProjectInput
    risk_category: str = Field(..., description="Predicted risk category: Low, Medium, or High")
    delay_probability: float = Field(..., description="Predicted delay probability (0-1)")


class RecommendResponse(BaseModel):
    risk_category: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def build_recommendation_prompt(project: ProjectInput, risk_category: str, delay_probability: float) -> str:
    """Build a structured prompt for the LLM."""
    return f"""You are an expert consultant in land acquisition, infrastructure project management, and conflict resolution in India.

A land acquisition project has been assessed by an AI system and classified as **{risk_category} Risk** with a delay probability of **{delay_probability * 100:.1f}%**.

Here are the project details:
- State: {project.state}, District: {project.district}
- Project Type: {project.project_type}
- Land Area: {project.land_area_hectares} hectares, Affected Families: {project.affected_families}
- Compensation Status: {project.compensation_status} ({project.compensation_disbursed_pct:.1f}% disbursed)
- Approval Stage: {project.approval_stage}
- Days Since Notification: {project.days_since_notification}
- Legal Disputes: {project.legal_disputes_count} ({project.legal_dispute_status or 'None'})
- Possession Status: {project.possession_status}
- Rehabilitation Progress: {project.rehabilitation_progress_pct:.1f}%
- Stakeholder Responsiveness Score: {project.stakeholder_responsiveness_score}/10
- Historical District Delay Rate: {project.historical_district_delay_rate * 100:.1f}%
- Inter-Department Coordination Issues: {project.inter_department_coordination_issues}
- Planned Duration: {project.planned_duration_days} days, Project Age: {project.project_age_days} days

Based on the **{risk_category} Risk** classification and the specific project parameters above, provide exactly 5 concise, actionable recommendations to reduce the land acquisition dispute and delay risk. 

Format your response as a numbered list (1. 2. 3. 4. 5.). Each recommendation should be specific to this project's data, not generic advice. Keep each point to 1-2 sentences."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": MODELS_LOADED, "ai_recommendations": groq_client is not None}


@app.get("/api/features")
async def get_features():
    """Returns input feature metadata for building clients/forms."""
    features = []
    for col in NUMERIC_COLS:
        features.append({"name": col, "type": "numeric"})
    for col in CATEGORICAL_COLS:
        features.append({"name": col, "type": "categorical", "options": VALID_VALUES[col]})
    return {"features": features}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(project: ProjectInput):
    """Predict risk category and delay probability for a project."""
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded.")

    data = {col: [getattr(project, col)] for col in NUMERIC_COLS + CATEGORICAL_COLS}
    if data["legal_dispute_status"][0] is None:
        data["legal_dispute_status"] = ["None"]

    df = pd.DataFrame(data)

    try:
        risk_enc = risk_classifier.predict(df)[0]
        risk_proba = risk_classifier.predict_proba(df)[0]
        risk_label = label_encoder.inverse_transform([risk_enc])[0]
        probs = {c: round(float(p), 4) for c, p in zip(label_encoder.classes_, risk_proba)}

        delay = float(delay_regressor.predict(df)[0])
        delay = max(0.0, min(1.0, delay))

        return PredictionResponse(
            risk_category=risk_label,
            delay_probability=round(delay, 4),
            risk_probabilities=probs,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Generate AI-powered recommendations to reduce land dispute risk."""
    if not groq_client:
        raise HTTPException(503, "Groq API key not configured. Set GROQ_API_KEY in your .env file.")

    prompt = build_recommendation_prompt(req.project, req.risk_category, req.delay_probability)

    try:
        chat = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are a land acquisition and conflict resolution expert for Indian infrastructure projects."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        raw = chat.choices[0].message.content.strip()

        # Parse numbered list into individual recommendations
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        recommendations = [
            line.lstrip("0123456789.-) ").strip()
            for line in lines
            if line and line[0].isdigit()
        ]
        if not recommendations:
            recommendations = [raw]  # fallback: return raw text as single item

        return RecommendResponse(
            risk_category=req.risk_category,
            recommendations=recommendations,
        )
    except Exception as e:
        raise HTTPException(500, f"AI recommendation failed: {e}")
