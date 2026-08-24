"""
Zameen Vivaad — FastAPI backend for Land Acquisition Delay Prediction.

Endpoints:
  GET  /docs        → Swagger UI (interactive API docs)
  GET  /redoc       → ReDoc (API docs)
  GET  /health      → Health check
  POST /api/predict  → Predict risk & delay
  GET  /api/features → List input features & valid options
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zameen Vivaad — Land Acquisition Delay Predictor",
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": MODELS_LOADED}


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
