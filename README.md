# 🏗️ Zameen Vivaad — Land Acquisition Delay Predictor

AI-powered Predictive Analytics System for Land Acquisition Delays (SIH Project).

Predicts **risk category** (Low / Medium / High) and **delay probability** for land acquisition projects using trained ML models (Random Forest classifier + XGBoost regressor).

---

## 🚀 Deployment Options

This project includes two components:
1. **Streamlit Frontend (`streamlit_app.py`)** — An interactive UI for making predictions.
2. **FastAPI Backend (`app.py`)** — A REST API with auto-generated Swagger documentation.

### Option A: Deploy to Streamlit Community Cloud (Frontend Only)
If you just need the web UI, you can deploy the Streamlit app for free on Streamlit Community Cloud. *Note: This will not deploy the FastAPI backend or Swagger UI.*

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Click **New app**.
4. Select this repository and set the **Main file path** to `streamlit_app.py`.
5. Click **Deploy!**

### Option B: Deploy to Render (Frontend + Backend API)
If you want **both** the interactive UI and the REST API (with Swagger Docs), you can deploy the entire project to Render.

1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com).
3. Click **New → Blueprint** and connect this repository.
4. Render will read `render.yaml` and automatically deploy both the UI and the API.

---

## 📡 API Endpoints (FastAPI Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI — interactive API documentation |
| `GET` | `/redoc` | ReDoc — API documentation |
| `GET` | `/api/features` | List of input features with valid options |
| `POST` | `/api/predict` | Submit project data, get risk + delay prediction |

---

## 🖥️ Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit frontend
streamlit run streamlit_app.py

# 3. Run the FastAPI backend (Swagger UI at http://localhost:8000/docs)
uvicorn app:app --reload
```

---

## 📁 Project Structure

```
├── app.py                      # FastAPI backend (API + Swagger docs)
├── streamlit_app.py            # Streamlit frontend (input form UI)
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── data.csv                    # Training dataset
├── risk_classifier.joblib      # Trained risk classifier model
├── delay_regressor.joblib      # Trained delay regressor model
├── risk_label_encoder.joblib   # Label encoder for risk categories
└── land_acquisition_prototype.ipynb  # ML training notebook
```