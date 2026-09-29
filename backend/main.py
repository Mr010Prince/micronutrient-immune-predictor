import json
import os

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "model.joblib")
FEATURE_ORDER_PATH = os.path.join(os.path.dirname(__file__), "ml", "feature_order.json")

app = FastAPI(title="Micronutrient Immune Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model_pipeline = None
feature_order = None
shap_explainer = None


@app.on_event("startup")
def load_model():
    global model_pipeline, feature_order, shap_explainer
    model_pipeline = joblib.load(MODEL_PATH)
    with open(FEATURE_ORDER_PATH) as f:
        feature_order = json.load(f)
    shap_explainer = shap.TreeExplainer(model_pipeline.named_steps["model"])


@app.post("/predict")
def predict(row: dict):
    x = pd.DataFrame([row], columns=feature_order)
    x_trans = model_pipeline[:-1].transform(x)
    proba = float(model_pipeline.predict_proba(x_trans)[0, 1])
    pred = int(proba >= 0.5)
    sv = shap_explainer.shap_values(x_trans)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    shap_vals = [round(float(v), 5) for v in sv[0]]
    return {
        "prediction": pred,
        "probability": round(proba, 5),
        "shap_values": dict(zip(feature_order, shap_vals)),
    }
