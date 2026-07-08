"""
credit risk 모델을 실시간 API로 서빙합니다.
실무에서는 학습된 모델이 이렇게 API 형태로 배포되어 다른 서비스와 연동됩니다.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import joblib
from credit_risk_v2 import CreditRiskNet

app = FastAPI(title="Credit Default Risk API", version="1.0")

scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
threshold = joblib.load("threshold.pkl")

model = CreditRiskNet(len(feature_cols))
model.load_state_dict(torch.load("credit_risk_model.pt"))
model.eval()

class CustomerFeatures(BaseModel):
    features: dict = Field(..., description=f"필요한 키: {feature_cols}")

class RiskResponse(BaseModel):
    default_probability: float
    is_high_risk: bool
    threshold_used: float

@app.get("/")
def root():
    return {"status": "ok", "model": "credit-default-risk-mlp", "features_required": feature_cols}

@app.post("/predict", response_model=RiskResponse)
def predict(payload: CustomerFeatures):
    missing = [c for c in feature_cols if c not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"누락된 피처: {missing}")

    x = np.array([[payload.features[c] for c in feature_cols]], dtype=np.float32)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x_scaled, dtype=torch.float32))).item()

    return RiskResponse(
        default_probability=round(prob, 4),
        is_high_risk=prob > threshold,
        threshold_used=round(float(threshold), 4)
    )
