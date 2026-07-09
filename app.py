"""
Streamlit 인터랙티브 데모: 신용카드 대손 위험 예측기
채용담당자가 코드를 몰라도 브라우저에서 직접 값을 조정하며 체험할 수 있습니다.
"""
import streamlit as st
import torch
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from credit_risk_v2 import CreditRiskNet

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="신용카드 대손 위험 예측기", layout="wide")
st.title("신용카드 대손(Default) 위험 예측 데모")
st.caption("PyTorch MLP + SHAP 기반 설명가능성 | 카드사 리스크관리 실무 시나리오")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    threshold = joblib.load("threshold.pkl")
    model = CreditRiskNet(len(feature_cols))
    model.load_state_dict(torch.load("credit_risk_model.pt"))
    model.eval()
    return model, scaler, feature_cols, threshold

model, scaler, feature_cols, threshold = load_artifacts()

st.sidebar.header("고객 프로필 입력")
input_values = {}
defaults = {"LIMIT_BAL": 50000, "AGE": 35, "PAY_0": 0, "BILL_AMT1": 30000, "PAY_AMT1": 3000}
for col in feature_cols:
    default_val = defaults.get(col, 0)
    input_values[col] = st.sidebar.number_input(col, value=float(default_val))

if st.sidebar.button("위험도 예측", type="primary"):
    x = np.array([[input_values[c] for c in feature_cols]], dtype=np.float32)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x_scaled, dtype=torch.float32))).item()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("예측 대손 확률", f"{prob:.1%}")
        if prob > threshold:
            st.error(f"고위험 고객 (임계값 {threshold:.1%} 초과)")
        else:
            st.success(f"정상 위험 수준 (임계값 {threshold:.1%} 이하)")

    with col2:
        st.write("**입력된 프로필**")
        st.json(input_values)

    st.subheader("이 예측의 근거 (SHAP)")
    def model_predict(x_numpy):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(x_numpy, dtype=torch.float32))).numpy().ravel()

    background = np.tile(x_scaled, (20, 1)) + np.random.normal(0, 0.1, (20, x_scaled.shape[1]))
    explainer = shap.KernelExplainer(model_predict, background)
    shap_val = explainer.shap_values(x_scaled, nsamples=50)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap_val_flat = np.array(shap_val).ravel()
    colors = ["crimson" if v > 0 else "steelblue" for v in shap_val_flat]
    ax.barh(feature_cols, shap_val_flat, color=colors)
    ax.set_title("각 피처가 대손 확률을 얼마나 높이거나(빨강) 낮췄는지(파랑)")
    ax.axvline(0, color="black", linewidth=0.8)
    st.pyplot(fig)

st.divider()
st.caption("이 데모는 실제 서비스가 아닌 포트폴리오 목적의 프로토타입입니다.")
