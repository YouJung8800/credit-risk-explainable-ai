"""
실무형 임계값 최적화: F1 점수가 아니라 실제 '비용'을 기준으로 결정 임계값을 찾습니다.
카드사/핀테크 리스크 실무의 핵심 원칙: 대손을 놓치는 비용(False Negative)이
정상 고객을 잘못 거절하는 비용(False Positive)보다 훨씬 큽니다.
"""
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from credit_risk_v2 import CreditRiskNet  # 기존 모델 클래스 재사용
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- 저장된 아티팩트 로드 ----------
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

model = CreditRiskNet(len(feature_cols))
model.load_state_dict(torch.load("credit_risk_model.pt"))
model.eval()
print("저장된 모델 로드 완료")

# ---------- 테스트 데이터 재구성 (동일 시드로 재현) ----------
try:
    df = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        header=1)
    df = df.rename(columns={"default payment next month": "default"})
except Exception:
    n = 6000
    np.random.seed(42)
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.lognormal(11, 0.8, n),
        "AGE": np.random.randint(21, 65, n),
        "PAY_0": np.random.choice([-1, 0, 1, 2, 3], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        "BILL_AMT1": np.random.lognormal(9, 1.2, n),
        "PAY_AMT1": np.random.lognormal(7, 1.5, n),
    })
    risk_score = (df["PAY_0"] * 0.5 + (df["BILL_AMT1"] / df["LIMIT_BAL"]) * 2
                  - (df["PAY_AMT1"] / (df["BILL_AMT1"] + 1)) * 1.5)
    prob = 1 / (1 + np.exp(-risk_score + 1))
    df["default"] = (np.random.rand(n) < prob).astype(int)

X = df[feature_cols].fillna(0).values
y = df["default"].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_test_s = scaler.transform(X_test)

with torch.no_grad():
    prob = torch.sigmoid(model(torch.tensor(X_test_s, dtype=torch.float32))).numpy().ravel()

# ---------- 비용 함수 정의 ----------
# 실무 가정: 대손 고객을 놓치는 비용(FN)이, 정상 고객을 잘못 거절하는 비용(FP)의 5배
# (이 비율은 실제로는 여신담당 부서의 데이터로 산정해야 하며, 여기선 업계 통념 반영)
COST_FN = 5.0  # 대손을 놓쳤을 때(실제 위험한데 승인) 손실
COST_FP = 1.0  # 정상 고객을 잘못 거절했을 때(기회비용, 고객 이탈)

thresholds = np.linspace(0.05, 0.95, 100)
total_costs = []
for th in thresholds:
    pred = (prob > th).astype(int)
    fn = ((pred == 0) & (y_test == 1)).sum()
    fp = ((pred == 1) & (y_test == 0)).sum()
    cost = fn * COST_FN + fp * COST_FP
    total_costs.append(cost)

best_idx = np.argmin(total_costs)
best_cost_threshold = thresholds[best_idx]

print(f"\n=== 비용 기반 임계값 최적화 (FN 비용={COST_FN} : FP 비용={COST_FP}) ===")
print(f"최적 임계값: {best_cost_threshold:.3f}")
print(f"이 임계값에서의 총 비용: {total_costs[best_idx]:.1f}")
print(f"기본 임계값(0.5) 대비 비용 절감률: "
      f"{(1 - total_costs[best_idx] / total_costs[np.argmin(np.abs(thresholds - 0.5))]) * 100:.1f}%")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, total_costs)
plt.axvline(best_cost_threshold, color="red", linestyle="--", label=f"최적 임계값 ({best_cost_threshold:.2f})")
plt.axvline(0.5, color="gray", linestyle=":", label="기본 임계값 (0.5)")
plt.xlabel("판단 임계값")
plt.ylabel("총 비용 (FN×5 + FP×1)")
plt.title("비용 기반 임계값 최적화")
plt.legend()
plt.tight_layout()
plt.savefig("cost_threshold_optimization.png", dpi=150)
print("\n시각화 저장: cost_threshold_optimization.png")
plt.show()

joblib.dump(float(best_cost_threshold), "threshold_cost_based.pkl")
