import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

torch.manual_seed(42)
np.random.seed(42)

def load_real_or_synthetic():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        df = pd.read_excel(url, header=1)
        df = df.rename(columns={"default payment next month": "default"})
        print(f"실제 데이터 로드: {len(df)}건")
        return df, True
    except Exception as e:
        print(f"실제 데이터 실패({e}) -> 합성 데이터 사용")
        n = 6000
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
        return df, False

df, is_real = load_real_or_synthetic()
feature_cols = [c for c in df.columns if c not in ["default", "ID"]]
X = df[feature_cols].fillna(0).values
y = df["default"].values

class CreditRiskNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_one_fold(X_tr, y_tr, X_val, y_val, patience=10):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)

    model = CreditRiskNet(X_tr_t.shape[1])
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_auc, best_state, no_improve = 0, None, 0
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_val_t)).numpy()
        auc = roc_auc_score(y_val, val_prob)

        if auc > best_auc:
            best_auc, best_state, no_improve = auc, model.state_dict(), 0
        else:
            no_improve += 1
        if no_improve >= patience:  # Early stopping
            break

    model.load_state_dict(best_state)
    return model, scaler, best_auc

# ---------- K-Fold 교차검증: 단일 분할이 아니라 성능의 "안정성"을 검증 ----------
print("\n=== 5-Fold 교차검증 (성능이 우연이 아님을 검증) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    _, _, auc = train_one_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    fold_aucs.append(auc)
    print(f"  Fold {fold+1}: ROC-AUC = {auc:.4f}")

print(f"\n평균 ROC-AUC: {np.mean(fold_aucs):.4f} (표준편차: {np.std(fold_aucs):.4f})")
print("표준편차가 작을수록 특정 데이터 분할에 우연히 잘 맞은 게 아니라는 신뢰도가 높습니다.")

# ---------- 최종 모델 (전체 학습/테스트 분할로 재학습, 이후 분석용) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
final_model, final_scaler, _ = train_one_fold(X_train, y_train, X_test, y_test)

X_test_s = final_scaler.transform(X_test)
with torch.no_grad():
    final_prob = torch.sigmoid(final_model(torch.tensor(X_test_s, dtype=torch.float32))).numpy().ravel()
final_auc = roc_auc_score(y_test, final_prob)

# ---------- 임계값 최적화: 실무에서는 0.5가 아니라 F1 최적 임계값을 씀 ----------
precisions, recalls, thresholds = precision_recall_curve(y_test, final_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]
print(f"\n최적 판단 임계값: {best_threshold:.3f} (기본값 0.5 대신 사용 시 F1 {f1_scores[best_idx]:.3f})")

# ---------- SHAP 기반 설명가능성 (permutation importance보다 실무 표준에 가까움) ----------
print("\nSHAP 분석 중 (모델의 각 예측을 피처별로 분해)...")
import shap

def model_predict(x_numpy):
    with torch.no_grad():
        out = torch.sigmoid(final_model(torch.tensor(x_numpy, dtype=torch.float32))).numpy()
    return out.ravel()

background = X_test_s[np.random.choice(len(X_test_s), min(50, len(X_test_s)), replace=False)]
explainer = shap.KernelExplainer(model_predict, background)
shap_sample = X_test_s[:30]
shap_values = explainer.shap_values(shap_sample, nsamples=100)

fig1 = plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("SHAP 요약 플롯 저장: shap_summary.png")

# ---------- 종합 대시보드 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

axes[0, 0].bar(range(1, 6), fold_aucs, color="steelblue")
axes[0, 0].axhline(np.mean(fold_aucs), color="red", linestyle="--", label="평균")
axes[0, 0].set_title("5-Fold 교차검증 ROC-AUC")
axes[0, 0].set_xlabel("Fold")
axes[0, 0].legend()

axes[0, 1].plot(recalls, precisions)
axes[0, 1].scatter(recalls[best_idx], precisions[best_idx], color="red", zorder=5, label="최적 임계값")
axes[0, 1].set_title("Precision-Recall Curve")
axes[0, 1].set_xlabel("Recall")
axes[0, 1].set_ylabel("Precision")
axes[0, 1].legend()

axes[1, 0].hist(final_prob[y_test == 0], bins=30, alpha=0.6, label="실제 정상", color="blue")
axes[1, 0].hist(final_prob[y_test == 1], bins=30, alpha=0.6, label="실제 대손", color="red")
axes[1, 0].axvline(best_threshold, color="black", linestyle="--", label="최적 임계값")
axes[1, 0].set_title("예측 확률 분포 + 최적 임계값")
axes[1, 0].legend()

shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[..., 0]
mean_abs_shap = np.abs(shap_values).mean(axis=0).ravel()
axes[1, 1].barh(feature_cols, mean_abs_shap, color="darkslateblue")
axes[1, 1].set_title("평균 |SHAP| 값 (전역 피처 중요도)")
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig("credit_risk_v2_dashboard.png", dpi=150)
print("\n최종 대시보드 저장: credit_risk_v2_dashboard.png")
plt.show()


# ---------- 모델 및 스케일러 저장 (배포용) ----------
import joblib
torch.save(final_model.state_dict(), "credit_risk_model.pt")
joblib.dump(final_scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(float(best_threshold), "threshold.pkl")
print("\n모델 아티팩트 저장 완료: credit_risk_model.pt, scaler.pkl, feature_cols.pkl, threshold.pkl")

print(f"\n=== 최종 요약 ===")
print(f"5-Fold 평균 ROC-AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
print(f"최적 임계값: {best_threshold:.3f}")
print("SHAP 분석으로 개별 예측의 근거를 피처 단위로 설명 가능")
