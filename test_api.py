"""
FastAPI 서버 자체를 검증하는 통합 테스트.
TestClient를 쓰면 실제 서버를 띄우지 않고도 API 로직을 테스트할 수 있습니다.
"""
from fastapi.testclient import TestClient
from api_server import app, feature_cols

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_input():
    sample = {c: 1000.0 for c in feature_cols}
    response = client.post("/predict", json={"features": sample})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["default_probability"] <= 1.0
    assert isinstance(data["is_high_risk"], bool)

def test_predict_missing_feature():
    """필수 피처가 누락되면 400 에러를 반환해야 함 (실무에서 중요한 방어 코드)"""
    incomplete = {feature_cols[0]: 1000.0}  # 일부러 하나만 넣음
    response = client.post("/predict", json={"features": incomplete})
    assert response.status_code == 400
    assert "누락된 피처" in response.json()["detail"]

def test_predict_extreme_values():
    """극단적 입력값에도 서버가 죽지 않고 정상 응답하는지 검증"""
    extreme = {c: 1e9 for c in feature_cols}
    response = client.post("/predict", json={"features": extreme})
    assert response.status_code == 200

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
