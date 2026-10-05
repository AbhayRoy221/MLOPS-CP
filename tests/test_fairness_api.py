from fastapi.testclient import TestClient
import pytest
from src.api.main import app

client = TestClient(app)

valid_payload = {
    "highest_education": "A Level or Equivalent",
    "num_of_prev_attempts": 0,
    "studied_credits": 60,
    "registration_day": -15,
    "code_module": "AAA",
    "code_presentation": "2013J",
    "module_presentation_length": 268,
    "asm_submission_count": 2.0,
    "asm_mean_score": 85.0,
    "asm_score_std": 5.0,
    "asm_failed_count": 0.0,
    "asm_average_delay": 0.0,
    "vle_total_clicks": 150.0,
    "vle_active_days": 10.0,
    "vle_days_since_last_activity": 2.0,
    "vle_clicks_last_7_days": 50.0,
    "vle_clicks_last_14_days": 100.0,
    "vle_forum_clicks": 20.0,
    "vle_resource_clicks": 80.0,
    "vle_quiz_clicks": 50.0,
    "vle_activity_type_diversity": 5.0
}

def test_missing_gender():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "fairness_adjusted_decision" in data
    assert data["fairness_adjusted_decision"] is None
    assert "Unavailable" in data["fairness_status"]

def test_valid_gender():
    payload = valid_payload.copy()
    payload["gender"] = "M"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fairness_status"] == "Applied"
    assert data["fairness_adjusted_decision"] in [0, 1]

def test_unsupported_gender():
    payload = valid_payload.copy()
    payload["gender"] = "UnknownGender"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Unavailable" in data["fairness_status"]
    assert data["fairness_adjusted_decision"] is None

def test_baseline_preservation():
    # Ensure risk probability and tiers are untouched
    response = client.post("/predict", json=valid_payload)
    data_no_gender = response.json()
    
    payload = valid_payload.copy()
    payload["gender"] = "M"
    response2 = client.post("/predict", json=payload)
    data_with_gender = response2.json()
    
    assert data_no_gender["risk_probability"] == data_with_gender["risk_probability"]
    assert data_no_gender["risk_tier"] == data_with_gender["risk_tier"]
    assert data_no_gender["human_review_required"] == data_with_gender["human_review_required"]
