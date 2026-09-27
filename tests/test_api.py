import unittest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.valid_payload = {
            "highest_education": "A Level or Equivalent",
            "num_of_prev_attempts": 0,
            "studied_credits": 60,
            "registration_day": -10,
            "code_module": "AAA",
            "code_presentation": "2013J",
            "module_presentation_length": 268,
            "asm_submission_count": 2,
            "asm_mean_score": 80.5,
            "asm_score_std": 5.0,
            "asm_failed_count": 0,
            "asm_average_delay": 0.5,
            "vle_total_clicks": 150,
            "vle_active_days": 20,
            "vle_days_since_last_activity": 2,
            "vle_clicks_last_7_days": 50,
            "vle_clicks_last_14_days": 100,
            "vle_forum_clicks": 20,
            "vle_resource_clicks": 100,
            "vle_quiz_clicks": 30,
            "vle_activity_type_diversity": 5
        }
        
    def test_root_endpoint(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("service", response.json())
        
    def test_health_endpoint(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertTrue(response.json()["model_loaded"])
        
    def test_predict_valid_input(self):
        response = client.post("/predict", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check required fields
        self.assertIn("risk_probability", data)
        self.assertIn("risk_tier", data)
        self.assertIn("priority_score", data)
        self.assertIn("human_review_required", data)
        self.assertIn("intervention_level", data)
        self.assertIn("recommended_actions", data)
        
        # Check priority equals prob
        self.assertEqual(data["risk_probability"], data["priority_score"])
        
        # No explanation should be present by default
        self.assertIsNone(data.get("top_positive_contributors"))
        self.assertIsNone(data.get("top_negative_contributors"))
        
    def test_predict_invalid_input(self):
        invalid_payload = self.valid_payload.copy()
        invalid_payload["studied_credits"] = "NOT_A_NUMBER"
        
        response = client.post("/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)
        
    def test_predict_with_shap(self):
        response = client.post("/predict?include_explanation=true", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIsNotNone(data.get("top_positive_contributors"))
        self.assertIsNotNone(data.get("top_negative_contributors"))
        
        # Ensure no ID student in SHAP
        for contrib in data["top_positive_contributors"]:
            self.assertNotEqual(contrib["feature"], "id_student")
        
    def test_target_cannot_be_supplied(self):
        payload_with_target = self.valid_payload.copy()
        payload_with_target["final_result"] = "Withdrawn"
        payload_with_target["is_withdrawn"] = 1
        
        # It will be ignored by Pydantic Extra behavior by default, 
        # or we just ensure the model does not require it. 
        # The schema doesn't have it, so if we post it, it's either ignored or rejected.
        # But let's check it doesn't break
        response = client.post("/predict", json=payload_with_target)
        self.assertEqual(response.status_code, 200) # Ignored extra fields
        
    def test_fairness_attributes_not_required(self):
        # We did not provide gender, disability, etc. in valid_payload. 
        # The fact that test_predict_valid_input works verifies they are not required.
        pass
        
    def test_deterministic_behavior(self):
        resp1 = client.post("/predict", json=self.valid_payload).json()
        resp2 = client.post("/predict", json=self.valid_payload).json()
        self.assertEqual(resp1["risk_probability"], resp2["risk_probability"])
        self.assertEqual(resp1["risk_tier"], resp2["risk_tier"])

if __name__ == '__main__':
    unittest.main()
