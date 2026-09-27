# API Deployment Documentation (Step 2.11)

## 1. Purpose
The Student Dropout Early Warning API exposes the existing Day-28 predictive pipeline via a fast, local REST interface. This API allows integration with potential frontend dashboards or student information systems, providing deterministic risk scores, human-readable risk tiers, recommended interventions, and optional SHAP-based context.

## 2. Architecture
- **Framework:** FastAPI (Python)
- **Web Server:** Uvicorn
- **Model Loader:** Loads the locked `models/baseline/xgb_pipeline.pkl` at startup to ensure high throughput.
- **Explainability:** Reuses the Step 2.10 `SHAPExplainer` class.
- **Interventions:** Reuses the Step 2.9 `InterventionEngine`.
- **Validation:** Pydantic models automatically enforce data types and structure for the 21 expected Day-28 predictive features.

## 3. API Endpoints
- `GET /` : General service info.
- `GET /health` : Returns model loading status.
- `POST /predict` : The primary inference endpoint.

## 4. Request Schema
The `/predict` endpoint expects a JSON body matching the exact 21 predictive features engineered up to Day-28:
- `highest_education` (string)
- `num_of_prev_attempts` (integer, >=0)
- `studied_credits` (integer, >=0)
- `registration_day` (integer)
- `code_module` (string)
- `code_presentation` (string)
- `module_presentation_length` (integer, >0)
- `asm_submission_count` (float, >=0)
- `asm_mean_score` (float)
- `asm_score_std` (float)
- `asm_failed_count` (float, >=0)
- `asm_average_delay` (float)
- `vle_total_clicks` (float, >=0)
- `vle_active_days` (float, >=0)
- `vle_days_since_last_activity` (float, >=0)
- `vle_clicks_last_7_days` (float, >=0)
- `vle_clicks_last_14_days` (float, >=0)
- `vle_forum_clicks` (float, >=0)
- `vle_resource_clicks` (float, >=0)
- `vle_quiz_clicks` (float, >=0)
- `vle_activity_type_diversity` (float, >=0)

*Note: Identifiers (`id_student`), fairness attributes (e.g. `gender`), and the target variable (`final_result`) are strictly excluded from the request schema.*

## 5. Response Schema
The `/predict` endpoint returns:
- `risk_probability` (float [0, 1]): Raw XGBoost prediction.
- `risk_tier` (string): e.g., LOW, MODERATE, HIGH, CRITICAL.
- `priority_score` (float): Equals risk probability for ranking.
- `human_review_required` (boolean): Flag from the intervention engine.
- `intervention_level` (string): e.g., standard, advisor, priority.
- `recommended_actions` (list of strings): Concrete actions for an advisor.
- (Optional) `top_positive_contributors`: Top 5 SHAP features increasing risk.
- (Optional) `top_negative_contributors`: Top 5 SHAP features decreasing risk.

## 6. Validation
Pydantic handles automatic validation. Out-of-bound counts (like negative submission counts) or missing required features result in a `422 Unprocessable Entity` HTTP response. Only constraints inherently supported by the original dataset definition are strictly enforced.

## 7. Model Loading
The pipeline (`RobustScaler` + `OneHotEncoder` + `XGBClassifier`) is loaded safely in memory when the FastAPI app spins up. It does not load from disk on every request, ensuring millisecond-level latency.

## 8. Risk-Tier Integration
The API directly imports and delegates to the authoritative `InterventionEngine`. Risk boundaries are NOT duplicated or hardcoded into the API logic.

## 9. Intervention Integration
The returned recommended actions align identically with the Step 2.9 configuration matrix.

## 10. SHAP Integration
Adding the query parameter `?include_explanation=true` will attach the model's global and local tree explanations. The API safely encapsulates the SHAP explainer, returning clear "positive" or "negative" direction strings alongside raw numeric contributions. 

## 11. Error Handling
- Invalid JSON or missing fields → `422 Unprocessable Entity`.
- General Inference Failure → `500 Internal Server Error`.
- If the SHAP backend crashes internally, the API isolates it and avoids crashing the primary prediction (unless entirely failed), or returns `500` strictly indicating SHAP generation failure without dropping connection context or leaking Python stack traces to the client.

## 12. Example Request/Response

**Request (`POST /predict?include_explanation=true`):**
```json
{
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
```

**Response:**
```json
{
  "risk_probability": 0.274,
  "risk_tier": "MODERATE",
  "priority_score": 0.274,
  "human_review_required": false,
  "intervention_level": "standard",
  "recommended_actions": [
    "academic check-in",
    "study planning"
  ],
  "top_positive_contributors": [
    {
      "feature": "num__asm_average_delay",
      "contribution": 0.05,
      "direction": "Positive"
    }
  ],
  "top_negative_contributors": [
    {
      "feature": "cat__code_module_AAA",
      "contribution": -0.15,
      "direction": "Negative"
    }
  ]
}
```

## 13. Security/Privacy Considerations
- The API implements minimal, safe logging.
- PII, student IDs, and raw feature payloads are intentionally excluded from logs.
- Sensitive demographics (e.g., gender, disability) are never processed or retained by this predictive endpoint.
- Internal file paths and Python exceptions are swallowed and converted to standard generic HTTP HTTP statuses.

## 14. Limitations
- **This is an inference API.** It does not retrain the model or update feature encodings.
- **It does not use the locked test/final holdout.** The model strictly acts on forward-inference payloads.
- **Risk predictions are model outputs, not guarantees.**
- **SHAP explanations are not causal explanations.** They describe feature influence strictly on the model's structure, not on the student's actual life.
- **Intervention recommendations are supportive suggestions** for human review, not mandatory enforcement mechanisms.
