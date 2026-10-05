import pandas as pd
import numpy as np
import joblib
import os
from fastapi import HTTPException
from src.intervention.intervention_engine import InterventionEngine
from src.explainability.shap_explainer import SHAPExplainer
from src.api.schemas import PredictiveFeatures, PredictionResponse, SHAPExplanation
from src.models.train_baselines import PREDICTIVE_FEATURES

class InferenceService:
    def __init__(self, model_path: str = 'models/baseline/xgb_pipeline.pkl', to_path: str = 'models/mitigated/xgb_threshold_optimizer_gender.pkl'):
        try:
            self.pipeline = joblib.load(model_path)
            self.to_pipeline = None
            if os.path.exists(to_path):
                self.to_pipeline = joblib.load(to_path)
            self.intervention_engine = InterventionEngine()
            self.shap_explainer = SHAPExplainer(model_path=model_path)
            self.loaded = True
        except Exception as e:
            self.loaded = False
            self.pipeline = None
            self.to_pipeline = None
            self.intervention_engine = None
            self.shap_explainer = None
            print(f"Failed to load model or explainers: {e}")

    def is_healthy(self) -> bool:
        return self.loaded

    def predict(self, features: PredictiveFeatures, include_explanation: bool = False) -> PredictionResponse:
        if not self.loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        # Convert input to DataFrame with correct column order
        input_data = {key: [getattr(features, key)] for key in PREDICTIVE_FEATURES}
        df = pd.DataFrame(input_data)
        
        # Ensure numerical casting if needed (for safe pipeline processing)
        num_cols = [c for c in PREDICTIVE_FEATURES if c not in ['highest_education', 'code_module', 'code_presentation']]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        try:
            # Inference
            prob = self.pipeline.predict_proba(df)[0, 1]
            
            # Fairness Decision
            fairness_decision = None
            fairness_status = "Unavailable: gender missing or mitigation model missing"
            
            if self.to_pipeline is not None and features.gender:
                if features.gender not in ['M', 'F']:
                    fairness_status = f"Unavailable: unsupported gender '{features.gender}'"
                else:
                    try:
                        # Pass the exact sensitive attribute structure expected by TO
                        A = pd.Series([features.gender])
                        fairness_decision = int(self.to_pipeline.predict(df, sensitive_features=A)[0])
                        fairness_status = "Applied"
                    except Exception as e:
                        fairness_status = f"Unavailable: {e}"
            
            # Intervention logic
            rec = self.intervention_engine.generate_recommendation(prob)
            
            top_pos = None
            top_neg = None
            
            # SHAP calculation
            if include_explanation:
                try:
                    shap_vals, feature_names = self.shap_explainer.explain_instances(df)
                    shap_vals = shap_vals[0]
                    feature_contributions = list(zip(feature_names, shap_vals))
                    
                    pos_contribs = sorted([(f, v) for f, v in feature_contributions if v > 0], key=lambda x: x[1], reverse=True)[:5]
                    neg_contribs = sorted([(f, v) for f, v in feature_contributions if v < 0], key=lambda x: x[1])[:5]
                    
                    top_pos = [SHAPExplanation(feature=f, contribution=v, direction="Positive") for f, v in pos_contribs]
                    top_neg = [SHAPExplanation(feature=f, contribution=v, direction="Negative") for f, v in neg_contribs]
                except Exception as e:
                    # Do not fail prediction if SHAP fails, but log it or return None for explanations
                    print(f"SHAP explanation failed: {e}")
                    raise HTTPException(status_code=500, detail="Failed to generate SHAP explanation")
            
            return PredictionResponse(
                risk_probability=prob,
                risk_tier=rec['risk_tier'],
                priority_score=prob,
                human_review_required=rec['human_review_required'],
                intervention_level=rec['intervention_level'],
                recommended_actions=rec['recommended_actions'],
                fairness_adjusted_decision=fairness_decision,
                fairness_status=fairness_status,
                top_positive_contributors=top_pos,
                top_negative_contributors=top_neg
            )
        except HTTPException as he:
            raise he
        except Exception as e:
            print(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail="Inference failed due to an internal error")

# Global singleton instance
inference_service = InferenceService()
