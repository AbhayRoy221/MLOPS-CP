from pydantic import BaseModel, Field
from typing import List, Optional

class PredictiveFeatures(BaseModel):
    """
    Schema for the 21 predictive features required by the dropout model.
    """
    highest_education: str = Field(..., description="Highest education level on entry")
    num_of_prev_attempts: int = Field(..., ge=0, description="Number of previous attempts")
    studied_credits: int = Field(..., ge=0, description="Total number of credits studied")
    registration_day: int = Field(..., description="Registration date relative to module presentation start")
    code_module: str = Field(..., description="Module code (e.g., AAA, BBB)")
    code_presentation: str = Field(..., description="Presentation code (e.g., 2013J, 2014B)")
    module_presentation_length: int = Field(..., gt=0, description="Length of the module presentation in days")
    
    asm_submission_count: float = Field(..., ge=0, description="Number of assessments submitted by day 28")
    asm_mean_score: float = Field(..., description="Mean score on assessments submitted by day 28")
    asm_score_std: float = Field(..., description="Standard deviation of assessment scores")
    asm_failed_count: float = Field(..., ge=0, description="Number of failed assessments by day 28")
    asm_average_delay: float = Field(..., description="Average delay in days for assessment submission")
    
    vle_total_clicks: float = Field(..., ge=0, description="Total VLE clicks by day 28")
    vle_active_days: float = Field(..., ge=0, description="Number of days active on the VLE by day 28")
    vle_days_since_last_activity: float = Field(..., ge=0, description="Days since last VLE activity before day 28")
    vle_clicks_last_7_days: float = Field(..., ge=0, description="VLE clicks between day 21 and 28")
    vle_clicks_last_14_days: float = Field(..., ge=0, description="VLE clicks between day 14 and 28")
    vle_forum_clicks: float = Field(..., ge=0, description="VLE clicks on forum resources")
    vle_resource_clicks: float = Field(..., ge=0, description="VLE clicks on standard resources")
    vle_quiz_clicks: float = Field(..., ge=0, description="VLE clicks on quizzes/assessments")
    vle_activity_type_diversity: float = Field(..., ge=0, description="Number of distinct VLE activity types accessed")


class SHAPExplanation(BaseModel):
    """
    Schema for local SHAP explanation features.
    """
    feature: str = Field(..., description="Name of the feature")
    contribution: float = Field(..., description="SHAP value for the feature")
    direction: str = Field(..., description="Positive (increases risk) or Negative (decreases risk)")


class PredictionResponse(BaseModel):
    """
    Schema for the overall API response.
    """
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of dropping out (withdrawing)")
    risk_tier: str = Field(..., description="Categorical risk tier (LOW, MODERATE, HIGH, CRITICAL)")
    priority_score: float = Field(..., description="Score used for prioritization, currently equals risk_probability")
    human_review_required: bool = Field(..., description="Whether human review is required")
    intervention_level: str = Field(..., description="Level of intervention recommended")
    recommended_actions: List[str] = Field(..., description="Recommended actions for the advisor")
    
    top_positive_contributors: Optional[List[SHAPExplanation]] = Field(None, description="Features pushing risk higher")
    top_negative_contributors: Optional[List[SHAPExplanation]] = Field(None, description="Features pushing risk lower")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
