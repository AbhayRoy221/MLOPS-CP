from typing import Dict, Any
from .risk_tiers import RiskTierMapper
from .intervention_rules import InterventionRuleEngine

class InterventionEngine:
    def __init__(self, config_path: str = 'configs/intervention_config.yaml'):
        self.tier_mapper = RiskTierMapper(config_path)
        self.rule_engine = InterventionRuleEngine(config_path)
        
    def generate_recommendation(self, risk_probability: float, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates a structured intervention recommendation based on risk probability.
        
        Args:
            risk_probability: A float between 0.0 and 1.0.
            context: Optional contextual features (e.g. vle_active_days), not used for score modification.
            
        Returns:
            Dictionary containing the full recommendation.
        """
        risk_tier, valid_prob = self.tier_mapper.get_tier(risk_probability)
        rules = self.rule_engine.get_rules(risk_tier)
        
        recommendation = {
            'risk_tier': risk_tier,
            'risk_probability': valid_prob,
            'intervention_level': rules['intervention_level'],
            'recommended_actions': rules['recommended_actions'],
            'human_review_required': rules['human_review_required'],
            'priority_score': valid_prob
        }
        
        if context:
            recommendation['context'] = context
            
        return recommendation
