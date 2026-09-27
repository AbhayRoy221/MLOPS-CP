import yaml
from typing import Dict, Any

class InterventionRuleEngine:
    def __init__(self, config_path: str = 'configs/intervention_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['intervention_rules']
            
    def get_rules(self, risk_tier: str) -> Dict[str, Any]:
        """
        Retrieves intervention rules for a given risk tier.
        
        Args:
            risk_tier: The categorical risk tier (e.g. 'LOW', 'MODERATE', 'HIGH', 'CRITICAL')
            
        Returns:
            Dictionary containing intervention level, review requirement, and actions.
        """
        tier_key = risk_tier.lower()
        if tier_key not in self.config:
            raise ValueError(f"Unknown risk tier: {risk_tier}")
            
        return self.config[tier_key]
