import yaml
import math
from typing import Tuple

import numpy as np

class RiskTierMapper:
    def __init__(self, config_path: str = 'configs/intervention_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['risk_tiers']
            
    def get_tier(self, risk_probability: float) -> Tuple[str, float]:
        """
        Maps a risk probability to a categorical risk tier.
        
        Args:
            risk_probability: A float between 0.0 and 1.0 inclusive.
            
        Returns:
            Tuple of (risk_tier_name_uppercase, risk_probability)
            
        Raises:
            ValueError: If risk_probability is missing, NaN, or outside [0,1].
        """
        if risk_probability is None:
            raise ValueError("risk_probability cannot be None.")
            
        if not isinstance(risk_probability, (int, float, np.number)):
            raise ValueError(f"risk_probability must be a number, got {type(risk_probability)}")
            
        risk_probability = float(risk_probability)
            
        if math.isnan(risk_probability):
            raise ValueError("risk_probability cannot be NaN.")
            
        if not (0.0 <= risk_probability <= 1.0):
            raise ValueError(f"risk_probability must be between 0.0 and 1.0. Got: {risk_probability}")
            
        # Determine tier
        # low: < 0.20
        # moderate: 0.20 <= p < 0.30
        # high: 0.30 <= p < 0.50
        # critical: >= 0.50
        
        if risk_probability < self.config['low']['max_probability']:
            return "LOW", risk_probability
        elif risk_probability < self.config['moderate']['max_probability']:
            return "MODERATE", risk_probability
        elif risk_probability < self.config['high']['max_probability']:
            return "HIGH", risk_probability
        else:
            return "CRITICAL", risk_probability
