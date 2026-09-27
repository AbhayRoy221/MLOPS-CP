import unittest
import numpy as np
import yaml
from src.intervention.risk_tiers import RiskTierMapper
from src.intervention.intervention_rules import InterventionRuleEngine
from src.intervention.intervention_engine import InterventionEngine

class TestIntervention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = RiskTierMapper()
        cls.rules = InterventionRuleEngine()
        cls.engine = InterventionEngine()
        
    def test_boundary_correctness(self):
        cases = [
            (0.00, 'LOW'),
            (0.199999, 'LOW'),
            (0.20, 'MODERATE'),
            (0.299999, 'MODERATE'),
            (0.30, 'HIGH'),
            (0.499999, 'HIGH'),
            (0.50, 'CRITICAL'),
            (1.00, 'CRITICAL')
        ]
        for prob, expected_tier in cases:
            tier, _ = self.mapper.get_tier(prob)
            self.assertEqual(tier, expected_tier, f"Failed at {prob}: expected {expected_tier}, got {tier}")
            
    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            self.mapper.get_tier(-0.01)
        with self.assertRaises(ValueError):
            self.mapper.get_tier(1.01)
        with self.assertRaises(ValueError):
            self.mapper.get_tier(float('nan'))
        with self.assertRaises(ValueError):
            self.mapper.get_tier(None)
            
    def test_intervention_mapping(self):
        low_rules = self.rules.get_rules('LOW')
        self.assertEqual(low_rules['intervention_level'], 'monitoring')
        
        mod_rules = self.rules.get_rules('MODERATE')
        self.assertEqual(mod_rules['intervention_level'], 'academic check-in')
        
        high_rules = self.rules.get_rules('HIGH')
        self.assertEqual(high_rules['intervention_level'], 'advisor/tutor outreach')
        
        crit_rules = self.rules.get_rules('CRITICAL')
        self.assertEqual(crit_rules['intervention_level'], 'priority human review')
        
    def test_human_review_flags(self):
        self.assertFalse(self.rules.get_rules('LOW')['human_review_required'])
        self.assertFalse(self.rules.get_rules('MODERATE')['human_review_required'])
        self.assertTrue(self.rules.get_rules('HIGH')['human_review_required'])
        self.assertTrue(self.rules.get_rules('CRITICAL')['human_review_required'])
        
    def test_priority_score(self):
        prob = 0.42
        rec = self.engine.generate_recommendation(prob)
        self.assertEqual(rec['priority_score'], prob)
        self.assertEqual(rec['risk_probability'], prob)
        
    def test_determinism(self):
        prob = 0.35
        rec1 = self.engine.generate_recommendation(prob)
        rec2 = self.engine.generate_recommendation(prob)
        self.assertEqual(rec1, rec2)
        
    def test_configuration(self):
        # Verify it loads from yaml
        with open('configs/intervention_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertEqual(config['risk_tiers']['low']['max_probability'], 0.20)
        
if __name__ == '__main__':
    unittest.main()
