import unittest
import yaml
import json
import os

class TestMonitoringConfigSyntax(unittest.TestCase):
    """Test that all newly added monitoring configuration files have valid syntax."""

    def test_prometheus_yaml_syntax(self):
        """Prometheus config should be valid YAML and have basic required keys."""
        config_path = 'monitoring/prometheus/prometheus.yml'
        self.assertTrue(os.path.exists(config_path), f"File {config_path} not found")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.assertIn('global', config)
        self.assertIn('scrape_configs', config)
        self.assertIsInstance(config['scrape_configs'], list)

    def test_alerts_yaml_syntax(self):
        """Alerts config should be valid YAML and have rules."""
        config_path = 'monitoring/prometheus/alerts.yml'
        self.assertTrue(os.path.exists(config_path), f"File {config_path} not found")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.assertIn('groups', config)
        groups = config['groups']
        self.assertIsInstance(groups, list)
        self.assertTrue(len(groups) > 0)
        
        # Check that our specific alerts exist
        all_alerts = [rule.get('alert') for group in groups for rule in group.get('rules', [])]
        self.assertIn('HighErrorRate', all_alerts)
        self.assertIn('HighLatency', all_alerts)
        self.assertIn('InstanceDown', all_alerts)

    def test_grafana_dashboard_json_syntax(self):
        """Dashboard should be valid JSON and contain expected panels."""
        config_path = 'monitoring/grafana/dashboards/prediction_dashboard.json'
        self.assertTrue(os.path.exists(config_path), f"File {config_path} not found")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.assertIn('panels', config)
        panels = config['panels']
        self.assertIsInstance(panels, list)
        
        # Make sure our titles are in there
        titles = [panel.get('title') for panel in panels if 'title' in panel]
        self.assertIn('Prediction Request Rate', titles)
        self.assertIn('Prediction Error Rate', titles)
        self.assertIn('Prediction Latency', titles)

if __name__ == '__main__':
    unittest.main()
