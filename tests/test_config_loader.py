"""
Tests for the ConfigLoader class.
"""

import os
import unittest
import tempfile
import json
from app.utils.config_loader import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    """Tests for the ConfigLoader class."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test configs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, 'config.json')

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_default_config(self):
        """Test loading default configuration when no file exists."""
        # Create a config loader with a non-existent path
        loader = ConfigLoader(config_path='nonexistent.json')
        config = loader.load_config()

        # Verify default configuration was loaded
        self.assertTrue(isinstance(config, dict))
        self.assertIn('general', config)
        self.assertIn('debug', config['general'])
        self.assertIn('server', config)
        self.assertIn('port', config['server'])

    def test_load_custom_config(self):
        """Test loading a custom configuration file."""
        # Create a test configuration file
        test_config = {
            'general': {
                'debug': False,
                'use_gpu': True
            },
            'server': {
                'port': 8080
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(test_config, f)

        # Load the configuration
        loader = ConfigLoader(config_path=self.config_path)
        config = loader.load_config()

        # Verify configuration was loaded correctly
        self.assertEqual(config['general']['debug'], False)
        self.assertEqual(config['general']['use_gpu'], True)
        self.assertEqual(config['server']['port'], 8080)

    def test_get_nested_values(self):
        """Test retrieving nested values from configuration."""
        # Create a test configuration
        test_config = {
            'level1': {
                'level2': {
                    'level3': 'value'
                }
            }
        }

        # Create a loader and set the config
        loader = ConfigLoader(config_path=self.config_path)
        loader.config = test_config

        # Test retrieving nested values
        self.assertEqual(loader.get('level1', 'level2', 'level3'), 'value')

        # Test retrieving non-existent values
        self.assertIsNone(loader.get('nonexistent'))
        self.assertEqual(loader.get('nonexistent', default='default'), 'default')


if __name__ == '__main__':
    unittest.main() 