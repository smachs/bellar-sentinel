#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Loader: Handles loading and parsing of configuration files for the Crisis Sentinel system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Handles loading and parsing of configuration files for the Crisis Sentinel system.
    
    This utility loads configuration settings from JSON files, environment variables,
    or provides sensible defaults when configuration is not available.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file, defaults to 'config/config.json'
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join('config', 'config.json')
        self.config = {}
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Dictionary containing configuration settings
        """
        try:
            # Try to load from file
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self.logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                self._load_default_config()
            
            # Apply any environment variable overrides
            self._apply_env_overrides()
            
            return self.config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.logger.warning("Loading default configuration")
            self._load_default_config()
            return self.config
    
    def _load_default_config(self):
        """Load default configuration settings."""
        self.config = {
            'general': {
                'debug': True,
                'use_gpu': False
            },
            'server': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': True
            },
            'data': {
                'historical_sentiment_path': 'data/historical_sentiment.json',
                'historical_market_path': 'data/historical_market.json',
                'max_sentiment_history': 365,
                'max_market_history': 365
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/crisis_sentinel.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'sentiment': {
                'days_back': 7,
                'transformer_model': 'distilbert-base-uncased-finetuned-sst-2-english',
                'vader_weight': 0.4,
                'transformer_weight': 0.6,
                'source_weights': {
                    'news': 0.5,
                    'twitter': 0.3,
                    'reddit': 0.2
                },
                'shift_threshold': 0.3,
                'risk_thresholds': [0.2, 0.4, 0.6, 0.8],
                'news_sources': [
                    'bloomberg', 'financial-times', 'the-wall-street-journal', 
                    'the-economist', 'reuters', 'cnbc', 'business-insider'
                ],
                'crisis_keywords': [
                    'recession', 'inflation', 'market crash', 'financial crisis',
                    'bank failure', 'debt default', 'economic collapse',
                    'currency devaluation', 'credit crunch', 'market volatility'
                ],
                'twitter_keywords': [
                    'recession', 'market crash', 'financial crisis',
                    'economic collapse', 'stock market', 'federal reserve',
                    'central bank', 'interest rates', 'inflation'
                ],
                'reddit_subreddits': [
                    'investing', 'finance', 'economics', 'stocks', 'wallstreetbets'
                ],
                'reddit_limit': 100
            },
            'market': {
                'days_back': 30,
                'risk_thresholds': [0.2, 0.4, 0.6, 0.8],
                'indicators': {
                    'yield_curve': {
                        'enabled': True,
                        'weight': 0.25,
                        'tickers': {
                            '3m': '^IRX',
                            '2y': '^UST2Y',
                            '10y': '^TNX'
                        }
                    },
                    'volatility': {
                        'enabled': True,
                        'weight': 0.20,
                        'tickers': {
                            'vix': '^VIX',
                            'vvix': '^VVIX'
                        }
                    },
                    'credit_spreads': {
                        'enabled': True,
                        'weight': 0.15,
                        'tickers': {
                            'high_yield': 'HYG',
                            'investment_grade': 'LQD'
                        }
                    },
                    'liquidity': {
                        'enabled': True,
                        'weight': 0.20,
                        'tickers': {
                            'ted_spread': {
                                'components': [
                                    '^IRX',
                                    'USD3MTD156N'
                                ],
                                'calculation': 'spread'
                            }
                        }
                    },
                    'market_indices': {
                        'enabled': True,
                        'weight': 0.10,
                        'tickers': {
                            'sp500': '^GSPC',
                            'nasdaq': '^IXIC',
                            'russell': '^RUT'
                        }
                    },
                    'commodities': {
                        'enabled': True,
                        'weight': 0.10,
                        'tickers': {
                            'gold': 'GC=F',
                            'oil': 'CL=F',
                            'vix': '^VIX'
                        }
                    }
                }
            },
            'apis': {
                'twitter': {
                    'bearer_token': '',
                    'consumer_key': '',
                    'consumer_secret': '',
                    'access_token': '',
                    'access_token_secret': ''
                },
                'reddit': {
                    'client_id': '',
                    'client_secret': '',
                    'user_agent': 'crisis_sentinel/1.0'
                },
                'news': {
                    'api_key': ''
                }
            }
        }
    
    def _apply_env_overrides(self):
        """Override configuration with environment variables."""
        # Environment variables take precedence over config file
        try:
            # Map of environment variables to config keys
            env_mappings = {
                'CS_DEBUG': ('general', 'debug', lambda x: x.lower() == 'true'),
                'CS_USE_GPU': ('general', 'use_gpu', lambda x: x.lower() == 'true'),
                'CS_SERVER_HOST': ('server', 'host', str),
                'CS_SERVER_PORT': ('server', 'port', int),
                'CS_LOG_LEVEL': ('logging', 'level', str),
                'CS_TWITTER_BEARER_TOKEN': ('apis', 'twitter', 'bearer_token', str),
                'CS_TWITTER_CONSUMER_KEY': ('apis', 'twitter', 'consumer_key', str),
                'CS_TWITTER_CONSUMER_SECRET': ('apis', 'twitter', 'consumer_secret', str),
                'CS_TWITTER_ACCESS_TOKEN': ('apis', 'twitter', 'access_token', str),
                'CS_TWITTER_ACCESS_TOKEN_SECRET': ('apis', 'twitter', 'access_token_secret', str),
                'CS_REDDIT_CLIENT_ID': ('apis', 'reddit', 'client_id', str),
                'CS_REDDIT_CLIENT_SECRET': ('apis', 'reddit', 'client_secret', str),
                'CS_NEWS_API_KEY': ('apis', 'news', 'api_key', str)
            }
            
            # Apply overrides
            for env_var, config_path in env_mappings.items():
                if env_var in os.environ:
                    value = os.environ[env_var]
                    
                    # Apply type conversion if provided
                    if len(config_path) > 2 and callable(config_path[-1]):
                        value = config_path[-1](value)
                        path = config_path[:-1]
                    else:
                        path = config_path
                    
                    # Traverse the config dictionary
                    current = self.config
                    for i, key in enumerate(path):
                        if i == len(path) - 1:
                            current[key] = value
                        else:
                            current = current[key]
            
            self.logger.debug("Applied environment variable overrides")
        except Exception as e:
            self.logger.error(f"Error applying environment variable overrides: {str(e)}")
    
    def get(self, *keys, default=None) -> Any:
        """
        Get a configuration value from nested keys.
        
        Args:
            *keys: Sequence of keys to access nested dictionaries
            default: Default value to return if keys not found
            
        Returns:
            The configuration value or default
        """
        if not self.config:
            self.load_config()
        
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, value, *keys) -> bool:
        """
        Set a configuration value at specified nested keys.
        
        Args:
            value: Value to set
            *keys: Sequence of keys to access nested dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not keys:
            return False
        
        if not self.config:
            self.load_config()
        
        current = self.config
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                current[key] = value
                return True
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
        
        return False
    
    def save_config(self, config_path: str = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration file, defaults to self.config_path
            
        Returns:
            True if successful, False otherwise
        """
        path = config_path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            return False 