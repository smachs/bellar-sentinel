#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DefconSystem: A simplified risk assessment system that calculates overall risk levels
based on sentiment and market data.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Tuple, Optional

# Local imports
from app.utils.data_processor import DataProcessor


class DefconSystem:
    """
    Calculates overall risk levels based on multiple data sources.
    
    This simplified version provides risk level calculation without
    the full Defcon warning system implementation.
    """
    
    def __init__(self, config):
        """Initialize the DefconSystem with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor()
        
        # Initialize state
        self.current_alert_level = 0
        self.last_updated = None
        self.risk_components = {}
        
        # Risk level thresholds (0-1 range maps to 1-5 levels)
        general_config = self.config.get('general', {})
        self.risk_thresholds = general_config.get('risk_thresholds', [0.2, 0.4, 0.6, 0.8])
    
    def initialize(self):
        """Initialize the system."""
        try:
            self.logger.info("Simplified Defcon system initialized")
            self.last_updated = datetime.datetime.now().isoformat()
        except Exception as e:
            self.logger.error(f"Failed to initialize Defcon system: {str(e)}")
            raise
    
    def calculate_alert_level(self, sentiment_data: Dict[str, Any], 
                             market_data: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Calculate the current alert level based on sentiment and market data.
        
        Args:
            sentiment_data: Sentiment analysis data
            market_data: Market monitoring data
            
        Returns:
            Tuple of (alert_level, insights)
        """
        try:
            # Extract risk indices
            sentiment_risk = sentiment_data.get('risk_indices', {}).get('risk_index', 0.0)
            market_risk = market_data.get('indicators', {}).get('aggregate', {}).get('risk_index', 0.0)
            
            # Get weights for each component
            defcon_config = self.config.get('defcon', {})
            component_weights = defcon_config.get('component_weights', {})
            sentiment_weight = component_weights.get('sentiment', 0.5)
            market_weight = component_weights.get('market', 0.5)
            
            # Calculate weighted average risk
            weighted_risk = (
                sentiment_risk * sentiment_weight +
                market_risk * market_weight
            )
            
            # Convert to alert level (1-5)
            alert_level = self._convert_to_alert_level(weighted_risk)
            
            # Store current state
            self.current_alert_level = alert_level
            self.last_updated = datetime.datetime.now().isoformat()
            self.risk_components = {
                'sentiment': sentiment_risk,
                'market': market_risk,
                'weighted': weighted_risk
            }
            
            # Generate insights
            insights = self._generate_insights(sentiment_data, market_data, alert_level)
            
            self.logger.info(f"Calculated alert level: {alert_level} (weighted risk: {weighted_risk:.2f})")
            return alert_level, insights
        
        except Exception as e:
            self.logger.error(f"Error calculating alert level: {str(e)}")
            return 0, []
    
    def _convert_to_alert_level(self, risk_index: float) -> int:
        """
        Convert risk index (0-1) to an alert level (1-5).
        
        Args:
            risk_index: Risk index value (0-1)
            
        Returns:
            Alert level (1-5)
        """
        if risk_index < self.risk_thresholds[0]:
            return 1  # Very Low
        elif risk_index < self.risk_thresholds[1]:
            return 2  # Low
        elif risk_index < self.risk_thresholds[2]:
            return 3  # Moderate
        elif risk_index < self.risk_thresholds[3]:
            return 4  # High
        else:
            return 5  # Very High
    
    def _generate_insights(self, sentiment_data: Dict[str, Any], 
                         market_data: Dict[str, Any], 
                         alert_level: int) -> List[Dict[str, Any]]:
        """
        Generate insights based on the current data and alert level.
        
        Args:
            sentiment_data: Sentiment analysis data
            market_data: Market monitoring data
            alert_level: Current alert level
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        # Add alert level insight
        level_descriptions = [
            "Unknown",
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
        
        alert_description = level_descriptions[alert_level] if 0 <= alert_level < len(level_descriptions) else "Unknown"
        
        insights.append({
            'type': 'alert_level',
            'title': f"Crisis Risk Level: {alert_description}",
            'description': f"The current crisis risk level is {alert_description} (Level {alert_level}/5)",
            'severity': alert_level
        })
        
        # Add sentiment insights
        sentiment_summary = sentiment_data.get('summary', {})
        if sentiment_summary:
            insights.append({
                'type': 'sentiment',
                'title': 'Sentiment Analysis',
                'description': sentiment_summary.get('text', 'No sentiment data available'),
                'severity': sentiment_summary.get('risk_level', 0)
            })
            
            # Add specific concerns
            concerns = sentiment_summary.get('concerns', [])
            for i, concern in enumerate(concerns):
                if i >= 3:  # Limit to top 3 concerns
                    break
                insights.append({
                    'type': 'sentiment_concern',
                    'title': f"Sentiment Concern #{i+1}",
                    'description': concern,
                    'severity': sentiment_summary.get('risk_level', 0)
                })
        
        # Add market insights
        market_summary = market_data.get('summary', {})
        if market_summary:
            insights.append({
                'type': 'market',
                'title': 'Market Analysis',
                'description': market_summary.get('text', 'No market data available'),
                'severity': market_summary.get('risk_level', 0)
            })
            
            # Add specific findings
            findings = market_summary.get('key_findings', [])
            for i, finding in enumerate(findings):
                if i >= 3:  # Limit to top 3 findings
                    break
                insights.append({
                    'type': 'market_finding',
                    'title': f"Market Finding #{i+1}",
                    'description': finding,
                    'severity': market_summary.get('risk_level', 0)
                })
        
        return insights
    
    def generate_report(self, report_type: str = 'comprehensive', 
                      time_horizon: str = 'medium') -> Dict[str, Any]:
        """
        Generate a crisis prediction report.
        
        Args:
            report_type: Type of report ('comprehensive', 'summary', 'technical')
            time_horizon: Time horizon ('short', 'medium', 'long')
            
        Returns:
            Report dictionary
        """
        # In this simplified version, we just return a basic report structure
        report = {
            'title': f"{report_type.capitalize()} Crisis Risk Report - {time_horizon.capitalize()} Term",
            'generated_at': datetime.datetime.now().isoformat(),
            'alert_level': self.current_alert_level,
            'risk_components': self.risk_components,
            'time_horizon': time_horizon,
            'report_type': report_type,
            'sections': [
                {
                    'title': 'Overview',
                    'content': f"This is a simplified crisis risk report. The current alert level is {self.current_alert_level}/5."
                },
                {
                    'title': 'Risk Assessment',
                    'content': "This simplified version does not include detailed risk assessments."
                },
                {
                    'title': 'Recommendations',
                    'content': "This simplified version does not include specific recommendations."
                }
            ]
        }
        
        # Log report generation
        self.logger.info(f"Generated {report_type} report for {time_horizon} term horizon")
        
        return report 