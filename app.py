#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crisis Sentinel: An AI-powered system for predicting financial, geopolitical, and institutional crises
by analyzing patterns across multiple data sources.
"""

import os
import logging
from flask import Flask, render_template, jsonify, request
import json

from app.core.sentiment_analyzer import SentimentAnalyzer
from app.core.market_monitor import MarketMonitor
from app.core.defcon_system import DefconSystem
from app.utils.config_loader import ConfigLoader
from app.utils.logger import setup_logger

# Initialize Flask application
app = Flask(__name__)

# Setup logging
logger = setup_logger()

# Load configuration
config = ConfigLoader().load_config()

# Initialize components
sentiment_analyzer = SentimentAnalyzer(config)
market_monitor = MarketMonitor(config)
defcon_system = DefconSystem(config)


@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')


@app.route('/api/current_status', methods=['GET'])
def current_status():
    """Get the current crisis alert status."""
    try:
        # Analyze global sentiment
        sentiment_data = sentiment_analyzer.analyze_current_sentiment()
        
        # Monitor market liquidity
        market_data = market_monitor.get_current_market_state()
        
        # Calculate Defcon level
        defcon_level, insights = defcon_system.calculate_alert_level(
            sentiment_data=sentiment_data,
            market_data=market_data
        )
        
        response = {
            'status': 'success',
            'defcon_level': defcon_level,
            'sentiment_summary': sentiment_data['summary'],
            'market_summary': market_data['summary'],
            'insights': insights,
            'timestamp': defcon_system.last_updated
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in current_status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to retrieve current status: {str(e)}"
        }), 500


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate a detailed crisis prediction report."""
    try:
        report_type = request.json.get('report_type', 'comprehensive')
        time_horizon = request.json.get('time_horizon', 'medium')  # short, medium, long
        
        report = defcon_system.generate_report(report_type, time_horizon)
        
        return jsonify({
            'status': 'success',
            'report': report
        })
    
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to generate report: {str(e)}"
        }), 500


def main():
    """Run the application."""
    try:
        # Initialize data collection
        sentiment_analyzer.initialize()
        market_monitor.initialize()
        defcon_system.initialize()
        
        # Start the Flask application
        app.run(host=config.get('server', 'host'),
                port=config.get('server', 'port'),
                debug=config.get('server', 'debug'))
                
    except Exception as e:
        logger.critical(f"Failed to start Crisis Sentinel: {str(e)}")
        raise


if __name__ == "__main__":
    main() 