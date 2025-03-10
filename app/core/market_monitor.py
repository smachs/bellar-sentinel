#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Monitor: Tracks financial indicators, market liquidity, and detects
abnormal patterns that may indicate upcoming crises.
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Financial data APIs
import yfinance as yf

# Local imports
from app.utils.data_processor import DataProcessor


class MarketMonitor:
    """
    Monitors financial markets for signs of instability or crisis.
    
    Tracks indicators such as liquidity, yield curves, volatility indices,
    and unusual trading patterns to detect early warning signs of financial crises.
    """
    
    def __init__(self, config):
        """Initialize the MarketMonitor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor()
        
        # Historical market data
        self.historical_data = {}
        self.last_updated = None
        
        # Risk thresholds
        self.risk_thresholds = self.config.get('market', 'risk_thresholds')
        
        # Market indicators to track
        self.indicators = self.config.get('market', 'indicators', {
            'yield_curve': {
                'enabled': True,
                'weight': 0.25,
                'tickers': {
                    '3m': '^IRX',  # 13-week Treasury yield
                    '2y': '^UST2Y',  # 2-year Treasury yield
                    '10y': '^TNX'  # 10-year Treasury yield
                }
            },
            'volatility': {
                'enabled': True,
                'weight': 0.20,
                'tickers': {
                    'vix': '^VIX',  # CBOE Volatility Index
                    'vvix': '^VVIX'  # CBOE VIX of VIX Index
                }
            },
            'credit_spreads': {
                'enabled': True,
                'weight': 0.15,
                'tickers': {
                    'high_yield': 'HYG',  # iShares iBoxx $ High Yield Corporate Bond ETF
                    'investment_grade': 'LQD'  # iShares iBoxx $ Investment Grade Corporate Bond ETF
                }
            },
            'liquidity': {
                'enabled': True,
                'weight': 0.20,
                'tickers': {
                    'ted_spread': {
                        'components': [
                            '^IRX',  # 3-month Treasury yield
                            'USD3MTD156N'  # 3-month LIBOR USD (might need direct source)
                        ],
                        'calculation': 'spread'
                    }
                }
            },
            'market_indices': {
                'enabled': True,
                'weight': 0.10,
                'tickers': {
                    'sp500': '^GSPC',  # S&P 500
                    'nasdaq': '^IXIC',  # NASDAQ Composite
                    'russell': '^RUT'  # Russell 2000 (small caps)
                }
            },
            'commodities': {
                'enabled': True,
                'weight': 0.10,
                'tickers': {
                    'gold': 'GC=F',  # Gold futures
                    'oil': 'CL=F',  # Crude oil futures
                    'vix': '^VIX'  # VIX as a fear indicator
                }
            }
        })
    
    def initialize(self):
        """Initialize market monitoring with historical data."""
        try:
            # Load historical market data
            historical_data_path = self.config.get('data', 'historical_market_path')
            if os.path.exists(historical_data_path):
                with open(historical_data_path, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                    
            # Initialize baseline market data
            if not self.historical_data:
                self.logger.warning("No historical market data found, initializing baseline")
                self._initialize_baseline_data()
                
            self.logger.info("Market monitor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize market monitor: {str(e)}")
            raise
    
    def _initialize_baseline_data(self):
        """Initialize baseline market data from current readings."""
        # Collect current market data
        market_data = self._collect_market_data()
        
        # Calculate indicators
        indicators = self._calculate_indicators(market_data)
        
        # Store as historical
        self.historical_data = {
            'baseline': indicators,
            'history': [
                {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'data': indicators
                }
            ]
        }
        
        # Save to disk
        self._save_historical_data()
    
    def _save_historical_data(self):
        """Save historical market data to disk."""
        try:
            historical_data_path = self.config.get('data', 'historical_market_path')
            os.makedirs(os.path.dirname(historical_data_path), exist_ok=True)
            
            with open(historical_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.historical_data, f, indent=2)
                
            self.logger.info("Historical market data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save historical market data: {str(e)}")
    
    def _collect_market_data(self) -> Dict[str, Any]:
        """Collect current market data from various sources."""
        market_data = {}
        
        try:
            # Get all unique tickers to fetch
            tickers = set()
            for indicator_group in self.indicators.values():
                if not indicator_group.get('enabled', True):
                    continue
                
                for ticker_key, ticker_val in indicator_group.get('tickers', {}).items():
                    if isinstance(ticker_val, str):
                        tickers.add(ticker_val)
                    elif isinstance(ticker_val, dict) and 'components' in ticker_val:
                        tickers.update(ticker_val['components'])
            
            # Determine date range
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            days_back = self.config.get('market', 'days_back', 30)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Fetch all ticker data at once
            if tickers:
                ticker_data = yf.download(
                    tickers=' '.join(tickers),
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                # Reorganize data by ticker
                if len(tickers) == 1:
                    # yfinance returns different format for single ticker
                    single_ticker = list(tickers)[0]
                    market_data[single_ticker] = ticker_data
                else:
                    # For multiple tickers, organize by ticker
                    for ticker in tickers:
                        try:
                            # Extract just this ticker's data
                            ticker_df = ticker_data.xs(ticker, level=1, axis=1).copy()
                            market_data[ticker] = ticker_df
                        except Exception as e:
                            self.logger.warning(f"Could not extract data for ticker {ticker}: {e}")
            
            self.logger.info(f"Collected market data for {len(market_data)} tickers")
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
        
        return market_data
    
    def _calculate_indicators(self, market_data) -> Dict[str, Any]:
        """Calculate financial indicators from raw market data."""
        indicators = {}
        
        try:
            # Calculate each indicator group
            if self.indicators.get('yield_curve', {}).get('enabled', False):
                indicators['yield_curve'] = self._calculate_yield_curve(market_data)
                
            if self.indicators.get('volatility', {}).get('enabled', False):
                indicators['volatility'] = self._calculate_volatility(market_data)
                
            if self.indicators.get('credit_spreads', {}).get('enabled', False):
                indicators['credit_spreads'] = self._calculate_credit_spreads(market_data)
                
            if self.indicators.get('liquidity', {}).get('enabled', False):
                indicators['liquidity'] = self._calculate_liquidity(market_data)
                
            if self.indicators.get('market_indices', {}).get('enabled', False):
                indicators['market_indices'] = self._calculate_market_indices(market_data)
                
            if self.indicators.get('commodities', {}).get('enabled', False):
                indicators['commodities'] = self._calculate_commodities(market_data)
            
            # Add timestamp
            indicators['timestamp'] = datetime.datetime.now().isoformat()
            
            # Calculate aggregate indicators
            indicators['aggregate'] = self._calculate_aggregate_indicators(indicators)
            
            self.logger.info("Calculated market indicators successfully")
        except Exception as e:
            self.logger.error(f"Error calculating market indicators: {str(e)}")
        
        return indicators
    
    def _calculate_yield_curve(self, market_data) -> Dict[str, Any]:
        """Calculate yield curve indicators."""
        result = {
            'current_values': {},
            'spreads': {},
            'is_inverted': False
        }
        
        tickers = self.indicators['yield_curve']['tickers']
        
        # Get current values for each maturity
        for maturity, ticker in tickers.items():
            if ticker in market_data and not market_data[ticker].empty:
                # Get most recent value
                current_value = market_data[ticker]['Close'].iloc[-1]
                result['current_values'][maturity] = current_value
        
        # Calculate spreads
        if '10y' in result['current_values'] and '2y' in result['current_values']:
            result['spreads']['10y_2y'] = result['current_values']['10y'] - result['current_values']['2y']
            result['is_inverted'] = result['spreads']['10y_2y'] < 0
        
        if '10y' in result['current_values'] and '3m' in result['current_values']:
            result['spreads']['10y_3m'] = result['current_values']['10y'] - result['current_values']['3m']
        
        # Calculate historical context
        if market_data.get(tickers.get('10y')) is not None and market_data.get(tickers.get('2y')) is not None:
            ten_year = market_data[tickers['10y']]['Close']
            two_year = market_data[tickers['2y']]['Close']
            
            if len(ten_year) > 0 and len(two_year) > 0:
                # Calculate spread over time
                spread_history = ten_year - two_year
                
                # Calculate statistics
                result['stats'] = {
                    'mean': spread_history.mean(),
                    'min': spread_history.min(),
                    'max': spread_history.max(),
                    'current': spread_history.iloc[-1],
                    'percentile': percentileofscore(spread_history, spread_history.iloc[-1])
                }
        
        # Calculate risk level based on curve inversion
        if result['is_inverted']:
            # Inverted yield curves are strong recession indicators
            result['risk_level'] = 4  # High risk
        elif result.get('spreads', {}).get('10y_2y', 1) < 0.2:
            # Flattening yield curve
            result['risk_level'] = 3  # Moderate risk
        else:
            # Normal yield curve
            result['risk_level'] = 1  # Low risk
        
        return result
    
    def _calculate_volatility(self, market_data) -> Dict[str, Any]:
        """Calculate volatility indicators."""
        result = {
            'current_values': {},
            'changes': {},
            'percentiles': {}
        }
        
        tickers = self.indicators['volatility']['tickers']
        
        for name, ticker in tickers.items():
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                
                # Current value
                current_value = df['Close'].iloc[-1]
                result['current_values'][name] = current_value
                
                # Calculate changes
                if len(df) > 1:
                    daily_change = df['Close'].pct_change().iloc[-1] * 100
                    result['changes'][f"{name}_daily"] = daily_change
                
                if len(df) > 5:
                    weekly_change = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
                    result['changes'][f"{name}_weekly"] = weekly_change
                
                # Calculate percentiles over the period
                all_values = df['Close'].dropna()
                percentile = percentileofscore(all_values, current_value)
                result['percentiles'][name] = percentile
        
        # VIX specific indicators
        if 'vix' in result['current_values']:
            vix_value = result['current_values']['vix']
            
            # Risk levels based on VIX
            if vix_value >= 30:
                result['risk_level'] = 4  # High risk - high volatility
            elif vix_value >= 20:
                result['risk_level'] = 3  # Moderate risk
            elif vix_value >= 15:
                result['risk_level'] = 2  # Low risk
            else:
                result['risk_level'] = 1  # Very low risk - complacency
        else:
            result['risk_level'] = 0  # Unknown
        
        return result
    
    def _calculate_credit_spreads(self, market_data) -> Dict[str, Any]:
        """Calculate credit spread indicators."""
        result = {
            'current_values': {},
            'spreads': {},
            'changes': {}
        }
        
        tickers = self.indicators['credit_spreads']['tickers']
        
        # Calculate current values
        for name, ticker in tickers.items():
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                current_value = df['Close'].iloc[-1]
                result['current_values'][name] = current_value
        
        # Calculate spread between high yield and investment grade
        if 'high_yield' in result['current_values'] and 'investment_grade' in result['current_values']:
            # We're using ETF prices as a proxy, so the spread is inverted
            # When credit risk increases, high yield ETFs drop faster than investment grade
            spread = result['current_values']['investment_grade'] / result['current_values']['high_yield']
            result['spreads']['high_yield_investment_grade'] = spread
            
            # Calculate historical context
            if (tickers['high_yield'] in market_data and tickers['investment_grade'] in market_data and
                    not market_data[tickers['high_yield']].empty and not market_data[tickers['investment_grade']].empty):
                high_yield_prices = market_data[tickers['high_yield']]['Close']
                inv_grade_prices = market_data[tickers['investment_grade']]['Close']
                
                # Calculate historical spread
                historical_spread = inv_grade_prices / high_yield_prices
                
                # Calculate statistics
                result['stats'] = {
                    'mean': historical_spread.mean(),
                    'min': historical_spread.min(),
                    'max': historical_spread.max(),
                    'current': historical_spread.iloc[-1],
                    'percentile': percentileofscore(historical_spread, historical_spread.iloc[-1])
                }
                
                # Calculate risk level based on spread percentile
                percentile = result['stats']['percentile']
                if percentile > 80:
                    result['risk_level'] = 4  # High risk - wide spreads
                elif percentile > 60:
                    result['risk_level'] = 3  # Moderate risk
                elif percentile > 40:
                    result['risk_level'] = 2  # Low risk
                else:
                    result['risk_level'] = 1  # Very low risk
        else:
            result['risk_level'] = 0  # Unknown
        
        return result
    
    def _calculate_liquidity(self, market_data) -> Dict[str, Any]:
        """Calculate liquidity indicators."""
        result = {
            'current_values': {},
            'spreads': {},
            'changes': {}
        }
        
        try:
            # TED Spread (3-month LIBOR - 3-month Treasury)
            # Note: Direct LIBOR data might not be available via yfinance
            # Using this as a placeholder - in a real system, we'd use a specific API for this
            liquidity_indicators = self.indicators['liquidity']['tickers']
            
            if 'ted_spread' in liquidity_indicators:
                components = liquidity_indicators['ted_spread']['components']
                if all(c in market_data and not market_data[c].empty for c in components):
                    # Calculate TED spread
                    # Placeholder calculation - would need actual LIBOR data
                    libor_proxy = market_data[components[1]]['Close'].iloc[-1]
                    treasury = market_data[components[0]]['Close'].iloc[-1]
                    result['spreads']['ted_spread'] = libor_proxy - treasury
                    
                    # Risk level based on TED spread
                    ted_spread = result['spreads']['ted_spread']
                    if ted_spread > 1.0:
                        result['risk_level'] = 5  # Very high risk - severe liquidity issues
                    elif ted_spread > 0.5:
                        result['risk_level'] = 4  # High risk
                    elif ted_spread > 0.3:
                        result['risk_level'] = 3  # Moderate risk
                    elif ted_spread > 0.2:
                        result['risk_level'] = 2  # Low risk
                    else:
                        result['risk_level'] = 1  # Very low risk
                else:
                    result['risk_level'] = 0  # Unknown
        except Exception as e:
            self.logger.error(f"Error calculating liquidity indicators: {str(e)}")
            result['risk_level'] = 0  # Unknown
        
        return result
    
    def _calculate_market_indices(self, market_data) -> Dict[str, Any]:
        """Calculate market index indicators."""
        result = {
            'current_values': {},
            'changes': {},
            'moving_averages': {},
            'technical_signals': {}
        }
        
        tickers = self.indicators['market_indices']['tickers']
        
        for name, ticker in tickers.items():
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                
                # Current value
                current_value = df['Close'].iloc[-1]
                result['current_values'][name] = current_value
                
                # Calculate changes
                if len(df) > 1:
                    daily_change = df['Close'].pct_change().iloc[-1] * 100
                    result['changes'][f"{name}_daily"] = daily_change
                
                if len(df) > 5:
                    weekly_change = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
                    result['changes'][f"{name}_weekly"] = weekly_change
                
                if len(df) > 20:
                    monthly_change = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100
                    result['changes'][f"{name}_monthly"] = monthly_change
                
                # Calculate moving averages
                if len(df) >= 50:
                    ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
                    result['moving_averages'][f"{name}_ma50"] = ma50
                    result['technical_signals'][f"{name}_above_ma50"] = current_value > ma50
                
                if len(df) >= 200:
                    ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
                    result['moving_averages'][f"{name}_ma200"] = ma200
                    result['technical_signals'][f"{name}_above_ma200"] = current_value > ma200
        
        # Calculate overall market direction
        if 'sp500_monthly' in result['changes']:
            monthly_change = result['changes']['sp500_monthly']
            
            # Risk level based on market direction
            if monthly_change < -10:
                result['risk_level'] = 4  # High risk - market correction
            elif monthly_change < -5:
                result['risk_level'] = 3  # Moderate risk - market pullback
            elif monthly_change < 0:
                result['risk_level'] = 2  # Low risk - slight decline
            else:
                result['risk_level'] = 1  # Very low risk - positive market
        else:
            result['risk_level'] = 0  # Unknown
        
        return result
    
    def _calculate_commodities(self, market_data) -> Dict[str, Any]:
        """Calculate commodity price indicators."""
        result = {
            'current_values': {},
            'changes': {}
        }
        
        tickers = self.indicators['commodities']['tickers']
        
        for name, ticker in tickers.items():
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                
                # Current value
                current_value = df['Close'].iloc[-1]
                result['current_values'][name] = current_value
                
                # Calculate changes
                if len(df) > 1:
                    daily_change = df['Close'].pct_change().iloc[-1] * 100
                    result['changes'][f"{name}_daily"] = daily_change
                
                if len(df) > 5:
                    weekly_change = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
                    result['changes'][f"{name}_weekly"] = weekly_change
                
                if len(df) > 20:
                    monthly_change = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100
                    result['changes'][f"{name}_monthly"] = monthly_change
        
        # Calculate gold/oil ratio as a stress indicator
        if 'gold' in result['current_values'] and 'oil' in result['current_values']:
            gold_oil_ratio = result['current_values']['gold'] / result['current_values']['oil']
            result['ratios'] = {'gold_oil': gold_oil_ratio}
            
            # Risk level based on gold price and gold/oil ratio
            # Gold tends to rise during crises
            if 'gold_monthly' in result['changes']:
                gold_change = result['changes']['gold_monthly']
                if gold_change > 10 and gold_oil_ratio > 30:
                    result['risk_level'] = 4  # High risk - flight to safety
                elif gold_change > 5 and gold_oil_ratio > 25:
                    result['risk_level'] = 3  # Moderate risk
                elif gold_change > 0:
                    result['risk_level'] = 2  # Low risk
                else:
                    result['risk_level'] = 1  # Very low risk
            else:
                result['risk_level'] = 0  # Unknown
        else:
            result['risk_level'] = 0  # Unknown
        
        return result
    
    def _calculate_aggregate_indicators(self, indicators) -> Dict[str, Any]:
        """Calculate aggregate indicators across all categories."""
        aggregate_result = {
            'timestamp': indicators['timestamp']
        }
        
        # Calculate weighted risk level
        weighted_risk = 0
        total_weight = 0
        
        for category, data in indicators.items():
            if category in ('timestamp', 'aggregate'):
                continue
                
            # Get category weight
            category_weight = self.indicators.get(category, {}).get('weight', 0)
            
            # Get category risk level
            category_risk = data.get('risk_level', 0)
            
            if category_risk > 0:
                weighted_risk += category_risk * category_weight
                total_weight += category_weight
        
        # Calculate final risk level
        if total_weight > 0:
            final_risk_index = weighted_risk / total_weight
            
            # Scale to 0-1 range
            final_risk_index = (final_risk_index - 1) / 4
            
            # Convert to risk level (1-5)
            final_risk_level = self._convert_to_risk_level(final_risk_index)
            
            aggregate_result['risk_index'] = final_risk_index
            aggregate_result['risk_level'] = final_risk_level
        else:
            aggregate_result['risk_index'] = 0
            aggregate_result['risk_level'] = 0
        
        return aggregate_result
    
    def _convert_to_risk_level(self, risk_index) -> int:
        """Convert risk index (0-1) to a 5-level risk scale."""
        thresholds = self.risk_thresholds
        
        if risk_index < thresholds[0]:
            return 1  # Very Low
        elif risk_index < thresholds[1]:
            return 2  # Low
        elif risk_index < thresholds[2]:
            return 3  # Moderate
        elif risk_index < thresholds[3]:
            return 4  # High
        else:
            return 5  # Very High
    
    def get_current_market_state(self) -> Dict[str, Any]:
        """Get the current market state with risk assessments."""
        try:
            # Collect current market data
            market_data = self._collect_market_data()
            
            # Calculate current indicators
            indicators = self._calculate_indicators(market_data)
            
            # Detect changes from historical data
            changes = self._detect_market_changes(indicators)
            
            # Update historical data
            self._update_historical_data(indicators)
            
            # Update last updated timestamp
            self.last_updated = datetime.datetime.now().isoformat()
            
            # Generate summary
            summary = self._generate_market_summary(indicators, changes)
            
            return {
                'indicators': indicators,
                'changes': changes,
                'summary': summary,
                'last_updated': self.last_updated
            }
        except Exception as e:
            self.logger.error(f"Error getting current market state: {str(e)}")
            return {
                'error': str(e),
                'last_updated': self.last_updated or datetime.datetime.now().isoformat()
            }
    
    def _detect_market_changes(self, current_indicators) -> Dict[str, Any]:
        """Detect significant changes in market indicators from historical data."""
        changes = {
            'detected': False,
            'categories': {}
        }
        
        if not self.historical_data.get('history'):
            return changes
        
        # Get most recent historical data
        historical = sorted(
            self.historical_data['history'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        if not historical:
            return changes
        
        # Compare current with historical for each category
        for category in current_indicators:
            if category in ('timestamp', 'aggregate'):
                continue
            
            category_changes = self._detect_category_changes(
                category,
                current_indicators.get(category, {}),
                historical
            )
            
            if category_changes.get('detected'):
                changes['detected'] = True
                changes['categories'][category] = category_changes
        
        # Detect changes in overall risk level
        if ('aggregate' in current_indicators and
                historical and 'data' in historical[0] and 'aggregate' in historical[0]['data']):
            current_risk = current_indicators['aggregate'].get('risk_level', 0)
            previous_risk = historical[0]['data']['aggregate'].get('risk_level', 0)
            
            if current_risk != previous_risk:
                changes['overall_risk_change'] = {
                    'previous': previous_risk,
                    'current': current_risk,
                    'direction': 'increased' if current_risk > previous_risk else 'decreased'
                }
        
        return changes
    
    def _detect_category_changes(self, category, current_data, historical) -> Dict[str, Any]:
        """Detect changes in specific category indicators."""
        changes = {
            'detected': False,
            'indicators': {}
        }
        
        # Skip if no current data
        if not current_data:
            return changes
        
        # Get historical data for this category
        historical_data = []
        for h in historical:
            if 'data' in h and category in h['data']:
                historical_data.append(h['data'][category])
                if len(historical_data) >= 10:  # Limit to last 10 data points
                    break
        
        if not historical_data:
            return changes
        
        # Check for category-specific changes
        if category == 'yield_curve':
            if ('is_inverted' in current_data and 
                    'is_inverted' in historical_data[0] and
                    current_data['is_inverted'] != historical_data[0]['is_inverted']):
                changes['detected'] = True
                changes['indicators']['inversion'] = {
                    'previous': historical_data[0]['is_inverted'],
                    'current': current_data['is_inverted']
                }
                
        elif category == 'volatility':
            if ('current_values' in current_data and 
                    'vix' in current_data['current_values'] and
                    'current_values' in historical_data[0] and
                    'vix' in historical_data[0]['current_values']):
                current_vix = current_data['current_values']['vix']
                previous_vix = historical_data[0]['current_values']['vix']
                
                # Detect large VIX changes
                pct_change = (current_vix - previous_vix) / previous_vix * 100
                if abs(pct_change) > 10:  # More than 10% change
                    changes['detected'] = True
                    changes['indicators']['vix'] = {
                        'previous': previous_vix,
                        'current': current_vix,
                        'pct_change': pct_change,
                        'direction': 'increased' if pct_change > 0 else 'decreased'
                    }
        
        elif category == 'market_indices':
            for index in ('sp500', 'nasdaq', 'russell'):
                if ('changes' in current_data and 
                        f'{index}_daily' in current_data['changes'] and
                        current_data['changes'][f'{index}_daily'] < -3):  # More than 3% daily drop
                    changes['detected'] = True
                    changes['indicators'][index] = {
                        'change': current_data['changes'][f'{index}_daily'],
                        'direction': 'decreased'
                    }
        
        return changes
    
    def _update_historical_data(self, current_indicators):
        """Update historical market data with current readings."""
        # Add current data to history
        self.historical_data.setdefault('history', []).append({
            'timestamp': current_indicators['timestamp'],
            'data': current_indicators
        })
        
        # Limit history size
        max_history = self.config.get('data', 'max_market_history', 365)  # Default 1 year
        if len(self.historical_data['history']) > max_history:
            # Sort by timestamp and keep most recent
            self.historical_data['history'] = sorted(
                self.historical_data['history'],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:max_history]
        
        # Save updated data
        self._save_historical_data()
    
    def _generate_market_summary(self, indicators, changes) -> Dict[str, Any]:
        """Generate a human-readable summary of market analysis."""
        # Risk level descriptions
        risk_descriptions = [
            "Unknown",
            "Very Low",
            "Low",
            "Moderate", 
            "High",
            "Very High"
        ]
        
        # Get aggregate risk level
        risk_level = indicators.get('aggregate', {}).get('risk_level', 0)
        risk_description = risk_descriptions[risk_level] if 0 <= risk_level < len(risk_descriptions) else "Unknown"
        
        # Build summary based on key indicators
        key_findings = []
        
        # Yield curve
        if indicators.get('yield_curve', {}).get('is_inverted'):
            key_findings.append("Treasury yield curve is inverted, historically a recession indicator")
        
        # Volatility
        if 'vix' in indicators.get('volatility', {}).get('current_values', {}):
            vix_value = indicators['volatility']['current_values']['vix']
            if vix_value > 30:
                key_findings.append(f"VIX at elevated level ({vix_value:.1f}), indicating high market anxiety")
        
        # Market indices
        market_indices = indicators.get('market_indices', {}).get('changes', {})
        for index in ('sp500_daily', 'nasdaq_daily', 'russell_daily'):
            if index in market_indices and market_indices[index] < -2:
                index_name = index.split('_')[0].upper()
                key_findings.append(f"{index_name} fell {abs(market_indices[index]):.1f}% today")
        
        # Credit spreads
        if 'high_yield_investment_grade' in indicators.get('credit_spreads', {}).get('spreads', {}):
            spread = indicators['credit_spreads']['spreads']['high_yield_investment_grade']
            if spread > 1.5:  # Using ETF ratio as proxy
                key_findings.append("Credit spreads are widening, indicating increased default concerns")
        
        # Changes in risk level
        if 'overall_risk_change' in changes:
            change = changes['overall_risk_change']
            prev_desc = risk_descriptions[change['previous']] if 0 <= change['previous'] < len(risk_descriptions) else "Unknown"
            key_findings.append(f"Market risk level has {change['direction']} from {prev_desc} to {risk_description}")
        
        # Generate summary text
        if key_findings:
            summary_text = f"Market risk level: {risk_description} ({risk_level}/5). Key findings: {'; '.join(key_findings)}."
        else:
            summary_text = f"Market risk level: {risk_description} ({risk_level}/5). No significant market anomalies detected."
        
        return {
            'text': summary_text,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'key_findings': key_findings,
            'timestamp': indicators['timestamp']
        }


# Utility function (normally from scipy.stats, included here for completeness)
def percentileofscore(a, score):
    """
    Calculate the percentile rank of a score relative to a list of scores.
    
    Parameters:
    -----------
    a : array-like
        Array of scores to which `score` is compared.
    score : float
        Score to compute percentile for.
        
    Returns:
    --------
    float : percentile rank of score (0-100)
    """
    a = np.asarray(a)
    n = len(a)
    if n == 0:
        return np.nan
    
    # Count number of scores below or equal to our score
    count = np.sum(a <= score)
    pct = (count / n) * 100.0
    return pct 