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

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types."""
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (set, tuple)):
                return list(obj)
            if hasattr(obj, '__dict__'):
                # For custom objects, convert to dict but with a safety net
                try:
                    return {k: v for k, v in obj.__dict__.items() 
                           if not k.startswith('_') and not callable(v)}
                except:
                    return str(obj)
            return super(CustomJSONEncoder, self).default(obj)
        except Exception:
            # Last resort, convert to string
            return str(obj)

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
        self.risk_thresholds = self.config.get('market', {}).get('risk_thresholds', [0.2, 0.4, 0.6, 0.8])
        
        # Market indicators to track
        market_config = self.config.get('market', {})
        self.indicators = market_config.get('indicators', {
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
            data_config = self.config.get('data', {})
            historical_data_path = data_config.get('historical_market_path', 'data/historical_market.json')
            
            if os.path.exists(historical_data_path):
                try:
                    with open(historical_data_path, 'r', encoding='utf-8') as f:
                        self.historical_data = json.load(f)
                    self.logger.info(f"Loaded historical market data from {historical_data_path}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Corrupted historical market data file: {str(e)}. Creating new baseline data.")
                    # Backup the corrupted file
                    backup_path = f"{historical_data_path}.bak"
                    try:
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                        os.rename(historical_data_path, backup_path)
                        self.logger.info(f"Backed up corrupted file to {backup_path}")
                    except Exception as backup_error:
                        self.logger.error(f"Failed to backup corrupted data file: {str(backup_error)}")
                    
                    # Reset historical data
                    self.historical_data = None
            
            # Initialize baseline market data if needed
            if not self.historical_data:
                self.logger.warning("No historical market data found or failed to load, initializing baseline")
                self._initialize_baseline_data()
                
            self.logger.info("Market monitor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize market monitor: {str(e)}")
            # Initialize with empty data rather than crashing the entire application
            self.historical_data = {
                'baseline': {},
                'history': []
            }
            # Don't re-raise the exception to allow the application to continue
    
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
            if not self.historical_data:
                self.logger.warning("No historical data to save")
                return
                
            data_config = self.config.get('data', {})
            historical_data_path = data_config.get('historical_market_path', 'data/historical_market.json')
            os.makedirs(os.path.dirname(historical_data_path), exist_ok=True)
            
            # Use atomic writing to prevent corruption:
            # First write to a temp file, then rename it
            temp_path = f"{historical_data_path}.tmp"
            
            # Serialize the data
            try:
                serialized_data = json.dumps(self.historical_data, indent=2, cls=CustomJSONEncoder)
                
                # Write to temp file
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(serialized_data)
                
                # If we got here, the write was successful, so rename to final path
                if os.path.exists(historical_data_path):
                    os.remove(historical_data_path)
                os.rename(temp_path, historical_data_path)
                
                self.logger.info("Historical market data saved successfully")
            except Exception as serialize_error:
                self.logger.error(f"Error serializing historical data: {str(serialize_error)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        except Exception as e:
            self.logger.error(f"Failed to save historical market data: {str(e)}")
    
    def _collect_market_data(self) -> Dict[str, Any]:
        """Collect current market data from various sources."""
        market_data = {}
        
        try:
            # Define replacement tickers for problematic ones
            ticker_replacements = {
                'USD3MTD156N': '^IRX',  # Replace with 13-week Treasury Bill
                '^UST2Y': '^TYX'       # Replace with 30-Year Treasury Yield
            }
            
            # Get all unique tickers to fetch
            tickers = set()
            for indicator_group in self.indicators.values():
                if not indicator_group.get('enabled', True):
                    continue
                
                for ticker_key, ticker_val in indicator_group.get('tickers', {}).items():
                    if isinstance(ticker_val, str):
                        # Use replacement if ticker is problematic
                        ticker_to_use = ticker_replacements.get(ticker_val, ticker_val)
                        tickers.add(ticker_to_use)
                    elif isinstance(ticker_val, dict) and 'components' in ticker_val:
                        # Apply replacements to components
                        components = [ticker_replacements.get(t, t) for t in ticker_val['components']]
                        tickers.update(components)
            
            # Determine date range
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            market_config = self.config.get('market', {})
            days_back = market_config.get('days_back', 30)
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
                    daily_change = df['Close'].pct_change(fill_method=None).iloc[-1] * 100
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
                    daily_change = df['Close'].pct_change(fill_method=None).iloc[-1] * 100
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
                    daily_change = df['Close'].pct_change(fill_method=None).iloc[-1] * 100
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
        """Calculate aggregate risk indicators from all indicator categories."""
        aggregate_result = {
            'risk_indices': {},
            'risk_levels': {}
        }
        
        try:
            # Calculate weighted risk index for each enabled indicator
            total_weight = 0
            weighted_risk_sum = 0
            
            for category in self.indicators:
                if not category in indicators:
                    continue
                
                data = indicators[category]
                
                # Get category weight
                category_weight = self.indicators.get(category, {}).get('weight', 0)
                
                # Get risk level for category
                category_risk = data.get('risk_level', 0)
                
                if category_weight > 0 and category_risk > 0:
                    # Store individual risk level
                    aggregate_result['risk_levels'][category] = category_risk
                    
                    # Convert risk level (1-5) to risk index (0-1)
                    category_risk_index = (category_risk - 1) / 4
                    
                    # Store individual risk index
                    aggregate_result['risk_indices'][category] = category_risk_index
                    
                    # Add to weighted sum
                    weighted_risk_sum += category_risk_index * category_weight
                    total_weight += category_weight
            
            # Calculate final risk index
            if total_weight > 0:
                final_risk_index = weighted_risk_sum / total_weight
            else:
                final_risk_index = 0
            
            # Scale to 0-1 range
            final_risk_index = (final_risk_index - 1) / 4
            
            # Convert to risk level (1-5)
            final_risk_level = self._convert_to_risk_level(final_risk_index)
            
            aggregate_result['risk_index'] = final_risk_index
            aggregate_result['risk_level'] = final_risk_level
        except Exception as e:
            self.logger.error(f"Error calculating aggregate indicators: {str(e)}")
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
        """Detect changes in market indicators from historical data."""
        changes = {
            'categories': {},
            'overall': {
                'direction': 'stable',
                'magnitude': 0
            },
            'alerts': []
        }
        
        try:
            # Check if we have historical data
            if not self.historical_data or 'history' not in self.historical_data or not self.historical_data['history']:
                return changes
            
            # Get most recent historical data point
            historical = sorted(self.historical_data['history'], key=lambda x: x['timestamp'], reverse=True)
            
            # Overall risk level change - check if aggregate exists
            if 'aggregate' in current_indicators and 'data' in historical[0] and 'aggregate' in historical[0]['data']:
                current_risk = current_indicators['aggregate'].get('risk_level', 0)
                previous_risk = historical[0]['data']['aggregate'].get('risk_level', 0)
                
                # Calculate overall direction and magnitude
                if current_risk > previous_risk:
                    changes['overall']['direction'] = 'deteriorating'
                    changes['overall']['magnitude'] = current_risk - previous_risk
                elif current_risk < previous_risk:
                    changes['overall']['direction'] = 'improving'
                    changes['overall']['magnitude'] = previous_risk - current_risk
                
                # Generate alert if risk level increased significantly
                if current_risk - previous_risk >= 1:
                    changes['alerts'].append({
                        'type': 'risk_increase',
                        'message': f"Overall risk level increased from {previous_risk} to {current_risk}",
                        'severity': min(current_risk, 5)
                    })
            
            # Check each category for changes
            for category in current_indicators:
                if category != 'timestamp' and category != 'aggregate' and category in historical[0]['data']:
                    category_changes = self._detect_category_changes(
                        category, 
                        current_indicators[category], 
                        historical[0]['data'][category]
                    )
                    
                    if category_changes:
                        changes['categories'][category] = category_changes
            
        except Exception as e:
            self.logger.error(f"Error detecting market changes: {str(e)}")
        
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
        data_config = self.config.get('data', {})
        max_history = data_config.get('max_market_history', 365)  # Default 1 year
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
        """Generate a human-readable summary of market indicators."""
        summary = {
            'text': "",
            'key_findings': [],
            'risk_level': 0,
            'direction': 'stable'
        }
        
        try:
            # Risk level descriptions
            risk_descriptions = [
                "Very Low",
                "Low",
                "Moderate", 
                "High",
                "Very High"
            ]
            
            # Get overall risk level
            risk_level = indicators.get('aggregate', {}).get('risk_level', 0)
            
            if risk_level > 0 and risk_level <= 5:
                risk_description = risk_descriptions[risk_level-1]
                summary['risk_level'] = risk_level
            else:
                risk_description = "Unknown"
                summary['risk_level'] = 0
            
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
            
            summary['text'] = summary_text
            summary['risk_description'] = risk_description
            summary['key_findings'] = key_findings
            summary['risk_level'] = risk_level
            summary['direction'] = changes['overall']['direction']
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            summary['text'] = "Unable to generate market summary due to an error."
        
        return summary


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