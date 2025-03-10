#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processor: Handles data transformation, cleaning, and preprocessing for the Crisis Sentinel system.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Union, Optional
import datetime


class DataProcessor:
    """
    Handles data transformation, cleaning, and preprocessing for different data sources.
    
    This utility class provides common data processing functions used across
    sentiment analysis and market monitoring components.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text data by removing special characters, URLs, etc.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers (keep letters, spaces, and basic punctuation)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_time_series(self, data: pd.Series) -> pd.Series:
        """
        Normalize time series data to the range [0, 1].
        
        Args:
            data: Time series data to normalize
            
        Returns:
            Normalized time series
        """
        if data.empty:
            return data
        
        min_val = data.min()
        max_val = data.max()
        
        if max_val == min_val:
            return pd.Series(0.5, index=data.index)
        
        return (data - min_val) / (max_val - min_val)
    
    def calculate_z_scores(self, data: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate z-scores for time series data.
        
        Args:
            data: Time series data
            window: Rolling window size for z-score calculation
            
        Returns:
            Z-scores series
        """
        if data.empty or len(data) < window:
            return pd.Series(index=data.index)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        # Calculate z-scores
        z_scores = (data - rolling_mean) / rolling_std
        
        return z_scores
    
    def detect_outliers(self, data: pd.Series, threshold: float = 2.0) -> pd.Series:
        """
        Detect outliers in time series data using z-scores.
        
        Args:
            data: Time series data
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean mask of outliers
        """
        z_scores = self.calculate_z_scores(data)
        
        # Mark values with absolute z-scores above threshold as outliers
        outliers = abs(z_scores) > threshold
        
        return outliers
    
    def merge_dataframes(self, dataframes: List[pd.DataFrame], on: str = 'date') -> pd.DataFrame:
        """
        Merge multiple dataframes on a common column.
        
        Args:
            dataframes: List of dataframes to merge
            on: Common column to merge on
            
        Returns:
            Merged dataframe
        """
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Start with the first dataframe
        result = dataframes[0]
        
        # Merge with the rest
        for df in dataframes[1:]:
            result = pd.merge(result, df, on=on, how='outer')
        
        return result
    
    def resample_time_series(self, data: pd.DataFrame, date_column: str, 
                             freq: str = 'D', agg_func: Dict[str, str] = None) -> pd.DataFrame:
        """
        Resample time series data to a specified frequency.
        
        Args:
            data: DataFrame containing time series data
            date_column: Column containing dates
            freq: Frequency to resample to ('D' for daily, 'W' for weekly, etc.)
            agg_func: Aggregation functions for each column
            
        Returns:
            Resampled dataframe
        """
        if data.empty:
            return data
        
        # Set date column as index
        df = data.copy()
        if date_column in df.columns:
            df.set_index(date_column, inplace=True)
        
        # Default aggregation: mean for numeric, first for others
        if agg_func is None:
            agg_func = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    agg_func[col] = 'mean'
                else:
                    agg_func[col] = 'first'
        
        # Resample
        resampled = df.resample(freq).agg(agg_func)
        
        return resampled
    
    def calculate_correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns in a dataframe.
        
        Args:
            data: DataFrame with numeric columns
            
        Returns:
            Correlation matrix
        """
        if data.empty:
            return pd.DataFrame()
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        return corr_matrix
    
    def calculate_rolling_correlation(self, series1: pd.Series, series2: pd.Series, 
                                     window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation between two time series.
        
        Args:
            series1: First time series
            series2: Second time series
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        if series1.empty or series2.empty or len(series1) < window or len(series2) < window:
            return pd.Series()
        
        # Ensure indexes match
        common_index = series1.index.intersection(series2.index)
        s1 = series1.loc[common_index]
        s2 = series2.loc[common_index]
        
        # Calculate rolling correlation
        rolling_corr = s1.rolling(window=window).corr(s2)
        
        return rolling_corr
    
    def preprocess_news_data(self, news_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess news data into a structured format.
        
        Args:
            news_data: List of news article dictionaries
            
        Returns:
            Preprocessed DataFrame
        """
        if not news_data:
            return pd.DataFrame()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(news_data)
            
            # Convert published_at to datetime
            if 'published_at' in df.columns:
                df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            
            # Clean text columns
            text_columns = ['title', 'description', 'content']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: self.clean_text(x) if x else "")
            
            # Create a combined text column for analysis
            df['combined_text'] = df['title'].fillna("") + ". " + df['description'].fillna("")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error preprocessing news data: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_social_media_data(self, social_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess social media data from different platforms.
        
        Args:
            social_data: Dictionary of social media data by platform
            
        Returns:
            Dictionary of preprocessed DataFrames by platform
        """
        result = {}
        
        try:
            # Process Twitter data
            if 'twitter' in social_data and social_data['twitter']:
                twitter_df = pd.DataFrame(social_data['twitter'])
                if 'created_at' in twitter_df.columns:
                    twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')
                if 'text' in twitter_df.columns:
                    twitter_df['clean_text'] = twitter_df['text'].apply(self.clean_text)
                result['twitter'] = twitter_df
            
            # Process Reddit data
            if 'reddit' in social_data and social_data['reddit']:
                reddit_df = pd.DataFrame(social_data['reddit'])
                if 'created_utc' in reddit_df.columns:
                    reddit_df['created_at'] = pd.to_datetime(reddit_df['created_utc'], unit='s', errors='coerce')
                
                # Clean text columns
                if 'title' in reddit_df.columns:
                    reddit_df['clean_title'] = reddit_df['title'].apply(self.clean_text)
                if 'text' in reddit_df.columns:
                    reddit_df['clean_text'] = reddit_df['text'].apply(self.clean_text)
                
                # Create a combined text column
                reddit_df['combined_text'] = reddit_df['title'].fillna("") + ". " + reddit_df['text'].fillna("")
                
                result['reddit'] = reddit_df
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error preprocessing social media data: {str(e)}")
            return {}
    
    def calculate_sentiment_change_rate(self, sentiment_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the rate of change in sentiment over time.
        
        Args:
            sentiment_history: List of historical sentiment data points
            
        Returns:
            Dictionary with change rate statistics
        """
        if not sentiment_history or len(sentiment_history) < 2:
            return {
                'short_term_rate': 0,
                'medium_term_rate': 0,
                'long_term_rate': 0,
                'acceleration': 0
            }
        
        try:
            # Convert to DataFrame
            data = []
            for item in sentiment_history:
                timestamp = pd.to_datetime(item.get('timestamp'))
                score = item.get('data', {}).get('aggregate', {}).get('score', default=0)
                data.append({'timestamp': timestamp, 'score': score})
            
            df = pd.DataFrame(data)
            df.sort_values('timestamp', inplace=True)
            
            # Calculate rates of change for different time periods
            short_term = 7  # 1 week
            medium_term = 30  # 1 month
            long_term = min(90, len(df) - 1)  # 3 months or all available data
            
            # Get current and historical values
            current = df['score'].iloc[-1]
            short_term_past = df['score'].iloc[-min(short_term + 1, len(df))]
            medium_term_past = df['score'].iloc[-min(medium_term + 1, len(df))]
            long_term_past = df['score'].iloc[-min(long_term + 1, len(df))]
            
            # Calculate change rates
            short_term_rate = (current - short_term_past) / max(abs(short_term_past), 0.01) * 100
            medium_term_rate = (current - medium_term_past) / max(abs(medium_term_past), 0.01) * 100
            long_term_rate = (current - long_term_past) / max(abs(long_term_past), 0.01) * 100
            
            # Calculate acceleration (change in rate of change)
            if len(df) > short_term * 2:
                previous_rate = (short_term_past - df['score'].iloc[-min(short_term * 2 + 1, len(df))]) / max(abs(df['score'].iloc[-min(short_term * 2 + 1, len(df))]), 0.01) * 100
                acceleration = short_term_rate - previous_rate
            else:
                acceleration = 0
            
            return {
                'short_term_rate': short_term_rate,
                'medium_term_rate': medium_term_rate,
                'long_term_rate': long_term_rate,
                'acceleration': acceleration
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating sentiment change rate: {str(e)}")
            return {
                'short_term_rate': 0,
                'medium_term_rate': 0,
                'long_term_rate': 0,
                'acceleration': 0
            }

    def _process_historical_sentiment(self, raw_data):
        """Process raw historical sentiment data into trend charts."""
        result = {
            'trends': {},
            'anomalies': []
        }
        
        if not raw_data or 'history' not in raw_data:
            return result
        
        # Get time series data
        time_series = []
        for item in raw_data['history']:
            if 'timestamp' in item and 'data' in item:
                # Convert timestamp to datetime
                try:
                    timestamp = datetime.datetime.fromisoformat(item['timestamp'])
                    
                    # Get aggregate score (normalize to -1 to 1 range)
                    score = item.get('data', {}).get('aggregate', {}).get('score', 0)
                    
                    time_series.append({
                        'timestamp': timestamp,
                        'score': score
                    })
                except ValueError:
                    continue
        
        # Sort by timestamp
        time_series.sort(key=lambda x: x['timestamp'])
        
        # Generate trend data
        if time_series:
            dates = [item['timestamp'].strftime('%Y-%m-%d') for item in time_series]
            scores = [item['score'] for item in time_series]
            
            result['trends']['sentiment'] = {
                'dates': dates,
                'values': scores
            }
            
            # Detect anomalies (simple threshold-based for now)
            threshold = 0.5
            for i, score in enumerate(scores):
                if abs(score) > threshold:
                    result['anomalies'].append({
                        'date': dates[i],
                        'score': score,
                        'direction': 'positive' if score > 0 else 'negative',
                        'magnitude': abs(score)
                    })
        
        return result 