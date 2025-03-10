#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analyzer: Analyzes news, social media, and political discourse to identify 
sentiment changes that may indicate upcoming crises.
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# NLP and sentiment analysis libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# APIs for data collection
import tweepy
import praw
from newsapi import NewsApiClient

# Local imports
from app.utils.data_processor import DataProcessor


class SentimentAnalyzer:
    """
    Analyzes sentiment across multiple sources to detect potential crisis signals.
    
    The analyzer collects data from news sources, social media platforms, and transcripts
    of political speeches to identify shifts in public sentiment that may indicate
    upcoming financial, geopolitical, or institutional crises.
    """
    
    def __init__(self, config):
        """Initialize the SentimentAnalyzer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor()
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Initialize API clients
        self._initialize_api_clients()
        
        # Historical sentiment data
        self.historical_sentiment = {}
        self.sentiment_indicators = []
        
        # Risk thresholds
        sentiment_config = self.config.get('sentiment', {})
        self.risk_thresholds = sentiment_config.get('risk_thresholds', [0.2, 0.4, 0.6, 0.8])
        
    def _initialize_nlp(self):
        """Initialize NLP tools for sentiment analysis."""
        try:
            # Download necessary NLTK data
            nltk.download('vader_lexicon', quiet=True)
            self.vader = SentimentIntensityAnalyzer()
            
            # Try to initialize transformers pipeline
            try:
                transformer_model = self.config.get('sentiment', {}).get('transformer_model', 'distilbert-base-uncased-finetuned-sst-2-english')
                use_gpu = self.config.get('general', {}).get('use_gpu', False)
                
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=transformer_model,
                    device=0 if use_gpu else -1
                )
                self.logger.info("Transformer pipeline initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize transformer pipeline: {str(e)}")
                self.logger.warning("Using VADER sentiment analysis only")
                self.sentiment_pipeline = None
            
            self.logger.info("NLP components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP components: {str(e)}")
            raise
    
    def _initialize_api_clients(self):
        """Initialize API clients for data collection."""
        try:
            apis_config = self.config.get('apis', {})
            
            # Twitter API
            twitter_config = apis_config.get('twitter', {})
            self.twitter_client = tweepy.Client(
                bearer_token=twitter_config.get('bearer_token', ''),
                consumer_key=twitter_config.get('consumer_key', ''),
                consumer_secret=twitter_config.get('consumer_secret', ''),
                access_token=twitter_config.get('access_token', ''),
                access_token_secret=twitter_config.get('access_token_secret', '')
            )
            
            # Reddit API
            reddit_config = apis_config.get('reddit', {})
            self.reddit_client = praw.Reddit(
                client_id=reddit_config.get('client_id', ''),
                client_secret=reddit_config.get('client_secret', ''),
                user_agent=reddit_config.get('user_agent', 'crisis_sentinel/1.0')
            )
            
            # News API
            news_config = apis_config.get('news', {})
            self.news_client = NewsApiClient(api_key=news_config.get('api_key', ''))
            
            self.logger.info("API clients initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {str(e)}")
            raise
    
    def initialize(self):
        """Initialize the sentiment analyzer with historical data."""
        try:
            # Load historical sentiment data
            data_config = self.config.get('data', {})
            historical_data_path = data_config.get('historical_sentiment_path', 'data/historical_sentiment.json')
            if os.path.exists(historical_data_path):
                with open(historical_data_path, 'r', encoding='utf-8') as f:
                    self.historical_sentiment = json.load(f)
                    
            # Initialize baseline sentiment
            if not self.historical_sentiment:
                self.logger.warning("No historical sentiment data found, initializing baseline")
                self._initialize_baseline_sentiment()
                
            self.logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise
    
    def _initialize_baseline_sentiment(self):
        """Initialize baseline sentiment from current data."""
        # Collect current data
        news_data = self._collect_news_data()
        social_media_data = self._collect_social_media_data()
        
        # Analyze current sentiment as baseline
        baseline = self._analyze_raw_sentiment(news_data, social_media_data)
        
        # Store as historical
        self.historical_sentiment = {
            'baseline': baseline,
            'history': [
                {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'data': baseline
                }
            ]
        }
        
        # Save to disk
        self._save_historical_data()
    
    def _save_historical_data(self):
        """Save historical sentiment data to disk."""
        try:
            data_config = self.config.get('data', {})
            historical_data_path = data_config.get('historical_sentiment_path', 'data/historical_sentiment.json')
            os.makedirs(os.path.dirname(historical_data_path), exist_ok=True)
            
            with open(historical_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.historical_sentiment, f, indent=2)
                
            self.logger.info("Historical sentiment data saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save historical sentiment data: {str(e)}")
    
    def _collect_news_data(self) -> List[Dict[str, Any]]:
        """Collect news data from various sources."""
        news_data = []
        
        try:
            # Get sources from config
            sentiment_config = self.config.get('sentiment', {})
            sources = sentiment_config.get('news_sources', [])
            keywords = sentiment_config.get('crisis_keywords', [])
            days_back = sentiment_config.get('days_back', 7)
            
            # Calculate date range
            to_date = datetime.datetime.now()
            from_date = to_date - datetime.timedelta(days=days_back)
            
            # Format dates for API
            to_date_str = to_date.strftime('%Y-%m-%d')
            from_date_str = from_date.strftime('%Y-%m-%d')
            
            # Query for multiple keywords
            for keyword in keywords:
                response = self.news_client.get_everything(
                    q=keyword,
                    sources=','.join(sources),
                    from_param=from_date_str,
                    to=to_date_str,
                    language='en',
                    sort_by='relevancy',
                    page_size=100
                )
                
                if response['status'] == 'ok':
                    for article in response['articles']:
                        # Process and add to news data
                        processed_article = {
                            'source': article['source']['name'],
                            'title': article['title'],
                            'description': article['description'],
                            'content': article['content'],
                            'url': article['url'],
                            'published_at': article['publishedAt'],
                            'keyword': keyword
                        }
                        news_data.append(processed_article)
                        
            self.logger.info(f"Collected {len(news_data)} news articles")
        except Exception as e:
            self.logger.error(f"Error collecting news data: {str(e)}")
        
        return news_data
    
    def _collect_social_media_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from social media platforms."""
        social_data = {
            'twitter': [],
            'reddit': []
        }
        
        try:
            # Twitter data collection
            sentiment_config = self.config.get('sentiment', {})
            twitter_keywords = sentiment_config.get('twitter_keywords', [])
            for keyword in twitter_keywords:
                tweets = self.twitter_client.search_recent_tweets(
                    query=keyword,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics', 'lang']
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        if tweet.lang == 'en':  # Filter English tweets
                            social_data['twitter'].append({
                                'id': tweet.id,
                                'text': tweet.text,
                                'created_at': tweet.created_at.isoformat(),
                                'metrics': tweet.public_metrics,
                                'keyword': keyword
                            })
            
            # Reddit data collection
            reddit_subreddits = sentiment_config.get('reddit_subreddits', [])
            limit = sentiment_config.get('reddit_limit', 100)
            
            for subreddit_name in reddit_subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Get top posts
                for post in subreddit.hot(limit=limit):
                    social_data['reddit'].append({
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'created_utc': post.created_utc,
                        'score': post.score,
                        'subreddit': subreddit_name
                    })
            
            self.logger.info(f"Collected {len(social_data['twitter'])} tweets and {len(social_data['reddit'])} Reddit posts")
        except Exception as e:
            self.logger.error(f"Error collecting social media data: {str(e)}")
        
        return social_data
    
    def _analyze_raw_sentiment(self, news_data, social_media_data) -> Dict[str, Any]:
        """Analyze raw sentiment from collected data."""
        results = {
            'news': self._analyze_news_sentiment(news_data),
            'twitter': self._analyze_twitter_sentiment(social_media_data['twitter']),
            'reddit': self._analyze_reddit_sentiment(social_media_data['reddit']),
            'aggregate': {},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Calculate aggregate sentiment
        results['aggregate'] = self._calculate_aggregate_sentiment(results)
        
        return results
    
    def _analyze_news_sentiment(self, news_data) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        if not news_data:
            return {'average_score': 0, 'scores': [], 'sources': {}, 'keywords': {}}
        
        results = {
            'scores': [],
            'sources': {},
            'keywords': {}
        }
        
        for article in news_data:
            # Combine title and description for better analysis
            text = f"{article['title']}. {article['description'] or ''}"
            
            # Get VADER sentiment
            vader_score = self.vader.polarity_scores(text)
            
            # Get transformer sentiment if available
            if self.sentiment_pipeline:
                transformer_result = self.sentiment_pipeline(text)[0]
                combined_score = self._normalize_sentiment_score(vader_score, transformer_result)
                
                sentiment_score = {
                    'vader': vader_score,
                    'transformer': {
                        'label': transformer_result['label'],
                        'score': transformer_result['score']
                    },
                    'combined_score': combined_score,
                    'source': article['source'],
                    'keyword': article['keyword'],
                    'url': article['url'],
                    'title': article['title'],
                    'published_at': article['published_at']
                }
            else:
                # Use only VADER if transformer is not available
                combined_score = vader_score['compound']
                
                sentiment_score = {
                    'vader': vader_score,
                    'transformer': None,
                    'combined_score': combined_score,
                    'source': article['source'],
                    'keyword': article['keyword'],
                    'url': article['url'],
                    'title': article['title'],
                    'published_at': article['published_at']
                }
            
            results['scores'].append(sentiment_score)
            
            # Aggregate by source
            if article['source'] not in results['sources']:
                results['sources'][article['source']] = []
            results['sources'][article['source']].append(sentiment_score['combined_score'])
            
            # Aggregate by keyword
            if article['keyword'] not in results['keywords']:
                results['keywords'][article['keyword']] = []
            results['keywords'][article['keyword']].append(sentiment_score['combined_score'])
        
        # Calculate averages
        results['average_score'] = np.mean([s['combined_score'] for s in results['scores']])
        
        # Calculate average by source
        for source in results['sources']:
            results['sources'][source] = np.mean(results['sources'][source])
        
        # Calculate average by keyword
        for keyword in results['keywords']:
            results['keywords'][keyword] = np.mean(results['keywords'][keyword])
        
        return results
    
    def _analyze_twitter_sentiment(self, tweets) -> Dict[str, Any]:
        """Analyze sentiment from Twitter data."""
        if not tweets:
            return {'average_score': 0, 'scores': [], 'keywords': {}}
        
        results = {
            'scores': [],
            'keywords': {}
        }
        
        for tweet in tweets:
            # Get VADER sentiment
            vader_score = self.vader.polarity_scores(tweet['text'])
            
            # Get transformer sentiment if available
            if self.sentiment_pipeline:
                transformer_result = self.sentiment_pipeline(tweet['text'])[0]
                combined_score = self._normalize_sentiment_score(vader_score, transformer_result)
                
                sentiment_score = {
                    'vader': vader_score,
                    'transformer': {
                        'label': transformer_result['label'],
                        'score': transformer_result['score']
                    },
                    'combined_score': combined_score,
                    'metrics': tweet['metrics'],
                    'keyword': tweet['keyword'],
                    'created_at': tweet['created_at']
                }
            else:
                # Use only VADER if transformer is not available
                combined_score = vader_score['compound']
                
                sentiment_score = {
                    'vader': vader_score,
                    'transformer': None,
                    'combined_score': combined_score,
                    'metrics': tweet['metrics'],
                    'keyword': tweet['keyword'],
                    'created_at': tweet['created_at']
                }
            
            results['scores'].append(sentiment_score)
            
            # Aggregate by keyword
            if tweet['keyword'] not in results['keywords']:
                results['keywords'][tweet['keyword']] = []
            results['keywords'][tweet['keyword']].append(sentiment_score['combined_score'])
        
        # Calculate averages
        results['average_score'] = np.mean([s['combined_score'] for s in results['scores']])
        
        # Calculate average by keyword
        for keyword in results['keywords']:
            results['keywords'][keyword] = np.mean(results['keywords'][keyword])
        
        return results
    
    def _analyze_reddit_sentiment(self, posts) -> Dict[str, Any]:
        """Analyze sentiment from Reddit data."""
        if not posts:
            return {'average_score': 0, 'scores': [], 'subreddits': {}}
        
        results = {
            'scores': [],
            'subreddits': {}
        }
        
        for post in posts:
            # Combine title and text for analysis
            text = f"{post['title']}. {post['text'] or ''}"
            
            # Get VADER sentiment
            vader_score = self.vader.polarity_scores(text)
            
            # Get transformer sentiment if available
            if self.sentiment_pipeline:
                transformer_result = self.sentiment_pipeline(text)[0]
                combined_score = self._normalize_sentiment_score(vader_score, transformer_result)
                
                sentiment_score = {
                    'vader': vader_score,
                    'transformer': {
                        'label': transformer_result['label'],
                        'score': transformer_result['score']
                    },
                    'combined_score': combined_score,
                    'score': post['score'],
                    'subreddit': post['subreddit'],
                    'created_utc': post['created_utc']
                }
            else:
                # Use only VADER if transformer is not available
                combined_score = vader_score['compound']
                
                sentiment_score = {
                    'vader': vader_score,
                    'transformer': None,
                    'combined_score': combined_score,
                    'score': post['score'],
                    'subreddit': post['subreddit'],
                    'created_utc': post['created_utc']
                }
            
            results['scores'].append(sentiment_score)
            
            # Aggregate by subreddit
            if post['subreddit'] not in results['subreddits']:
                results['subreddits'][post['subreddit']] = []
            results['subreddits'][post['subreddit']].append(sentiment_score['combined_score'])
        
        # Calculate averages
        results['average_score'] = np.mean([s['combined_score'] for s in results['scores']])
        
        # Calculate average by subreddit
        for subreddit in results['subreddits']:
            results['subreddits'][subreddit] = np.mean(results['subreddits'][subreddit])
        
        return results
    
    def _normalize_sentiment_score(self, vader_score, transformer_result=None) -> float:
        """Normalize and combine sentiment scores from different algorithms."""
        # If transformer result is not available, just use VADER
        if transformer_result is None:
            return vader_score['compound']
            
        # Weight for each algorithm
        sentiment_config = self.config.get('sentiment', {})
        vader_weight = sentiment_config.get('vader_weight', 0.4)
        transformer_weight = sentiment_config.get('transformer_weight', 0.6)
        
        # Normalize vader score (-1 to 1 range)
        normalized_vader = vader_score['compound']
        
        # Normalize transformer score (0 to 1 range, convert to -1 to 1)
        normalized_transformer = (
            transformer_result['score'] if transformer_result['label'] == 'POSITIVE' 
            else -transformer_result['score']
        )
        
        # Weighted average
        return (vader_weight * normalized_vader) + (transformer_weight * normalized_transformer)
    
    def _calculate_aggregate_sentiment(self, sentiment_results) -> Dict[str, Any]:
        """Calculate aggregate sentiment across all sources."""
        # Weights for each source
        sentiment_config = self.config.get('sentiment', {})
        weights = sentiment_config.get('source_weights', {
            'news': 0.5,
            'twitter': 0.3,
            'reddit': 0.2
        })
        
        # Calculate weighted average
        aggregate_score = (
            weights['news'] * sentiment_results['news']['average_score'] +
            weights['twitter'] * sentiment_results['twitter']['average_score'] +
            weights['reddit'] * sentiment_results['reddit']['average_score']
        )
        
        # Calculate standard deviation of scores
        all_scores = []
        if sentiment_results['news']['scores']:
            all_scores.extend([s['combined_score'] for s in sentiment_results['news']['scores']])
        if sentiment_results['twitter']['scores']:
            all_scores.extend([s['combined_score'] for s in sentiment_results['twitter']['scores']])
        if sentiment_results['reddit']['scores']:
            all_scores.extend([s['combined_score'] for s in sentiment_results['reddit']['scores']])
        
        std_dev = np.std(all_scores) if all_scores else 0
        
        return {
            'score': aggregate_score,
            'std_dev': std_dev,
            'volatility': std_dev / max(abs(aggregate_score), 0.01),  # Normalize volatility
            'confidence': 1.0 / (1.0 + std_dev)  # Higher std_dev = lower confidence
        }
    
    def analyze_current_sentiment(self) -> Dict[str, Any]:
        """Analyze current sentiment and detect potential crisis signals."""
        # Collect current data
        news_data = self._collect_news_data()
        social_media_data = self._collect_social_media_data()
        
        # Analyze raw sentiment
        current_sentiment = self._analyze_raw_sentiment(news_data, social_media_data)
        
        # Detect shifts from historical data
        sentiment_shifts = self._detect_sentiment_shifts(current_sentiment)
        
        # Calculate risk indices
        risk_indices = self._calculate_risk_indices(current_sentiment, sentiment_shifts)
        
        # Update historical data
        self._update_historical_data(current_sentiment)
        
        # Prepare response
        response = {
            'current': current_sentiment,
            'shifts': sentiment_shifts,
            'risk_indices': risk_indices,
            'summary': self._generate_sentiment_summary(current_sentiment, sentiment_shifts, risk_indices)
        }
        
        return response
    
    def _detect_sentiment_shifts(self, current_sentiment) -> Dict[str, Any]:
        """Detect shifts in sentiment compared to historical data."""
        if not self.historical_sentiment.get('history'):
            return {'detected': False, 'magnitude': 0, 'direction': 'neutral'}
        
        # Get most recent historical data
        historical = sorted(
            self.historical_sentiment['history'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        if not historical:
            return {'detected': False, 'magnitude': 0, 'direction': 'neutral'}
        
        # Compare current with historical
        current_aggregate = current_sentiment['aggregate']['score']
        
        # Calculate shifts for different time periods
        shifts = {
            'short_term': self._calculate_shift(current_aggregate, historical[:7]),  # 1 week
            'medium_term': self._calculate_shift(current_aggregate, historical[:30]),  # 1 month
            'long_term': self._calculate_shift(current_aggregate, historical)  # All history
        }
        
        # Determine if a significant shift is detected
        sentiment_config = self.config.get('sentiment', {})
        threshold = sentiment_config.get('shift_threshold', 0.3)
        max_shift = max([abs(s['magnitude']) for s in shifts.values()])
        
        detected = max_shift > threshold
        
        return {
            'detected': detected,
            'short_term': shifts['short_term'],
            'medium_term': shifts['medium_term'],
            'long_term': shifts['long_term'],
            'max_shift': max_shift,
            'threshold': threshold
        }
    
    def _calculate_shift(self, current_score, historical_data) -> Dict[str, Any]:
        """Calculate sentiment shift magnitude and direction."""
        if not historical_data:
            return {'magnitude': 0, 'direction': 'neutral'}
        
        # Calculate average historical score
        historical_scores = [h['data']['aggregate']['score'] for h in historical_data]
        avg_historical = np.mean(historical_scores)
        std_historical = np.std(historical_scores)
        
        # Calculate shift
        magnitude = (current_score - avg_historical) / max(std_historical, 0.01)  # Normalize by std dev
        
        # Determine direction
        direction = 'neutral'
        if magnitude > 0.1:
            direction = 'positive'
        elif magnitude < -0.1:
            direction = 'negative'
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'z_score': magnitude,  # Z-score is standardized shift
            'raw_change': current_score - avg_historical,
            'historical_avg': avg_historical,
            'historical_std': std_historical
        }
    
    def _calculate_risk_indices(self, current_sentiment, sentiment_shifts) -> Dict[str, Any]:
        """Calculate risk indices based on sentiment and shifts."""
        # Base risk from current sentiment (-1 to 1 scale)
        # Negative sentiment = higher risk
        base_risk = max(0, -current_sentiment['aggregate']['score'] * 5 + 0.5)  # Scale to 0-1
        
        # Adjust for volatility
        volatility_factor = current_sentiment['aggregate']['volatility'] * 2
        
        # Adjust for shifts
        shift_factor = 0
        if sentiment_shifts['detected']:
            # Negative shifts increase risk
            short_shift = sentiment_shifts['short_term']['magnitude']
            medium_shift = sentiment_shifts['medium_term']['magnitude']
            
            # Recent shifts have higher weight
            shift_factor = max(0, -short_shift * 0.7 - medium_shift * 0.3)
        
        # Calculate overall risk index (0-1 scale)
        risk_index = min(1.0, max(0.0, base_risk + volatility_factor * 0.3 + shift_factor * 0.5))
        
        # Convert to the 5-level crisis risk scale
        risk_level = self._convert_to_risk_level(risk_index)
        
        return {
            'risk_index': risk_index,
            'risk_level': risk_level,
            'components': {
                'base_risk': base_risk,
                'volatility_factor': volatility_factor,
                'shift_factor': shift_factor
            }
        }
    
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
    
    def _update_historical_data(self, current_sentiment):
        """Update historical sentiment data with current readings."""
        # Add current data to history
        self.historical_sentiment.setdefault('history', []).append({
            'timestamp': current_sentiment['timestamp'],
            'data': current_sentiment
        })
        
        # Limit history size
        data_config = self.config.get('data', {})
        max_history = data_config.get('max_sentiment_history', 365)  # Default 1 year
        if len(self.historical_sentiment['history']) > max_history:
            # Sort by timestamp and keep most recent
            self.historical_sentiment['history'] = sorted(
                self.historical_sentiment['history'],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:max_history]
        
        # Save updated data
        self._save_historical_data()
    
    def _generate_sentiment_summary(self, current, shifts, risks) -> Dict[str, Any]:
        """Generate a human-readable summary of sentiment analysis."""
        # Risk level descriptions
        risk_descriptions = [
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
        
        # Sentiment direction
        sentiment_direction = "neutral"
        if current['aggregate']['score'] > 0.2:
            sentiment_direction = "positive"
        elif current['aggregate']['score'] < -0.2:
            sentiment_direction = "negative"
        
        # Shift description
        shift_description = "stable"
        if shifts['detected']:
            if shifts['short_term']['direction'] == 'negative':
                shift_description = "deteriorating"
            elif shifts['short_term']['direction'] == 'positive':
                shift_description = "improving"
        
        # Generate summary text
        summary_text = (
            f"Current sentiment is {sentiment_direction} ({current['aggregate']['score']:.2f}) "
            f"and {shift_description}. Overall risk is "
            f"{risk_descriptions[risks['risk_level']-1]} (level {risks['risk_level']}/5)."
        )
        
        # Key concerns
        concerns = []
        
        # Check news sources with most negative sentiment
        if current['news']['scores']:
            most_negative_news = sorted(
                [(source, score) for source, score in current['news']['sources'].items()],
                key=lambda x: x[1]
            )[:3]
            
            if most_negative_news[0][1] < -0.3:
                concerns.append(f"Negative sentiment in {most_negative_news[0][0]} news")
        
        # Check keywords with most negative sentiment
        news_keywords = current['news'].get('keywords', {})
        twitter_keywords = current['twitter'].get('keywords', {})
        
        all_keywords = {}
        for k, v in news_keywords.items():
            all_keywords[k] = all_keywords.get(k, 0) + v * 0.6
        
        for k, v in twitter_keywords.items():
            all_keywords[k] = all_keywords.get(k, 0) + v * 0.4
        
        most_negative_keywords = sorted(
            [(keyword, score) for keyword, score in all_keywords.items()],
            key=lambda x: x[1]
        )[:3]
        
        if most_negative_keywords and most_negative_keywords[0][1] < -0.4:
            concerns.append(f"Strong negative sentiment around '{most_negative_keywords[0][0]}'")
        
        # Check for sudden shifts
        if shifts['short_term']['magnitude'] < -0.5:
            concerns.append("Rapid deterioration in public sentiment")
        
        return {
            'text': summary_text,
            'sentiment_direction': sentiment_direction,
            'shift_description': shift_description,
            'risk_level': risks['risk_level'],
            'risk_description': risk_descriptions[risks['risk_level']-1],
            'concerns': concerns,
            'timestamp': current['timestamp']
        } 