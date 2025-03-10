# Crisis Sentinel

An AI-powered system for predicting financial, geopolitical, and institutional crises by analyzing patterns across multiple data sources.

![Crisis Sentinel Dashboard](https://via.placeholder.com/800x450.png?text=Crisis+Sentinel+Dashboard)

## Overview

Crisis Sentinel is a sophisticated crisis prediction platform that leverages artificial intelligence to detect early warning signs of financial, geopolitical, and institutional crises. By analyzing data from news sources, social media, financial markets, and other indicators, the system can identify patterns that may indicate upcoming instability before they become evident to the public and markets.

### Key Features

- **Global Sentiment Analysis**: Collects and analyzes data from news, social media, and political discourse to identify shifts in public sentiment
- **Market Liquidity Monitoring**: Tracks financial indicators, yield curves, volatility indices, and unusual trading patterns
- **Risk Assessment System**: Assigns a risk level (1-5) based on the convergence of economic, social, and political data
- **Comprehensive Reporting**: Generates detailed reports with insights and possible future scenarios

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- API keys for Twitter, Reddit, and News API (for full functionality)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crisis-sentinel.git
   cd crisis-sentinel
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a configuration file:
   ```
   mkdir -p config
   cp config/config.example.json config/config.json
   ```

4. Edit the configuration file (`config/config.json`) to add your API keys and customize settings.

5. Create necessary directories:
   ```
   mkdir -p data logs
   ```

## Usage

### Running the Application

Start the Crisis Sentinel application:

```
python app.py
```

The application will be available at `http://localhost:5000` by default.

### Configuration

The system can be configured through the `config/config.json` file or by setting environment variables:

- `CS_DEBUG`: Enable debug mode (true/false)
- `CS_SERVER_PORT`: Set the server port
- `CS_TWITTER_BEARER_TOKEN`: Twitter API bearer token
- `CS_REDDIT_CLIENT_ID`: Reddit API client ID
- `CS_NEWS_API_KEY`: News API key

See the configuration file for all available options.

## System Components

### Sentiment Analyzer

The Sentiment Analyzer module collects and processes data from:
- News articles from major financial and general news sources
- Social media posts from Twitter and Reddit
- Political speeches and statements

It uses natural language processing techniques to identify sentiment shifts that may indicate upcoming crises.

### Market Monitor

The Market Monitor tracks financial indicators including:
- Yield curve inversions
- Volatility indices (VIX, VVIX)
- Credit spreads
- Liquidity measures
- Market indices
- Commodity prices

It detects abnormal patterns and correlations that have historically preceded financial crises.

### Risk Assessment System

The Risk Assessment System:
- Combines data from sentiment analysis and market monitoring
- Calculates a weighted risk index
- Assigns a risk level from 1 (Very Low) to 5 (Very High)
- Generates insights and recommendations based on the current situation

## Development

### Project Structure

```
crisis-sentinel/
├── app/
│   ├── core/
│   │   ├── sentiment_analyzer.py
│   │   ├── market_monitor.py
│   │   └── defcon_system.py
│   ├── data/
│   ├── utils/
│   │   ├── config_loader.py
│   │   ├── data_processor.py
│   │   └── logger.py
│   ├── api/
│   └── config/
├── static/
│   ├── css/
│   └── js/
├── templates/
├── data/
├── logs/
├── config/
├── app.py
├── requirements.txt
└── README.md
```

### Running Tests

```
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial crisis prediction research by [Researcher Name]
- Sentiment analysis techniques from [Source]
- Market indicators based on [Reference]

## Disclaimer

This system is provided for informational purposes only. The predictions and risk assessments should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

---

© 2025 Crisis Sentinel | AI-Powered Crisis Prediction System 