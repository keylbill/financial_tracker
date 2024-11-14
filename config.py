# config.py

# API Keys
ALPHA_VANTAGE_API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'

# Alert Criteria Thresholds
SENTIMENT_THRESHOLD = 0.7  # Adjust based on sentiment scoring system
MOVEMENT_THRESHOLD = 10    # Percentage
EARNINGS_WINDOW_DAYS = 14  # Days ahead to look for earnings 

# Technical Analysis Settings
TECHNICAL_ANALYSIS_SETTINGS = {
    'RSI_PERIOD': 14,
    'RSI_OVERSOLD': 30,
    'RSI_OVERBOUGHT': 70,
    'MOVING_AVERAGE_PERIODS': [20, 50, 200],
    'CACHE_TTL_MINUTES': 15  # Add cache timeout setting
}

# Risk Management Settings
RISK_MANAGEMENT_SETTINGS = {
    'ACCOUNT_SIZE': 100000,          # Your trading account size
    'MAX_POSITION_SIZE': 0.02,       # 2% of portfolio
    'RISK_PER_TRADE': 0.02,         # 2% risk per trade
    'STOP_LOSS_PERCENTAGE': 0.05,    # 5% stop loss
    'TAKE_PROFIT_RATIO': 2.0,        # Risk:Reward ratio (add .0 for clarity)
    'MIN_POSITION_SIZE': 100         # Add minimum position size
}

# Add logging configuration
LOGGING_CONFIG = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(levelname)s - %(message)s',
    'FILENAME': 'financial_tracker.log'
}