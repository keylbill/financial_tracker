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
    'MOVING_AVERAGE_PERIODS': [20, 50, 200]
}

# Risk Management Settings
RISK_MANAGEMENT_SETTINGS = {
    'MAX_POSITION_SIZE': 0.02,  # 2% of portfolio
    'STOP_LOSS_PERCENTAGE': 0.05,
    'TAKE_PROFIT_RATIO': 2  # Risk:Reward ratio
}

# Account and Risk Management Settings
ACCOUNT_SIZE = 100000  # Your trading account size
RISK_PER_TRADE = 0.02  # 2% risk per trade
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss