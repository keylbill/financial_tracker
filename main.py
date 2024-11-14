from data_fetcher import (
    get_upcoming_earnings,
    get_options_data,
    get_historical_earnings_movement
)
from sentiment_analyzer import SentimentAnalyzer
from technical_analyzer import TechnicalAnalyzer
from alert_sender import send_telegram_alert
from risk_analyzer import RiskAnalyzer
from config import (
    SENTIMENT_THRESHOLD, 
    MOVEMENT_THRESHOLD, 
    EARNINGS_WINDOW_DAYS,
    ACCOUNT_SIZE,
    RISK_PER_TRADE,
    STOP_LOSS_PERCENTAGE
)
import schedule
import time
import logging
import numpy as np
from typing import Dict, Optional, Union, Any, List
from functools import lru_cache
from timeout_decorator import timeout, TimeoutError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='financial_tracker.log'
)

# Add type hints for clarity
AnalyzerDict = Dict[str, Union[SentimentAnalyzer, TechnicalAnalyzer, RiskAnalyzer]]
StockData = Dict[str, Any]

@lru_cache(maxsize=100)
def safe_process_stock(ticker: str, analyzers: AnalyzerDict) -> Optional[StockData]:
    """
    Safely process a single stock with proper error handling and timeout protection.
    
    Args:
        ticker: Stock symbol to process
        analyzers: Dictionary containing initialized analyzer instances
        
    Returns:
        Processed stock data dictionary or None if processing fails
    """
    try:
        with timeout(30):
            return process_stock(ticker, analyzers)
    except TimeoutError:
        logging.error(f"Processing timeout for {ticker}")
        return None
    except Exception as e:
        logging.error(f"Error processing {ticker}: {str(e)}")
        return None

def process_stock(ticker: str, analyzers: AnalyzerDict) -> Optional[StockData]:
    """
    Process a single stock and gather all relevant analysis data.
    
    Args:
        ticker: Stock symbol to analyze
        analyzers: Dictionary containing initialized analyzer instances
        
    Returns:
        Processed stock data dictionary or None if processing fails
    """
    try:
        # Extract analyzers with type hints for better IDE support
        sentiment_analyzer: SentimentAnalyzer = analyzers['sentiment']
        technical_analyzer: TechnicalAnalyzer = analyzers['technical']
        risk_analyzer: RiskAnalyzer = analyzers['risk']

        # Get sentiment data with error handling
        try:
            news_data = sentiment_analyzer.get_news_sentiment(ticker)
            social_data = sentiment_analyzer.get_social_sentiment(ticker)
        except Exception as e:
            logging.error(f"Sentiment analysis failed for {ticker}: {e}")
            return None

        # Calculate sentiment score with null safety
        sentiment_score = calculate_sentiment_score(news_data, social_data)

        # Get technical analysis with validation
        technical_analysis = technical_analyzer.analyze_technical_indicators()
        if not technical_analysis:
            logging.warning(f"No technical analysis data available for {ticker}")
            return None

        current_price = technical_analysis.get('current_price')
        if not current_price:
            logging.warning(f"No current price available for {ticker}")
            return None

        # Calculate risk metrics
        try:
            position_info = risk_analyzer.calculate_position_size(
                account_size=ACCOUNT_SIZE,
                risk_per_trade=RISK_PER_TRADE,
                stop_loss=STOP_LOSS_PERCENTAGE
            )

            risk_analysis = risk_analyzer.analyze_trade_risk(
                ticker=ticker,
                entry_price=current_price,
                position_size=position_info['position_size']
            )
        except Exception as e:
            logging.error(f"Risk analysis failed for {ticker}: {e}")
            return None

        # Get options data with error handling
        try:
            options_data = get_options_data(ticker)
        except Exception as e:
            logging.warning(f"Options data fetch failed for {ticker}: {e}")
            options_data = None

        return {
            'ticker': ticker,
            'sentiment_score': sentiment_score,
            'technical_analysis': technical_analysis,
            'risk_analysis': risk_analysis,
            'sentiment_details': {
                'overall_score': sentiment_score,
                'twitter': social_data['twitter'],
                'reddit': social_data['reddit'],
                'news_score': calculate_news_score(news_data),
                'news_count': len(news_data)
            },
            'options_data': options_data
        }
    except Exception as e:
        logging.error(f"Error processing stock {ticker}: {e}")
        return None

def calculate_sentiment_score(news_data: List[Dict], social_data: Dict) -> float:
    """Calculate weighted sentiment score from news and social data"""
    news_sentiment = np.mean([item['sentiment'] for item in news_data]) if news_data else 0.0
    return social_data['weighted_average'] * 0.6 + news_sentiment * 0.4

def calculate_news_score(news_data: List[Dict]) -> Optional[float]:
    """Calculate average news sentiment score"""
    return np.mean([item['sentiment'] for item in news_data]) if news_data else None

def identify_potential_stocks():
    try:
        earnings_calendar = get_upcoming_earnings(EARNINGS_WINDOW_DAYS)
        potential_stocks = []
        
        # Create analyzers once outside the loop for better performance
        analyzers = {
            'sentiment': SentimentAnalyzer(),
            'technical': TechnicalAnalyzer(None),  # Will be updated per ticker
            'risk': RiskAnalyzer()
        }

        for _, row in earnings_calendar.iterrows():
            ticker = row['ticker']
            logging.info(f"Processing {ticker}...")
            
            # Update technical analyzer for current ticker
            analyzers['technical'] = TechnicalAnalyzer(ticker)
            
            # Process stock with all necessary data
            stock_data = safe_process_stock(ticker, analyzers)
            if not stock_data:
                continue

            # Add earnings date to the stock data
            stock_data['earnings_date'] = row['earnings_date'].strftime('%Y-%m-%d')
            potential_stocks.append(stock_data)
            
            logging.info(f"Added {ticker} to potential stocks")
            
        return potential_stocks
    except Exception as e:
        logging.error(f"Error in stock identification: {str(e)}")
        return []

def job():
    try:
        potential_stocks = identify_potential_stocks()
        for stock_info in potential_stocks:
            try:
                send_telegram_alert(stock_info)
            except Exception as e:
                logging.error(f"Error sending alert for {stock_info['ticker']}: {e}")
    except Exception as e:
        logging.error(f"Error in main job: {e}")

def format_alert_message(stock_info):
    # Update sentiment details section
    sentiment_section = (
        f"*📊 Sentiment Analysis:*\n"
        f"• Overall Score: {stock_info['sentiment_details']['overall_score']:.2f}\n"
        f"• Twitter: {stock_info['sentiment_details']['twitter']:.2f}\n"
        f"• Reddit: {stock_info['sentiment_details']['reddit']:.2f}\n"
        f"• News: {stock_info['sentiment_details'].get('news_score', 'N/A')}\n"
        f"• News Articles: {stock_info['sentiment_details']['news_count']}\n\n"
    )
    # Rest of the function remains the same

if __name__ == "__main__":
    # Schedule the job to run once a day at a specific time
    schedule.every().day.at("09:00").do(job)  # Adjust the time as needed

    while True:
        schedule.run_pending()
        time.sleep(60) 