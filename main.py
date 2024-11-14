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
from typing import Dict, Optional
from functools import lru_cache
from timeout_decorator import timeout, TimeoutError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='financial_tracker.log'
)

def safe_process_stock(ticker: str, analyzers: Dict) -> Optional[Dict]:
    """Safely process a single stock with proper error handling"""
    try:
        with timeout(30):  # Add timeout protection
            return process_stock(ticker, analyzers)
    except TimeoutError:
        logging.error(f"Processing timeout for {ticker}")
        return None
    except Exception as e:
        logging.error(f"Error processing {ticker}: {str(e)}")
        return None

def identify_potential_stocks():
    try:
        earnings_calendar = get_upcoming_earnings(EARNINGS_WINDOW_DAYS)
        potential_stocks = []
        risk_analyzer = RiskAnalyzer()

        for _, row in earnings_calendar.iterrows():
            ticker = row['ticker']
            print(f"Processing {ticker}...")
            
            # Create analyzer instances
            sentiment_analyzer = SentimentAnalyzer()
            technical_analyzer = TechnicalAnalyzer(ticker)
            
            # Get sentiment analysis
            news_data = sentiment_analyzer.get_news_sentiment(ticker)
            social_data = sentiment_analyzer.get_social_sentiment(ticker)
            sentiment_score = social_data['weighted_average']
            
            if sentiment_score < SENTIMENT_THRESHOLD:
                continue
                
            # Get technical analysis
            technical_analysis = technical_analyzer.analyze_technical_indicators()
            if technical_analysis.get('overall_signal') in ['strong_sell', 'sell']:
                continue

            avg_movement = get_historical_earnings_movement(ticker)
            if avg_movement < MOVEMENT_THRESHOLD:
                continue

            options_data = get_options_data(ticker)
            if options_data.empty:
                continue

            current_price = technical_analysis.get('current_price', 0)
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

            if risk_analysis.get('risk_level', 'high') == 'high':
                logging.info(f"Skipping {ticker} due to high risk level")
                continue

            stock_info = {
                'ticker': ticker,
                'earnings_date': row['earnings_date'].strftime('%Y-%m-%d'),
                'sentiment_score': round(sentiment_score, 2),
                'avg_movement': round(avg_movement, 2),
                'options_data': options_data,
                'technical_analysis': technical_analysis,
                'risk_analysis': {
                    'position_size': position_info['position_size'],
                    'stop_loss': position_info['stop_loss_price'],
                    'take_profit': position_info['take_profit_price'],
                    'risk_reward_ratio': position_info['risk_reward_ratio'],
                    'risk_level': risk_analysis['risk_level'],
                    'volatility': risk_analysis['volatility'],
                    'max_potential_loss': risk_analysis['max_potential_loss']
                },
                'sentiment_details': {
                    'overall_score': sentiment_score,
                    'twitter_sentiment': social_data['twitter'],
                    'reddit_sentiment': social_data['reddit'],
                    'stocktwits_sentiment': social_data['stocktwits'],
                    'news_score': np.mean([item['sentiment'] for item in news_data]) if news_data else None,
                    'news_count': len(news_data)
                }
            }
            potential_stocks.append(stock_info)
            
            logging.info(f"Added {ticker} to potential stocks with {risk_analysis['risk_level']} risk level")
            
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

if __name__ == "__main__":
    # Schedule the job to run once a day at a specific time
    schedule.every().day.at("09:00").do(job)  # Adjust the time as needed

    while True:
        schedule.run_pending()
        time.sleep(60) 