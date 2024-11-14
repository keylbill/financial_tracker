import requests
import pandas as pd
from datetime import datetime, timedelta
from config import ALPHA_VANTAGE_API_KEY
import yfinance as yf

def get_upcoming_earnings(days_ahead=14):
    today = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    # Use Alpha Vantage's Earnings Calendar API
    url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    earnings_list = []
    for item in data.get('earningsCalendar', []):
        earnings_date = datetime.strptime(item['reportDate'], '%Y-%m-%d')
        if today <= earnings_date <= end_date:
            earnings_list.append({
                'ticker': item['symbol'],
                'earnings_date': earnings_date
            })
    earnings_df = pd.DataFrame(earnings_list)
    return earnings_df 

def get_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return pd.DataFrame()
        nearest_expiry = expirations[0]
        options_chain = stock.option_chain(nearest_expiry)
        calls = options_chain.calls
        # Get current stock price
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        # Filter slightly out-of-the-money calls
        otm_calls = calls[(calls['strike'] > current_price * 1.05) & (calls['strike'] < current_price * 1.10)]
        # Select necessary columns
        otm_calls = otm_calls[['strike', 'impliedVolatility', 'bid', 'ask']]
        return otm_calls
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {e}")
        return pd.DataFrame()

def get_historical_earnings_movement(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings
        price_history = stock.history(period='1y')
        movements = []
        for date in earnings.index:
            date = pd.to_datetime(date)
            if date in price_history.index:
                before_price = price_history.loc[date - pd.Timedelta(days=1)]['Close']
                after_price = price_history.loc[date + pd.Timedelta(days=1)]['Close']
                movement = ((after_price - before_price) / before_price) * 100
                movements.append(abs(movement))
        if movements:
            avg_movement = sum(movements) / len(movements)
            return avg_movement
        else:
            return 0
    except Exception as e:
        print(f"Error calculating historical movement for {ticker}: {e}")
        return 0