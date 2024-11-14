import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Tuple
import logging
from config import TECHNICAL_ANALYSIS_SETTINGS
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CachedData:
    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp

    @property
    def is_valid(self, ttl_minutes: int = 15):
        return datetime.now() - self.timestamp < timedelta(minutes=ttl_minutes)

class TechnicalAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.settings = TECHNICAL_ANALYSIS_SETTINGS
        self._cache = {}
        
    @lru_cache(maxsize=100)
    def get_historical_data(self, period: str = '1y') -> pd.DataFrame:
        """Cached version of historical data retrieval"""
        return self._fetch_historical_data(self.ticker, period)

    def _fetch_historical_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch historical data for technical analysis"""
        try:
            return self.stock.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Relative Strength Index"""
        if period is None:
            period = self.settings['RSI_PERIOD']
            
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data: pd.DataFrame, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and MACD histogram"""
        exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_moving_averages(self, data: pd.DataFrame) -> Dict[int, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        for period in self.settings['MOVING_AVERAGE_PERIODS']:
            mas[period] = data['Close'].rolling(window=period).mean()
        return mas

    def analyze_technical_indicators(self) -> Dict:
        """Main function to analyze all technical indicators"""
        try:
            data = self.get_historical_data()
            if data.empty:
                return {}

            # Calculate RSI
            rsi = self.calculate_rsi(data)
            current_rsi = rsi.iloc[-1]
            rsi_signal = (
                'oversold' if current_rsi <= self.settings['RSI_OVERSOLD']
                else 'overbought' if current_rsi >= self.settings['RSI_OVERBOUGHT']
                else 'neutral'
            )

            # Calculate MACD
            macd, signal, histogram = self.calculate_macd(data)
            macd_signal = 'bullish' if histogram.iloc[-1] > 0 else 'bearish'

            # Calculate Moving Averages
            mas = self.calculate_moving_averages(data)
            current_price = data['Close'].iloc[-1]
            ma_signals = {}
            for period, ma in mas.items():
                ma_signals[period] = {
                    'value': ma.iloc[-1],
                    'position': 'above' if current_price > ma.iloc[-1] else 'below'
                }

            # Compile analysis results
            analysis = {
                'ticker': self.ticker,
                'current_price': current_price,
                'rsi': {
                    'value': current_rsi,
                    'signal': rsi_signal
                },
                'macd': {
                    'value': macd.iloc[-1],
                    'signal': macd_signal,
                    'histogram': histogram.iloc[-1]
                },
                'moving_averages': ma_signals,
                'overall_signal': self._generate_overall_signal(
                    rsi_signal, macd_signal, ma_signals
                )
            }
            
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing technical indicators for {self.ticker}: {e}")
            return {}

    def _generate_overall_signal(self, rsi_signal: str, 
                               macd_signal: str, 
                               ma_signals: Dict) -> str:
        """Generate overall trading signal based on all indicators"""
        signals = []
        
        # RSI weight
        if rsi_signal == 'oversold':
            signals.append(1)
        elif rsi_signal == 'overbought':
            signals.append(-1)
        else:
            signals.append(0)
            
        # MACD weight
        signals.append(1 if macd_signal == 'bullish' else -1)
        
        # Moving averages weight
        ma_count = 0
        for ma_data in ma_signals.values():
            if ma_data['position'] == 'above':
                ma_count += 1
            else:
                ma_count -= 1
                
        signals.append(1 if ma_count > 0 else -1 if ma_count < 0 else 0)
        
        # Calculate final signal
        signal_sum = sum(signals)
        if signal_sum >= 2:
            return 'strong_buy'
        elif signal_sum >= 1:
            return 'buy'
        elif signal_sum <= -2:
            return 'strong_sell'
        elif signal_sum <= -1:
            return 'sell'
        else:
            return 'neutral'

    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        if key in self._cache:
            cached = self._cache[key]
            if cached.is_valid:
                return cached.data
        return None

def analyze_technical_indicators(ticker: str) -> Dict:
    """Wrapper function for external calls"""
    analyzer = TechnicalAnalyzer(ticker)
    return analyzer.analyze_technical_indicators() 