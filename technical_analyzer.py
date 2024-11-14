import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, Tuple, Union, List
import logging
from config import TECHNICAL_ANALYSIS_SETTINGS
from functools import lru_cache
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class CachedData:
    def __init__(self, data, timestamp, ttl_minutes: int = 15):
        self.data = data
        self.timestamp = timestamp
        self.ttl_minutes = ttl_minutes

    @property
    def is_valid(self) -> bool:
        return datetime.now() - self.timestamp < timedelta(minutes=self.ttl_minutes)

class TechnicalAnalyzer:
    STRONG_BUY_THRESHOLD: int = 2
    STRONG_SELL_THRESHOLD: int = -2
    BUY_THRESHOLD: int = 1
    SELL_THRESHOLD: int = -1
    
    SIGNAL_TYPES: Dict[str, str] = {
        'STRONG_BUY': 'strong_buy',
        'BUY': 'buy',
        'STRONG_SELL': 'strong_sell',
        'SELL': 'sell',
        'NEUTRAL': 'neutral'
    }
    
    def __init__(self, ticker: str):
        """
        Initialize TechnicalAnalyzer with a ticker symbol.
        
        Args:
            ticker: Stock ticker symbol to analyze
        """
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("Ticker must be a non-empty string")
            
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.settings = TECHNICAL_ANALYSIS_SETTINGS
        self._cache: Dict[str, CachedData] = {}
        
    @lru_cache(maxsize=100)
    def get_historical_data(self, period: str = '1y') -> pd.DataFrame:
        """Cached version of historical data retrieval"""
        cache_key = f"{self.ticker}_{period}"
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        data = self._fetch_historical_data(self.ticker, period)
        self._cache[cache_key] = CachedData(data, datetime.now())
        return data

    def _fetch_historical_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch historical data for technical analysis"""
        try:
            return self.stock.history(period=period)
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, data: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: DataFrame containing price data
            period: RSI calculation period (defaults to settings value)
            
        Returns:
            pd.Series containing RSI values
            
        Raises:
            ValueError: If data is empty or missing required columns
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if 'Close' not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column")
            
        if period is None:
            period = self.settings['RSI_PERIOD']
            
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero
        rs = gain / loss.replace(0, float('inf'))
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
        return {
            period: data['Close'].rolling(window=period).mean()
            for period in self.settings['MOVING_AVERAGE_PERIODS']
        }

    def analyze_technical_indicators(self) -> Dict:
        """Main function to analyze all technical indicators"""
        try:
            data = self.get_historical_data()
            if data.empty:
                logger.warning(f"No historical data available for {self.ticker}")
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

        except pd.errors.EmptyDataError:
            logger.error(f"Empty data received for {self.ticker}")
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while analyzing {self.ticker}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error analyzing {self.ticker}: {e}")
            return {}

    def _generate_overall_signal(
        self,
        rsi_signal: str,
        macd_signal: str,
        ma_signals: Dict[int, Dict[str, Union[float, str]]]
    ) -> str:
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
        if signal_sum >= self.STRONG_BUY_THRESHOLD:
            return self.SIGNAL_TYPES['STRONG_BUY']
        elif signal_sum >= self.BUY_THRESHOLD:
            return self.SIGNAL_TYPES['BUY']
        elif signal_sum <= self.STRONG_SELL_THRESHOLD:
            return self.SIGNAL_TYPES['STRONG_SELL']
        elif signal_sum <= self.SELL_THRESHOLD:
            return self.SIGNAL_TYPES['SELL']
        else:
            return self.SIGNAL_TYPES['NEUTRAL']

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