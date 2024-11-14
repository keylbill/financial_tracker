import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from config import RISK_MANAGEMENT_SETTINGS
from technical_analyzer import TechnicalAnalyzer
import yfinance as yf

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    def __init__(self):
        self.settings = RISK_MANAGEMENT_SETTINGS
        
    def calculate_position_size(self, account_size: float, risk_per_trade: float, stop_loss: float) -> Dict:
        """
        Calculate optimal position size based on account size and risk parameters
        
        Args:
            account_size: Total account value
            risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 for 2%)
            stop_loss: Stop loss percentage as decimal
            
        Returns:
            Dict containing position size and risk metrics
        """
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [account_size, risk_per_trade, stop_loss]):
                raise ValueError("All inputs must be numerical values")
            
            if risk_per_trade > self.settings['MAX_POSITION_SIZE']:
                risk_per_trade = self.settings['MAX_POSITION_SIZE']
                logger.warning(f"Risk per trade adjusted to maximum allowed: {risk_per_trade*100}%")
            
            # Calculate maximum dollar risk
            max_risk_amount = account_size * risk_per_trade
            
            # Calculate position size based on stop loss
            position_size = max_risk_amount / stop_loss
            
            # Calculate take profit based on risk:reward ratio
            take_profit = stop_loss * self.settings['TAKE_PROFIT_RATIO']
            
            return {
                'position_size': position_size,
                'max_risk_amount': max_risk_amount,
                'stop_loss_price': stop_loss,
                'take_profit_price': take_profit,
                'risk_reward_ratio': self.settings['TAKE_PROFIT_RATIO']
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {}

    def analyze_trade_risk(self, ticker: str, entry_price: float, position_size: float) -> Dict:
        try:
            technical_analyzer = TechnicalAnalyzer(ticker)
            technical_data = technical_analyzer.analyze_technical_indicators()
            
            # Calculate volatility risk
            historical_data = technical_analyzer.get_historical_data()
            daily_returns = historical_data['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate volatility trend
            vol_trend = self.analyze_volatility_trend(historical_data)
            
            # Calculate market correlation
            market_correlation = self.analyze_market_correlation(ticker)
            
            # Calculate stop loss and take profit levels
            stop_loss = entry_price * (1 - self.settings['STOP_LOSS_PERCENTAGE'])
            take_profit = entry_price + ((entry_price - stop_loss) * self.settings['TAKE_PROFIT_RATIO'])
            
            # Calculate maximum drawdown risk
            max_loss = (entry_price - stop_loss) * position_size
            
            risk_analysis = {
                'volatility': volatility,
                'volatility_trend': vol_trend,
                'market_correlation': market_correlation,
                'stop_loss_level': stop_loss,
                'take_profit_level': take_profit,
                'max_potential_loss': max_loss,
                'technical_signals': technical_data.get('overall_signal', 'neutral'),
                'risk_level': self._calculate_risk_level(volatility, technical_data)
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trade risk: {e}")
            return {}

    def _calculate_risk_level(self, volatility: float, technical_data: Dict, sentiment_data: Dict = None) -> str:
        """
        Calculate overall risk level based on various factors including sentiment
        """
        risk_score = 0
        
        # Volatility risk factor
        if volatility > 0.4:  # High volatility
            risk_score += 3
        elif volatility > 0.2:  # Medium volatility
            risk_score += 2
        else:
            risk_score += 1
            
        # Technical analysis risk factors
        signal = technical_data.get('overall_signal', 'neutral')
        if signal in ['strong_sell', 'strong_buy']:
            risk_score += 1  # Strong signals might indicate lower risk
        elif signal == 'neutral':
            risk_score += 2
            
        # RSI risk factor
        rsi = technical_data.get('rsi', {}).get('value', 50)
        if rsi > 70 or rsi < 30:
            risk_score += 2  # Overbought/oversold conditions
            
        # Add sentiment risk factor
        if sentiment_data:
            sentiment_score = sentiment_data.get('overall_score', 0.5)
            if sentiment_score < 0.3:  # Very negative sentiment
                risk_score += 3
            elif sentiment_score < 0.4:  # Negative sentiment
                risk_score += 2
            elif sentiment_score > 0.7:  # Very positive sentiment
                risk_score += 1  # Lower risk for positive sentiment
            
        # Risk level categorization
        if risk_score >= 6:
            return 'high'
        elif risk_score >= 4:
            return 'medium'
        else:
            return 'low'

    def get_risk_adjusted_returns(self, historical_returns: pd.Series) -> Dict:
        """
        Calculate risk-adjusted return metrics
        """
        try:
            # Calculate basic metrics
            mean_return = historical_returns.mean()
            std_dev = historical_returns.std()
            
            # Sharpe Ratio (assuming risk-free rate of 0.02)
            risk_free_rate = 0.02
            sharpe_ratio = (mean_return - risk_free_rate) / std_dev
            
            # Sortino Ratio (downside deviation)
            negative_returns = historical_returns[historical_returns < 0]
            downside_dev = negative_returns.std()
            sortino_ratio = (mean_return - risk_free_rate) / downside_dev if len(negative_returns) > 0 else np.nan
            
            # Maximum Drawdown
            cumulative_returns = (1 + historical_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'annualized_volatility': std_dev * np.sqrt(252),
                'risk_adjusted_return': sharpe_ratio * mean_return
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {e}")
            return {}

    def analyze_volatility_trend(self, historical_data: pd.DataFrame, window: int = 20) -> float:
        """Analyze volatility trend to identify increasing risk patterns"""
        returns = historical_data['Close'].pct_change()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        vol_trend = rolling_vol.diff().mean()
        return vol_trend 

    def analyze_market_correlation(self, ticker: str, market_index: str = 'SPY') -> float:
        """Calculate correlation with broader market"""
        try:
            stock_data = self.get_historical_data()
            market_data = yf.download(market_index, start=stock_data.index[0])
            
            stock_returns = stock_data['Close'].pct_change()
            market_returns = market_data['Close'].pct_change()
            
            # Align dates
            combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
            correlation = combined.corr().iloc[0,1]
            
            return correlation
        except Exception as e:
            logger.error(f"Error calculating market correlation: {e}")
            return 0.0