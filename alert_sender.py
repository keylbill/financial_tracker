import telegram
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import logging

logger = logging.getLogger(__name__)

def send_telegram_alert(stock_info):
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        message = format_alert_message(stock_info)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)
    except telegram.error.TelegramError as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error sending alert: {e}")
        raise

def format_alert_message(stock_info):
    try:
        sections = [
            _format_header(stock_info),
            _format_sentiment_section(stock_info),
            _format_technical_section(stock_info),
            _format_options_section(stock_info)
        ]
        return ''.join(sections)
    except KeyError as e:
        logger.error(f"Missing required data in stock_info: {e}")
        raise
    except Exception as e:
        logger.error(f"Error formatting alert message: {e}")
        raise

def _format_header(stock_info):
    return f"*📣 {stock_info['ticker']}* is reporting earnings on *{stock_info['earnings_date']}*\n\n"

def _format_sentiment_section(stock_info):
    return (
        f"*📊 Sentiment Analysis:*\n"
        f"• Overall Score: {stock_info['sentiment_details']['overall_score']:.2f}\n"
        f"• Twitter: {stock_info['sentiment_details']['twitter']:.2f}\n"
        f"• Reddit: {stock_info['sentiment_details']['reddit']:.2f}\n"
        f"• News: {stock_info['sentiment_details'].get('news_score', 'N/A')}\n"
        f"• News Articles: {stock_info['sentiment_details']['news_count']}\n\n"
    )

def _format_technical_section(stock_info):
    return (
        f"*📈 Technical Analysis:*\n"
        f"• RSI: {stock_info['technical_analysis']['rsi']['value']:.2f} "
        f"({stock_info['technical_analysis']['rsi']['signal']})\n"
        f"• MACD: {stock_info['technical_analysis']['macd']['signal']}\n"
        f"• Moving Averages: {', '.join([f'{period}MA: {data['position']}' for period, data in stock_info['technical_analysis']['moving_averages'].items()])}\n"
        f"• Overall Signal: {stock_info['technical_analysis']['overall_signal'].upper()}\n\n"
    )

def _format_options_section(stock_info):
    options_section = "*⚙️ Options Data:*\n"
    for _, option in stock_info['options_data'].iterrows():
        bid_ask_spread = option['ask'] - option['bid']
        options_section += (
            f"• Strike: {option['strike']}, "
            f"IV: {option['impliedVolatility']:.2f}, "
            f"Bid-Ask Spread: {bid_ask_spread:.2f}\n"
        )
    return options_section

def format_risk_message(risk_analysis):
    return f"""
Risk Analysis:
• Position Size: ${risk_analysis['position_size']:,.2f}
• Stop Loss: ${risk_analysis['stop_loss']:,.2f}
• Take Profit: ${risk_analysis['take_profit']:,.2f}
• Risk/Reward: {risk_analysis['risk_reward_ratio']}:1
• Risk Level: {risk_analysis['risk_level'].upper()}
• Volatility: {risk_analysis['volatility']*100:.1f}%
• Max Loss: ${risk_analysis['max_potential_loss']:,.2f}
""" 