import telegram
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_alert(stock_info):
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    message = format_alert_message(stock_info)
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)

def format_alert_message(stock_info):
    message = (
        f"*📣 {stock_info['ticker']}* is reporting earnings on *{stock_info['earnings_date']}*\n\n"
        f"*📊 Sentiment Analysis:*\n"
        f"• Overall Score: {stock_info['sentiment_details']['overall_score']:.2f}\n"
        f"• Twitter: {stock_info['sentiment_details']['twitter_sentiment']:.2f}\n"
        f"• Reddit: {stock_info['sentiment_details']['reddit_sentiment']:.2f}\n"
        f"• News: {stock_info['sentiment_details'].get('news_score', 'N/A')}\n"
        f"• News Articles: {stock_info['sentiment_details']['news_count']}\n\n"
        f"*📈 Technical Analysis:*\n"
        f"• RSI: {stock_info['technical_analysis']['rsi']['value']:.2f} ({stock_info['technical_analysis']['rsi']['signal']})\n"
        f"• MACD: {stock_info['technical_analysis']['macd']['signal']}\n"
        f"• Moving Averages: {', '.join(f'{period}MA: {data['position']}' for period, data in stock_info['technical_analysis']['moving_averages'].items())}\n"
        f"• Overall Signal: {stock_info['technical_analysis']['overall_signal'].upper()}\n\n"
        f"*⚙️ Options Data:*\n"
    )
    for _, option in stock_info['options_data'].iterrows():
        bid_ask_spread = option['ask'] - option['bid']
        message += (
            f"• Strike: {option['strike']}, IV: {round(option['impliedVolatility'], 2)}, "
            f"Bid-Ask Spread: {round(bid_ask_spread, 2)}\n"
        )
    return message 

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