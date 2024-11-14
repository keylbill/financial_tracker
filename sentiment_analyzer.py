import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
import praw
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import json

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API keys from config
try:
    with open('api_keys.json', 'r') as f:
        API_KEYS = json.load(f)
except FileNotFoundError:
    logger.error("api_keys.json not found. Please create this file with your API keys.")
    API_KEYS = {}

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.initialize_apis()

    def initialize_apis(self):
        """Initialize API clients"""
        # Twitter API setup
        try:
            auth = tweepy.OAuthHandler(
                API_KEYS.get('TWITTER_API_KEY'),
                API_KEYS.get('TWITTER_API_SECRET')
            )
            auth.set_access_token(
                API_KEYS.get('TWITTER_ACCESS_TOKEN'),
                API_KEYS.get('TWITTER_ACCESS_TOKEN_SECRET')
            )
            self.twitter_api = tweepy.API(auth)
        except Exception as e:
            logger.error(f"Twitter API initialization failed: {e}")
            self.twitter_api = None

        # Reddit API setup
        try:
            self.reddit_api = praw.Reddit(
                client_id=API_KEYS.get('REDDIT_CLIENT_ID'),
                client_secret=API_KEYS.get('REDDIT_CLIENT_SECRET'),
                user_agent='FinancialSentimentBot 1.0'
            )
        except Exception as e:
            logger.error(f"Reddit API initialization failed: {e}")
            self.reddit_api = None

    def analyze_text(self, text: str) -> float:
        """Analyze sentiment of a text using both VADER and TextBlob"""
        if not text:
            return 0.0
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob sentiment
        textblob_sentiment = TextBlob(text).sentiment.polarity
        
        # Weighted average (VADER given more weight as it's better for social media)
        return (vader_compound * 0.7) + (textblob_sentiment * 0.3)

    def get_news_sentiment(self, ticker: str) -> List[Dict]:
        """Fetch and analyze news articles"""
        news_sources = [
            self._get_finnhub_news,
            self._get_newsapi_news,
            self._get_marketwatch_news
        ]
        
        all_news = []
        with ThreadPoolExecutor() as executor:
            future_to_source = {
                executor.submit(source, ticker): source.__name__
                for source in news_sources
            }
            
            for future in future_to_source:
                try:
                    articles = future.result()
                    all_news.extend(articles)
                except Exception as e:
                    logger.error(f"Error fetching news from {future_to_source[future]}: {e}")
        
        return all_news

    def get_social_sentiment(self, ticker: str) -> Dict[str, float]:
        """Analyze social media sentiment"""
        twitter_sentiment = self._analyze_twitter(ticker)
        reddit_sentiment = self._analyze_reddit(ticker)
        stocktwits_sentiment = self._analyze_stocktwits(ticker)
        
        weights = {
            'twitter': 0.4,
            'reddit': 0.4,
            'stocktwits': 0.2
        }
        
        return {
            'twitter': twitter_sentiment,
            'reddit': reddit_sentiment,
            'stocktwits': stocktwits_sentiment,
            'weighted_average': sum(
                s * weights[k] for k, s in {
                    'twitter': twitter_sentiment,
                    'reddit': reddit_sentiment,
                    'stocktwits': stocktwits_sentiment
                }.items()
            )
        }
    
    def _analyze_twitter(self, ticker: str) -> float:
        """Analyze Twitter sentiment"""
        if not self.twitter_api:
            return 0.0
            
        try:
            tweets = self.twitter_api.search_tweets(
                q=f"${ticker} -filter:retweets",
                lang="en",
                count=100,
                tweet_mode="extended"
            )
            
            sentiments = [
                self.analyze_text(tweet.full_text)
                for tweet in tweets
            ]
            
            return np.mean(sentiments) if sentiments else 0.0
        except Exception as e:
            logger.error(f"Twitter analysis error: {e}")
            return 0.0

    def _analyze_reddit(self, ticker: str) -> float:
        """Analyze Reddit sentiment"""
        if not self.reddit_api:
            return 0.0
            
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        sentiments = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit_api.subreddit(subreddit_name)
                posts = subreddit.search(f"{ticker}", limit=50, time_filter='week')
                
                for post in posts:
                    sentiments.append(self.analyze_text(post.title))
                    sentiments.append(self.analyze_text(post.selftext))
                    
                    # Analyze top comments
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:20]:
                        sentiments.append(self.analyze_text(comment.body))
                        
            except Exception as e:
                logger.error(f"Reddit analysis error in r/{subreddit_name}: {e}")
                
        return np.mean(sentiments) if sentiments else 0.0

    def calculate_weighted_sentiment(self, sentiments: List[Dict[str, float]], 
                                   source_weights: Dict[str, float]) -> float:
        """Calculate weighted sentiment score based on source reliability"""
        weighted_scores = []
        for sentiment in sentiments:
            source = sentiment['source']
            score = sentiment['score']
            weight = source_weights.get(source, 1.0)
            weighted_scores.append(score * weight)
        return sum(weighted_scores) / sum(source_weights.values())

    def batch_analyze_texts(self, texts: List[str], batch_size: int = 100) -> List[float]:
        """Analyze multiple texts in batches for better performance"""
        sentiments = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_sentiments = [self.analyze_text(text) for text in batch]
            sentiments.extend(batch_sentiments)
        return sentiments

    def _get_finnhub_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Finnhub"""
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"Finnhub news fetch error: {e}")
            return []

    def _get_newsapi_news(self, ticker: str) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []

    def _get_marketwatch_news(self, ticker: str) -> List[Dict]:
        """Fetch news from MarketWatch"""
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"MarketWatch news fetch error: {e}")
            return []

def get_media_sentiment(ticker: str) -> float:
    """Main function to get overall sentiment score"""
    analyzer = SentimentAnalyzer()
    
    # Get sentiment from different sources
    news_data = analyzer.get_news_sentiment(ticker)
    social_data = analyzer.get_social_sentiment(ticker)
    
    # Calculate news sentiment
    news_sentiments = [item['sentiment'] for item in news_data]
    news_score = np.mean(news_sentiments) if news_sentiments else 0.0
    
    # Combine scores with weights
    weights = {
        'news': 0.4,
        'social': 0.6
    }
    
    final_score = (
        news_score * weights['news'] +
        social_data['weighted_average'] * weights['social']
    )
    
    # Normalize to range [0, 1]
    normalized_score = (final_score + 1) / 2
    
    logger.info(f"Final sentiment score for {ticker}: {normalized_score:.2f}")
    return normalized_score 