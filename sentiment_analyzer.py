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
import os
from dotenv import load_dotenv
import yfinance as yf
import time
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_keys():
    """Load API keys from environment variables with JSON fallback"""
    keys = {
        'TWITTER_API_KEY': os.getenv('TWITTER_API_KEY'),
        'TWITTER_API_SECRET': os.getenv('TWITTER_API_SECRET'),
        'TWITTER_ACCESS_TOKEN': os.getenv('TWITTER_ACCESS_TOKEN'),
        'TWITTER_ACCESS_TOKEN_SECRET': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID'),
        'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET'),
        'NEWSAPI_KEY': os.getenv('NEWSAPI_KEY')
    }
    
    # Fallback to JSON if env vars not found
    if not all(keys.values()):
        try:
            with open('api_keys.json', 'r') as f:
                keys.update(json.load(f))
        except FileNotFoundError:
            logger.error("Neither environment variables nor api_keys.json found")
    
    return keys

class SentimentAnalyzer:
    # Add class-level constants for better maintainability
    WEIGHTS = {
        'VADER': 0.7,
        'TEXTBLOB': 0.3,
        'NEWS': 0.4,
        'SOCIAL': 0.6,
        'TWITTER': 0.5,
        'REDDIT': 0.5
    }

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.api_keys = load_api_keys()
        self.initialize_apis()
        self._cache = {}  # Add caching for API responses

    def initialize_apis(self):
        """Initialize API clients"""
        try:
            auth = tweepy.OAuthHandler(
                self.api_keys.get('TWITTER_API_KEY'),
                self.api_keys.get('TWITTER_API_SECRET')
            )
            auth.set_access_token(
                self.api_keys.get('TWITTER_ACCESS_TOKEN'),
                self.api_keys.get('TWITTER_ACCESS_TOKEN_SECRET')
            )
            self.twitter_api = tweepy.API(auth)
        except Exception as e:
            logger.error(f"Twitter API initialization failed: {e}")
            self.twitter_api = None

        # Reddit API setup
        try:
            self.reddit_api = praw.Reddit(
                client_id=self.api_keys.get('REDDIT_CLIENT_ID'),
                client_secret=self.api_keys.get('REDDIT_CLIENT_SECRET'),
                user_agent='FinancialSentimentBot 1.0'
            )
        except Exception as e:
            logger.error(f"Reddit API initialization failed: {e}")
            self.reddit_api = None

    def analyze_text(self, text: str) -> float:
        """Analyze sentiment of a text using both VADER and TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Weighted sentiment score between -1 and 1
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob sentiment
        textblob_sentiment = TextBlob(text).sentiment.polarity
        
        # Weighted average using class constants
        return (vader_compound * self.WEIGHTS['VADER'] + 
                textblob_sentiment * self.WEIGHTS['TEXTBLOB'])

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
        """Analyze social media sentiment
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores for each platform
                            and weighted average
        """
        twitter_sentiment = self._analyze_twitter(ticker)
        reddit_sentiment = self._analyze_reddit(ticker)
        
        return {
            'twitter': twitter_sentiment,
            'reddit': reddit_sentiment,
            'weighted_average': (
                twitter_sentiment * self.WEIGHTS['TWITTER'] +
                reddit_sentiment * self.WEIGHTS['REDDIT']
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
        """Analyze Reddit sentiment with batch processing"""
        if not self.reddit_api:
            return 0.0
        
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        texts_to_analyze = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit_api.subreddit(subreddit_name)
                posts = subreddit.search(f"{ticker}", limit=50, time_filter='week')
                
                for post in posts:
                    texts_to_analyze.extend([post.title, post.selftext])
                    
                    # Add top comments
                    post.comments.replace_more(limit=0)
                    texts_to_analyze.extend([c.body for c in post.comments.list()[:20]])
                    
            except Exception as e:
                logger.error(f"Reddit analysis error in r/{subreddit_name}: {e}")
        
        # Batch process all texts
        sentiments = self.batch_analyze_texts(texts_to_analyze)
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
        if not self.api_keys.get('FINNHUB_API_KEY'):
            logger.warning("Finnhub API key not found")
            return []
        
        try:
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': self.api_keys['FINNHUB_API_KEY']
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            articles = response.json()
            return [{
                'title': article.get('headline', ''),
                'text': article.get('summary', ''),
                'source': 'finnhub',
                'sentiment': self.analyze_text(f"{article.get('headline', '')} {article.get('summary', '')}")
            } for article in articles]
            
        except Exception as e:
            logger.error(f"Finnhub news fetch error: {e}")
            return []

    def _get_newsapi_news(self, ticker: str) -> List[Dict]:
        """Fetch news from NewsAPI"""
        if not self.api_keys.get('NEWSAPI_KEY'):
            logger.warning("NewsAPI key not found")
            return []
        
        try:
            # Build search query with company name and ticker
            company_info = yf.Ticker(ticker).info
            company_name = company_info.get('longName', '')
            search_query = f"({ticker} OR {company_name}) AND (stock OR market OR trading OR earnings)"
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': search_query,
                'language': 'en',
                'sortBy': 'relevancy',
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'apiKey': self.api_keys['NEWSAPI_KEY'],
                'pageSize': 50  # Limit to most relevant 50 articles
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            
            # Process and filter articles
            processed_articles = []
            for article in articles:
                # Combine title and description for better sentiment analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Skip articles with insufficient content
                if len(text.split()) < 10:
                    continue
                    
                # Skip articles that don't mention the ticker or company name
                if not any(term.lower() in text.lower() 
                          for term in [ticker, company_name]):
                    continue
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'text': article.get('description', ''),
                    'source': 'newsapi',
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'sentiment': self.analyze_text(text)
                })
            
            return processed_articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request error: {e}")
            return []
        except Exception as e:
            logger.error(f"NewsAPI processing error: {e}")
            return []

    def _get_marketwatch_news(self, ticker: str) -> List[Dict]:
        """Fetch news from MarketWatch
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            List[Dict]: List of processed news articles with sentiment scores
        """
        try:
            # Build MarketWatch URL
            base_url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Fetch the page
            response = requests.get(base_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Use BeautifulSoup to parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles container
            articles_container = soup.find('div', {'class': 'collection__elements'})
            if not articles_container:
                logger.warning(f"No news articles found for {ticker} on MarketWatch")
                return []
            
            articles = []
            # Process each article
            for article in articles_container.find_all('div', {'class': 'element--article'}):
                try:
                    # Extract article details
                    title = article.find('h3', {'class': 'article__headline'})
                    title_text = title.get_text().strip() if title else ''
                    
                    timestamp = article.find('span', {'class': 'article__timestamp'})
                    pub_date = timestamp.get_text().strip() if timestamp else ''
                    
                    # Get article URL
                    link = article.find('a', {'class': 'link'})
                    url = link.get('href') if link else ''
                    
                    # Only process articles with sufficient content
                    if not title_text or not url:
                        continue
                    
                    # Fetch full article text if URL is available
                    article_text = ''
                    if url and url.startswith('http'):
                        try:
                            article_response = requests.get(url, headers=headers, timeout=10)
                            if article_response.status_code == 200:
                                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                article_body = article_soup.find('div', {'class': 'article__body'})
                                if article_body:
                                    article_text = ' '.join([p.get_text().strip() 
                                                           for p in article_body.find_all('p')])
                        except Exception as e:
                            logger.debug(f"Error fetching full article text: {e}")
                    
                    # Combine title and article text for sentiment analysis
                    full_text = f"{title_text} {article_text}"
                    
                    articles.append({
                        'title': title_text,
                        'text': article_text[:500],  # Limit text length
                        'source': 'marketwatch',
                        'url': url,
                        'published_at': pub_date,
                        'sentiment': self.analyze_text(full_text)
                    })
                    
                    # Rate limiting
                    time.sleep(0.5)  # 500ms delay between article fetches
                    
                except Exception as e:
                    logger.debug(f"Error processing individual MarketWatch article: {e}")
                    continue
                
            return articles[:25]  # Limit to 25 most recent articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MarketWatch request error for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"MarketWatch processing error for {ticker}: {e}")
            return []

    def _analyze_stocktwits(self, ticker: str) -> float:
        """Analyze StockTwits sentiment"""
        logger.warning("StockTwits analysis not implemented")
        return 0.0

def get_media_sentiment(ticker: str) -> float:
    """Calculate overall media sentiment score
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Normalized sentiment score between 0 and 1
    """
    if not ticker or not isinstance(ticker, str):
        logger.error("Invalid ticker provided")
        return 0.0
        
    analyzer = SentimentAnalyzer()
    
    # Get sentiment from different sources
    news_data = analyzer.get_news_sentiment(ticker)
    social_data = analyzer.get_social_sentiment(ticker)
    
    # Calculate news sentiment
    news_sentiments = [item['sentiment'] for item in news_data]
    news_score = np.mean(news_sentiments) if news_sentiments else 0.0
    
    # Combine scores using class constants
    final_score = (
        news_score * analyzer.WEIGHTS['NEWS'] +
        social_data['weighted_average'] * analyzer.WEIGHTS['SOCIAL']
    )
    
    # Normalize to range [0, 1]
    normalized_score = (final_score + 1) / 2
    
    logger.info(f"Final sentiment score for {ticker}: {normalized_score:.2f}")
    return normalized_score 