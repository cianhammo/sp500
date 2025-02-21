import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from readability import Document
from googleapiclient.discovery import build
from anthropic import Anthropic
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient  
from alpaca.trading.requests import GetCalendarRequest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging
import warnings
from googleapiclient.discovery_cache.base import Cache
import re
from typing import List, Union, Dict, Optional
import time
from dateutil import parser





################################################
#Secret keys from environment variables

alpaca_api_key = os.environ['ALPACA_API_KEY']
alpaca_secret_key = os.environ['ALPACA_SECRET_KEY']
google_search_api_key = os.environ['GOOGLE_SEARCH_API_KEY']
google_search_id = os.environ['GOOGLE_SEARCH_ID']
anthropic_api_key = os.environ['ANTHROPIC_API_KEY']
serpapi_api_key = os.environ['SERPAPI_API_KEY']





################################################
#Logging and Alpaca set up

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a log filename with today's date
log_filename = f"logs/sp500_{datetime.now().strftime('%Y-%m-%d')}.log"

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log the start of the script
logger.info("Starting SP500 analysis script")

# Suppress the file_cache warning
warnings.filterwarnings('ignore', message='file_cache is only supported with oauth2client<4.0.0')

# Create a no-op cache class to silence the file_cache warning
class NoOpCache(Cache):
    def get(self, url):
        return None
    
    def set(self, url, content):
        pass


# Initialize both clients with your API credentials
trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=True)
data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)

################################################
#Finding which stocks are currently part of the SP500

def download_sp500_list():
    """Downloads current S&P 500 components from Wikipedia """
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        
        # Get company names
        return {
            'tickers': sp500_table['Symbol'].tolist(),
            'company_names': sp500_table[['Symbol', 'Security']].set_index('Symbol').to_dict()['Security'],
        }
    except Exception as e:
        print(f"Error downloading S&P 500 list: {str(e)}")
        return None

def update_sp500_list():
    """Updates stored list of S&P 500 companies if needed"""
    filename = '../sp500_components.json'
    
    # Load existing data if available
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                stored_data = json.load(f)
            old_tickers = set(stored_data.get('tickers', []))
            old_company_names = stored_data.get('company_names', {})
            last_update = datetime.strptime(stored_data.get('last_update', '2000-01-01'), '%Y-%m-%d')
        except Exception as e:
            print(f"Error reading stored data: {str(e)}")
            return None, None
    else:
        old_tickers = set()
        old_company_names = {}
        last_update = datetime.min
    
    # Check if update is needed
    if (datetime.now() - last_update).days >= 7:
        new_data = download_sp500_list()
        
        # If download failed, return stored data if available
        if new_data is None:
            print("Using previously stored S&P 500 list due to download failure")
            return list(old_tickers), old_company_names
            
        new_tickers = set(new_data['tickers'])
        
        # Print to verify company names are present
        print("\nFirst few company names being stored:")
        print(dict(list(new_data['company_names'].items())[:5]))
        
        data_to_store = {
            'tickers': list(new_tickers),
            'company_names': new_data['company_names'],  # Make sure we store company names
            'last_update': datetime.now().strftime('%Y-%m-%d'),
            'last_changes': {
                'added': list(new_tickers - old_tickers),
                'removed': list(old_tickers - new_tickers)
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data_to_store, f, indent=2)
                print("\nStored data sample:")
                print(json.dumps(dict(list(data_to_store['company_names'].items())[:5]), indent=2))
        except Exception as e:
            print(f"Error saving updated data: {str(e)}")
            
        return list(new_tickers), new_data['company_names']
    else:
        return stored_data['tickers'], stored_data.get('company_names', {})


################################################
#Deciding whether the code should run today

def should_collect_data(trading_client, test_mode=False):
    """
    Determines if we need to collect data.
    Handles weekend gaps by looking back further for the most recent trading day.
    """
    ny_now = datetime.now(ZoneInfo("America/New_York"))
    
    try:
        # Look back 10 days instead of 5 to ensure we catch the previous trading day
        calendar_request = GetCalendarRequest(
            start=(ny_now - timedelta(days=10)).strftime("%Y-%m-%d"),
            end=ny_now.strftime("%Y-%m-%d")
        )
        calendar = trading_client.get_calendar(calendar_request)
        
        if not calendar:
            logger.info("No recent market days found.")
            return False, None
        
        # Get the two most recent completed market days
        completed_days = []
        for day in reversed(calendar):
            close_time_str = f"{day.date} {day.close.strftime('%H:%M')}"
            close_time = datetime.strptime(close_time_str, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
            
            if close_time < ny_now:
                completed_days.append(close_time)
                if len(completed_days) == 2:  # We found both days we need
                    logger.info(f"Found last two trading days: {completed_days}")
                    return True, completed_days
        
        logger.info("Could not find two completed market days")
        return False, None
        
    except Exception as e:
        logger.error(f"Error in should_collect_data: {e}")
        return False, None


################################################
#Getting the daily movements of the stocks
def get_sp500_movements(trading_client, data_client, tickers, test_mode=False):
    """
    Fetches and analyzes recent stock movements using the latest market close data
    with explicit timezone handling
    """
    should_collect, market_days = should_collect_data(trading_client, test_mode)
    if not should_collect or not market_days:
        logger.error("Should not collect data or no market days available")
        return None, None
    
    try:
        # Log the runtime environment timezone
        logger.info(f"Runtime timezone: {datetime.now().astimezone().tzinfo}")
        
        end = market_days[0]  # Most recent
        start = market_days[1]  # Previous
        
        # Log original timestamps and their timezones
        logger.info(f"Original start: {start} ({start.tzinfo})")
        logger.info(f"Original end: {end} ({end.tzinfo})")
        
        # Convert to midnight UTC for complete day coverage
        start_utc = start.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(ZoneInfo("UTC"))
        end_utc = (end + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(ZoneInfo("UTC"))
        
        # Log UTC timestamps
        logger.info(f"UTC start: {start_utc}")
        logger.info(f"UTC end: {end_utc}")
        
        logger.info(f"Attempting to fetch data for {len(tickers)} tickers")
        
        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_utc,
            end=end_utc,
            feed='iex',
            adjustment='all'
        )
        
        bars = data_client.get_stock_bars(request)
        
        if not bars.data:
            logger.error("No data returned from Alpaca API request")
            return pd.DataFrame(columns=['Price', 'Change_Percent', 'Volume']), None
            
        logger.info(f"Received data for {len(bars.data)} symbols")
        
        # Log sample data with explicit timezone information
        sample_symbol = list(bars.data.keys())[0]
        logger.info(f"Sample data structure for {sample_symbol}:")
        logger.info(f"Number of bars: {len(bars.data[sample_symbol])}")
        for bar in bars.data[sample_symbol]:
            logger.info(f"Bar timestamp: {bar.timestamp} ({bar.timestamp.tzinfo})")
        
        # Process the stock data using the actual market days
        df = pd.DataFrame(columns=['Price', 'Change_Percent', 'Volume', 'Volume_Change_Percent'])
        formatted_date = end.strftime("market close on %B %d, %Y")
        
        processed_count = 0
        data_issues = []
        
        for symbol, stock_data in bars.data.items():
            try:
                if len(stock_data) >= 2:
                    # Sort by timestamp to ensure correct order
                    stock_data = sorted(stock_data, key=lambda x: x.timestamp)
                    today_data = stock_data[-1]
                    yesterday_data = stock_data[-2]
                    
                    # Log timestamps for first few stocks to verify correct dates
                    if processed_count < 5:
                        logger.info(f"{symbol} timestamps - Today: {today_data.timestamp}, Yesterday: {yesterday_data.timestamp}")
                    
                    latest_price = today_data.close
                    previous_price = yesterday_data.close
                    change_pct = ((latest_price - previous_price) / previous_price) * 100
                    
                    volume_change_pct = ((today_data.volume - yesterday_data.volume) / yesterday_data.volume) * 100
                    
                    df.at[symbol, 'Price'] = round(latest_price, 2)
                    df.at[symbol, 'Change_Percent'] = round(change_pct, 2)
                    df.at[symbol, 'Volume'] = today_data.volume
                    df.at[symbol, 'Volume_Change_Percent'] = round(volume_change_pct, 2)
                    processed_count += 1
                else:
                    data_issues.append(f"{symbol}: Insufficient data points ({len(stock_data)})")
                    
            except Exception as e:
                data_issues.append(f"{symbol}: Processing error - {str(e)}")
                continue
        
        logger.info(f"Successfully processed {processed_count} stocks")
        logger.info(f"Data issues encountered: {len(data_issues)}")
        if data_issues:
            logger.info("Sample of data issues:")
            for issue in data_issues[:5]:
                logger.info(issue)
        
        if processed_count == 0:
            logger.error("No stocks were successfully processed")
            return pd.DataFrame(columns=['Price', 'Change_Percent', 'Volume']), None
        
        result_df = df.sort_values('Change_Percent', ascending=False)
        logger.info(f"Final dataframe contains {len(result_df)} stocks")
        
        return result_df, formatted_date
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame(columns=['Price', 'Change_Percent', 'Volume']), None


################################################
#Finding news stories for LLM 

def search_bing_news(query: str, 
                     subscription_key: str, 
                     count: int = 10,
                     market: str = 'en-US',
                     category: Optional[str] = None) -> Dict:
    """
    Search Bing News API with the given query.
    
    Args:
        query (str): Search query string
        subscription_key (str): Bing API subscription key
        count (int): Number of results to return (default: 10)
        market (str): Market code (default: 'en-US')
        category (str, optional): News category filter (e.g., 'Business', 'Entertainment', etc.)
    
    Returns:
        dict: JSON response from the Bing News API
    """
    endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
    
    params = {
        'q': f'{query} NOT (site:msn.com OR site:www.msn.com)',  # Exclude MSN domains
        'count': count,
        'mkt': market,
        'freshness': 'Day',  # Restrict to last 24 hours
        'sortBy': 'relevance'  # Get newest articles first
    }
    
    if category:
        params['category'] = category
    
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return {}

def parse_serpapi_date(date_str):
    """Parse the specific date format returned by SerpApi"""
    try:
        # Remove the UTC suffix as it's redundant with the +0000
        date_str = date_str.replace(", +0000 UTC", "")
        # Now parse with dateutil
        return parser.parse(date_str)
    except Exception as e:
        logger.warning(f"Could not parse date: {str(e)}")
        return None

def search_stock_movement(stock_name, percent_change, num_links=6):
    """Searches for relevant news articles about stock movement using Google News via SerpApi"""
    logger.info(f"Searching for news about {stock_name}")
    
    try:
        # Calculate the cutoff time (12 hours ago)
        cutoff_time = datetime.now() - timedelta(hours=12)
        
        # Build query with business terms
        query = f'"{stock_name}" (stock OR price OR business OR finance OR market OR earnings) when:1d -site:msn.com -site:www.msn.com'
        
        # Parameters for the API request
        params = {
            'engine': 'google_news',
            'q': query,
            'api_key': serpapi_api_key,
            'num': num_links,  # Fixed to always get 6 results
            'gl': 'us',
            'tbm': 'nws'
        }
        
        # Make the API request
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or 'news_results' not in data:
            logger.warning(f"No valid news results from Google News for {stock_name}")
            return []
        
        # Extract URLs, checking the date of each article
        urls = []
        for article in data['news_results']:
            if 'link' not in article or 'date' not in article:
                continue
                
            article_date = parse_serpapi_date(article['date'])
            if article_date and article_date >= cutoff_time:
                urls.append(article['link'])
                    
        # Take at most 6 URLs
        urls = urls[:6]
        logger.info(f"Found {len(urls)} recent search results for {stock_name}")
        return urls
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error for {stock_name}: {str(e)}")
        return []
    except KeyError as e:
        logger.error(f"Unexpected API response format for {stock_name}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Search error for {stock_name}: {str(e)}")
        return []
    
def is_valid_article(content):
    """Check if article content meets our quality criteria"""
    if not content or content == "Could not retrieve article content":
        return False
    
    # Check minimum length (roughly 100 words)
    if len(content.split()) < 100:
        return False
        
    return True


def get_articles_content(urls):
    """Retrieves and processes article content from URLs"""
    articles = []
    for url in urls:
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            doc = Document(response.text)
            content = doc.summary()
            
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            if len(text) > 100:
                logger.info(f"Successfully extracted article content ({len(text)} chars)")
                articles.append(text)
            else:
                logger.warning(f"Article content too short ({len(text)} chars)")
                articles.append("Could not retrieve meaningful article content")
        except Exception as e:
            logger.error(f"Failed to fetch article: {str(e)}")
            articles.append("Could not retrieve article content")
    
    return articles

def get_three_valid_articles(ticker, percent_change, company_name):
    """Try to get 3 valid articles by first checking the first half of results, then the second if needed"""
    logger.info(f"Searching for articles about {ticker}")
    
    valid_articles = []
    valid_urls = []
    
    try:
        # Get all 6 URLs at once
        urls = search_stock_movement(company_name, percent_change)
        
        if not urls:
            # Try with ticker if company name search failed
            urls = search_stock_movement(ticker, percent_change)
            
        if not urls:
            logger.warning(f"No URLs found for {ticker}")
            return [], []
            
        logger.info(f"Found {len(urls)} URLs to try for {ticker}")
        
        # Split URLs into two groups
        first_batch = urls[:3]
        second_batch = urls[3:]
        
        # Try first batch
        if first_batch:
            articles = get_articles_content(first_batch)
            for i, article in enumerate(articles):
                if is_valid_article(article):
                    valid_articles.append(article)
                    valid_urls.append(first_batch[i])
            
        # If we don't have enough valid articles, try second batch
        if len(valid_articles) < 2 and second_batch:
            articles = get_articles_content(second_batch)
            for i, article in enumerate(articles):
                if is_valid_article(article):
                    valid_articles.append(article)
                    valid_urls.append(second_batch[i])
                if len(valid_articles) >= 3:
                    break
        
        logger.info(f"Found total of {len(valid_articles)} valid articles for {ticker}")
        return valid_articles, valid_urls
        
    except Exception as e:
        logger.error(f"Error in get_three_valid_articles for {ticker}: {str(e)}")
        return [], []



def format_articles(articles, max_chars_per_article=15000):
    """Formats article content for analysis with length limit"""
    formatted = []
    for i, text in enumerate(articles):
        # If text is longer than limit, truncate and add ellipsis
        if len(text) > max_chars_per_article:
            truncated_text = text[:max_chars_per_article] + "..."
            formatted.append(f"Article {i+1}:\n{truncated_text}")
        else:
            formatted.append(f"Article {i+1}:\n{text}")
    return "\n\n".join(formatted)


def ask_claude_if_article_is_relevant(articles: List[str], stock_info: Dict, date: str) -> List[bool]:
    """
    Uses Claude to determine if each article is relevant to today's stock movement.
    
    Args:
        articles: List of article texts to analyze
        stock_info: Dictionary containing stock information (ticker, company_name, percent_change)
        date: String representing the date of the stock movement
    
    Returns:
        List of booleans indicating if each article is relevant
    """
    anthropic = Anthropic(api_key=anthropic_api_key)
    relevance_scores = []
    
    for article in articles:
        prompt = f"""You are analyzing an article about {stock_info['company_name']} ({stock_info['ticker']}) to determine if it's relevant to explaining the stock's {abs(stock_info['percent_change']):.2f}% {'increase' if stock_info['percent_change'] > 0 else 'decrease'}.

Article content:
{article}

Things for you to consider:
- Does this article mention {stock_info['company_name']}?
- Does it give any reason why the stock price of {stock_info['company_name']} might have changed?
You must respond with EXACTLY one character: 1 if the article is relevant to today's stock movement, 0 if it's not. Do not include any other text, explanation, or characters in your response. Thank you so much!


You must respond with EXACTLY one character: 1 if the article is relevant to today's stock movement, 0 if it's not. Do not include any other text, explanation, or characters in your response."""


        try:
            message = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=50,
                temperature=0,
                system="You are a financial analyst. You must respond with EXACTLY one character: either 1 or 0. No other response is acceptable.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            
            # Extract just the first digit if it exists
            if response_text and response_text[0] in ('0', '1'):
                score = int(response_text[0])
            else:
                logger.warning(f"Unexpected response from Claude: {response_text}. Treating as irrelevant.")
                score = 0
                
            relevance_scores.append(score == 1)
            
        except Exception as e:
            logger.error(f"Error getting relevance score for article: {str(e)}")
            relevance_scores.append(False)
            
    return relevance_scores


################################################
# Analysis of news stories by LLM

def analyze_news(formatted_articles, stock_info, article_urls, market_metrics=None):
    """
    Generates analysis using Claude API using all available information
    
    Args:
        formatted_articles: News articles text (may be empty)
        stock_info: Dict containing stock information (name, ticker, percent_change)
        article_urls: List of article URLs
        market_metrics: Dict containing additional market context
    """
    anthropic = Anthropic(api_key=anthropic_api_key)
    stock_name = stock_info['company_name']
    ticker = stock_info['ticker']
    percent_change = stock_info['percent_change']
    
    # Build context sections
    prompt_parts = [
        f"Here is all available information about {stock_name} ({ticker}) stock's {abs(percent_change):.2f}% {'increase' if percent_change > 0 else 'decrease'} today:\n"
    ]
    
    # Add news articles if available
    if formatted_articles:
        prompt_parts.extend([
            "News Articles:",
            formatted_articles,
            "\n"
        ])
    
    # Add market context if available
    if market_metrics:
        prompt_parts.append("Market Context:")
        if market_metrics.get('market_change') is not None:
            prompt_parts.append(f"S&P 500 movement today: {market_metrics['market_change']:+.2f}%")
        if market_metrics.get('volatility_metrics'):
            prompt_parts.append(market_metrics['volatility_metrics'])
        if market_metrics.get('beta') is not None:
            prompt_parts.append(f"Beta measures a stock\'s volatility compared to the overall market (in this case the S&P500). A beta of 1.5 means that when the SP500 goes up by 1%, the stock goes up on average by 1.5%. A negative value indicates the stock and the market are anti-correlated.")
            prompt_parts.append(f"Beta relative to S&P 500: {market_metrics['beta']}")
        if market_metrics.get('RSI') is not None:
            prompt_parts.append(f"Relative Strength Index (RSI) is a momentum indicator that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions. If the RSI is greater than 70 it is considered overbought, and if it's under 30 it's considered oversold. If it's between 30 and 70 it is considered neutral.")
            prompt_parts.append(f"RSI: {market_metrics['RSI']}")
        prompt_parts.append("\n")
    
    # Add analysis request
    prompt_parts.append("Based on all available information above, please provide a 1-2 sentence analysis explaining the stock movement. If the news articles aren't clearly relevant to today's price action, focus on the market metrics and technical factors. Your explanation should be concise and focus only on the most likely drivers of today's specific price movement. Thank you very much for your hard work.")
    
    message = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        temperature=0,
        system="You are a financial analyst. Synthesize available information to explain stock movements. Focus on the most relevant factors that explain today's specific price action, whether that's news, market context, or technical factors.",
        messages=[
            {"role": "user", "content": "\n".join(prompt_parts)}
        ]
    )
    
    response = message.content[0].text
    return response, article_urls

def analyze_stock_with_retries(finance_data, position, data_client, company_names, wiki_urls):
    """Analyzes stock movement using market metrics, news, and company information"""
    try:
        stock_name = finance_data.index[position]
        change_percent = float(finance_data.iloc[position]['Change_Percent'])
        company_name = company_names.get(stock_name, stock_name)
        
        logger.info(f"Analyzing {company_name} ({stock_name})")
        
        # Gather market metrics
        market_metrics = {
            'market_change': get_market_movement(data_client),
            'volatility_metrics': get_volatility_and_volume_metrics(stock_name, data_client),
            'beta': calculate_betas([stock_name], ['SPY'], data_client).get(stock_name),
            "RSI": calculate_rsi(stock_name)
        }
        
        # Prepare stock info
        stock_info = {
            'ticker': stock_name,
            'company_name': company_name,
            'percent_change': change_percent
        }
        
        # Gather news articles
        articles, urls = get_three_valid_articles(stock_name, change_percent, company_name)
        
        if articles:
            # Filter articles by relevance
            relevance_scores = ask_claude_if_article_is_relevant(
                articles, 
                stock_info,
                datetime.now().strftime("%B %d, %Y")
            )
            
            # Keep only relevant articles and their URLs
            relevant_articles = [art for art, relevant in zip(articles, relevance_scores) if relevant]
            relevant_urls = [url for url, relevant in zip(urls, relevance_scores) if relevant]
            
            logger.info(f"Found {len(relevant_articles)} relevant articles out of {len(articles)} total")
            
            # Format articles if any relevant ones exist
            formatted_articles = format_articles(relevant_articles) if relevant_articles else ""
            urls_to_use = relevant_urls
        else:
            formatted_articles = ""
            urls_to_use = []
        
        # Always use analyze_news, letting the LLM determine what's relevant
        return analyze_news(formatted_articles, stock_info, urls_to_use, market_metrics)
            
    except Exception as e:
        logger.error(f"Analysis failed for {stock_name}: {e}")
        return (f"Unable to analyze {stock_name}'s movement due to technical issues.", [])


################################################
#Market metrics (wikipedia context, performance of a stock's sector, beta)

def get_wiki_content_from_url(wiki_url, max_paragraphs=3):
    """
    Gets the first few paragraphs of a Wikipedia page using its URL.
    
    Args:
        wiki_url (str): Full Wikipedia URL (e.g., "https://en.wikipedia.org/wiki/Apple_Inc.")
        max_paragraphs (int): Maximum number of paragraphs to return
    
    Returns:
        str or None: First few paragraphs of the Wikipedia page, or None if unavailable
    """
    if not wiki_url:  # Handle None or empty URL
        logger.warning("No Wikipedia URL provided")
        return None
        
    try:
        # Extract the page title from the URL
        page_title = wiki_url.split('/wiki/')[-1]
        
        # URL decode the page title
        page_title = requests.utils.unquote(page_title)
        logger.debug(f"Extracted page title: {page_title}")
        
        # API endpoint
        WIKI_API = "https://en.wikipedia.org/w/api.php"
        
        # Get the page content using the title
        content_params = {
            "action": "parse",
            "format": "json",
            "page": page_title,
            "prop": "text",
            "section": "0",
            "formatversion": "2"
        }
        
        content_response = requests.get(WIKI_API, params=content_params)
        content_data = content_response.json()
        
        if 'error' in content_data:
            logger.warning(f"Wikipedia API error: {content_data['error']}")
            return None
            
        # Parse HTML content
        html_content = content_data["parse"]["text"]
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Get paragraphs
        paragraphs = []
        for p in soup.find_all("p"):
            # Skip empty paragraphs or those with only reference tags
            text = p.text.strip()
            
            if not text or text.startswith("[") or len(text) < 50:
                continue
                
            # Clean up the text
            text = re.sub(r'\[\d+\]', '', text)  # Remove reference numbers
            text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
            paragraphs.append(text)
            
            if len(paragraphs) >= max_paragraphs:
                break
        
        if paragraphs:
            return " ".join(paragraphs)
        else:
            logger.warning("No valid paragraphs found in the content")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching Wikipedia content: {str(e)}")
        return None

def get_company_descriptions(wiki_urls):
    """
    Gets Wikipedia descriptions for companies using their Wikipedia URLs.
    
    Args:
        wiki_urls (dict): Dictionary mapping tickers to Wikipedia URLs
        
    Returns:
        dict: Mapping of tickers to company descriptions
    """
    descriptions = {}
    
    for ticker, url in wiki_urls.items():
        description = get_wiki_content_from_url(url)
        if description:
            descriptions[ticker] = description
            
    return descriptions


def get_volatility_and_volume_metrics(ticker, data_client):
    """Get volatility and volume metrics for a stock when news analysis fails"""
    try:
        # Get last month of daily data
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=30)
        
        request = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
            adjustment='all'
        )
        
        bars = data_client.get_stock_bars(request)
        if not bars.data:
            return None
            
        stock_data = bars.data[ticker]
        
        # Calculate price metrics
        prices = [bar.close for bar in stock_data]
        volatility = np.std(np.diff(prices) / prices[:-1]) * 100
        
        # Calculate volume metrics
        volumes = [bar.volume for bar in stock_data]
        avg_volume = np.mean(volumes[:-1])  # Excluding today
        today_volume = volumes[-1]
        volume_ratio = (today_volume / avg_volume) * 100
        
        # Format volume description
        if volume_ratio > 200:
            volume_desc = "extremely heavy"
        elif volume_ratio > 150:
            volume_desc = "heavy"
        elif volume_ratio > 75:
            volume_desc = "normal"
        else:
            volume_desc = "light"
            
        return f"Trading Metrics: {ticker} has shown {volatility:.1f}% daily volatility over the past month. Today's trading volume was {volume_desc} at {volume_ratio:.0f}% of the 30-day average ({today_volume:,} vs {avg_volume:,.0f} average)."
    except Exception as e:
        logger.error(f"Error getting volatility and volume metrics: {e}")
        return None
    
    


def calculate_betas(
    tickers: List[str], 
    reference_tickers: Union[str, List[str]], 
    data_client: StockHistoricalDataClient,
    lookback_months: int = 24
) -> Dict[str, Optional[float]]:
    """
    Calculate betas for a list of stocks against reference ticker(s)
    """
    logger.info(f"Calculating betas for {len(tickers)} stocks against {reference_tickers}")
    
    # Set time window
    end = datetime.now(ZoneInfo("America/New_York"))
    start = end - timedelta(days=lookback_months * 30)  # Convert months to days
    
    # Make sure reference_tickers is a list
    if isinstance(reference_tickers, str):
        reference_tickers = [reference_tickers]
        
    all_symbols = tickers + reference_tickers
    
    # Get monthly data for all stocks
    request = StockBarsRequest(
        symbol_or_symbols=all_symbols,
        timeframe=TimeFrame.Month,
        start=start,
        end=end,
        feed="iex",
        adjustment='all'
    )
    
    try:
        bars = data_client.get_stock_bars(request)
        
        # Convert to DataFrame
        monthly_prices = {}
        for symbol in all_symbols:
            if symbol in bars.data:
                prices = pd.DataFrame([
                    {
                        'date': bar.timestamp,
                        'close': bar.close
                    } for bar in bars.data[symbol]
                ])
                prices.set_index('date', inplace=True)
                monthly_prices[symbol] = prices
            else:
                logger.warning(f"No data available for {symbol}")
        
        # Calculate monthly returns
        monthly_returns = {}
        for symbol, prices in monthly_prices.items():
            try:
                returns = prices['close'].pct_change().dropna()
                if len(returns) < 2:
                    logger.warning(f"Insufficient return data for {symbol}")
                    continue
                monthly_returns[symbol] = returns
            except Exception as e:
                logger.error(f"Error calculating returns for {symbol}: {e}")
        
        # Calculate average reference returns
        try:
            reference_returns_df = pd.DataFrame({symbol: monthly_returns[symbol] 
                                              for symbol in reference_tickers 
                                              if symbol in monthly_returns})
            if reference_returns_df.empty:
                logger.error("No valid reference ticker data available")
                return {}
                
            reference_returns = reference_returns_df.mean(axis=1)
            reference_var = reference_returns.var()
            
            if reference_var == 0:
                logger.error("Reference variance is zero, cannot calculate betas")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating reference returns: {e}")
            return {}
        
        # Calculate betas
        betas = {}
        for symbol in tickers:
            try:
                if symbol in monthly_returns:
                    stock_returns = monthly_returns[symbol]
                    aligned_returns = pd.concat([stock_returns, reference_returns], axis=1).dropna()
                    
                    if len(aligned_returns) < 3:
                        logger.warning(f"Insufficient aligned data for {symbol}")
                        betas[symbol] = None
                        continue
                        
                    covariance = aligned_returns.iloc[:, 0].cov(aligned_returns.iloc[:, 1])
                    beta = covariance / reference_var
                    betas[symbol] = round(beta, 2)
                    
                    logger.debug(f"Calculated beta for {symbol}: {beta}")
                else:
                    logger.warning(f"No returns data available for {symbol}")
                    betas[symbol] = None
            except Exception as e:
                logger.error(f"Error calculating beta for {symbol}: {e}")
                betas[symbol] = None
                
        return betas
        
    except Exception as e:
        logger.error(f"Error in beta calculation process: {str(e)}")
        return {}

    
def calculate_rsi(ticker, period=10):
    """
    Calculate RSI for a given stock using Alpaca API.
    
    Parameters:
    - ticker (str): Stock ticker symbol
    - api_key (str): Alpaca API key
    - api_secret (str): Alpaca API secret
    - period (int): RSI period (default 10)
    
    Returns:
    - float: Current RSI value
    """
    # Initialize client
    client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
    
    # Set time period (get 3x period length to ensure enough data for calculation)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period * 3)
    
    # Request daily bars
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment='all',  # Get adjusted prices
        feed = "iex"
    )
    
    try:
        # Get historical data
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if df.empty:
            raise ValueError("No data received for ticker")
        
        # Calculate price changes
        df['change'] = df['close'].diff()
        
        # Create gains (positive) and losses (negative) Series
        df['gain'] = df['change'].clip(lower=0)
        df['loss'] = -df['change'].clip(upper=0)
        
        # Calculate average gains and losses over period
        avg_gain = df['gain'].rolling(window=period).mean()
        avg_loss = df['loss'].rolling(window=period).mean()
        
        # Calculate RS (Relative Strength) and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Get the most recent RSI value
        current_rsi = rsi.iloc[-1]
        
        return current_rsi
        
    except Exception as e:
        raise Exception(f"Error calculating RSI: {str(e)}")    
    

################################################
#Bonus Features (wikipedia links, graph)


def get_market_movement(data_client):
    """
    Get the daily movement of SPY as a proxy for the broader market
    Now handles weekend gaps correctly
    """
    try:
        # Look back 10 days to ensure we catch the last two trading days
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=10)
        
        request = StockBarsRequest(
            symbol_or_symbols=['SPY'],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed='iex',
            adjustment='all'
        )
        
        bars = data_client.get_stock_bars(request)
        
        if not bars.data or 'SPY' not in bars.data:
            logger.warning("No SPY data available")
            return None
            
        # Get the last two trading days' data
        spy_data = sorted(bars.data['SPY'], key=lambda x: x.timestamp)
        if len(spy_data) < 2:
            logger.warning("Insufficient SPY data available")
            return None
            
        # Use the last two available trading days
        today = spy_data[-1]
        yesterday = spy_data[-2]
        
        change_pct = ((today.close - yesterday.close) / yesterday.close) * 100
        return round(change_pct, 2)
        
    except Exception as e:
        logger.error(f"Error getting market movement: {e}")
        return None

def get_wiki_urls(company_names, symbols):
    """Gets Wikipedia URLs for specific companies using Google Search API"""
    service = build('customsearch', 'v1', developerKey=google_search_api_key)
    wiki_urls = {}
    
    for symbol in symbols:
        company_name = company_names.get(symbol, symbol)
        try:
            result = service.cse().list(
                q=f"{company_name} company",
                cx=google_search_id,
                num=1,
                siteSearch="wikipedia.org"
            ).execute()
            
            if 'items' in result:
                wiki_urls[symbol] = result['items'][0]['link']
        except Exception as e:
            print(f"Error searching for {company_name}: {e}")
            continue
            
    return wiki_urls  


def get_yearly_stock_data(ticker):
    """
    Fetches one year of historical daily stock data for a given ticker.
    Returns data formatted for the web UI graph
    """
    end = datetime.now(ZoneInfo("America/New_York"))
    start = end - timedelta(days=365)
    
    try:
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed='iex',
            adjustment='all'
        )
        
        bars = data_client.get_stock_bars(request)
        
        if not bars.data or ticker not in bars.data:
            return None
            
        # Format data for the chart
        chart_data = [{
            'date': bar.timestamp.strftime('%Y-%m-%d'),
            'close': round(bar.close, 2)
        } for bar in bars.data[ticker]]
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        return None


################################################
# Save the data for the website to json file
def save_to_json(timestamp, market_data, top_analysis, bottom_analysis, company_names, wiki_urls):
    """Updated save_to_json function to include historical data"""
    top_stock = market_data.index[0]
    bottom_stock = market_data.index[-1]
    
    # Get historical data
    top_history = get_yearly_stock_data(top_stock)
    bottom_history = get_yearly_stock_data(bottom_stock)
    
    analysis_data = {
        "timestamp": timestamp,
        "top_performer": {
            "stock": top_stock,
            "company_name": company_names.get(top_stock, top_stock),
            "wiki_url": wiki_urls.get(top_stock, ""),
            "change_percent": float(market_data.loc[top_stock, 'Change_Percent']),
            "price": float(market_data.loc[top_stock, 'Price']),
            "analysis": top_analysis[0],
            "article_urls": top_analysis[1],
            "historical_data": top_history
        },
        "bottom_performer": {
            "stock": bottom_stock,
            "company_name": company_names.get(bottom_stock, bottom_stock),
            "wiki_url": wiki_urls.get(bottom_stock, ""),
            "change_percent": float(market_data.loc[bottom_stock, 'Change_Percent']),
            "price": float(market_data.loc[bottom_stock, 'Price']),
            "analysis": bottom_analysis[0],
            "article_urls": bottom_analysis[1],
            "historical_data": bottom_history
        }
    }
    
    with open('output/market_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
        
        
        
        
################################################

# Modified main function to accept test_mode parameter
def main(test_mode=False):
    # Initialize Alpaca clients
    data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
    trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=True)
    
    # Get SP500 tickers and company names
    tickers, company_names = update_sp500_list()
    logger.info(f"Retrieved {len(tickers)} tickers from SP500 list")
    if not tickers:
        logger.error("No tickers retrieved from SP500 list")
        return
    
    # Get market data with test_mode enabled
    market_data, timestamp = get_sp500_movements(trading_client, data_client, tickers, test_mode=test_mode)
    
    if market_data is None or market_data.empty:
        logger.warning("No market data available")
        return

    logger.info(f"Analysis for {timestamp}")
    logger.info(f"Found {len(market_data)} stocks with price data")

    try:
        # Get top and bottom performers
        top_stock = market_data.index[0]
        bottom_stock = market_data.index[-1]
        
        # Only fetch Wikipedia URLs for these two stocks
        wiki_urls = {}
        service = build('customsearch', 'v1', developerKey=google_search_api_key, cache=NoOpCache())
        
        for stock in [top_stock, bottom_stock]:
            company_name = company_names.get(stock, stock)
            try:
                result = service.cse().list(
                    q=f"{company_name} company site:wikipedia.org",
                    cx=google_search_id,
                    num=1
                ).execute()
                
                if 'items' in result:
                    wiki_urls[stock] = result['items'][0]['link']
                    logger.info(f"Found Wikipedia URL for {stock}: {wiki_urls[stock]}")
                time.sleep(2)  # Short delay between the two requests
                    
            except Exception as e:
                logger.warning(f"Error getting Wikipedia URL for {company_name}: {e}")
                wiki_urls[stock] = ""

        # Perform analysis
        logger.info("Analyzing top performer...")
        top_analysis = analyze_stock_with_retries(market_data, 0, data_client, company_names, wiki_urls)
        logger.info("Top performer analysis complete")

        logger.info("Analyzing bottom performer...")
        bottom_analysis = analyze_stock_with_retries(market_data, -1, data_client, company_names, wiki_urls)
        logger.info("Bottom performer analysis complete")
        
        # Save results
        logger.info("Saving results to JSON...")
        save_to_json(timestamp, market_data, top_analysis, bottom_analysis, company_names, wiki_urls)
        logger.info("Analysis complete and saved to market_analysis.json")
        
    except Exception as e:
        logger.error(f"Error in main analysis loop: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
