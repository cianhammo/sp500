# %%
import os
import json
import logging
import feedparser
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import google.generativeai as genai
from bs4 import BeautifulSoup
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient  
from alpaca.trading.requests import GetCalendarRequest
from zoneinfo import ZoneInfo
from time import sleep, strftime, time
import warnings
import re
import math
import requests
from googleapiclient.discovery import build
import pytz
from typing import Dict, List, Optional

logger = None



alpaca_api_key = os.environ['ALPACA_API_KEY']
alpaca_secret_key = os.environ['ALPACA_SECRET_KEY']
google_search_api_key = os.environ['GOOGLE_SEARCH_API_KEY']
google_search_id = os.environ['GOOGLE_SEARCH_ID']
gemini_api_key = os.environ['GEMINI_API_KEY']

genai.configure(api_key=gemini_api_key)


# Configuration
@dataclass
class Config:
    """Application configuration"""
    # API Keys and Credentials

    
    # SEC Configuration
    SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "Company Name company@example.com")
    ALLOWED_FILING_TYPES: set = frozenset({"8-K", "10-Q", "10-K", "6-K", "20-F"})
    
    # File Paths
    OUTPUT_DIR: Path = Path("output")
    ARCHIVE_DIR: Path = Path("archive")
    LOG_DIR: Path = Path("sec_logs")
    
    # API Limits and Timeouts
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    REQUEST_TIMEOUT: int = 30
    
    # Processing Settings
    CHUNK_OVERLAP_RATIO: float = 0.1
    MAX_CHUNK_TOKENS: int = 100000
    
    def __post_init__(self):
        """Create necessary directories"""
        for directory in [self.OUTPUT_DIR, self.ARCHIVE_DIR, self.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)



def setup_logging(config: Config) -> logging.Logger:
    """Configure application logging"""
    global logger
        
    # Create default config if none provided
    if config is None:
        config = Config()
    
    # Remove all existing handlers from root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    logger = logging.getLogger('sec_monitor')
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logging
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File Handler
    log_file = config.LOG_DIR / f"sec_monitor_{datetime.now():%Y%m%d}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Custom Exceptions
class SECMonitorException(Exception):
    """Base exception for SEC monitoring application"""
    pass

class APIError(SECMonitorException):
    """API-related errors"""
    pass

class DataProcessingError(SECMonitorException):
    """Data processing related errors"""
    pass

class ConfigurationError(SECMonitorException):
    """Configuration related errors"""
    pass

# Base Classes
class APIClient(ABC):
    """Abstract base class for API clients"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Setup the API client"""
        pass
    
    def _make_request(self, 
                     method: str, 
                     url: str, 
                     **kwargs) -> Optional[Any]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = method(url, timeout=self.config.REQUEST_TIMEOUT, **kwargs)
                response.raise_for_status()
                return response
            except Exception as e:
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.MAX_RETRIES}): {str(e)}"
                )
                if attempt == self.config.MAX_RETRIES - 1:
                    raise APIError(f"Request failed after {self.config.MAX_RETRIES} attempts: {str(e)}")
        return None

class MetricsCollector:
    """Collect and store application metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: Dict[str, Any] = {
            "filings_processed": 0,
            "api_calls": 0,
            "errors": 0,
            "processing_time": 0,
            "input_words": 0,
            "output_words": 0
        }
    
    def increment(self, metric: str, value: int = 1):
        """Increment a metric"""
        if metric in self.metrics:
            self.metrics[metric] += value
            self.logger.debug(f"Metric {metric} increased by {value}")
    
    def set(self, metric: str, value: Any):
        """Set a metric value"""
        self.metrics[metric] = value
        self.logger.debug(f"Metric {metric} set to {value}")
    
    def get_report(self) -> Dict[str, Any]:
        """Get metrics report"""
        return self.metrics.copy()

class FileManager:
    """Manage file operations"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def save_json(self, data: Dict, filename: str):
        """Save data to JSON file"""
        try:
            filepath = self.config.OUTPUT_DIR / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Successfully saved data to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON file: {str(e)}")
            raise DataProcessingError(f"Failed to save JSON file: {str(e)}")
    
    def load_json(self, filename: str) -> Dict:
        """Load data from JSON file"""
        try:
            filepath = self.config.OUTPUT_DIR / filename
            if not filepath.exists():
                return {}
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {str(e)}")
            raise DataProcessingError(f"Failed to load JSON file: {str(e)}")
    
    def archive_file(self, filename: str, archive_suffix: str):
        """Archive a file"""
        try:
            source = self.config.OUTPUT_DIR / filename
            if not source.exists():
                return
            
            destination = self.config.ARCHIVE_DIR / f"{filename.stem}_{archive_suffix}{filename.suffix}"
            source.rename(destination)
            self.logger.info(f"Archived {filename} to {destination}")
        except Exception as e:
            self.logger.error(f"Failed to archive file: {str(e)}")
            raise DataProcessingError(f"Failed to archive file: {str(e)}")

# Health Check
class HealthCheck:
    """System health monitoring"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.status = {
            "api_status": {},
            "disk_space": {},
            "last_successful_run": None
        }
    
    def check_api_access(self) -> bool:
        """Check if all required APIs are accessible"""
        apis_to_check = {
            "SEC": "https://www.sec.gov/",
            "Alpaca": "https://api.alpaca.markets/v2/account",
            "Google": "https://customsearch.googleapis.com/"
        }
        
        all_healthy = True
        for name, url in apis_to_check.items():
            try:
                # Implement actual health check logic here
                self.status["api_status"][name] = "healthy"
            except Exception as e:
                self.status["api_status"][name] = "unhealthy"
                all_healthy = False
                self.logger.error(f"API health check failed for {name}: {str(e)}")
        
        return all_healthy
    
    def check_disk_space(self) -> bool:
        """Check if sufficient disk space is available"""
        dirs_to_check = [
            self.config.OUTPUT_DIR,
            self.config.ARCHIVE_DIR,
            self.config.LOG_DIR
        ]
        
        all_healthy = True
        for directory in dirs_to_check:
            try:
                total, used, free = Path(directory).absolute().stat().st_size
                free_gb = free / (1024 ** 3)
                
                if free_gb < 1.0:  # Less than 1GB free
                    all_healthy = False
                    self.status["disk_space"][str(directory)] = "low"
                    self.logger.warning(f"Low disk space in {directory}: {free_gb:.2f}GB free")
                else:
                    self.status["disk_space"][str(directory)] = "healthy"
            except Exception as e:
                all_healthy = False
                self.logger.error(f"Failed to check disk space for {directory}: {str(e)}")
        
        return all_healthy
    
    def update_last_successful_run(self):
        """Update timestamp of last successful run"""
        self.status["last_successful_run"] = datetime.now().isoformat()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.status.copy()

def initialize_system() -> Tuple[Config, logging.Logger, MetricsCollector, FileManager, HealthCheck]:
    """Initialize all system components"""
    try:
        # Load configuration
        config = Config()
        
        # Setup logging
        logger = LoggerSetup.setup(config)
        logger.info("Starting SEC monitoring system initialization")
        
        # Initialize components
        metrics = MetricsCollector(logger)
        file_manager = FileManager(config, logger)
        health_check = HealthCheck(config, logger)
        
        # Perform health checks
        if not health_check.check_api_access():
            raise ConfigurationError("One or more required APIs are not accessible")
        
        if not health_check.check_disk_space():
            raise ConfigurationError("Insufficient disk space available")
        
        logger.info("System initialization completed successfully")
        return config, logger, metrics, file_manager, health_check
        
    except Exception as e:
        # If logger isn't set up yet, print to console
        error_msg = f"Failed to initialize system: {str(e)}"
        if 'logger' in locals():
            logger.critical(error_msg)
        else:
            print(error_msg)
        raise ConfigurationError(error_msg)
        
        
def read_sp500_list():
    """Reads the stored list of S&P 500 companies from JSON file"""
    # Get directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'sp500_components.json')
    
    print(f"Script directory: {script_dir}")
    print(f"Attempting to read from: {filename}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir(os.getcwd())}")
    
    try:
        with open(filename, 'r') as f:
            stored_data = json.load(f)
            
        print(f"Successfully loaded JSON data")
        print(f"Data type: {type(stored_data)}")
        print(f"Keys in data: {stored_data.keys() if isinstance(stored_data, dict) else 'Not a dictionary'}")
        
        return stored_data
    except Exception as e:
        print(f"Error reading stored data: {str(e)}")
        return None

# %%
trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=True)
data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)

# %%
class SECFilingAnalyzer:
    def __init__(self, model, logger, max_retries=3, sleep_duration=30, conservative_coefficient=0.6):
        """
        Initialize the SEC filing analyzer.
        
        Args:
            model: The LLM model to use for analysis
            logger: Logger instance for tracking progress
            max_retries (int): Maximum number of retries for API calls
            sleep_duration (int): Seconds to wait between API calls
            conservative_coefficient (float): Safety factor for token limits
        """
        self.model = model
        self.max_retries = max_retries
        self.sleep_duration = sleep_duration
        self.conservative_coefficient = conservative_coefficient
        self.now = datetime.now()
        self.logger = logger


    def analyze_filing(self, text):
        """
        Main method that coordinates chunking and summarizing a filing.
        
        Args:
            text (str): The full text of the SEC filing
            
        Returns:
            str: The final summarized analysis
        """
        chunks = self._chunk_text(text)
        previous_summary = None
        all_summaries = []
        
        for i, chunk in enumerate(chunks):
            sleep(self.sleep_duration)
            summary = self._summarize_chunk(
                chunk,
                previous_summary=previous_summary, 
                is_first=i==0,
                is_last=i==len(chunks)-1
            )
            
            if summary:
                previous_summary = summary
                all_summaries.append(summary)
        
        if len(all_summaries) > 1:
            return self._combine_summaries(all_summaries)
        return all_summaries[0] if all_summaries else "Failed to analyze filing"

    def _chunk_text(self, text, overlap_ratio=0.01):
        """
        Splits text into chunks based on token limits.
        
        Args:
            text (str): Input text to chunk
            overlap_ratio (float): Ratio of text to overlap between chunks
            
        Returns:
            list[str]: List of text chunks
        """
        total_tokens = self.model.count_tokens(text).total_tokens * 1.2  # 20% safety margin
        tokens_per_chunk = int(999000 * self.conservative_coefficient)
        
        if total_tokens < tokens_per_chunk:
            return [text]
            
        summary_tokens = 200
        tokens_per_apparent_chunk = tokens_per_chunk - summary_tokens - int(tokens_per_chunk * overlap_ratio)
        n_chunks = math.ceil(total_tokens / tokens_per_apparent_chunk)
        total_chars = len(text)
        
        chars_per_token = total_chars / total_tokens
        chars_per_chunk = int(tokens_per_chunk * chars_per_token)
        chars_per_overlap = int(chars_per_chunk * overlap_ratio)
        
        chunks = []
        start = 0
        end = chars_per_chunk
        
        for _ in range(n_chunks-1):
            chunks.append(text[start:end])
            diff = chars_per_chunk - chars_per_overlap
            start = start + diff
            if end < total_chars - 1 - chars_per_chunk:
                end = end + diff
            else:
                end = total_chars - 1
        
        if start < total_chars:
            chunks.append(text[start:])
        
        return chunks

    def _summarize_chunk(self, chunk, previous_summary=None, is_first=True, is_last=True):
        """
        Summarize a chunk while maintaining context from previous chunks.
        
        Args:
            chunk (str): Text chunk to summarize
            previous_summary (str, optional): Summary from previous chunk
            is_first (bool): Whether this is the first chunk
            is_last (bool): Whether this is the last chunk
            
        Returns:
            str: Summarized chunk
        """
        self.logger.info("="*50)
        self.logger.info(f"Processing {'first' if is_first else 'middle' if not is_last else 'last'} chunk")
        self.logger.info(f"Timestamp: {strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"token percentage = {round(self.model.count_tokens(chunk).total_tokens/10000)}%")
        formatted_date = self.now.strftime("%B %d, %Y")

        context_prompt = ""
        if not is_first:
            context_prompt = f"""
            Important: This is a continuation of a longer document. 
            Here's what we know from the previous section:
            {previous_summary}
            
            Only mention new information that wasn't covered in the previous summary.
            """
        
        if not is_last:
            context_prompt += "\nNote: This is not the end of the document."
        
        prompt = f"""
        Analyze this section of an SEC filing. {context_prompt}
        Provide maximum 3 points. Consider the following guidance:
        1. The most noteworthy information (e.g. focus on changes, surprises, or significant developments)
        2. The most meaningful numerical comparison (e.g. prioritize year-over-year changes)
        3. Specific future impacts or risks (include numbers where available)
        4. Regulatory or legal developments

        Key guidelines:
        - There should be at most 3 points. If there's not that much interesting info then just returning 1 or 2 points is fine.
        - The file may contain lots of annoying binary or XRBL data. Ignore this and focus on the human-readable info.
        - Be careful not to fall for marketing spin. We are only interested in the facts.
        - If nothing is unusual, focus on the largest changes or most concrete information.
        - Each point should be new information, so don't repeat yourself.
        - Limit each point to max 1 sentence.
        - Be very concise.
        - Today's date is {formatted_date}, keep this in mind when summarising info relating to predictions or past performance, and make it clear which you are referring to.
        - Don't introduce the points, just start the analysis straight away.
        - Number each point.
        
        {chunk}
        """
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Making API call - Attempt {attempt + 1}/{self.max_retries}")
                self.logger.info(f"Timestamp: {strftime('%Y-%m-%d %H:%M:%S')}")
                response = self.model.generate_content(prompt)
                self.logger.info("API call successful")
                return response.text
            except Exception as e:
                self.logger.error(f"API call failed - Attempt {attempt + 1}")
                self.logger.error(f"Error: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to summarize chunk after {self.max_retries} attempts: {e}")
                    return None
                self.logger.info(f"Sleeping for {self.sleep_duration*2} seconds before retry...")
                sleep(self.sleep_duration * 2)

    def _combine_summaries(self, summaries):
        """
        Combines multiple chunk summaries into one coherent summary.
        
        Args:
            summaries (list[str]): List of chunk summaries
            
        Returns:
            str: Combined final summary
        """
        print(f"\n{'='*50}")
        print("Starting summary combination")
        print(f"Number of summaries to combine: {len(summaries)}")
        print(f"Timestamp: {strftime('%Y-%m-%d %H:%M:%S')}")
        formatted_date = self.now.strftime("%B %d, %Y")
        
        combined_text = "\n\nSection Summary #".join(
            f"{i+1}:\n{summary}" for i, summary in enumerate(summaries)
        )
        
        prompt = f"""
        You are analyzing multiple summaries from different sections of the same SEC filing.
        Create a final analysis with at most 3 key points. Follow these rules:

        Priority order for information:
        1. The most noteworthy information (e.g. focus on changes, surprises, or significant developments)
        2. The most meaningful numerical comparison (e.g. prioritize year-over-year changes)
        3. Specific future impacts or risks (include numbers where available)
        4. Regulatory or legal developments

        Key requirements:
        - There should be at most 3 points. If there's not that much interesting info then just returning 1 or 2 points is fine.
        - The file may contain lots of annoying binary or XRBL data. Ignore this and focus on the human-readable info.
        - Be careful not to fall for marketing spin. We are only interested in the facts.
        - If nothing is unusual, focus on the largest changes or most concrete information.
        - Each point should be new information, so don't repeat yourself.
        - Limit each point to max 1 sentence.
        - Be very concise.
        - Today's date is {formatted_date}, keep this in mind when summarising info relating to predictions or past performance, and make it clear which you are referring to.
        - Don't introduce the points, just start the analysis straight away.
        - Number each point.

        Summaries from different sections:
        {combined_text}
        """
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Making combine API call - Attempt {attempt + 1}/{self.max_retries}")
                self.logger.info(f"Timestamp: {strftime('%Y-%m-%d %H:%M:%S')}")
                response = self.model.generate_content(prompt)
                self.logger.info("Combine API call successful")
                if response.text:
                    return response.text
            except Exception as e:
                self.logger.info(f"Combine API call failed - Attempt {attempt + 1}")
                self.logger.info(f"Error: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.info(f"Error combining summaries: {e}")
                    return summaries[0]  # Fallback to first summary
                self.logger.info(f"Sleeping for {self.sleep_duration} seconds before retry...")
                sleep(self.sleep_duration)

# %%

headers = {
    'User-Agent': 'Cian Hamilton cianhammo@gmail.com'
}
sec_tickers_url = "https://www.sec.gov/files/company_tickers.json"
tickers_response = requests.get(sec_tickers_url, headers=headers)
cik_to_ticker = {str(company_info['cik_str']).zfill(10): company_info['ticker'] 
                 for company_info in tickers_response.json().values()}

# %%

class SECFilingFetcher:
    def __init__(self, user_agent, sp500_tickers, sp500_company_names, cik_to_ticker, logger):
        """
        Initialize the SEC filing fetcher.
        
        Args:
            user_agent (str): Email/name for SEC requests
            sp500_tickers (set): Set of SP500 ticker symbols
            sp500_company_names (dict): Mapping of tickers to company names
            cik_to_ticker (dict): Mapping of CIK numbers to tickers
            logger: Logger instance
        """
        self.headers = {'User-Agent': user_agent}
        self.sp500_tickers = sp500_tickers
        self.sp500_company_names = sp500_company_names
        self.cik_to_ticker = cik_to_ticker
        self.allowed_types = {"8-K", "10-Q", "10-K", "6-K", "20-F"}
        self.timezone = pytz.timezone('US/Eastern')
        self.logger = logger

    def fetch_recent_filings(self, max_requests=100, request_delay=0.5):
        """
        Fetches SEC filings from last 24h for S&P 500 companies.
        """
        sp500_filings = []
        start = 0
        eastern = pytz.timezone('US/Eastern')
        one_day_ago = datetime.now(eastern) - timedelta(days=1)

        while start < max_requests * 100:
            try:
                rss_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start={start}&count=100&output=atom"
                
                self.logger.debug(f"Fetching filings from: {rss_url}")
                
                # Fetch batch of 100 filings
                response = requests.get(rss_url, headers=self.headers)
                feed = feedparser.parse(response.text)

                # Debug logging
                self.logger.debug(f"Number of entries in feed: {len(feed.entries)}")
                if feed.entries:
                    self.logger.debug(f"Sample entry title: {feed.entries[0].title}")

                if not feed.entries:
                    self.logger.info("No more entries found")
                    break

                # Check age of last filing in batch
                last_filing_date = datetime.strptime(feed.entries[-1].updated, "%Y-%m-%dT%H:%M:%S%z")
                self.logger.debug(f"Last filing date: {last_filing_date}")
                self.logger.debug(f"Cutoff date: {one_day_ago}")

                if last_filing_date >= one_day_ago:
                    # Process whole batch
                    for entry in feed.entries:
                        cik_match = re.search(r'/data/(\d+)/', entry.link)
                        accession_match = re.search(r'/(\d{10}-\d{2}-\d{6})', entry.link)

                        if cik_match and accession_match:
                            cik = cik_match.group(1).zfill(10)
                            ticker = self.cik_to_ticker.get(cik)

                            if ticker and ticker in self.sp500_tickers:
                                self.logger.debug(f"Found matching filing for {ticker}")
                                filing_data = self._create_filing_data(ticker, entry, cik, accession_match.group(1))
                                sp500_filings.append(filing_data)

                    sleep(request_delay)
                    start += 100
                else:
                    # Process entries until we hit an old one
                    for entry in feed.entries:
                        filing_date = datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%S%z")

                        if filing_date < one_day_ago:
                            self.logger.info(f"Reached old filing, stopping. Found {len(sp500_filings)} total filings")
                            return sp500_filings

                        cik_match = re.search(r'/data/(\d+)/', entry.link)
                        accession_match = re.search(r'/(\d{10}-\d{2}-\d{6})', entry.link)

                        if cik_match and accession_match:
                            cik = cik_match.group(1).zfill(10)
                            ticker = self.cik_to_ticker.get(cik)

                            if ticker and ticker in self.sp500_tickers:
                                self.logger.debug(f"Found matching filing for {ticker}")
                                filing_data = self._create_filing_data(ticker, entry, cik, accession_match.group(1))
                                sp500_filings.append(filing_data)

                    return sp500_filings

            except Exception as e:
                self.logger.error(f"Error fetching page {start}: {str(e)}")
                break

        self.logger.info(f"Completed fetching. Found {len(sp500_filings)} total filings")
        return sp500_filings

    def _create_filing_data(self, ticker, entry, cik, accession_number):
        """Helper to create filing data dictionary"""
        return {
            'ticker': ticker,
            'company_name': self.sp500_company_names.get(ticker, "Unknown Company"),
            'filing_title': entry.title,
            'filing_date': datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%S%z"),
            'accession_number': accession_number,
            'link': entry.link,
            'txt_url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-','')}/{accession_number}.txt",
            'html_url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace('-','')}/{accession_number}.html",
        }

    def get_text_from_link(self, input_link):
        """Get text content from SEC filing URL using html5lib parser"""
        try:
            response = requests.get(input_link, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html5lib")
            return soup.get_text()
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching URL: {e}")
            return None

    def get_interesting_filings(self, filings_inputs: list, processed_accessions: set = None) -> list:
        """
        Filter filings to only include interesting types and unprocessed filings.

        Args:
            self: Instance of SECFilingFetcher
            filings_inputs (list): List of filing dictionaries
            processed_accessions (set, optional): Set of already processed accession numbers

        Returns:
            list: Filtered list of filings
        """
        if processed_accessions is None:
            processed_accessions = set()

        filtered_filings = []

        for filing in filings_inputs:
            # Check filing type
            filing_type = filing["filing_title"].split(" - ")[0]
            if filing_type not in self.allowed_types:
                continue

            # Skip if already processed
            if filing['accession_number'] in processed_accessions:
                self.logger.debug(f"Skipping already processed filing: {filing['accession_number']}")
                continue

            filtered_filings.append(filing)

        self.logger.info(f"Found {len(filtered_filings)} new interesting filings")
        return filtered_filings

    def _filter_filing_type(self, filing_text):
        """Check if filing type is in allowed types"""
        filing_type = filing_text.split(" - ")[0]
        return filing_type in self.allowed_types

# %%
def get_stock_movements(trading_client, data_client, filing_tickers, filing_data, test_mode=False):
    """
    Fetches stock movements for specific tickers considering market hours and filing times.
    
    Args:
        trading_client: Alpaca trading client
        data_client: Alpaca data client
        filing_tickers: List of tickers to check
        filing_data: List of filing dictionaries containing filing times
        test_mode: Boolean for test mode
    
    Returns:
        DataFrame with stock price data, formatted market close date string
    """
    if not filing_tickers:
        logger.warning("No tickers provided for stock movement lookup")
        return pd.DataFrame(columns=['Price', 'Change_Percent']), None

    try:
        
        # Check if market is currently open
        clock = trading_client.get_clock()
        market_open = clock.is_open
        current_time = clock.timestamp.astimezone(pytz.timezone('US/Eastern'))

        # DEBUG
        logger.info(f"filing_tickers type: {type(filing_tickers)}")
        logger.info(f"filing_data type: {type(filing_data)}")

        # Create mapping of tickers to filing times
        filing_times = {
            filing['ticker']: filing['filing_date'] 
            for filing in filing_data 
            if filing['ticker'] in filing_tickers
        }

        # Initialize result DataFrame
        df = pd.DataFrame(columns=['Price', 'Change_Percent'])

        # Get the current/last trading day
        calendar_request = GetCalendarRequest(
            start=current_time.date() - timedelta(days=5),
            end=current_time.date()
        )
        calendar = trading_client.get_calendar(calendar_request)
        last_trading_day = calendar[-1].date
        
        # For each ticker, determine what price data to show
        for ticker in filing_tickers:
            try:
                filing_time = filing_times[ticker]
                filing_date = filing_time.date()
                
                # Get historical bars for price data
                bars_request = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Day,
                    start=(filing_date - timedelta(days=5)).strftime('%Y-%m-%d'),
                    end=current_time.strftime('%Y-%m-%d'),
                    feed='iex',
                    adjustment='all'
                )
                bars = data_client.get_stock_bars(bars_request)
                
                if not bars.data or ticker not in bars.data:
                    logger.warning(f"No price data available for {ticker}")
                    continue
                
                bar_data = bars.data[ticker]
                
                if market_open:
                    # Market is open - show current price vs previous close
                    latest_price = bar_data[-1].close
                    prev_close = bar_data[-2].close if len(bar_data) > 1 else None
                    
                    if prev_close:
                        change_pct = ((latest_price - prev_close) / prev_close) * 100
                        df.at[ticker, 'Price'] = round(latest_price, 2)
                        df.at[ticker, 'Change_Percent'] = round(change_pct, 2)
                
                else:
                    # Market is closed - check filing time
                    if is_during_market_hours(filing_time) and filing_date == last_trading_day:
                        # Filing was during market hours on the last trading day
                        # Show change from previous close to that day's close
                        filing_day_close = bar_data[-1].close
                        prev_close = bar_data[-2].close if len(bar_data) > 1 else None
                        
                        if prev_close:
                            change_pct = ((filing_day_close - prev_close) / prev_close) * 100
                            df.at[ticker, 'Price'] = round(filing_day_close, 2)
                            df.at[ticker, 'Change_Percent'] = round(change_pct, 2)
                    else:
                        # Filing was outside market hours or on a different day
                        # Show last price but no change
                        latest_price = bar_data[-1].close
                        df.at[ticker, 'Price'] = round(latest_price, 2)
                        df.at[ticker, 'Change_Percent'] = None
                
            except Exception as e:
                logger.error(f"Error processing price data for {ticker}: {str(e)}")
                continue
        
        # Format date string based on market state
        if market_open:
            formatted_date = current_time.strftime("%B %d, %Y %I:%M %p ET")
        else:
            formatted_date = last_trading_day.strftime("market close on %B %d, %Y")
        
        return df, formatted_date
        
    except Exception as e:
        logger.error(f"Error fetching stock movements: {str(e)}")
        return pd.DataFrame(columns=['Price', 'Change_Percent']), None
    
    

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


def create_output_json(interesting_filings, summary_results, wiki_urls, stock_data, input_word_count, output_word_count):
    """
    Creates formatted JSON output from filing data and summaries.
    
    Args:
        interesting_filings (list): List of filing dictionaries
        summary_results (list): List of dicts containing 'filing' and 'summary' keys
        wiki_urls (dict): Dictionary mapping tickers to Wikipedia URLs
        stock_data (pd.DataFrame): DataFrame with stock price data indexed by ticker
        input_word_count (int): Total input words processed
        output_word_count (int): Total output words in summaries
    
    Returns:
        dict: Formatted output ready for JSON serialization
    """
    # Define filing type priority (lower number = higher priority)
    filing_priority = {
        '10-K': 0,
        '10-Q': 1,
        '8-K': 2,
        '6-K': 3,
        '20-F': 4
    }
    
    # Create a mapping of filing URLs to their summaries for easy lookup
    summary_map = {
        result['filing']['link']: result['summary']
        for result in summary_results
    }
    
    # Create list of filings
    filings = []
    for filing in interesting_filings:
        ticker = filing['ticker']
        filing_type = filing['filing_title'].split(' - ')[0]
        
        # Get summary points for this filing
        summary_text = summary_map.get(filing['link'], '')
        summary_points = [
            point.strip() 
            for point in summary_text.split('\n') 
            if point.strip()
        ]
        
        # Create filing entry
        filing_entry = {
            "company_name": filing['company_name'],
            "stock": ticker,
            "wiki_url": wiki_urls.get(ticker, ""),
            "change_percent": stock_data.get(ticker, {}).get('Change_Percent', 0),
            "price": stock_data.get(ticker, {}).get('Price', 0),
            "filing_type": filing_type,
            "filing_time": filing['filing_date'].isoformat(),
            "filing_url": filing['link'],
            "summary_points": summary_points
        }
        filings.append(filing_entry)
    
    # Sort filings by type priority
    filings.sort(key=lambda x: filing_priority.get(x['filing_type'], 999))
    
    # Create final output structure
    output = {
        "timestamp": datetime.now(pytz.timezone('US/Eastern')).strftime('%b %-d, %Y %-I:%M %p EST'),
        "stats": {
            "input_words": input_word_count,
            "output_words": output_word_count,
            "filings_processed": len(interesting_filings)
        },
        "filings": filings
    }
    
    return output

def save_json(output_json, filename="sec_filings.json"):
    """
    Saves JSON data to a file with pretty formatting
    
    Args:
        output_json: The JSON data to save
        filename: Name of the file to save to (default: 'sec_filings.json')
    """
    with open(filename, 'w') as f:
        json.dump(output_json, f, indent=4)
        


# %%
eastern = pytz.timezone('US/Eastern')
today = datetime.now(eastern)
yesterday = today - timedelta(days=1)

market_days = [today, yesterday]

def word_counter(input_text):
    return len(re.findall(r'(?<!\s)\s(?!\s)', input_text)) + 1

# %%



def process_filings(fetcher, analyzer, trading_client, data_client, logger):
    """Process SEC filings and generate summaries"""
    logger.info("Fetching recent SEC filings")
    filings = fetcher.fetch_recent_filings()
    interesting_filings = fetcher.get_interesting_filings(filings)
    logger.info(f"Found {len(interesting_filings)} interesting filings to process")

    summary_results = []
    input_word_count = 0
    output_word_count = 0

    for i, filing in enumerate(interesting_filings, 1):
        try:
            logger.info(f"Processing filing {i}/{len(interesting_filings)}: {filing['ticker']}")
            text = fetcher.get_text_from_link(filing["txt_url"])

            if not text:
                logger.warning(f"No text content for {filing['ticker']}")
                continue

            input_word_count += word_counter(text)
            summary = analyzer.analyze_filing(text)
            
            if not summary:
                logger.warning(f"No summary generated for {filing['ticker']}")
                continue

            output_word_count += word_counter(summary)
            summary_results.append({
                'filing': filing,
                'summary': summary
            })

            sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error processing filing {filing['ticker']}: {str(e)}")
            continue

    return summary_results, interesting_filings, input_word_count, output_word_count

def save_output(interesting_filings, summary_results, trading_client, data_client, 
                input_word_count, output_word_count, logger):
    """Save processed data to JSON"""
    tickers = [f['ticker'] for f in interesting_filings]
    company_names = {f['ticker']: f['company_name'] for f in interesting_filings}

    stock_data, market_date = get_stock_movements(
        trading_client, 
        data_client, 
        tickers
    )

    wiki_urls = get_wiki_urls(company_names, tickers)
    
    json_output = create_output_json(
        interesting_filings,
        summary_results,
        wiki_urls,
        stock_data,
        input_word_count,
        output_word_count
    )

    # Setup directories
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sec_filings.json"

    # Archive existing file
    if output_path.exists():
        archive_dir = Path("archive")
        archive_dir.mkdir(exist_ok=True)
        archive_name = f"sec_filings_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_path.rename(archive_dir / archive_name)

    # Save new output
    save_json(json_output, output_path)
    logger.info(f"Successfully saved output to {output_path}")
    logger.info(f"Processing complete. Stats: {json_output['stats']}")

    return json_output


# %%
def initialize_sp500_data(logger):
    """Initialize S&P 500 data with proper error handling and logging"""
    try:
        logger.info("Starting S&P 500 data initialization")
        
        # Download latest S&P 500 list
        sp500_info = read_sp500_list()
        if not sp500_info:
            logger.error("Failed to download S&P 500 list")
            raise Exception("Failed to download S&P 500 list")
            
        # Validate the data structure
        if not isinstance(sp500_info, dict) or 'tickers' not in sp500_info or 'company_names' not in sp500_info:
            logger.error(f"Invalid S&P 500 data structure: {sp500_info}")
            raise ValueError("Invalid S&P 500 data structure")
            
        # Convert tickers to set for efficient lookup
        sp500_tickers = set(sp500_info["tickers"])
        
        logger.info(f"Successfully loaded {len(sp500_tickers)} S&P 500 tickers")
        logger.debug(f"First 5 tickers: {list(sp500_tickers)[:5]}")
        
        return {
            'tickers': sp500_tickers,
            'company_names': sp500_info["company_names"]
        }
        
    except Exception as e:
        logger.error(f"Error initializing S&P 500 data: {str(e)}")
        raise

def initialize_components(logger, sp500_data):
    """Initialize all components with proper error handling"""
    try:
        logger.info("Initializing components")
        
        # Initialize SEC fetcher with explicit S&P 500 data
        fetcher = SECFilingFetcher(
            user_agent="Cian Hamilton cianhammo@gmail.com",
            sp500_tickers=sp500_data['tickers'],
            sp500_company_names=sp500_data['company_names'],
            cik_to_ticker=cik_to_ticker,
            logger = logger
        )
        logger.info("SEC fetcher initialized")
        
        # Initialize Gemini model
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        analyzer = SECFilingAnalyzer(
        model=model,
        logger=logger)
        logger.info("Filing analyzer initialized")
        
        # Initialize Alpaca clients
        trading_client = TradingClient(alpaca_api_key, alpaca_secret_key, paper=True)
        data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        logger.info("Alpaca clients initialized")
        
        return fetcher, analyzer, trading_client, data_client
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise


# %%
def is_json_from_today(json_path: Path, logger: logging.Logger) -> bool:
    """
    Checks if the existing JSON file is from today.
    
    Args:
        json_path (Path): Path to the JSON file
        logger (logging.Logger): Logger instance
        
    Returns:
        bool: True if file is from today, False otherwise
    """
    try:
        if not json_path.exists():
            return False
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Parse the timestamp from the JSON
        timestamp = datetime.strptime(data['timestamp'], '%b %d, %Y %I:%M %p EST')
        timestamp = pytz.timezone('US/Eastern').localize(timestamp)
        
        # Check if timestamp is from today
        today = datetime.now(pytz.timezone('US/Eastern')).date()
        return timestamp.date() == today
        
    except Exception as e:
        logger.error(f"Error checking JSON timestamp: {str(e)}")
        return False
    
    
def get_processed_accessions(existing_filings: dict, logger: logging.Logger) -> set:
    """
    Extracts set of processed accession numbers from existing filings.
    
    Args:
        existing_filings (dict): Loaded JSON data
        logger (logging.Logger): Logger instance
        
    Returns:
        set: Set of accession numbers
    """
    processed = set()
    try:
        for filing in existing_filings.get('filings', []):
            accession = filing.get('accession_number')
            if accession:
                processed.add(accession)
        
        logger.info(f"Found {len(processed)} previously processed filings")
        return processed
        
    except Exception as e:
        logger.error(f"Error extracting accession numbers: {str(e)}")
        return set()
    
def handle_existing_json(config: Config, logger: logging.Logger) -> Tuple[set, bool]:
    """
    Handles existing JSON file - checks if from today, archives if old.
    
    Args:
        config (Config): Application configuration
        logger (logging.Logger): Logger instance
        
    Returns:
        Tuple[set, bool]: Set of processed accessions and whether file is from today
    """
    output_path = config.OUTPUT_DIR / "sec_filings.json"
    
    try:
        if not output_path.exists():
            logger.info("No existing JSON file found")
            return set(), False
            
        is_today = is_json_from_today(output_path, logger)
        
        if is_today:
            logger.info("Found JSON file from today")
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            return get_processed_accessions(existing_data, logger), True
            
        else:
            logger.info("Found old JSON file, archiving")
            archive_dir = config.ARCHIVE_DIR
            archive_dir.mkdir(exist_ok=True)
            
            archive_name = f"sec_filings_{datetime.now():%Y%m%d_%H%M%S}.json"
            output_path.rename(archive_dir / archive_name)
            
            return set(), False
            
    except Exception as e:
        logger.error(f"Error handling existing JSON: {str(e)}")
        return set(), False

def merge_with_existing_json(new_filings: list, config: Config, logger: logging.Logger, new_input_words: int, new_output_words: int) -> None:
    """
    Merges new filings with existing JSON file from today.
    
    Args:
        new_filings (list): List of new filing entries
        config (Config): Application configuration
        logger (logging.Logger): Logger instance
        new_input_words (int): Word count from new filings' input text
        new_output_words (int): Word count from new filings' summaries
    """
    output_path = config.OUTPUT_DIR / "sec_filings.json"
    
    try:
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            
        # Update timestamp
        existing_data['timestamp'] = datetime.now(pytz.timezone('US/Eastern')).strftime('%b %-d, %Y %-I:%M %p EST')
        
        # Update stats by adding new counts to existing ones
        existing_data['stats']['input_words'] += new_input_words
        existing_data['stats']['output_words'] += new_output_words
        existing_data['stats']['filings_processed'] += len(new_filings)
        
        # Add new filings
        existing_data['filings'].extend(new_filings)
        
        # Sort filings by filing time (newest first)
        existing_data['filings'].sort(key=lambda x: x['filing_time'], reverse=True)
        
        # Save updated JSON
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        logger.info(f"Successfully merged {len(new_filings)} new filings with existing JSON")
        logger.info(f"Cumulative stats - Input words: {existing_data['stats']['input_words']}, " 
                   f"Output words: {existing_data['stats']['output_words']}")
        
    except Exception as e:
        logger.error(f"Error merging with existing JSON: {str(e)}")
        



# %%
def is_during_market_hours(filing_time):
    """
    Check if filing occurred during regular market hours (9:30 AM - 4:00 PM ET)
    """
    # Convert to Eastern time if not already
    if filing_time.tzinfo != pytz.timezone('US/Eastern'):
        filing_time = filing_time.astimezone(pytz.timezone('US/Eastern'))
    
    # Market hours: 9:30 AM to 4:00 PM ET
    market_start = filing_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = filing_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= filing_time <= market_end

# %%
def main(test_mode=False):
    """
    Main function to process SEC filings with proper error handling
    
    Args:
        test_mode (bool): If True, only process the first filing for testing
    """
    logger = None
    
    try:
        # Stage 1: Initialize logging
        config = Config()
        logger = setup_logging(config)
        logger.info(f"Starting SEC filing monitor {'(TEST MODE)' if test_mode else ''}")
        
        # Stage 2: Load S&P 500 data
        try:
            logger.info("Loading S&P 500 data...")
            sp500_data = initialize_sp500_data(logger)
            logger.info(f"Successfully loaded {len(sp500_data['tickers'])} S&P 500 tickers")
        except Exception as e:
            logger.error(f"Failed to initialize S&P 500 data: {str(e)}")
            raise
        
        # Stage 3: Initialize components
        try:
            logger.info("Initializing components...")
            fetcher, analyzer, trading_client, data_client = initialize_components(logger, sp500_data)
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
            
        # Stage 4: Handle existing JSON and get processed accessions
        try:
            processed_accessions, is_today = handle_existing_json(config, logger)
            logger.info(f"Found {len(processed_accessions)} previously processed filings today" if is_today 
                       else "Starting new filing collection for today")
        except Exception as e:
            logger.error(f"Error handling existing JSON: {str(e)}")
            raise
        
        # Stage 5: Fetch and filter new filings
        try:
            logger.info("Fetching SEC filings...")
            filings = fetcher.fetch_recent_filings()
            interesting_filings = fetcher.get_interesting_filings(filings, processed_accessions)
            
            if test_mode and interesting_filings:
                logger.info("Test mode: Processing only first filing")
                interesting_filings = [interesting_filings[0]]
                logger.info(f"Selected test filing: {interesting_filings[0]['ticker']} - {interesting_filings[0]['filing_title']}")
            
            if not interesting_filings:
                logger.info("No new filings to process")
                return None
            
            logger.info(f"Found {len(interesting_filings)} new filings to process")
            
        except Exception as e:
            logger.error(f"Error fetching/filtering filings: {str(e)}")
            raise
        
        # Stage 6: Process filings and generate summaries
        try:
            summary_results = []
            input_word_count = 0
            output_word_count = 0
            
            for i, filing in enumerate(interesting_filings, 1):
                try:
                    logger.info(f"Processing filing {i}/{len(interesting_filings)}: {filing['ticker']}")
                    text = fetcher.get_text_from_link(filing["txt_url"])
                    
                    if text:
                        input_word_count += word_counter(text)
                        summary = analyzer.analyze_filing(text)
                        
                        if summary:
                            output_word_count += word_counter(summary)
                            summary_results.append({
                                'filing': filing,
                                'summary': summary
                            })
                        else:
                            logger.warning(f"No summary generated for {filing['ticker']}")
                    else:
                        logger.warning(f"No text content for {filing['ticker']}")
                        
                    sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing filing {filing['ticker']}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in filing processing stage: {str(e)}")
            raise
        
        # Stage 7: Generate supplementary data
        try:
            logger.info("Fetching supplementary data...")
            tickers = [f['ticker'] for f in interesting_filings]
            company_names = {f['ticker']: f['company_name'] for f in interesting_filings}
            
            stock_data, market_date = get_stock_movements(
                trading_client, 
                data_client, 
                tickers,
                interesting_filings,  # <- Pass the filing data here
                test_mode=test_mode  # <- Move test_mode to a named parameter
            )
            
            wiki_urls = get_wiki_urls(company_names, tickers)
            
        except Exception as e:
            logger.error(f"Error fetching supplementary data: {str(e)}")
            raise
        
        # Stage 8: Create output and save/merge
        try:
            logger.info("Generating final output...")
            json_output = create_output_json(
                interesting_filings,
                summary_results,
                wiki_urls,
                stock_data,
                input_word_count,
                output_word_count
            )
            
            output_path = config.OUTPUT_DIR / ("sec_filings_test.json" if test_mode else "sec_filings.json")
            
            if is_today and not test_mode:
                merge_with_existing_json(
                json_output['filings'],
                config,
                logger,
                input_word_count,  # We already have this in main()
                output_word_count  # We already have this in main()
            )
            else:
                save_json(json_output, output_path)
                
            logger.info(f"Successfully saved/merged output to {output_path}")
            logger.info(f"Processing complete. Stats: {json_output['stats']}")
            
            return json_output
            
        except Exception as e:
            logger.error(f"Error in output generation/saving stage: {str(e)}")
            raise
            
    except Exception as e:
        # Handle all uncaught exceptions
        if logger is None:
            print(f"Fatal error before logger initialization: {str(e)}")
        else:
            logger.critical(f"Fatal error in SEC monitor: {str(e)}")
        raise

if __name__ == "__main__":
    main(test_mode=False)

# %%


# %%



