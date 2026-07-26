"""
Enhanced Multi-Region Scraping Agent with Regional Tech Stock Coverage
Matches the scope of the API Agent with sentiment analysis for Asian tech stocks
Version 3.0 - Regional and multi-stock support
"""

import requests
import feedparser
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import urllib.parse
from bs4 import BeautifulSoup
import os
import json
import re
from collections import Counter

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ .env file loaded successfully")
except ImportError:
    print("⚠️  python-dotenv not installed")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ============================================================================
# REGIONAL STOCK DEFINITIONS (Matches API Agent)
# ============================================================================

ASIAN_TECH_STOCKS = {
    'East Asia': {
        'China': ['BABA', '9888.HK', 'TCEHY', '0700.HK', 'BIDU'],
        'Japan': ['6758.T', '6501.T', '9984.T', '7974.T'],
        'South Korea': ['005930.KS', '000660.KS', '035720.KS'],
        'Taiwan': ['TSM', '2330.TW', '2454.TW'],
        'Hong Kong': ['0700.HK', '1211.HK', '9888.HK']
    },
    'South Asia': {
        'India': ['TCS.BO', 'INFY.BO', 'WIPRO.BO', 'HCLTECH.BO'],
        'Pakistan': ['PTCL.KA']
    },
    'Southeast Asia': {
        'Singapore': ['U11.SI', 'D05.SI'],
        'Malaysia': ['5185.KL'],
        'Thailand': ['ADVANC.BK', 'TRUEIT.BK'],
        'Indonesia': ['TLKM.JK'],
        'Vietnam': ['VNM']
    },
    'Central Asia': {
        'Kazakhstan': ['KZRYO.KZ'],
        'Uzbekistan': ['UZINDEX.UZ']
    },
    'Western Asia': {
        'Israel': ['TRNFP'],
        'Saudi Arabia': ['2222.SR'],
        'UAE': ['EMAAR.AE']
    }
}

# Sentiment keywords
POSITIVE_KEYWORDS = {
    "gain": 1, "rise": 1, "surge": 2, "beat": 2, "strong": 1,
    "growth": 2, "profit": 1, "exceed": 2, "bullish": 2,
    "outperform": 2, "upgrade": 2, "positive": 1, "up": 1,
    "rally": 2, "boost": 1, "momentum": 1, "breakout": 2,
    "success": 1, "promising": 1, "excellent": 2, "robust": 1,
    "record": 1, "expansion": 1, "innovation": 1
}

NEGATIVE_KEYWORDS = {
    "fall": 1, "drop": 1, "loss": 1, "miss": 2, "weak": 1,
    "decline": 1, "bearish": 2, "downgrade": 2, "concern": 1,
    "warning": 2, "negative": 1, "down": 1, "crash": 2,
    "slump": 2, "plunge": 2, "risk": 1, "challenge": 1,
    "struggle": 1, "trouble": 1, "poor": 1, "threat": 2,
    "delay": 1, "shortage": 1, "volatility": 1
}

# ============================================================================
# SENTIMENT ANALYSIS CLASS
# ============================================================================

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with weighted scoring"""

    def __init__(self):
        self.positive_keywords = POSITIVE_KEYWORDS
        self.negative_keywords = NEGATIVE_KEYWORDS

    def analyze_text(self, text: str) -> Tuple[str, float]:
        """Analyze text sentiment with confidence score"""
        if not text:
            return "neutral", 0.5

        text_lower = text.lower()

        positive_score = sum(
            weight for keyword, weight in self.positive_keywords.items()
            if keyword in text_lower
        )

        negative_score = sum(
            weight for keyword, weight in self.negative_keywords.items()
            if keyword in text_lower
        )

        total_score = positive_score + negative_score

        if total_score == 0:
            return "neutral", 0.5

        if positive_score > negative_score:
            sentiment = "positive"
            confidence = positive_score / (positive_score + negative_score)
        elif negative_score > positive_score:
            sentiment = "negative"
            confidence = negative_score / (positive_score + negative_score)
        else:
            sentiment = "neutral"
            confidence = 0.5

        return sentiment, round(confidence, 2)

    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """Extract key financial phrases"""
        phrases = []
        keywords = list(self.positive_keywords.keys()) + list(self.negative_keywords.keys())

        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                pattern = rf'\b\w+\s+{re.escape(keyword)}\s+\w+\b'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    phrases.extend(matches[:1])

        return phrases[:max_phrases]


# ============================================================================
# FIRECRAWL CLIENT
# ============================================================================

class FirecrawlMCPClient:
    """Client for Firecrawl API with fallback"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        self.base_url = "https://api.firecrawl.dev/v0"
        self.session = requests.Session()

        if self.api_key:
            logger.info(f"Firecrawl API key loaded: {self.api_key[:10]}...{self.api_key[-4:]}")
        else:
            logger.warning("No Firecrawl API key found in environment")

    def scrape_url(self, url: str, formats: List[str] = None, timeout: int = 30) -> Dict:
        """Scrape URL using Firecrawl API with fallback"""
        if not self.api_key:
            return self._fallback_scrape(url)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "url": url,
                "formats": formats or ["markdown", "html"],
                "timeout": timeout
            }

            response = self.session.post(
                f"{self.base_url}/scrape",
                headers=headers,
                json=payload,
                timeout=timeout + 5
            )

            if response.status_code == 200:
                logger.info(f"✓ Firecrawl scraped: {url[:50]}...")
                return response.json()
            elif response.status_code == 429:
                logger.warning("Firecrawl rate limit, using fallback")
                time.sleep(5)
                return self._fallback_scrape(url)
            else:
                return self._fallback_scrape(url)

        except Exception as e:
            logger.warning(f"Firecrawl failed: {e}")
            return self._fallback_scrape(url)

    def _fallback_scrape(self, url: str) -> Dict:
        """Fallback scraping using BeautifulSoup"""
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()

            text = soup.get_text(separator="\n", strip=True)

            return {
                "success": True,
                "data": {
                    "markdown": text,
                    "metadata": {
                        "title": soup.title.string if soup.title else "No title",
                        "sourceURL": url
                    }
                }
            }
        except Exception as e:
            logger.error(f"Fallback scraping failed: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# ENHANCED SCRAPING AGENT
# ============================================================================

class EnhancedMultiRegionScrapingAgent:
    """Scraping agent with regional and multi-stock support"""

    def __init__(self, use_firecrawl: bool = True):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.use_firecrawl = use_firecrawl
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.firecrawl = FirecrawlMCPClient() if use_firecrawl else None

    def _fetch_with_retry(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Fetch URL with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.content
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def scrape_google_news(self, symbol: str, max_articles: int = 10) -> List[Dict]:
        """Scrape Google News RSS feed for a stock"""
        try:
            query = f"{symbol} stock news"
            encoded_query = urllib.parse.quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            content = self._fetch_with_retry(rss_url)
            if not content:
                return []

            feed = feedparser.parse(content)
            articles = []

            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", entry.get("updated", "N/A")),
                    "summary": entry.get("summary", ""),
                    "source": entry.get("source", {}).get("title", "Google News RSS"),
                    "symbol": symbol
                }
                articles.append(article)

            logger.info(f"✓ RSS: {len(articles)} articles for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"RSS scraping failed for {symbol}: {e}")
            return []

    def scrape_article_content(self, url: str) -> Dict:
        """Scrape full article content"""
        if self.firecrawl and self.firecrawl.api_key:
            logger.info(f"Using Firecrawl to scrape: {url[:60]}...")
            result = self.firecrawl.scrape_url(url)

            if result.get("success"):
                return {
                    "url": url,
                    "content": result["data"].get("markdown", ""),
                    "title": result["data"].get("metadata", {}).get("title", ""),
                    "method": "firecrawl",
                    "success": True
                }

        logger.info(f"Using fallback scraping for: {url[:60]}...")
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)

            return {
                "url": url,
                "content": text,
                "title": soup.title.string if soup.title else "No title",
                "method": "fallback",
                "success": True
            }
        except Exception as e:
            logger.error(f"Article scraping failed: {e}")
            return {"url": url, "error": str(e), "method": "failed", "success": False}

    def analyze_single_stock(self, symbol: str, max_articles: int = 5) -> Dict:
        """Analyze sentiment for a single stock"""
        logger.info(f"Analyzing {symbol}")

        articles = self.scrape_google_news(symbol, max_articles)

        if not articles:
            return {
                "symbol": symbol,
                "error": "No articles found",
                "articles": [],
                "timestamp": datetime.now().isoformat()
            }

        # Analyze headline sentiment
        headline_sentiments = []
        for article in articles:
            sentiment, confidence = self.sentiment_analyzer.analyze_text(article["title"])
            article["headline_sentiment"] = sentiment
            article["headline_confidence"] = confidence
            headline_sentiments.append(sentiment)

        # Scrape full content for first 3 articles
        detailed_articles = []
        for i, article in enumerate(articles[:3], 1):
            content_data = self.scrape_article_content(article["link"])

            if content_data.get("success"):
                content_sentiment, content_confidence = self.sentiment_analyzer.analyze_text(
                    content_data.get("content", "")
                )
                article["content_sentiment"] = content_sentiment
                article["content_confidence"] = content_confidence
                article["full_content"] = content_data.get("content", "")[:1000]
                article["scraping_method"] = content_data.get("method", "unknown")
                key_phrases = self.sentiment_analyzer.extract_key_phrases(
                    content_data.get("content", "")
                )
                article["key_phrases"] = key_phrases
            else:
                article["scraping_method"] = "failed"
                article["content_sentiment"] = "unknown"

            detailed_articles.append(article)
            if i < 3:
                time.sleep(1)

        # Calculate overall sentiment
        positive = headline_sentiments.count("positive")
        negative = headline_sentiments.count("negative")
        neutral = headline_sentiments.count("neutral")
        total = len(headline_sentiments)

        if positive > negative and positive > neutral:
            overall = "positive"
        elif negative > positive and negative > neutral:
            overall = "negative"
        else:
            overall = "neutral"

        confidence = max(positive, negative, neutral) / total if total > 0 else 0

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "overall_sentiment": overall,
                "confidence": round(confidence, 2),
                "article_count": total
            },
            "sentiment_breakdown": {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "positive_percentage": round(positive / total * 100, 1) if total > 0 else 0,
                "negative_percentage": round(negative / total * 100, 1) if total > 0 else 0,
                "neutral_percentage": round(neutral / total * 100, 1) if total > 0 else 0
            },
            "articles": detailed_articles,
            "recommendation": self._generate_recommendation(overall, confidence)
        }

    def analyze_regional_sentiment(self, region: str) -> Dict:
        """Analyze sentiment for all stocks in a region"""
        if region not in ASIAN_TECH_STOCKS:
            return {"error": f"Region '{region}' not found"}

        print(f"\n{'='*70}")
        print(f"📊 Sentiment Analysis: {region}")
        print(f"{'='*70}")

        stocks = []
        for country, symbols in ASIAN_TECH_STOCKS[region].items():
            stocks.extend(symbols)

        results = {
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "total_stocks": len(stocks),
            "stocks": {}
        }

        for i, symbol in enumerate(stocks, 1):
            print(f"Processing {i}/{len(stocks)}: {symbol}")
            stock_analysis = self.analyze_single_stock(symbol, max_articles=3)
            results["stocks"][symbol] = stock_analysis
            time.sleep(1)  # Rate limiting

        # Calculate regional sentiment
        positive_count = sum(
            1 for s in results["stocks"].values()
            if s.get("analysis_summary", {}).get("overall_sentiment") == "positive"
        )
        negative_count = sum(
            1 for s in results["stocks"].values()
            if s.get("analysis_summary", {}).get("overall_sentiment") == "negative"
        )

        results["regional_sentiment"] = {
            "positive_stocks": positive_count,
            "negative_stocks": negative_count,
            "neutral_stocks": len(stocks) - positive_count - negative_count,
            "overall_trend": "positive" if positive_count > negative_count else (
                "negative" if negative_count > positive_count else "neutral"
            )
        }

        return results

    def _generate_recommendation(self, sentiment: str, confidence: float) -> str:
        """Generate trading recommendation"""
        if sentiment == "positive" and confidence > 0.7:
            return "🟢 STRONG BUY"
        elif sentiment == "positive" and confidence > 0.5:
            return "🟢 BUY"
        elif sentiment == "negative" and confidence > 0.7:
            return "🔴 STRONG SELL"
        elif sentiment == "negative" and confidence > 0.5:
            return "🔴 SELL"
        else:
            return "🟡 HOLD"

    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """Save analysis results to JSON"""
        if filename is None:
            filename = f"regional_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🤖 ENHANCED MULTI-REGION SCRAPING AGENT")
    print("Regional Sentiment Analysis for Asian Tech Stocks - Version 3.0")
    print("="*70 + "\n")

    api_key = os.getenv('FIRECRAWL_API_KEY')
    if api_key:
        print(f"✅ Firecrawl API Key: Found\n")
    else:
        print(f"⚠️  Firecrawl API Key: Not found (will use fallback)\n")

    # Initialize agent
    agent = EnhancedMultiRegionScrapingAgent(use_firecrawl=api_key is not None)

    # Analyze each region
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "regions": {}
    }

    for region in ASIAN_TECH_STOCKS.keys():
        region_results = agent.analyze_regional_sentiment(region)
        all_results["regions"][region] = region_results
        time.sleep(2)  # Rate limiting between regions

    # Print summary
    print("\n" + "="*70)
    print("📈 REGIONAL SENTIMENT SUMMARY")
    print("="*70)
    print(f"{'Region':<20} {'Positive':<12} {'Negative':<12} {'Neutral':<12} {'Trend':<12}")
    print("-" * 68)

    for region, data in all_results["regions"].items():
        if "regional_sentiment" in data:
            sentiment = data["regional_sentiment"]
            print(f"{region:<20} {sentiment['positive_stocks']:<12} "
                  f"{sentiment['negative_stocks']:<12} {sentiment['neutral_stocks']:<12} "
                  f"{sentiment['overall_trend'].upper():<12}")

    print("="*70 + "\n")

    # Save results
    agent.save_results(all_results)

    print("✅ Analysis complete")


if __name__ == "__main__":
    main()