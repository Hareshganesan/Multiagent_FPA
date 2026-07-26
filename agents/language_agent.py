"""
Enhanced Language Agent - LLM-powered Market Brief Generator
FIXED VERSION - UTF-8 Encoding for Windows Compatibility
Integrates ALL agents: API → Scraping → Retriever → Analysis → Language
Handles Gemini rate limits (2 requests/minute) with caching and queuing
Version 2.1 - Encoding Fix for Windows
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import google.generativeai as genai
from pathlib import Path
from collections import deque

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITER CLASS
# ============================================================================

class RateLimiter:
    """Rate limiter for Gemini API (2 requests per minute)"""
    
    def __init__(self, max_requests: int = 2, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = deque()
        logger.info(f"Rate limiter: {max_requests} requests per {time_window}s")
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove old requests outside time window
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        
        # If at limit, wait until oldest request expires
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (now - self.requests[0]) + 1
            if wait_time > 0:
                logger.warning(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.requests.popleft()
        
        # Record this request
        self.requests.append(time.time())

# ============================================================================
# ENHANCED LANGUAGE AGENT CLASS
# ============================================================================

class EnhancedLanguageAgent:
    """
    Enhanced LLM agent with full multi-agent integration and rate limiting
    FIXED: UTF-8 encoding for Windows compatibility
    """
    
    def __init__(self, api_key: Optional[str] = None, max_requests: int = 2):
        """Initialize Enhanced Language Agent"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Rate limiter
        self.rate_limiter = RateLimiter(max_requests=max_requests, time_window=60)
        
        # Response cache
        self.cache = {}
        self.cache_file = "llm_cache.json"
        self._load_cache()
        
        # Generation config
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        logger.info(f"✅ Enhanced Language Agent initialized (max {max_requests} req/min)")
    
    def _load_cache(self):
        """Load cached responses with UTF-8 encoding"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached responses")
            except:
                self.cache = {}
    
    def _save_cache(self):
        """Save response cache with UTF-8 encoding"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return str(hash(prompt))
    
    def _check_cache(self, prompt: str) -> Optional[str]:
        """Check if response is cached (within 1 hour)"""
        key = self._get_cache_key(prompt)
        if key in self.cache:
            cached = self.cache[key]
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=1):
                logger.info("✓ Using cached response")
                return cached['response']
        return None
    
    def _cache_response(self, prompt: str, response: str):
        """Cache response with timestamp"""
        key = self._get_cache_key(prompt)
        self.cache[key] = {
            'timestamp': datetime.now().isoformat(),
            'response': response
        }
        self._save_cache()
    
    # ========================================================================
    # DATA LOADING (ALL AGENTS) - FIXED UTF-8 ENCODING
    # ========================================================================
    
    def load_all_agent_data(self) -> Dict:
        """Load outputs from all 4 agents with UTF-8 encoding"""
        logger.info("Loading data from all agents...")
        
        data = {
            'api_agent': None,
            'scraping_agent': None,
            'analysis_agent': None,
            'retriever_available': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Load API agent data (market data)
        api_files = sorted(Path('.').glob('multi_region_results_*.json'))
        if api_files:
            try:
                with open(api_files[-1], 'r', encoding='utf-8') as f:
                    data['api_agent'] = json.load(f)
                logger.info(f"✓ Loaded API agent data: {api_files[-1].name}")
            except Exception as e:
                logger.error(f"Failed to load API data: {e}")
        else:
            logger.warning("⚠️  API agent data not found")
        
        # Load scraping agent data (sentiment)
        sentiment_files = sorted(Path('.').glob('regional_sentiment_*.json'))
        if sentiment_files:
            try:
                with open(sentiment_files[-1], 'r', encoding='utf-8') as f:
                    data['scraping_agent'] = json.load(f)
                logger.info(f"✓ Loaded scraping agent data: {sentiment_files[-1].name}")
            except Exception as e:
                logger.error(f"Failed to load scraping data: {e}")
        else:
            logger.warning("⚠️  Scraping agent data not found")
        
        # Load analysis agent data (quantitative analysis)
        analysis_files = sorted(Path('.').glob('morning_brief_*.json'))
        if analysis_files:
            try:
                with open(analysis_files[-1], 'r', encoding='utf-8') as f:
                    data['analysis_agent'] = json.load(f)
                logger.info(f"✓ Loaded analysis agent data: {analysis_files[-1].name}")
            except Exception as e:
                logger.error(f"Failed to load analysis data: {e}")
        else:
            logger.warning("⚠️  Analysis agent data not found")
        
        # Check retriever agent availability
        if os.path.exists('vector_store/faiss_index.bin'):
            data['retriever_available'] = True
            logger.info("✓ Retriever agent index found")
        else:
            logger.warning("⚠️  Retriever agent index not found")
        
        return data
    
    def _load_api_data(self) -> Optional[Dict]:
        """Load API agent data (market data) with UTF-8 encoding"""
        try:
            api_files = sorted(Path('.').glob('multi_region_results_*.json'))
            if not api_files:
                logger.warning("No API data files found")
                return None
            
            with open(api_files[-1], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Loaded API data: {api_files[-1].name}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load API data: {e}")
            return None
    
    def _load_sentiment_data(self) -> Optional[Dict]:
        """Load scraping agent data (sentiment) with UTF-8 encoding"""
        try:
            sentiment_files = sorted(Path('.').glob('regional_sentiment_*.json'))
            if not sentiment_files:
                logger.warning("No sentiment data files found")
                return None
            
            with open(sentiment_files[-1], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Loaded sentiment data: {sentiment_files[-1].name}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")
            return None
    
    # ========================================================================
    # ENHANCED PROMPT TEMPLATES
    # ========================================================================
    
    def _build_comprehensive_context(self, all_data: Dict) -> str:
        """Build comprehensive context from all agents"""
        context = "**COMPREHENSIVE PORTFOLIO DATA FROM ALL AGENTS:**\n\n"
        
        # API Agent - Market Data
        if all_data.get('api_agent'):
            api_data = all_data['api_agent']
            context += "**1. MARKET DATA (API Agent):**\n"
            for region, region_data in api_data.get('regions', {}).items():
                context += f"\n{region}:\n"
                context += f"  - Average Change: {region_data.get('average_change_percent', 0):.2f}%\n"
                context += f"  - Market Cap: ${region_data.get('total_market_cap', 0):,.0f}\n"
                context += f"  - Stocks: {region_data.get('portfolio_size', 0)}\n"
        
        # Scraping Agent - Sentiment
        if all_data.get('scraping_agent'):
            scrape_data = all_data['scraping_agent']
            context += "\n**2. SENTIMENT ANALYSIS (Scraping Agent):**\n"
            for region, region_data in scrape_data.get('regions', {}).items():
                sentiment = region_data.get('regional_sentiment', {})
                context += f"\n{region}:\n"
                context += f"  - Positive: {sentiment.get('positive_stocks', 0)} stocks\n"
                context += f"  - Negative: {sentiment.get('negative_stocks', 0)} stocks\n"
                context += f"  - Trend: {sentiment.get('overall_trend', 'neutral').upper()}\n"
        
        # Analysis Agent - Quantitative Metrics
        if all_data.get('analysis_agent'):
            analysis = all_data['analysis_agent']
            summary = analysis.get('summary', {})
            context += "\n**3. QUANTITATIVE ANALYSIS (Analysis Agent):**\n"
            context += f"  - Total AUM: ${summary.get('total_aum', 0):,.0f}\n"
            context += f"  - Mean Return: {summary.get('mean_return', 0):.2f}%\n"
            context += f"  - Volatility: {summary.get('volatility', 0):.2f}%\n"
            context += f"  - Earnings Beats: {summary.get('earnings_beats', 0)}\n"
            context += f"  - Earnings Misses: {summary.get('earnings_misses', 0)}\n"
            context += f"  - Recommendation: {summary.get('recommendation', 'N/A')}\n"
        
        return context
    
    def _get_comprehensive_brief_prompt(self, all_data: Dict) -> str:
        """Generate prompt using ALL agent data"""
        context = self._build_comprehensive_context(all_data)
        
        return f"""You are a professional portfolio manager delivering a comprehensive morning market brief.

{context}

**INSTRUCTIONS:**
Generate a professional morning market brief (3-4 paragraphs) that:

1. **Opening**: Start with overall portfolio status (total AUM, number of stocks, top region)
2. **Market Performance**: Highlight regional performance differences using API agent data
3. **Sentiment & News**: Incorporate sentiment trends from scraping agent
4. **Earnings & Fundamentals**: Mention key earnings beats/misses from analysis
5. **Risk Assessment**: Include volatility and risk metrics
6. **Actionable Recommendation**: Clear buy/sell/hold guidance with reasoning

Use natural, conversational language suitable for spoken delivery. Include specific numbers and percentages. Keep it concise but comprehensive.

Generate the brief now:"""
    
    def _get_rag_enhanced_prompt(self, query: str, retrieved_docs: List[Dict], all_data: Dict) -> str:
        """Generate RAG-enhanced prompt with retriever context"""
        # Format retrieved documents
        rag_context = "\n\n".join([
            f"**Document {i+1}** (Relevance: {doc.get('relevance', 0):.2f}):\n{doc.get('content', '')[:400]}"
            for i, doc in enumerate(retrieved_docs[:3])
        ])
        
        # Add structured data context
        structured_context = self._build_comprehensive_context(all_data)
        
        return f"""You are a financial assistant answering portfolio questions using retrieved data and analysis.

**USER QUESTION:** {query}

**RETRIEVED CONTEXT (from Vector Store):**
{rag_context}

**STRUCTURED DATA (from Analysis Pipeline):**
{structured_context}

**INSTRUCTIONS:**
1. Answer using BOTH retrieved documents AND structured data
2. Prioritize recent data and specific metrics
3. Be conversational but precise
4. Include numbers and percentages
5. If information is missing, say so clearly
6. Provide actionable insights

Answer:"""
    
    # ========================================================================
    # GENERATION METHODS WITH RATE LIMITING
    # ========================================================================
    
    def generate_comprehensive_brief(self, all_data: Dict) -> Dict:
        """Generate comprehensive brief from ALL agents"""
        logger.info("Generating comprehensive morning brief...")
        
        try:
            prompt = self._get_comprehensive_brief_prompt(all_data)
            
            # Check cache first
            cached = self._check_cache(prompt)
            if cached:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'query_type': 'comprehensive_brief',
                    'brief_text': cached,
                    'model': 'gemini-2.0-flash-exp',
                    'cached': True,
                    'success': True
                }
            
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            brief_text = response.text
            
            # Cache response
            self._cache_response(prompt, brief_text)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'query_type': 'comprehensive_brief',
                'brief_text': brief_text,
                'model': 'gemini-2.0-flash-exp',
                'cached': False,
                'success': True,
                'agents_used': [k for k, v in all_data.items() if v is not None]
            }
            
            logger.info(f"✓ Generated comprehensive brief ({len(brief_text)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate brief: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'query_type': 'comprehensive_brief',
                'error': str(e),
                'success': False
            }
    
    def answer_with_rag(self, query: str, retriever=None, all_data: Dict = None) -> Dict:
        """Answer query using RAG + all agent data"""
        logger.info(f"Answering with RAG: '{query}'")
        
        try:
            # Retrieve relevant documents if retriever available
            retrieved_docs = []
            if retriever:
                retrieved_docs = retriever.retrieve(query, top_k=3)
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Build prompt
            prompt = self._get_rag_enhanced_prompt(query, retrieved_docs, all_data or {})
            
            # Check cache
            cached = self._check_cache(prompt)
            if cached:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'answer': cached,
                    'model': 'gemini-2.0-flash-exp',
                    'cached': True,
                    'success': True
                }
            
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            answer_text = response.text
            
            # Cache response
            self._cache_response(prompt, answer_text)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'answer': answer_text,
                'model': 'gemini-2.0-flash-exp',
                'documents_used': len(retrieved_docs),
                'cached': False,
                'success': True
            }
            
            logger.info(f"✓ Generated answer ({len(answer_text)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to answer query: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'error': str(e),
                'success': False
            }
    
    def save_response(self, response: Dict, filename: Optional[str] = None):
        """Save response to file with UTF-8 encoding"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            query_type = response.get('query_type', 'query')
            filename = f"llm_response_{query_type}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Response saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save response: {e}")
            return None


# ============================================================================
# MAIN EXECUTION - LIMITED TO 2 QUESTIONS
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🤖 ENHANCED LANGUAGE AGENT - FULL MULTI-AGENT INTEGRATION")
    print("Powered by Google Gemini (Rate Limited: 2 requests/minute)")
    print("Version 2.1 - UTF-8 Encoding Fix")
    print("="*80 + "\n")
    
    # Initialize agent with rate limiting
    try:
        agent = EnhancedLanguageAgent(max_requests=2)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nPlease set GEMINI_API_KEY in your .env file")
        return
    
    # Load all agent data
    print("📂 Loading data from all agents...\n")
    all_data = agent.load_all_agent_data()
    
    if not any([all_data['api_agent'], all_data['scraping_agent'], all_data['analysis_agent']]):
        print("❌ No agent data found. Please run:")
        print("   1. api_agent.py")
        print("   2. scraping_agent.py")
        print("   3. retriever_agent.py")
        print("   4. analysis_agent.py")
        return
    
    print("\n" + "="*80)
    print("📊 GENERATING COMPREHENSIVE MORNING BRIEF")
    print("Using data from: API Agent + Scraping Agent + Analysis Agent")
    print("="*80 + "\n")
    
    # Question 1: Morning Brief (uses all agents)
    brief_result = agent.generate_comprehensive_brief(all_data)
    
    if brief_result['success']:
        print("🌅 MORNING MARKET BRIEF\n")
        print(brief_result['brief_text'])
        print(f"\n{'─'*80}")
        print(f"Agents Used: {', '.join(brief_result.get('agents_used', []))}")
        print(f"Cached: {'Yes' if brief_result.get('cached') else 'No'}")
        print(f"Model: {brief_result['model']}")
        print(f"{'─'*80}\n")
        
        agent.save_response(brief_result, 'comprehensive_morning_brief.json')
    else:
        print(f"❌ Error: {brief_result.get('error', 'Unknown error')}\n")
    
    # Question 2: RAG-enhanced query (if retriever available)
    print("="*80)
    print("❓ ANSWERING SPECIFIC QUERY (RAG + All Agents)")
    print("="*80 + "\n")
    
    # Load retriever if available
    retriever = None
    if all_data['retriever_available']:
        try:
            from retriever_agent import FinancialRetrieverAgent
            retriever = FinancialRetrieverAgent()
            retriever.load_index()
            print("✓ Retriever agent loaded\n")
        except Exception as e:
            print(f"⚠️  Could not load retriever: {e}\n")
    
    # ONLY 2 QUESTIONS AS REQUESTED
    query = "In which stock should i invest today ?"
    
    print(f"Query: {query}\n")
    print("Retrieving relevant context and generating answer...\n")
    
    rag_result = agent.answer_with_rag(query, retriever, all_data)
    
    if rag_result['success']:
        print("📝 ANSWER:\n")
        print(rag_result['answer'])
        print(f"\n{'─'*80}")
        print(f"Documents Retrieved: {rag_result.get('documents_used', 0)}")
        print(f"Cached: {'Yes' if rag_result.get('cached') else 'No'}")
        print(f"Model: {rag_result['model']}")
        print(f"{'─'*80}\n")
        
        agent.save_response(rag_result, 'rag_query_response.json')
    else:
        print(f"❌ Error: {rag_result.get('error', 'Unknown error')}\n")
    
    print("="*80)
    print("✅ COMPLETE - 2 Questions Asked (Rate Limit Respected)")
    print("="*80)
    print("\nResponses cached for 1 hour. Re-run to use cached responses.")
    print("Cache file: llm_cache.json")


if __name__ == "__main__":
    main()