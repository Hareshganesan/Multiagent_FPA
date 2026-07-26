"""
Analysis Agent - Quantitative Financial Analysis (FIXED VERSION)
Uses structured JSON data directly instead of string parsing from Retriever
Version 2.0 - Proper data handling with optional RAG for semantic queries
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import retriever agent (optional, for semantic search only)
try:
    from retriever_agent import FinancialRetrieverAgent
    RETRIEVER_AVAILABLE = True
except ImportError:
    RETRIEVER_AVAILABLE = False
    print("⚠️  Retriever agent not available. Semantic search disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# FIXED ANALYSIS AGENT CLASS
# ============================================================================

class FinancialAnalysisAgent:
    """
    Quantitative analysis agent using STRUCTURED DATA
    - Loads JSON files directly for fast, accurate calculations
    - Uses Retriever only for semantic search queries (optional)
    """
    
    def __init__(self, retriever: Optional['FinancialRetrieverAgent'] = None):
        """Initialize analysis agent"""
        self.retriever = retriever
        
        # Load structured data from JSON files
        self.api_data = self._load_api_data()
        self.sentiment_data = self._load_sentiment_data()
        
        logger.info("✅ Analysis Agent initialized with structured data")
    
    # ========================================================================
    # DATA LOADING - STRUCTURED JSON
    # ========================================================================
    
    def _load_api_data(self) -> Optional[Dict]:
        """Load API agent data (market data)"""
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
        """Load scraping agent data (sentiment)"""
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
    # PORTFOLIO ANALYSIS - USING STRUCTURED DATA
    # ========================================================================
    
    def calculate_portfolio_allocation(self, region: Optional[str] = None) -> Dict:
        """Calculate portfolio allocation by region, sector, and holdings"""
        logger.info(f"Calculating portfolio allocation{f' for {region}' if region else ''}")
        
        if not self.api_data:
            return {'error': 'No API data available'}
        
        allocation = {
            'by_region': {},
            'by_sector': defaultdict(lambda: {'count': 0, 'market_cap': 0, 'symbols': []}),
            'total_market_cap': 0,
            'total_stocks': 0,
            'stocks': []
        }
        
        # Iterate through regions in structured data
        for region_name, region_data in self.api_data.get('regions', {}).items():
            # Filter by region if specified
            if region and region != region_name:
                continue
            
            # Extract region-level data (already aggregated!)
            region_info = {
                'count': region_data.get('portfolio_size', 0),
                'market_cap': region_data.get('total_market_cap', 0),
                'avg_change': region_data.get('average_change_percent', 0),
                'symbols': []
            }
            
            # Process individual stocks
            for symbol, market_data in region_data.get('market_data', {}).items():
                if 'error' in market_data:
                    continue
                
                # Extract clean structured data
                market_cap = market_data.get('market_cap', 0)
                if market_cap == 'N/A':
                    market_cap = 0
                
                sector = market_data.get('sector', 'Unknown')
                
                stock_info = {
                    'symbol': symbol,
                    'region': region_name,
                    'sector': sector,
                    'market_cap': market_cap,
                    'current_price': market_data.get('current_price', 0),
                    'change_percent': market_data.get('change_percent', 0),
                    'volume': market_data.get('volume', 0),
                    'company_name': market_data.get('company_name', symbol)
                }
                
                # Aggregate by sector
                allocation['by_sector'][sector]['count'] += 1
                allocation['by_sector'][sector]['market_cap'] += market_cap
                allocation['by_sector'][sector]['symbols'].append(symbol)
                
                allocation['total_market_cap'] += market_cap
                allocation['total_stocks'] += 1
                allocation['stocks'].append(stock_info)
                
                region_info['symbols'].append(symbol)
            
            # Store region data
            allocation['by_region'][region_name] = region_info
        
        # Calculate percentages
        total_cap = allocation['total_market_cap']
        
        if total_cap > 0:
            for region_name, data in allocation['by_region'].items():
                data['percentage'] = (data['market_cap'] / total_cap * 100)
            
            for sector, data in allocation['by_sector'].items():
                data['percentage'] = (data['market_cap'] / total_cap * 100)
        
        # Convert defaultdict to regular dict
        allocation['by_sector'] = dict(allocation['by_sector'])
        
        return allocation
    
    def calculate_risk_metrics(self, region: Optional[str] = None) -> Dict:
        """Calculate portfolio risk metrics using structured data"""
        logger.info(f"Calculating risk metrics{f' for {region}' if region else ''}")
        
        if not self.api_data:
            return {'error': 'No API data available'}
        
        changes = []
        volumes = []
        stocks = []
        
        # Iterate through regions
        for region_name, region_data in self.api_data.get('regions', {}).items():
            # Filter by region if specified
            if region and region != region_name:
                continue
            
            # Process market data
            for symbol, market_data in region_data.get('market_data', {}).items():
                if 'error' in market_data:
                    continue
                
                change_pct = market_data.get('change_percent', 0)
                volume = market_data.get('volume', 0)
                market_cap = market_data.get('market_cap', 0)
                
                if market_cap == 'N/A':
                    market_cap = 0
                
                changes.append(change_pct)
                volumes.append(volume)
                
                stocks.append({
                    'symbol': symbol,
                    'region': region_name,
                    'change': change_pct,
                    'volume': volume,
                    'market_cap': market_cap
                })
        
        if not changes:
            return {'error': 'No data available for risk calculation'}
        
        changes_array = np.array(changes)
        
        # Calculate risk metrics
        risk_metrics = {
            'volatility': float(np.std(changes_array)),
            'mean_return': float(np.mean(changes_array)),
            'max_drawdown': float(np.min(changes_array)),
            'max_gain': float(np.max(changes_array)),
            'positive_stocks': int(np.sum(changes_array > 0)),
            'negative_stocks': int(np.sum(changes_array < 0)),
            'neutral_stocks': int(np.sum(changes_array == 0)),
            'total_stocks': len(changes_array),
            'sharpe_ratio': self._calculate_sharpe_ratio(changes_array),
            'value_at_risk_95': float(np.percentile(changes_array, 5)),
            'value_at_risk_99': float(np.percentile(changes_array, 1)),
            'stocks': stocks
        }
        
        # Calculate concentration risk
        risk_metrics['concentration_risk'] = self._calculate_concentration_risk(stocks)
        
        # Calculate correlation (if enough data)
        if len(changes) > 2:
            risk_metrics['return_correlation'] = self._calculate_return_correlation(stocks)
        
        return risk_metrics
    
    def analyze_earnings_surprises(self, threshold: float = 5.0) -> Dict:
        """Identify earnings surprises (beats/misses) using structured data"""
        logger.info(f"Analyzing earnings surprises (threshold: {threshold}%)")
        
        if not self.api_data:
            return {'error': 'No API data available'}
        
        surprises = {
            'beats': [],
            'misses': [],
            'in_line': [],
            'total_analyzed': 0
        }
        
        # Iterate through regions
        for region_name, region_data in self.api_data.get('regions', {}).items():
            # Process earnings data
            for symbol, earnings in region_data.get('earnings_data', {}).items():
                if 'error' in earnings:
                    continue
                
                # Extract earnings metrics
                earnings_growth = earnings.get('earnings_growth')
                
                # Convert to percentage if needed
                if earnings_growth is not None and earnings_growth != 'N/A':
                    if isinstance(earnings_growth, (int, float)):
                        growth_pct = earnings_growth * 100  # Convert decimal to percentage
                    else:
                        continue
                    
                    surprise_info = {
                        'symbol': symbol,
                        'region': region_name,
                        'company_name': earnings.get('company_name', symbol),
                        'earnings_growth': growth_pct,
                        'eps': earnings.get('last_reported_eps', 'N/A'),
                        'forward_pe': earnings.get('forward_pe', 'N/A'),
                        'pe_ratio': earnings.get('pe_ratio', 'N/A'),
                        'revenue_growth': earnings.get('revenue_growth', 'N/A'),
                        'profit_margins': earnings.get('profit_margins', 'N/A')
                    }
                    
                    if growth_pct > threshold:
                        surprises['beats'].append(surprise_info)
                    elif growth_pct < -threshold:
                        surprises['misses'].append(surprise_info)
                    else:
                        surprises['in_line'].append(surprise_info)
                    
                    surprises['total_analyzed'] += 1
        
        # Sort by growth
        surprises['beats'].sort(key=lambda x: x['earnings_growth'], reverse=True)
        surprises['misses'].sort(key=lambda x: x['earnings_growth'])
        
        return surprises
    
    def analyze_sentiment_trends(self, region: Optional[str] = None) -> Dict:
        """Analyze sentiment trends using structured data"""
        logger.info(f"Analyzing sentiment trends{f' for {region}' if region else ''}")
        
        if not self.sentiment_data:
            return {'error': 'No sentiment data available'}
        
        trends = {
            'by_region': {},
            'by_stock': {},
            'overall': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        }
        
        # Iterate through regions
        for region_name, region_data in self.sentiment_data.get('regions', {}).items():
            # Filter by region if specified
            if region and region != region_name:
                continue
            
            # Get regional sentiment (already aggregated!)
            regional_sentiment = region_data.get('regional_sentiment', {})
            
            trends['by_region'][region_name] = {
                'positive': regional_sentiment.get('positive_stocks', 0),
                'negative': regional_sentiment.get('negative_stocks', 0),
                'neutral': regional_sentiment.get('neutral_stocks', 0),
                'total': region_data.get('total_stocks', 0),
                'overall_trend': regional_sentiment.get('overall_trend', 'neutral')
            }
            
            # Process individual stocks
            for symbol, stock_data in region_data.get('stocks', {}).items():
                if 'error' in stock_data:
                    continue
                
                analysis = stock_data.get('analysis_summary', {})
                sentiment = analysis.get('overall_sentiment', 'neutral')
                confidence = analysis.get('confidence', 0)
                
                trends['by_stock'][symbol] = {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'region': region_name,
                    'article_count': analysis.get('article_count', 0),
                    'recommendation': stock_data.get('recommendation', 'HOLD')
                }
                
                # Update overall counts
                trends['overall'][sentiment] += 1
                trends['overall']['total'] += 1
        
        # Calculate percentages
        for region_name, data in trends['by_region'].items():
            if data['total'] > 0:
                data['positive_pct'] = data['positive'] / data['total'] * 100
                data['negative_pct'] = data['negative'] / data['total'] * 100
                data['neutral_pct'] = data['neutral'] / data['total'] * 100
        
        if trends['overall']['total'] > 0:
            trends['overall']['positive_pct'] = trends['overall']['positive'] / trends['overall']['total'] * 100
            trends['overall']['negative_pct'] = trends['overall']['negative'] / trends['overall']['total'] * 100
            trends['overall']['neutral_pct'] = trends['overall']['neutral'] / trends['overall']['total'] * 100
        
        return trends
    
    def generate_morning_brief(self) -> Dict:
        """Generate comprehensive morning market brief"""
        logger.info("Generating morning market brief")
        
        brief = {
            'timestamp': datetime.now().isoformat(),
            'allocation': self.calculate_portfolio_allocation(),
            'risk_metrics': self.calculate_risk_metrics(),
            'earnings_surprises': self.analyze_earnings_surprises(),
            'sentiment_trends': self.analyze_sentiment_trends(),
            'summary': {}
        }
        
        # Generate summary insights
        allocation = brief['allocation']
        risk = brief['risk_metrics']
        earnings = brief['earnings_surprises']
        sentiment = brief['sentiment_trends']
        
        # Determine top region by allocation
        top_region = 'N/A'
        top_region_pct = 0
        if allocation.get('by_region'):
            top_region_data = max(
                allocation['by_region'].items(),
                key=lambda x: x[1].get('market_cap', 0)
            )
            top_region = top_region_data[0]
            top_region_pct = top_region_data[1].get('market_cap', 0) / allocation['total_market_cap'] * 100 if allocation['total_market_cap'] > 0 else 0
        
        # Key insights
        brief['summary'] = {
            'total_aum': allocation.get('total_market_cap', 0),
            'total_stocks': allocation.get('total_stocks', 0),
            'top_region': top_region,
            'top_region_allocation': top_region_pct,
            'mean_return': risk.get('mean_return', 0),
            'volatility': risk.get('volatility', 0),
            'sharpe_ratio': risk.get('sharpe_ratio', 0),
            'concentration_risk': risk.get('concentration_risk', 'Unknown'),
            'earnings_beats': len(earnings.get('beats', [])),
            'earnings_misses': len(earnings.get('misses', [])),
            'sentiment_positive_pct': sentiment['overall'].get('positive_pct', 0),
            'sentiment_negative_pct': sentiment['overall'].get('negative_pct', 0),
            'recommendation': self._generate_recommendation(risk, sentiment)
        }
        
        return brief
    
    # ========================================================================
    # SEMANTIC SEARCH (OPTIONAL - USES RETRIEVER)
    # ========================================================================
    
    def semantic_stock_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search for stocks matching query
        Example: "Find stocks with high growth potential"
        """
        if not self.retriever:
            return [{'error': 'Retriever not available. Install retriever_agent.py'}]
        
        logger.info(f"Semantic search: '{query}'")
        
        try:
            results = self.retriever.retrieve(query, top_k=top_k)
            
            # Filter for stock-related documents
            stock_results = [
                r for r in results 
                if r['metadata'].get('type') in ['stock_data', 'stock_sentiment']
            ]
            
            return stock_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return [{'error': str(e)}]
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        try:
            # Assume daily returns, annualize
            daily_rf = risk_free_rate / 252
            excess_returns = returns - daily_rf
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            # Annualize: multiply by sqrt(252) for daily data
            sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
            return float(sharpe)
        except:
            return 0.0
    
    def _calculate_concentration_risk(self, stocks: List[Dict]) -> str:
        """Calculate portfolio concentration risk using HHI"""
        if not stocks:
            return 'Unknown'
        
        total_cap = sum(s.get('market_cap', 0) for s in stocks)
        if total_cap == 0:
            return 'Unknown'
        
        # Calculate HHI (Herfindahl-Hirschman Index)
        hhi = sum((s.get('market_cap', 0) / total_cap) ** 2 for s in stocks)
        
        if hhi < 0.15:
            return 'Low (Diversified)'
        elif hhi < 0.25:
            return 'Moderate'
        else:
            return 'High (Concentrated)'
    
    def _calculate_return_correlation(self, stocks: List[Dict]) -> float:
        """Calculate average pairwise correlation of returns"""
        try:
            returns = [s.get('change', 0) for s in stocks]
            if len(returns) < 2:
                return 0.0
            
            # Simple correlation estimate (would need historical data for accuracy)
            return float(np.corrcoef(returns, returns)[0, 1])
        except:
            return 0.0
    
    def _generate_recommendation(self, risk: Dict, sentiment: Dict) -> str:
        """Generate trading recommendation"""
        mean_return = risk.get('mean_return', 0)
        volatility = risk.get('volatility', 0)
        positive_pct = sentiment['overall'].get('positive_pct', 0)
        negative_pct = sentiment['overall'].get('negative_pct', 0)
        
        if mean_return > 2 and positive_pct > 60:
            return '🟢 STRONG BUY - Positive momentum with bullish sentiment'
        elif mean_return > 0 and positive_pct > 50:
            return '🟢 BUY - Moderate gains with positive sentiment'
        elif mean_return < -2 and negative_pct > 60:
            return '🔴 STRONG SELL - Negative momentum with bearish sentiment'
        elif mean_return < 0 and negative_pct > 50:
            return '🔴 SELL - Declining performance with negative sentiment'
        elif volatility > 3:
            return '🟡 HOLD - High volatility, wait for stability'
        else:
            return '🟡 HOLD - Neutral market conditions'
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_morning_brief(self, brief: Dict):
        """Print formatted morning brief"""
        print("\n" + "="*80)
        print("📊 MORNING MARKET BRIEF")
        print(f"Generated: {datetime.fromisoformat(brief['timestamp']).strftime('%Y-%m-%d %I:%M %p')}")
        print("="*80)
        
        summary = brief['summary']
        
        print(f"\n💼 PORTFOLIO OVERVIEW")
        print(f"{'─'*80}")
        print(f"Total AUM: ${summary['total_aum']:,.0f}")
        print(f"Total Stocks: {summary['total_stocks']}")
        print(f"Top Region: {summary['top_region']} ({summary['top_region_allocation']:.1f}% of AUM)")
        print(f"Mean Return: {summary['mean_return']:.2f}%")
        print(f"Volatility: {summary['volatility']:.2f}%")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Concentration Risk: {summary['concentration_risk']}")
        
        print(f"\n📈 EARNINGS ANALYSIS")
        print(f"{'─'*80}")
        earnings = brief['earnings_surprises']
        print(f"Earnings Beats: {summary['earnings_beats']}")
        print(f"Earnings Misses: {summary['earnings_misses']}")
        print(f"In-Line: {len(earnings.get('in_line', []))}")
        
        if earnings['beats']:
            print(f"\nTop Performers:")
            for stock in earnings['beats'][:3]:
                print(f"  • {stock['symbol']} ({stock['company_name']}): +{stock['earnings_growth']:.1f}% growth")
        
        if earnings['misses']:
            print(f"\nUnderperformers:")
            for stock in earnings['misses'][:3]:
                print(f"  • {stock['symbol']} ({stock['company_name']}): {stock['earnings_growth']:.1f}% growth")
        
        print(f"\n💭 SENTIMENT ANALYSIS")
        print(f"{'─'*80}")
        print(f"Positive Sentiment: {summary['sentiment_positive_pct']:.1f}%")
        print(f"Negative Sentiment: {summary['sentiment_negative_pct']:.1f}%")
        print(f"Neutral Sentiment: {100 - summary['sentiment_positive_pct'] - summary['sentiment_negative_pct']:.1f}%")
        
        print(f"\n🎯 RECOMMENDATION")
        print(f"{'─'*80}")
        print(f"{summary['recommendation']}")
        
        print("\n" + "="*80)
    
    def save_brief(self, brief: Dict, filename: Optional[str] = None):
        """Save brief to JSON file"""
        if filename is None:
            filename = f"morning_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(brief, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Brief saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save brief: {e}")
            return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🤖 ANALYSIS AGENT - QUANTITATIVE PORTFOLIO ANALYSIS (FIXED)")
    print("Using Structured JSON Data (Fast & Accurate)")
    print("="*80 + "\n")
    
    # Initialize retriever (optional)
    retriever = None
    if RETRIEVER_AVAILABLE:
        try:
            retriever = FinancialRetrieverAgent()
            if retriever.load_index():
                print("✅ Retriever loaded (semantic search available)\n")
            else:
                print("⚠️  No vector index (semantic search disabled)\n")
                retriever = None
        except:
            print("⚠️  Retriever unavailable (semantic search disabled)\n")
    
    # Initialize analysis agent
    analyst = FinancialAnalysisAgent(retriever)
    
    # Check if data is available
    if not analyst.api_data and not analyst.sentiment_data:
        print("❌ No data files found. Please run:")
        print("   1. api_agent.py")
        print("   2. scraping_agent.py")
        return
    
    # Generate morning brief
    print("📊 Generating morning market brief...\n")
    brief = analyst.generate_morning_brief()
    
    # Print brief
    analyst.print_morning_brief(brief)
    
    # Save brief
    filename = analyst.save_brief(brief)
    print(f"\n💾 Full analysis saved to: {filename}")
    
    # Additional detailed analysis
    print("\n" + "="*80)
    print("📋 DETAILED REGIONAL ANALYSIS")
    print("="*80)
    
    regions_to_analyze = ['East Asia', 'South Asia', 'Southeast Asia', 'Western Asia']
    
    for region in regions_to_analyze:
        print(f"\n{'─'*80}")
        print(f"Region: {region}")
        print(f"{'─'*80}")
        
        allocation = analyst.calculate_portfolio_allocation(region)
        risk = analyst.calculate_risk_metrics(region)
        sentiment = analyst.analyze_sentiment_trends(region)
        
        if allocation.get('total_stocks', 0) > 0:
            print(f"Stocks: {allocation['total_stocks']}")
            print(f"Market Cap: ${allocation['total_market_cap']:,.0f}")
            print(f"Mean Return: {risk.get('mean_return', 0):.2f}%")
            print(f"Volatility: {risk.get('volatility', 0):.2f}%")
            print(f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
            
            region_sentiment = sentiment.get('by_region', {}).get(region, {})
            print(f"Sentiment - Positive: {region_sentiment.get('positive', 0)} | "
                  f"Negative: {region_sentiment.get('negative', 0)} | "
                  f"Neutral: {region_sentiment.get('neutral', 0)}")
            print(f"Trend: {region_sentiment.get('overall_trend', 'N/A').upper()}")
        else:
            print("No data available")
    
    # Test semantic search if available
    if retriever:
        print("\n" + "="*80)
        print("🔍 SEMANTIC SEARCH TEST")
        print("="*80)
        
        test_queries = [
            "stocks with high earnings growth",
            "negative sentiment tech stocks",
            "volatile stocks in Asia"
        ]
        
        for query in test_queries:
            print(f"\n Query: '{query}'")
            results = analyst.semantic_stock_search(query, top_k=3)
            
            if results and 'error' not in results[0]:
                print(f" Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    symbol = result['metadata'].get('symbol', 'Unknown')
                    relevance = result.get('relevance', 0)
                    print(f"  {i}. {symbol} (relevance: {relevance:.3f})")
            else:
                print("  No results found")
    
    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()