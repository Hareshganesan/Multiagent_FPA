import os
from dotenv import load_dotenv
import logging
import requests
from crewai.tools import tool
import yfinance as yf
import time
from datetime import datetime
import json
import numpy as np
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)
fmp_api_key = os.getenv("FMP_API_KEY")

print("✅ Configuration loaded successfully")

# ============================================================================
# REGIONAL & COUNTRY DEFINITIONS
# ============================================================================

ASIAN_REGIONS = {
    'east_asia': {
        'name': 'East Asia',
        'countries': ['China', 'Japan', 'South Korea', 'Taiwan', 'Hong Kong', 'Mongolia'],
        'major_exchanges': ['SSE', 'TSE', 'KRX', 'TWSE', 'HKEX', 'MNX'],
        'timezone': 'Asia/Shanghai'
    },
    'south_asia': {
        'name': 'South Asia',
        'countries': ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal'],
        'major_exchanges': ['NSE', 'BSE', 'PSX', 'DSE', 'CSE', 'NEPSE'],
        'timezone': 'Asia/Kolkata'
    },
    'southeast_asia': {
        'name': 'Southeast Asia',
        'countries': ['Singapore', 'Malaysia', 'Indonesia', 'Thailand', 'Vietnam', 'Philippines'],
        'major_exchanges': ['SGX', 'KLSE', 'IDX', 'SET', 'HNX', 'PSE'],
        'timezone': 'Asia/Bangkok'
    },
    'central_asia': {
        'name': 'Central Asia',
        'countries': ['Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Kyrgyzstan', 'Tajikistan'],
        'major_exchanges': ['KASE', 'UZX', 'TMSX', 'KGX', 'TJX'],
        'timezone': 'Asia/Almaty'
    },
    'west_asia': {
        'name': 'Western Asia (Middle East)',
        'countries': ['Saudi Arabia', 'UAE', 'Israel', 'Turkey', 'Iran', 'Qatar'],
        'major_exchanges': ['TASI', 'ADX', 'TASE', 'BIST', 'TSE', 'QSE'],
        'timezone': 'Asia/Riyadh'
    }
}

# Major tech companies across Asian regions - EXPANDED
ASIAN_TECH_COMPANIES = {
    # East Asia - Tech Giants
    'east_asia': {
        'China': {
            'symbols': ['BABA', '9888.HK', 'TCEHY', '0700.HK', 'BIDU', 'JD', 'PDD'],
            'names': ['Alibaba', 'Baidu HK', 'Tencent ADR', 'Tencent HK', 'Baidu ADR', 'JD.com', 'Pinduoduo']
        },
        'Japan': {
            'symbols': ['6758.T', '6501.T', '9984.T', '7974.T', '6861.T'],
            'names': ['Sony', 'Hitachi', 'SoftBank', 'Nintendo', 'Keyence']
        },
        'South Korea': {
            'symbols': ['005930.KS', '000660.KS', '035720.KS', '035420.KS'],
            'names': ['Samsung Electronics', 'SK Hynix', 'Kakao', 'NAVER']
        },
        'Taiwan': {
            'symbols': ['TSM', '2330.TW', '2454.TW', '2357.TW'],
            'names': ['TSMC', 'MediaTek', 'Acer', 'ASUS']
        },
        'Hong Kong': {
            'symbols': ['0700.HK', '1211.HK', '9888.HK', '9618.HK'],
            'names': ['Tencent', 'BYD', 'Baidu', 'JD.com']
        }
    },
    # South Asia - Growing Tech Hubs
    'south_asia': {
        'India': {
            'symbols': ['TCS.BO', 'INFY.BO', 'WIPRO.BO', 'HCLTECH.BO', 'TECHM.BO'],
            'names': ['Tata Consultancy', 'Infosys', 'Wipro', 'HCL Technologies', 'Tech Mahindra']
        },
        'Pakistan': {
            'symbols': ['PTCL.KA'],
            'names': ['PTCL (Pakistan Telecom)']
        }
    },
    # Southeast Asia - Emerging Hubs
    'southeast_asia': {
        'Singapore': {
            'symbols': ['U11.SI', 'D05.SI', 'S58.SI'],
            'names': ['Venture Corp', 'DBS Bank (Tech)', 'SATS']
        },
        'Malaysia': {
            'symbols': ['5185.KL', '1082.KL'],
            'names': ['Axiata Group', 'Genting Malaysia']
        },
        'Thailand': {
            'symbols': ['ADVANC.BK', 'TRUEIT.BK', 'INTUCH.BK'],
            'names': ['Advanced Info Service', 'True Internet', 'Intouch Holdings']
        },
        'Indonesia': {
            'symbols': ['TLKM.JK', 'GOTO.JK'],
            'names': ['Telekomunikasi Indonesia', 'GoTo']
        },
        'Vietnam': {
            'symbols': ['VNM', 'FPT.VN'],
            'names': ['Vinamilk', 'FPT Corporation']
        }
    },
    # Western Asia - Tech Pioneers
    'west_asia': {
        'Israel': {
            'symbols': ['TRNFP', 'GMDA', 'WLFRG', 'NICE'],
            'names': ['Tronox', 'Goldilocks', 'Willfried Ventures', 'Nice Systems']
        },
        'Saudi Arabia': {
            'symbols': ['2222.SR', '7010.SR'],
            'names': ['Saudi Aramco', 'Saudi Telecom']
        },
        'UAE': {
            'symbols': ['EMAAR.AE', 'DU.AE'],
            'names': ['Emaar Properties', 'Emirates Integrated Telecom']
        }
    }
}

# Convert NumPy types to Python types
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_market_data_yahoo(symbols: List[str], force_fresh: bool = True) -> Dict:
    """Fetch real-time market data from Yahoo Finance API with fresh data option"""
    results = {}
    
    print(f"\n🔄 Fetching {'FRESH' if force_fresh else 'cached'} market data for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        try:
            # Force fresh data by creating new Ticker instance each time
            ticker = yf.Ticker(symbol)
            
            # Use period="1d" for today's data only (fresh)
            hist = ticker.history(period="5d" if force_fresh else "2d")
            
            if not hist.empty:
                info = ticker.info
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                
                # Get intraday high/low for today
                today_high = hist['High'].iloc[-1]
                today_low = hist['Low'].iloc[-1]
                
                results[symbol] = {
                    'current_price': round(float(current_price), 2),
                    'previous_close': round(float(prev_price), 2),
                    'change': round(float(current_price - prev_price), 2),
                    'change_percent': round(((current_price - prev_price) / prev_price) * 100, 2),
                    'volume': int(hist['Volume'].iloc[-1]),
                    'today_high': round(float(today_high), 2),
                    'today_low': round(float(today_low), 2),
                    'company_name': info.get('longName', symbol),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'source': 'Yahoo Finance',
                    'fetch_timestamp': datetime.now().isoformat(),
                    'data_date': hist.index[-1].strftime('%Y-%m-%d')
                }
                logger.info(f"✓ {symbol}: ${current_price:.2f} ({results[symbol]['change_percent']:+.2f}%)")
            else:
                results[symbol] = {'error': 'No data available', 'timestamp': datetime.now().isoformat()}
                
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            results[symbol] = {'error': str(e), 'timestamp': datetime.now().isoformat()}
        
        # Rate limiting - be respectful to Yahoo Finance
        if i < len(symbols) - 1:
            time.sleep(0.5)
    
    return results

def fetch_earnings_data(symbols: List[str], force_fresh: bool = True) -> Dict:
    """Fetch earnings data from Yahoo Finance with fresh data option"""
    results = {}
    
    print(f"\n📊 Fetching {'FRESH' if force_fresh else 'cached'} earnings data for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        try:
            # Force fresh data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            results[symbol] = {
                'company_name': info.get('longName', symbol),
                'last_reported_eps': info.get('trailingEps', 'N/A'),
                'estimated_eps': info.get('forwardEps', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'earnings_growth': info.get('earningsGrowth', 'N/A'),
                'revenue_growth': info.get('revenueGrowth', 'N/A'),
                'profit_margins': info.get('profitMargins', 'N/A'),
                'fetch_timestamp': datetime.now().isoformat()
            }
            logger.info(f"✓ Earnings for {symbol} fetched")
            
        except Exception as e:
            logger.warning(f"Earnings fetch failed for {symbol}: {e}")
            results[symbol] = {'error': str(e), 'timestamp': datetime.now().isoformat()}
        
        if i < len(symbols) - 1:
            time.sleep(0.5)
    
    return results

def get_regional_portfolio(region: str = None) -> Dict:
    """Get portfolio filtered by region - EXPANDED TO 19 STOCKS"""
    portfolio = {
        # East Asia - 8 stocks
        'TSM': {'shares': 1000, 'avg_cost': 95.50, 'region': 'East Asia', 'country': 'Taiwan', 'sector': 'Technology'},
        '005930.KS': {'shares': 500, 'avg_cost': 71000, 'region': 'East Asia', 'country': 'South Korea', 'sector': 'Technology'},
        '0700.HK': {'shares': 200, 'avg_cost': 400.00, 'region': 'East Asia', 'country': 'Hong Kong', 'sector': 'Technology'},
        '6758.T': {'shares': 300, 'avg_cost': 4500.00, 'region': 'East Asia', 'country': 'Japan', 'sector': 'Technology'},
        '9888.HK': {'shares': 400, 'avg_cost': 120.00, 'region': 'East Asia', 'country': 'Hong Kong', 'sector': 'Technology'},
        'BABA': {'shares': 250, 'avg_cost': 85.00, 'region': 'East Asia', 'country': 'China', 'sector': 'Technology'},
        '9984.T': {'shares': 150, 'avg_cost': 5800.00, 'region': 'East Asia', 'country': 'Japan', 'sector': 'Technology'},
        '000660.KS': {'shares': 300, 'avg_cost': 128000, 'region': 'East Asia', 'country': 'South Korea', 'sector': 'Technology'},
        
        # South Asia - 4 stocks
        'TCS.BO': {'shares': 100, 'avg_cost': 3500.00, 'region': 'South Asia', 'country': 'India', 'sector': 'Technology'},
        'INFY.BO': {'shares': 150, 'avg_cost': 1800.00, 'region': 'South Asia', 'country': 'India', 'sector': 'Technology'},
        'WIPRO.BO': {'shares': 200, 'avg_cost': 450.00, 'region': 'South Asia', 'country': 'India', 'sector': 'Technology'},
        'HCLTECH.BO': {'shares': 120, 'avg_cost': 1200.00, 'region': 'South Asia', 'country': 'India', 'sector': 'Technology'},
        
        # Southeast Asia - 4 stocks
        'ADVANC.BK': {'shares': 1000, 'avg_cost': 200.00, 'region': 'Southeast Asia', 'country': 'Thailand', 'sector': 'Telecommunications'},
        'U11.SI': {'shares': 500, 'avg_cost': 3.50, 'region': 'Southeast Asia', 'country': 'Singapore', 'sector': 'Technology'},
        'TLKM.JK': {'shares': 2000, 'avg_cost': 3500.00, 'region': 'Southeast Asia', 'country': 'Indonesia', 'sector': 'Telecommunications'},
        '5185.KL': {'shares': 800, 'avg_cost': 4.20, 'region': 'Southeast Asia', 'country': 'Malaysia', 'sector': 'Telecommunications'},
        
        # Western Asia - 3 stocks
        'EMAAR.AE': {'shares': 500, 'avg_cost': 50.00, 'region': 'Western Asia', 'country': 'UAE', 'sector': 'Real Estate/Tech'},
        '2222.SR': {'shares': 100, 'avg_cost': 28.50, 'region': 'Western Asia', 'country': 'Saudi Arabia', 'sector': 'Energy/Tech'},
        'TRNFP': {'shares': 300, 'avg_cost': 12.00, 'region': 'Western Asia', 'country': 'Israel', 'sector': 'Technology'},
    }
    
    if region:
        return {k: v for k, v in portfolio.items() if v['region'] == region}
    
    return portfolio

# ============================================================================
# CREWAI TOOLS
# ============================================================================

@tool("Fetches regional market summary")
def regional_market_summary_tool(region: str) -> Dict:
    """
    Fetch market data for all tech stocks in a specific Asian region.
    
    Args:
        region: One of 'East Asia', 'South Asia', 'Southeast Asia', 'Central Asia', 'Western Asia'
    
    Returns:
        Dictionary with market data for all stocks in that region
    """
    portfolio = get_regional_portfolio(region)
    symbols = list(portfolio.keys())
    
    if not symbols:
        return {'error': f'No stocks found for region: {region}'}
    
    market_data = fetch_market_data_yahoo(symbols, force_fresh=True)
    
    return {
        'region': region,
        'stocks_analyzed': len(symbols),
        'market_data': market_data,
        'portfolio': portfolio,
        'fetch_timestamp': datetime.now().isoformat()
    }

@tool("Fetches cross-regional comparison")
def cross_regional_comparison_tool(regions: str) -> Dict:
    """
    Compare market performance across multiple Asian regions.
    
    Args:
        regions: Comma-separated regions (e.g., 'East Asia,South Asia,Southeast Asia')
    
    Returns:
        Dictionary with comparative analysis
    """
    region_list = [r.strip() for r in regions.split(',')]
    comparison = {}
    
    for region in region_list:
        portfolio = get_regional_portfolio(region)
        symbols = list(portfolio.keys())
        
        if symbols:
            market_data = fetch_market_data_yahoo(symbols, force_fresh=True)
            successful = sum(1 for d in market_data.values() if 'error' not in d)
            avg_change = np.mean([d['change_percent'] for d in market_data.values() if 'error' not in d])
            
            comparison[region] = {
                'stocks_analyzed': len(symbols),
                'successful_fetches': successful,
                'avg_change_percent': round(avg_change, 2),
                'data': market_data,
                'fetch_timestamp': datetime.now().isoformat()
            }
    
    return comparison

@tool("Fetches earnings analysis for a region")
def regional_earnings_analysis_tool(region: str) -> Dict:
    """
    Analyze earnings data for tech stocks in a specific region.
    
    Args:
        region: One of 'East Asia', 'South Asia', 'Southeast Asia', etc.
    
    Returns:
        Dictionary with earnings analysis
    """
    portfolio = get_regional_portfolio(region)
    symbols = list(portfolio.keys())
    
    if not symbols:
        return {'error': f'No stocks found for region: {region}'}
    
    earnings_data = fetch_earnings_data(symbols, force_fresh=True)
    
    return {
        'region': region,
        'stocks_analyzed': len(symbols),
        'earnings_data': earnings_data,
        'portfolio': portfolio,
        'fetch_timestamp': datetime.now().isoformat()
    }

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_regional_performance(force_fresh: bool = True) -> Dict:
    """Analyze performance across all Asian regions with fresh data"""
    print("\n" + "="*80)
    print("🌏 COMPREHENSIVE ASIAN MARKET ANALYSIS - 19 STOCKS")
    print(f"Data Freshness: {'LIVE/FRESH' if force_fresh else 'Cached'}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    print("="*80 + "\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_freshness': 'fresh' if force_fresh else 'cached',
        'total_stocks': 19,
        'regions': {}
    }
    
    for region_name in ['East Asia', 'South Asia', 'Southeast Asia', 'Western Asia']:
        print(f"\n📊 Analyzing {region_name}...")
        print("-" * 60)
        
        portfolio = get_regional_portfolio(region_name)
        symbols = list(portfolio.keys())
        
        if not symbols:
            print(f"No stocks configured for {region_name}")
            continue
        
        print(f"Fetching FRESH data for {len(symbols)} stocks...")
        market_data = fetch_market_data_yahoo(symbols, force_fresh=force_fresh)
        earnings_data = fetch_earnings_data(symbols, force_fresh=force_fresh)
        
        # Calculate region statistics
        successful_markets = sum(1 for d in market_data.values() if 'error' not in d)
        successful_earnings = sum(1 for d in earnings_data.values() if 'error' not in d)
        
        if successful_markets > 0:
            avg_change = np.mean([
                d['change_percent'] for d in market_data.values() 
                if 'error' not in d
            ])
            market_cap_total = sum([
                d['market_cap'] for d in market_data.values() 
                if 'error' not in d and d['market_cap'] != 'N/A'
            ])
        else:
            avg_change = 0
            market_cap_total = 0
        
        region_summary = {
            'portfolio_size': len(symbols),
            'successful_market_fetches': successful_markets,
            'successful_earnings_fetches': successful_earnings,
            'average_change_percent': round(avg_change, 2),
            'total_market_cap': market_cap_total,
            'market_data': market_data,
            'earnings_data': earnings_data,
            'portfolio': portfolio,
            'fetch_timestamp': datetime.now().isoformat()
        }
        
        results['regions'][region_name] = region_summary
        
        print(f"\n✓ {region_name} Summary:")
        print(f"  Stocks: {len(symbols)}")
        print(f"  Market Data: {successful_markets}/{len(symbols)} fetched")
        print(f"  Avg Change: {avg_change:.2f}%")
        print(f"  Market Cap: ${market_cap_total:,.0f}")
    
    return results

def print_regional_comparison(results: Dict):
    """Print comparative analysis across regions"""
    print("\n" + "="*80)
    print("📈 REGIONAL COMPARISON SUMMARY - 19 STOCKS")
    print(f"Data Timestamp: {results.get('timestamp', 'N/A')}")
    print("="*80)
    print(f"{'Region':<20} {'Stocks':<10} {'Avg Change %':<15} {'Market Cap':<20}")
    print("-" * 65)
    
    for region, data in results['regions'].items():
        print(f"{region:<20} {data['portfolio_size']:<10} "
              f"{data['average_change_percent']:<14.2f}% "
              f"${data['total_market_cap']:<18,.0f}")
    
    print("="*80 + "\n")

def print_detailed_analysis(results: Dict):
    """Print detailed analysis for each region"""
    for region, data in results['regions'].items():
        print(f"\n{'='*80}")
        print(f"🌏 {region.upper()}")
        print(f"{'='*80}")
        
        print(f"\n📊 Market Data ({data['successful_market_fetches']}/{data['portfolio_size']} successful):")
        print(f"{'Symbol':<12} {'Price':<12} {'Change %':<12} {'Volume':<15} {'Sector':<20}")
        print("-" * 71)
        
        for symbol, market_data in data['market_data'].items():
            if 'error' not in market_data:
                price = market_data.get('current_price', 'N/A')
                change = market_data.get('change_percent', 'N/A')
                volume = market_data.get('volume', 'N/A')
                sector = market_data.get('sector', 'N/A')[:18]
                print(f"{symbol:<12} ${price:<11.2f} {change:<11.2f}% {volume:<14,d} {sector:<20}")
        
        print(f"\n📋 Earnings Data ({data['successful_earnings_fetches']}/{data['portfolio_size']} successful):")
        print(f"{'Symbol':<12} {'EPS':<12} {'Forward PE':<12} {'Growth':<12}")
        print("-" * 48)
        
        for symbol, earnings in data['earnings_data'].items():
            if 'error' not in earnings:
                eps = earnings.get('last_reported_eps', 'N/A')
                fwd_pe = earnings.get('forward_pe', 'N/A')
                growth = earnings.get('earnings_growth', 'N/A')
                
                eps_str = f"{eps:.2f}" if isinstance(eps, (int, float)) else str(eps)
                fwd_pe_str = f"{fwd_pe:.2f}" if isinstance(fwd_pe, (int, float)) else str(fwd_pe)
                growth_str = f"{growth*100:.1f}%" if isinstance(growth, (int, float)) else str(growth)
                
                print(f"{symbol:<12} {eps_str:<11} {fwd_pe_str:<11} {growth_str:<12}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🤖 MULTI-REGION ASIAN PORTFOLIO & MARKET DATA AGENT")
    print("="*80)
    print("\nConfiguration: Direct Data Fetching with FRESH Data")
    print("Portfolio Size: 19 Tech Stocks")
    print("Regions: East Asia | South Asia | Southeast Asia | Western Asia")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n")
    
    # Run comprehensive analysis with FRESH data
    results = analyze_regional_performance(force_fresh=True)
    
    # Print summaries
    print_regional_comparison(results)
    print_detailed_analysis(results)
    
    # Save results
    output_file = f"multi_region_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        results_clean = convert_numpy(results)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Results saved to: {output_file}\n")
        print(f"📊 Summary: 19 stocks analyzed across 4 regions")
        print(f"🕐 Data fetched at: {datetime.now().strftime('%I:%M:%S %p')}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()