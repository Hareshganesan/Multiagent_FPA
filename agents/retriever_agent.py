"""
Retriever Agent - RAG Pipeline for Financial Data
Indexes market data and sentiment analysis into vector store (FAISS)
Retrieves relevant context for portfolio queries
Version 1.0 - Multi-source indexing with hybrid search
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path

# Vector store and embeddings
import faiss
from sentence_transformers import SentenceTransformer

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

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
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, efficient model
VECTOR_STORE_PATH = "vector_store"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# ============================================================================
# RETRIEVER AGENT CLASS
# ============================================================================

class FinancialRetrieverAgent:
    """
    RAG-based retriever for financial market data and sentiment analysis
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize retriever with embedding model"""
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self.documents = []
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"✅ Retriever initialized (dim={self.embedding_dim})")
    
    # ========================================================================
    # DATA INGESTION
    # ========================================================================
    
    def load_api_data(self, json_file: str) -> List[Document]:
        """Load and process API agent market data"""
        logger.info(f"Loading API data from: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            timestamp = data.get('timestamp', 'Unknown')
            
            # Process each region
            for region, region_data in data.get('regions', {}).items():
                # Regional summary document
                summary_text = self._format_regional_summary(region, region_data)
                documents.append(Document(
                    page_content=summary_text,
                    metadata={
                        'source': 'api_agent',
                        'type': 'regional_summary',
                        'region': region,
                        'timestamp': timestamp
                    }
                ))
                
                # Individual stock documents
                for symbol, market_data in region_data.get('market_data', {}).items():
                    if 'error' not in market_data:
                        stock_text = self._format_stock_data(
                            symbol, 
                            market_data,
                            region_data.get('earnings_data', {}).get(symbol, {}),
                            region
                        )
                        documents.append(Document(
                            page_content=stock_text,
                            metadata={
                                'source': 'api_agent',
                                'type': 'stock_data',
                                'symbol': symbol,
                                'region': region,
                                'timestamp': timestamp
                            }
                        ))
            
            logger.info(f"✓ Loaded {len(documents)} documents from API data")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load API data: {e}")
            return []
    
    def load_sentiment_data(self, json_file: str) -> List[Document]:
        """Load and process scraping agent sentiment data"""
        logger.info(f"Loading sentiment data from: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            timestamp = data.get('timestamp', 'Unknown')
            
            # Process each region
            for region, region_data in data.get('regions', {}).items():
                # Regional sentiment summary
                if 'regional_sentiment' in region_data:
                    sentiment_text = self._format_regional_sentiment(region, region_data)
                    documents.append(Document(
                        page_content=sentiment_text,
                        metadata={
                            'source': 'scraping_agent',
                            'type': 'regional_sentiment',
                            'region': region,
                            'timestamp': timestamp
                        }
                    ))
                
                # Individual stock sentiment
                for symbol, stock_data in region_data.get('stocks', {}).items():
                    if 'error' not in stock_data:
                        sentiment_text = self._format_stock_sentiment(symbol, stock_data, region)
                        documents.append(Document(
                            page_content=sentiment_text,
                            metadata={
                                'source': 'scraping_agent',
                                'type': 'stock_sentiment',
                                'symbol': symbol,
                                'region': region,
                                'timestamp': timestamp
                            }
                        ))
                        
                        # Index individual news articles
                        for article in stock_data.get('articles', [])[:3]:
                            article_text = self._format_article(symbol, article, region)
                            documents.append(Document(
                                page_content=article_text,
                                metadata={
                                    'source': 'scraping_agent',
                                    'type': 'news_article',
                                    'symbol': symbol,
                                    'region': region,
                                    'timestamp': article.get('published', timestamp)
                                }
                            ))
            
            logger.info(f"✓ Loaded {len(documents)} documents from sentiment data")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")
            return []
    
    # ========================================================================
    # FORMATTING HELPERS
    # ========================================================================
    
    def _format_regional_summary(self, region: str, data: Dict) -> str:
        """Format regional market summary"""
        return f"""Regional Market Summary: {region}
Timestamp: {data.get('timestamp', 'N/A')}
Portfolio Size: {data.get('portfolio_size', 0)} stocks
Average Change: {data.get('average_change_percent', 0):.2f}%
Total Market Cap: ${data.get('total_market_cap', 0):,.0f}
Successful Fetches: {data.get('successful_market_fetches', 0)}/{data.get('portfolio_size', 0)}

This summary provides overview of {region} tech stocks performance including price movements and market capitalization."""
    
    def _format_stock_data(self, symbol: str, market_data: Dict, earnings_data: Dict, region: str) -> str:
        """Format individual stock market data"""
        text = f"""Stock: {symbol} ({region})
Company: {market_data.get('company_name', 'N/A')}
Sector: {market_data.get('sector', 'N/A')}

Market Data:
- Current Price: ${market_data.get('current_price', 0):.2f}
- Previous Close: ${market_data.get('previous_close', 0):.2f}
- Change: {market_data.get('change_percent', 0):.2f}%
- Volume: {market_data.get('volume', 0):,}
- Market Cap: ${market_data.get('market_cap', 0):,}
"""
        
        # Add earnings if available
        if earnings_data and 'error' not in earnings_data:
            eps = earnings_data.get('last_reported_eps', 'N/A')
            fwd_pe = earnings_data.get('forward_pe', 'N/A')
            growth = earnings_data.get('earnings_growth', 'N/A')
            
            text += f"""
Earnings Data:
- EPS: {eps if isinstance(eps, str) else f'{eps:.2f}'}
- Forward P/E: {fwd_pe if isinstance(fwd_pe, str) else f'{fwd_pe:.2f}'}
- Earnings Growth: {growth if isinstance(growth, str) else f'{growth*100:.1f}%'}
"""
        
        return text
    
    def _format_regional_sentiment(self, region: str, data: Dict) -> str:
        """Format regional sentiment summary"""
        sentiment = data.get('regional_sentiment', {})
        return f"""Regional Sentiment Analysis: {region}
Timestamp: {data.get('timestamp', 'N/A')}
Total Stocks Analyzed: {data.get('total_stocks', 0)}

Sentiment Breakdown:
- Positive Stocks: {sentiment.get('positive_stocks', 0)}
- Negative Stocks: {sentiment.get('negative_stocks', 0)}
- Neutral Stocks: {sentiment.get('neutral_stocks', 0)}
- Overall Trend: {sentiment.get('overall_trend', 'neutral').upper()}

This analysis provides market sentiment overview for {region} based on recent news and articles."""
    
    def _format_stock_sentiment(self, symbol: str, data: Dict, region: str) -> str:
        """Format individual stock sentiment"""
        summary = data.get('analysis_summary', {})
        breakdown = data.get('sentiment_breakdown', {})
        
        return f"""Stock Sentiment: {symbol} ({region})
Timestamp: {data.get('timestamp', 'N/A')}

Overall Sentiment: {summary.get('overall_sentiment', 'neutral').upper()}
Confidence: {summary.get('confidence', 0):.2f}
Articles Analyzed: {summary.get('article_count', 0)}

Sentiment Distribution:
- Positive: {breakdown.get('positive_percentage', 0):.1f}%
- Negative: {breakdown.get('negative_percentage', 0):.1f}%
- Neutral: {breakdown.get('neutral_percentage', 0):.1f}%

Recommendation: {data.get('recommendation', 'HOLD')}
"""
    
    def _format_article(self, symbol: str, article: Dict, region: str) -> str:
        """Format news article"""
        return f"""News Article: {symbol} ({region})
Title: {article.get('title', 'No title')}
Published: {article.get('published', 'N/A')}
Source: {article.get('source', 'Unknown')}

Headline Sentiment: {article.get('headline_sentiment', 'neutral').upper()}
Confidence: {article.get('headline_confidence', 0):.2f}

Summary: {article.get('summary', 'No summary available')[:300]}
"""
    
    # ========================================================================
    # INDEXING
    # ========================================================================
    
    def index_documents(self, documents: List[Document]) -> int:
        """Index documents into FAISS vector store"""
        if not documents:
            logger.warning("No documents to index")
            return 0
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Chunk documents
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ))
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        
        # Generate embeddings
        texts = [doc.page_content for doc in chunked_docs]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata and documents
        self.metadata.extend([doc.metadata for doc in chunked_docs])
        self.documents.extend(chunked_docs)
        
        logger.info(f"✅ Indexed {len(chunked_docs)} chunks (total: {self.index.ntotal})")
        return len(chunked_docs)
    
    def build_index_from_files(self, api_file: str, sentiment_file: str) -> Dict:
        """Build complete index from both data sources"""
        logger.info("Building index from data files...")
        
        stats = {
            'api_documents': 0,
            'sentiment_documents': 0,
            'total_chunks': 0
        }
        
        # Load API data
        if os.path.exists(api_file):
            api_docs = self.load_api_data(api_file)
            stats['api_documents'] = len(api_docs)
            self.index_documents(api_docs)
        else:
            logger.warning(f"API file not found: {api_file}")
        
        # Load sentiment data
        if os.path.exists(sentiment_file):
            sentiment_docs = self.load_sentiment_data(sentiment_file)
            stats['sentiment_documents'] = len(sentiment_docs)
            self.index_documents(sentiment_docs)
        else:
            logger.warning(f"Sentiment file not found: {sentiment_file}")
        
        stats['total_chunks'] = self.index.ntotal
        
        return stats
    
    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    
    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS, 
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant documents for query"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty. Please build index first.")
            return []
        
        logger.info(f"Retrieving top-{top_k} results for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k * 3, self.index.ntotal)  # Get more for filtering
        )
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                metadata = self.metadata[idx]
                
                # Apply metadata filter if specified
                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                results.append({
                    'content': doc.page_content,
                    'metadata': metadata,
                    'score': float(dist),
                    'relevance': 1.0 / (1.0 + float(dist))  # Convert distance to similarity
                })
        
        # Sort by relevance and return top-k
        results.sort(key=lambda x: x['relevance'], reverse=True)
        results = results[:top_k]
        
        logger.info(f"✓ Retrieved {len(results)} relevant documents")
        return results
    
    def retrieve_by_symbol(self, symbol: str, top_k: int = 5) -> List[Dict]:
        """Retrieve all data for a specific stock symbol"""
        return self.retrieve(
            query=f"Stock data and sentiment for {symbol}",
            top_k=top_k,
            filter_metadata={'symbol': symbol}
        )
    
    def retrieve_by_region(self, region: str, top_k: int = 5) -> List[Dict]:
        """Retrieve data for a specific region"""
        return self.retrieve(
            query=f"Market data and sentiment for {region}",
            top_k=top_k,
            filter_metadata={'region': region}
        )
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save_index(self, directory: str = VECTOR_STORE_PATH):
        """Save FAISS index and metadata to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, INDEX_FILE)
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, METADATA_FILE)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.metadata,
                'documents': [{'content': d.page_content, 'metadata': d.metadata} 
                             for d in self.documents]
            }, f, indent=2)
        
        logger.info(f"✅ Index saved to: {directory}")
    
    def load_index(self, directory: str = VECTOR_STORE_PATH):
        """Load FAISS index and metadata from disk"""
        index_path = os.path.join(directory, INDEX_FILE)
        metadata_path = os.path.join(directory, METADATA_FILE)
        
        if not os.path.exists(index_path):
            logger.warning(f"Index file not found: {index_path}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = data['metadata']
            self.documents = [Document(page_content=d['content'], metadata=d['metadata']) 
                            for d in data['documents']]
        
        logger.info(f"✅ Index loaded from: {directory} ({self.index.ntotal} vectors)")
        return True
    
    # ========================================================================
    # QUERY UTILITIES
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        regions = set(m.get('region') for m in self.metadata if 'region' in m)
        symbols = set(m.get('symbol') for m in self.metadata if 'symbol' in m)
        sources = set(m.get('source') for m in self.metadata if 'source' in m)
        
        return {
            'total_vectors': self.index.ntotal,
            'total_documents': len(self.documents),
            'regions': sorted(list(regions)),
            'unique_symbols': len(symbols),
            'sources': sorted(list(sources)),
            'embedding_dimension': self.embedding_dim
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🤖 RETRIEVER AGENT - RAG PIPELINE")
    print("Financial Data Indexing & Retrieval System")
    print("="*70 + "\n")
    
    # Initialize agent
    agent = FinancialRetrieverAgent()
    
    # Find latest data files
    api_files = sorted(Path('.').glob('multi_region_results_*.json'))
    sentiment_files = sorted(Path('.').glob('regional_sentiment_*.json'))
    
    if not api_files or not sentiment_files:
        print("❌ Data files not found. Please run api_agent.py and scraping_agent.py first.")
        return
    
    api_file = str(api_files[-1])
    sentiment_file = str(sentiment_files[-1])
    
    print(f"📂 Using data files:")
    print(f"  API Data: {api_file}")
    print(f"  Sentiment Data: {sentiment_file}\n")
    
    # Build index
    print("🔨 Building vector index...")
    stats = agent.build_index_from_files(api_file, sentiment_file)
    
    print(f"\n✅ Index built successfully!")
    print(f"  API Documents: {stats['api_documents']}")
    print(f"  Sentiment Documents: {stats['sentiment_documents']}")
    print(f"  Total Chunks Indexed: {stats['total_chunks']}\n")
    
    # Save index
    agent.save_index()
    
    # Display stats
    index_stats = agent.get_stats()
    print("="*70)
    print("📊 INDEX STATISTICS")
    print("="*70)
    print(f"Total Vectors: {index_stats['total_vectors']}")
    print(f"Total Documents: {index_stats['total_documents']}")
    print(f"Unique Symbols: {index_stats['unique_symbols']}")
    print(f"Regions: {', '.join(index_stats['regions'])}")
    print(f"Sources: {', '.join(index_stats['sources'])}")
    print(f"Embedding Dimension: {index_stats['embedding_dimension']}")
    print("="*70 + "\n")
    
    # Test queries
    print("🔍 Testing retrieval with sample queries...\n")
    
    test_queries = [
        "What is the risk exposure in Asia tech stocks?",
        "Show me earnings data for TSMC",
        "What is the sentiment for Samsung?",
        "Market performance in South Asia"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"Query: '{query}'")
        print(f"{'─'*70}")
        
        results = agent.retrieve(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] Relevance: {result['relevance']:.4f}")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"Type: {result['metadata'].get('type', 'Unknown')}")
            if 'symbol' in result['metadata']:
                print(f"Symbol: {result['metadata']['symbol']}")
            if 'region' in result['metadata']:
                print(f"Region: {result['metadata']['region']}")
            print(f"\nContent Preview:\n{result['content'][:300]}...")
    
    print("\n" + "="*70)
    print("✅ Retriever Agent Ready!")
    print("="*70)


if __name__ == "__main__":
    main()