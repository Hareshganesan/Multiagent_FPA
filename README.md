# Multi-Agent Financial Portfolio Assistant

A sophisticated AI-powered system for analyzing Asian tech stock markets using a multi-agent architecture that combines real-time market data, sentiment analysis, quantitative metrics, and natural language processing.

## Project Overview

This project implements a comprehensive financial analysis system using 6 specialized AI agents working together to provide intelligent portfolio insights through multiple interfaces including voice interaction.

Coverage: 19 tech stocks across 4 Asian regions (East Asia, South Asia, Southeast Asia, Western Asia)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Key Features](#key-features)
- [Agent Descriptions](#agent-descriptions)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Enhancements](#future-enhancements)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Problem Statement

### Core Challenges:

- **Fragmented Data Sources:** Market data scattered across multiple APIs and platforms
- **Manual Analysis Burden:** Time-consuming to track 19+ stocks across multiple regions
- **Sentiment Blindness:** Missing market sentiment signals from news and social media
- **Lack of Integration:** No unified system combining quantitative metrics with qualitative sentiment
- **Accessibility Gap:** Complex portfolio analysis not accessible through natural interfaces (voice, chat)
- **Real-time Decision Making:** Difficulty in getting actionable insights quickly for trading decisions
- **Regional Complexity:** Asian markets span multiple timezones, languages, and market dynamics
- **Information Overload:** Too much data, not enough actionable intelligence

### Target Users:

- Retail investors managing personal portfolios
- Portfolio managers overseeing multiple Asian tech stocks
- Financial analysts requiring comprehensive regional insights
- Traders needing quick sentiment + quantitative analysis

---

## Solution Architecture

### Multi-Agent System Design:

The solution implements 6 specialized agents working in a coordinated pipeline:

1. **API Agent** - Fetches real-time market data
2. **Scraping Agent** - Analyzes news sentiment
3. **Retriever Agent** - Enables semantic search (RAG)
4. **Analysis Agent** - Performs quantitative calculations
5. **Language Agent** - Generates natural language insights
6. **Voice Agent** - Provides hands-free interaction
7. **Orchestrator Agent** - Coordinates workflows

Each agent is autonomous, specialized, and outputs structured JSON for downstream consumption.

---

## Key Features

### Data Collection & Analysis:

- **Real-time Market Data:** Yahoo Finance API integration for prices, volumes, earnings
- **Sentiment Analysis:** Web scraping with Firecrawl/BeautifulSoup for news sentiment
- **Multi-source Fusion:** Combines quantitative + qualitative data
- **19 Stock Coverage:** Major tech companies across 4 Asian regions

### Intelligence Layer:

- **RAG Pipeline:** FAISS vector store + sentence transformers for semantic search
- **Quantitative Metrics:** Portfolio allocation, Sharpe ratio, volatility, VaR, HHI concentration
- **Earnings Analysis:** Identifies beats/misses with threshold-based alerts
- **Risk Assessment:** Multi-dimensional risk scoring with actionable insights

### Natural Language Processing:

- **LLM Integration:** Google Gemini 2.0 for natural language generation
- **Morning Briefs:** Automated comprehensive portfolio summaries
- **Q&A System:** Answer specific portfolio questions using RAG
- **Rate Limiting:** Intelligent caching + request queuing

### Voice Interface:

- **Speech Recognition:** Google Speech API with ambient noise calibration
- **Text-to-Speech:** gTTS for audio responses
- **Keyword Validation:** Ensures portfolio-related queries
- **Graceful Fallback:** Text input when recognition fails

### User Interfaces:

- **Streamlit Dashboard:** Visual analytics with interactive charts (Plotly)
- **AI Chatbot:** Text-based portfolio assistant
- **Voice Assistant:** Hands-free queries with audio recorder
- **Command-line Tools:** Batch processing for automation

---

## Agent Descriptions

### 1. API Agent (api_agent.py)

**Purpose:** Fetch real-time market data from Yahoo Finance

**Key Functions:**
- Fetches current prices, previous close, change %, volume for 19 stocks
- Retrieves market cap, sector, P/E ratio, earnings data
- Supports regional filtering (East Asia, South Asia, etc.)
- Handles rate limiting (0.5s delay between requests)
- **Outputs:** multi_region_results_TIMESTAMP.json

**Technologies:**
- yfinance for Yahoo Finance API
- numpy for numerical calculations
- JSON persistence with NumPy type conversion

---

### 2. Scraping Agent (scraping_agent.py)

**Purpose:** Analyze market sentiment from news articles

**Key Functions:**
- Scrapes Google News RSS feeds for each stock
- Extracts headlines + full article content (Firecrawl API or BeautifulSoup fallback)
- Weighted sentiment scoring with positive/negative keywords
- Regional sentiment aggregation
- **Outputs:** regional_sentiment_TIMESTAMP.json

**Technologies:**
- feedparser for RSS parsing
- BeautifulSoup for HTML scraping
- Firecrawl MCP API for advanced scraping
- Custom sentiment analyzer with keyword weighting

---

### 3. Retriever Agent (retriever_agent.py)

**Purpose:** Implement RAG pipeline for semantic search

**Key Functions:**
- Loads data from API + Scraping agents
- Chunks documents (512 chars, 50 overlap)
- Generates embeddings using sentence-transformers
- Indexes in FAISS vector store
- Supports semantic queries like "stocks with high growth"
- **Outputs:** vector_store/faiss_index.bin + metadata.json

**Technologies:**
- FAISS for vector similarity search
- sentence-transformers/all-MiniLM-L6-v2 for embeddings
- langchain for document processing

---

### 4. Analysis Agent (analysis_agent.py)

**Purpose:** Quantitative portfolio analysis

**Key Functions:**
- Calculates portfolio allocation by region/sector
- Computes risk metrics: volatility, Sharpe ratio, VaR (95%, 99%), HHI concentration
- Identifies earnings surprises (beats/misses)
- Analyzes sentiment trends across regions
- Generates morning briefs with recommendations
- **Outputs:** morning_brief_TIMESTAMP.json

**Technologies:**
- Direct JSON parsing (no string manipulation)
- numpy for statistical calculations
- pandas for data aggregation
- Optional RAG integration via Retriever Agent

---

### 5. Language Agent (language_agent.py)

**Purpose:** Generate natural language insights using LLM

**Key Functions:**
- Loads data from all agents (API, Scraping, Analysis)
- Constructs comprehensive context for LLM
- Generates morning briefs (3-4 paragraphs)
- Answers specific queries using RAG
- Implements rate limiting (2 req/min for Gemini)
- Response caching (1-hour TTL)
- **Outputs:** language_report_TIMESTAMP.json, llm_cache.json

**Technologies:**
- google-generativeai (Gemini 2.0 Flash)
- Rate limiter with request queue
- UTF-8 encoding for Windows compatibility

---

### 6. Voice Agent (voice_agent.py)

**Purpose:** Hands-free portfolio interaction

**Key Functions:**
- Speech-to-text using Google Speech Recognition
- Ambient noise calibration
- Query validation with portfolio keywords
- Text-to-speech using gTTS
- Audio playback with pygame
- Conversation history tracking
- Fallback to text input
- **Outputs:** voice_conversation_TIMESTAMP.json

**Technologies:**
- speech_recognition for STT
- gTTS for TTS
- pygame for audio playback
- Integrates with Language Agent for processing

---

### 7. Orchestrator Agent (orchestrator.py)

**Purpose:** Coordinate multi-agent workflows

**Key Functions:**
- Checks agent availability
- Executes predefined workflows:
  - **Full Pipeline:** All 6 agents sequentially
  - **Morning Brief:** Quick daily summary
  - **Voice Query:** Interactive voice session
  - **Quick Analysis:** API + Analysis only
  - **Sentiment Check:** Scraping for specific stocks/regions
- Handles file paths across directories
- Logs execution history
- **Outputs:** orchestrator_log_TIMESTAMP.json

**Technologies:**
- Python pathlib for cross-platform paths
- Dynamic agent loading
- Workflow state management

---

## Technical Stack

### Core Technologies:

- **Python 3.8+**
- **Data Collection:** yfinance, requests, feedparser, BeautifulSoup
- **Sentiment Analysis:** Custom keyword-based analyzer
- **Vector Store:** FAISS, sentence-transformers
- **LLM:** google-generativeai (Gemini 2.0 Flash)
- **Voice:** speech_recognition, gTTS, pygame
- **Web UI:** streamlit, plotly, pandas
- **Scraping:** Firecrawl MCP API (optional)

### Development Tools:

- **Environment Management:** python-dotenv
- **Logging:** Python logging module
- **Data Formats:** JSON (UTF-8 encoded)
- **Version Control:** Git

---

## Installation

### Prerequisites:

```bash
# Python 3.8 or higher
python --version

# pip (Python package manager)
pip --version
```

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/financial-assistant.git
cd financial-assistant
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Or install manually:
pip install yfinance requests feedparser beautifulsoup4
pip install google-generativeai
pip install sentence-transformers faiss-cpu
pip install langchain
pip install streamlit plotly pandas
pip install python-dotenv

# Voice dependencies (optional)
pip install SpeechRecognition gTTS pygame
pip install audio-recorder-streamlit  # For Streamlit voice interface
```

### Step 4: Configure Environment Variables

Create .env file in project root:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for enhanced scraping)
FIRECRAWL_API_KEY=your_firecrawl_api_key_here

# Optional (for FMP alternative data source)
FMP_API_KEY=your_fmp_api_key_here
```

**Get API Keys:**
- **Gemini:** https://aistudio.google.com/app/apikey
- **Firecrawl:** https://firecrawl.dev (optional)
- **FMP:** https://financialmodelingprep.com/developer/docs (optional)

---

## Usage

### Option 1: Run Full Pipeline via Orchestrator

```bash
cd orchestrator
python orchestrator.py
```

**Select Workflow:**
1. Full Pipeline (All Agents) - Recommended for first run
2. Morning Brief (Quick)
3. Voice Query
4. Quick Analysis
5. Sentiment Check

---

### Option 2: Run Individual Agents

**Step 1: Fetch Market Data**

```bash
cd agents
python api_agent.py
```

Output: multi_region_results_TIMESTAMP.json

**Step 2: Analyze Sentiment**

```bash
python scraping_agent.py
```

Output: regional_sentiment_TIMESTAMP.json

**Step 3: Build Vector Index**

```bash
python retriever_agent.py
```

Output: vector_store/faiss_index.bin, metadata.json

**Step 4: Generate Analysis**

```bash
python analysis_agent.py
```

Output: morning_brief_TIMESTAMP.json

**Step 5: Generate Language Report**

```bash
python language_agent.py
```

Output: language_report_TIMESTAMP.json, llm_cache.json

**Step 6: Use Voice Assistant**

```bash
python voice_agent.py
```

Requires: Microphone + speakers

---

### Option 3: Launch Streamlit Dashboard

```bash
cd streamlit_app
streamlit run app.py
```

**Features:**
- Dashboard: Overview with charts
- Market Data: Stock prices and changes
- Sentiment Analysis: News sentiment by region
- Portfolio Analysis: Risk metrics and recommendations
- AI Assistant: Text-based chatbot
- Voice Assistant: Audio recording + playback
- Settings: Configuration and cache management

**Access:** Browser opens automatically at http://localhost:8501

---

## Project Structure

```
financial-assistant/
│
├── agents/                          # Core agent modules
│   ├── api_agent.py                # Market data fetcher
│   ├── scraping_agent.py           # Sentiment analyzer
│   ├── retriever_agent.py          # RAG pipeline
│   ├── analysis_agent.py           # Quantitative analysis
│   ├── language_agent.py           # LLM-powered insights
│   ├── voice_agent.py              # Voice interface
│   ├── multi_region_results_*.json # API outputs
│   ├── regional_sentiment_*.json   # Scraping outputs
│   ├── morning_brief_*.json        # Analysis outputs
│   ├── language_report_*.json      # LLM outputs
│   ├── llm_cache.json              # Response cache
│   └── vector_store/               # FAISS index
│       ├── faiss_index.bin
│       └── metadata.json
│
├── orchestrator/                    # Workflow coordinator
│   ├── orchestrator.py             # Main orchestrator
│   └── orchestrator_log_*.json     # Execution logs
│
├── streamlit_app/                   # Web dashboard
│   └── app.py                      # Streamlit application
│
├── .env                             # Environment variables (NOT in repo)
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Challenges & Solutions

### Challenge 1: Rate Limiting from APIs

**Problem:**
- Yahoo Finance: Aggressive rate limiting
- Google Gemini: 2 requests/minute limit
- Firecrawl: Usage-based limits

**Solution:**
- Implemented 0.5s delays between Yahoo Finance requests
- Rate limiter class with request queue for Gemini
- Response caching (1-hour TTL) to minimize API calls
- Exponential backoff for failed requests
- Fallback to BeautifulSoup when Firecrawl unavailable

---

### Challenge 2: UTF-8 Encoding Issues on Windows

**Problem:**
- Windows default encoding (CP1252) corrupted JSON files
- Non-ASCII characters (Chinese, Japanese company names) caused crashes

**Solution:**
- Enforced encoding='utf-8' in all file I/O operations
- Updated Language Agent to use UTF-8 explicitly
- Added ensure_ascii=False to JSON dumps
- Tested on both Windows and Linux systems

---

### Challenge 3: Speech Recognition Accuracy

**Problem:**
- Background noise interfered with recognition
- Low confidence for portfolio-specific terms
- Google API errors for unclear speech

**Solution:**
- Ambient noise calibration (2s before recording)
- Keyword validation against portfolio terms
- Confidence scoring for recognition quality
- User confirmation for low-confidence results
- Graceful fallback to text input
- Tips displayed for better recognition

---

### Challenge 4: Data Synchronization Across Agents

**Problem:**
- Agents run independently, may have stale data
- Race conditions when multiple agents write simultaneously
- No guarantee of data consistency

**Solution:**
- Timestamped all JSON outputs (ISO 8601 format)
- File modification time checks for staleness
- Atomic file writes (write to temp, then rename)
- Orchestrator validates data availability before dependent agents
- Morning Brief workflow checks for data age (24-hour threshold)

---

### Challenge 5: Agent Dependency Management

**Problem:**
- Some agents require outputs from others
- Missing agents break workflows
- Unclear error messages for missing dependencies

**Solution:**
- Orchestrator checks agent availability at startup
- Graceful degradation (skip unavailable agents)
- Clear error messages indicating which agent to run first
- Optional dependencies (e.g., Retriever for semantic search)
- Standalone mode for individual agents

---

### Challenge 6: Memory & Performance

**Problem:**
- FAISS index grows large with many documents
- LLM responses slow for long contexts
- Streamlit reloads entire app on interaction

**Solution:**
- FAISS IndexFlatL2 for fast approximate search
- Document chunking (512 chars) to reduce embedding size
- LLM response caching to avoid redundant generation
- Streamlit @st.cache_data for expensive operations
- Lazy loading of agents (only when needed)

---

### Challenge 7: Cross-Platform Path Handling

**Problem:**
- Windows uses \, Linux/macOS use /
- Relative imports failed across different directories
- Orchestrator couldn't find agents

**Solution:**
- Used pathlib.Path for all file operations
- Explicit sys.path.insert(0, str(AGENTS_DIR))
- os.chdir() to agents directory before imports
- Restored original directory after execution
- Tested on Windows 10, Ubuntu 20.04, macOS

---

## Future Enhancements

### Short-term (Next 3 months):

- **Real-time Data Streaming:** WebSocket integration for live market updates
- **Email Alerts:** Automated notifications for price thresholds, sentiment shifts, earnings surprises
- **PDF Report Generation:** Daily/weekly reports with charts and analysis
- **Portfolio Backtesting:** Historical performance simulation with different strategies
- **Technical Indicators:** RSI, MACD, Bollinger Bands integration

### Medium-term (3-6 months):

- **Predictive Modeling:** LSTM/Transformer for price forecasting
- **Portfolio Optimization:** Modern Portfolio Theory (MPT) for allocation recommendations
- **Multi-language Support:** Chinese, Japanese, Korean for news and voice
- **Mobile App:** React Native for iOS/Android
- **Database Migration:** PostgreSQL for structured data, MongoDB for documents

### Long-term (6-12 months):

- **Options Analysis:** Greeks calculation, volatility surface
- **Correlation Analysis:** Cross-asset relationships
- **Automated Trading:** Paper trading integration with Interactive Brokers
- **Social Sentiment:** Twitter/Reddit integration
- **Multi-asset Support:** Expand beyond tech stocks to bonds, commodities, crypto
- **Enterprise Features:** Multi-user support, role-based access, audit logs

---

## Performance Metrics

### Data Processing:

- **API Agent:** ~30 seconds for 19 stocks (with rate limiting)
- **Scraping Agent:** ~5 minutes for 19 stocks (3 articles each, 2s delay)
- **Retriever Agent:** ~10 seconds to index ~200 documents
- **Analysis Agent:** <5 seconds for full portfolio analysis
- **Language Agent:** ~10 seconds per query (with caching)
- **Full Pipeline:** ~10-15 minutes end-to-end

### Accuracy:

- **Market Data:** 100% (direct API, no parsing)
- **Sentiment Analysis:** ~75-80% (keyword-based, no fine-tuned model)
- **Speech Recognition:** ~85-90% (with ambient noise calibration)
- **LLM Responses:** Qualitative (depends on context quality)

---

## Contributing

Contributions welcome! Please follow:

1. Fork the repository
2. Create feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open Pull Request

### Areas needing help:

- Fine-tuned sentiment models (FinBERT, etc.)
- Additional data sources (Bloomberg, Reuters)
- Mobile app development
- Multi-language support
- Performance optimization

---

## Author

**Your Name**

- GitHub: [@Hemanth31052006](https://github.com/Hemanth31052006)
- GitHub: [@Hareshganesan](https://github.com/Hareshganesan)

---

## Acknowledgments

- [Yahoo Finance API](https://github.com/ranaroussi/yfinance) for market data
- [Google Gemini](https://ai.google.dev/) for LLM capabilities
- [Firecrawl](https://firecrawl.dev/) for web scraping
- [Streamlit](https://streamlit.io/) for rapid UI development
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- Open-source community for all dependencies