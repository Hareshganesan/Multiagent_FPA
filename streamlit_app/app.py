"""
Streamlit Financial Assistant - COMPLETE VERSION WITH WORKING VOICE
Includes full voice integration using streamlit-audio-recorder
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import tempfile
import base64

# Audio components
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    st.warning("⚠️ Install audio-recorder-streamlit: pip install audio-recorder-streamlit")

try:
    import speech_recognition as sr
    from gtts import gTTS
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

STREAMLIT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = STREAMLIT_DIR.parent.absolute()
AGENTS_DIR = PROJECT_ROOT / 'agents'

# Add to Python path
sys.path.insert(0, str(AGENTS_DIR))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Financial Assistant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .voice-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .listening {
        background-color: #ffebee;
        color: #c62828;
    }
    .processing {
        background-color: #fff3e0;
        color: #e65100;
    }
    .speaking {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'language_agent' not in st.session_state:
    st.session_state.language_agent = None

if 'retriever_agent' not in st.session_state:
    st.session_state.retriever_agent = None

if 'voice_mode' not in st.session_state:
    st.session_state.voice_mode = False

if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None

if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def find_and_load_file(pattern: str):
    """Find ANY file matching pattern and load it"""
    try:
        files = list(AGENTS_DIR.glob(pattern))
        
        if not files:
            return None
        
        latest_file = sorted(files)[-1]
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'data': data,
            'filename': latest_file.name,
            'timestamp': datetime.fromtimestamp(latest_file.stat().st_mtime),
            'path': str(latest_file)
        }
    except Exception as e:
        st.sidebar.error(f"Error loading {pattern}: {str(e)[:50]}")
        return None

def load_api_data():
    """Load ANY multi_region_results_*.json file"""
    return find_and_load_file('multi_region_results*.json')

def load_sentiment_data():
    """Load ANY regional_sentiment_*.json file"""
    return find_and_load_file('regional_sentiment*.json')

def load_analysis_data():
    """Load ANY morning_brief_*.json file"""
    return find_and_load_file('morning_brief*.json')

# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

def initialize_language_agent():
    """Initialize Language Agent once"""
    if st.session_state.language_agent is None:
        try:
            original_dir = os.getcwd()
            os.chdir(AGENTS_DIR)
            
            try:
                from language_agent import EnhancedLanguageAgent
                st.session_state.language_agent = EnhancedLanguageAgent(max_requests=2)
                return True
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            st.error(f"❌ Failed to initialize Language Agent: {str(e)}")
            return False
    
    return True

def initialize_retriever_agent():
    """Initialize Retriever Agent once"""
    if st.session_state.retriever_agent is None:
        try:
            original_dir = os.getcwd()
            os.chdir(AGENTS_DIR)
            
            try:
                from retriever_agent import FinancialRetrieverAgent
                retriever = FinancialRetrieverAgent()
                
                if retriever.load_index():
                    st.session_state.retriever_agent = retriever
                    return True
                else:
                    return False
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            return False
    
    return True

# ============================================================================
# VOICE PROCESSING FUNCTIONS
# ============================================================================

def transcribe_audio(audio_bytes):
    """Convert audio bytes to text using speech recognition"""
    if not VOICE_AVAILABLE:
        return None, "Speech recognition not available. Install: pip install SpeechRecognition"
    
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            fp.write(audio_bytes)
            temp_audio_path = fp.name
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
        
        # Recognize speech
        try:
            text = recognizer.recognize_google(audio_data)
            os.unlink(temp_audio_path)
            return text, None
        except sr.UnknownValueError:
            os.unlink(temp_audio_path)
            return None, "Could not understand audio. Please speak clearly."
        except sr.RequestError as e:
            os.unlink(temp_audio_path)
            return None, f"Speech recognition service error: {e}"
            
    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

def text_to_speech(text):
    """Convert text to speech and return audio file path"""
    if not VOICE_AVAILABLE:
        return None
    
    try:
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

def get_audio_player_html(audio_path):
    """Generate HTML audio player"""
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    audio_html = f"""
    <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    
    return audio_html

def process_voice_query(query_text):
    """Process voice query and return response"""
    try:
        original_dir = os.getcwd()
        os.chdir(AGENTS_DIR)
        
        try:
            # Load all data
            all_data = st.session_state.language_agent.load_all_agent_data()
            
            # Check for morning brief keywords
            brief_keywords = ['brief', 'summary', 'overview', 'morning', 'update', 'report']
            
            if any(keyword in query_text.lower() for keyword in brief_keywords):
                result = st.session_state.language_agent.generate_comprehensive_brief(all_data)
            else:
                result = st.session_state.language_agent.answer_with_rag(
                    query_text,
                    st.session_state.retriever_agent,
                    all_data
                )
            
            if result['success']:
                return result.get('brief_text') or result.get('answer'), None
            else:
                return None, result.get('error', 'Unknown error')
        
        finally:
            os.chdir(original_dir)
            
    except Exception as e:
        return None, str(e)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_regional_performance_chart(api_data):
    """Create regional performance chart"""
    if not api_data or 'regions' not in api_data:
        return None
    
    regions = []
    avg_changes = []
    
    for region, data in api_data['regions'].items():
        regions.append(region)
        avg_changes.append(data.get('average_change_percent', 0))
    
    df = pd.DataFrame({
        'Region': regions,
        'Avg Change %': avg_changes
    })
    
    fig = go.Figure()
    
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in df['Avg Change %']]
    
    fig.add_trace(go.Bar(
        x=df['Region'],
        y=df['Avg Change %'],
        marker_color=colors,
        text=[f"{x:+.2f}%" for x in df['Avg Change %']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Change: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Regional Performance Comparison',
        xaxis_title='Region',
        yaxis_title='Average Change %',
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """Create sentiment chart"""
    if not sentiment_data or 'regions' not in sentiment_data:
        return None
    
    regions = []
    positive = []
    negative = []
    neutral = []
    
    for region, data in sentiment_data['regions'].items():
        regional_sentiment = data.get('regional_sentiment', {})
        regions.append(region)
        positive.append(regional_sentiment.get('positive_stocks', 0))
        negative.append(regional_sentiment.get('negative_stocks', 0))
        neutral.append(regional_sentiment.get('neutral_stocks', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive',
        x=regions,
        y=positive,
        marker_color='#00CC96',
        hovertemplate='<b>%{x}</b><br>Positive: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative',
        x=regions,
        y=negative,
        marker_color='#EF553B',
        hovertemplate='<b>%{x}</b><br>Negative: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Neutral',
        x=regions,
        y=neutral,
        marker_color='#636EFA',
        hovertemplate='<b>%{x}</b><br>Neutral: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Sentiment Analysis by Region',
        xaxis_title='Region',
        yaxis_title='Number of Stocks',
        barmode='stack',
        height=400,
        hovermode='x unified'
    )
    
    return fig

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_dashboard():
    """Dashboard page"""
    st.markdown('<div class="main-header">📊 Portfolio Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    api_result = load_api_data()
    sentiment_result = load_sentiment_data()
    analysis_result = load_analysis_data()
    
    # Check if data exists
    if not any([api_result, sentiment_result, analysis_result]):
        st.warning("⚠️ No data files found!")
        
        st.info("""
        **To generate data files:**
        
        1. Run API Agent:
        ```bash
        cd agents
        python api_agent.py
        ```
        
        2. Run Scraping Agent:
        ```bash
        python scraping_agent.py
        ```
        
        3. Run Analysis Agent:
        ```bash
        python analysis_agent.py
        ```
        
        **Or use Orchestrator:**
        ```bash
        cd orchestrator
        python orchestrator.py
        ```
        Then select option 1 (Full Pipeline)
        """)
        return
    
    # Show what data is available
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if api_result:
            st.success(f"✅ Market Data")
            st.caption(f"File: {api_result['filename']}")
            st.caption(f"Updated: {api_result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.error("❌ Market data missing")
    
    with col2:
        if sentiment_result:
            st.success(f"✅ Sentiment Data")
            st.caption(f"File: {sentiment_result['filename']}")
            st.caption(f"Updated: {sentiment_result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.error("❌ Sentiment data missing")
    
    with col3:
        if analysis_result:
            st.success(f"✅ Analysis Data")
            st.caption(f"File: {analysis_result['filename']}")
            st.caption(f"Updated: {analysis_result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.error("❌ Analysis data missing")
    
    st.markdown("---")
    
    # Key metrics from analysis
    if analysis_result and 'summary' in analysis_result['data']:
        summary = analysis_result['data']['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total AUM", f"${summary.get('total_aum', 0):,.0f}")
        
        with col2:
            mean_return = summary.get('mean_return', 0)
            st.metric("Mean Return", f"{mean_return:.2f}%", delta=f"{mean_return:+.2f}%")
        
        with col3:
            st.metric("Total Stocks", summary.get('total_stocks', 0))
        
        with col4:
            st.metric("Volatility", f"{summary.get('volatility', 0):.2f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if api_result:
            fig = create_regional_performance_chart(api_result['data'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Market data chart unavailable")
    
    with col2:
        if sentiment_result:
            fig = create_sentiment_chart(sentiment_result['data'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💭 Sentiment chart unavailable")

def show_market_data():
    """Market data page"""
    st.title("📊 Market Data")
    st.markdown("---")
    
    api_result = load_api_data()
    
    if not api_result:
        st.warning("⚠️ No market data files found!")
        st.info("""
        Run the API agent to generate market data:
        ```bash
        cd agents
        python api_agent.py
        ```
        """)
        return
    
    api_data = api_result['data']
    
    st.success(f"✅ Using: {api_result['filename']}")
    st.caption(f"Last updated: {api_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    # Display regional data
    for region, data in api_data.get('regions', {}).items():
        with st.expander(f"📊 {region}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stocks", data.get('portfolio_size', 0))
            
            with col2:
                avg_change = data.get('average_change_percent', 0)
                st.metric("Avg Change", f"{avg_change:.2f}%", delta=f"{avg_change:+.2f}%")
            
            with col3:
                st.metric("Market Cap", f"${data.get('total_market_cap', 0):,.0f}")
            
            # Stock details table
            if 'market_data' in data:
                st.markdown("##### 📈 Individual Stocks")
                
                stock_data = []
                for symbol, info in data['market_data'].items():
                    if 'error' not in info:
                        stock_data.append({
                            'Symbol': symbol,
                            'Price': f"${info.get('current_price', 0):.2f}",
                            'Change %': f"{info.get('change_percent', 0):.2f}%",
                            'Volume': f"{info.get('volume', 0):,}",
                            'Market Cap': f"${info.get('market_cap', 0):,}"
                        })
                
                if stock_data:
                    df = pd.DataFrame(stock_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

def show_sentiment_analysis():
    """Sentiment analysis page"""
    st.title("💭 Sentiment Analysis")
    st.markdown("---")
    
    sentiment_result = load_sentiment_data()
    
    if not sentiment_result:
        st.warning("⚠️ No sentiment data files found!")
        st.info("""
        Run the scraping agent to generate sentiment data:
        ```bash
        cd agents
        python scraping_agent.py
        ```
        """)
        return
    
    sentiment_data = sentiment_result['data']
    
    st.success(f"✅ Using: {sentiment_result['filename']}")
    st.caption(f"Last updated: {sentiment_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    # Overall sentiment chart
    fig = create_sentiment_chart(sentiment_data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Display sentiment by region
    for region, data in sentiment_data.get('regions', {}).items():
        with st.expander(f"💭 {region}", expanded=False):
            sentiment = data.get('regional_sentiment', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Positive", sentiment.get('positive_stocks', 0))
            
            with col2:
                st.metric("Negative", sentiment.get('negative_stocks', 0))
            
            with col3:
                st.metric("Neutral", sentiment.get('neutral_stocks', 0))
            
            with col4:
                trend = sentiment.get('overall_trend', 'neutral').upper()
                color = "🟢" if trend == "POSITIVE" else "🔴" if trend == "NEGATIVE" else "🟡"
                st.metric("Trend", f"{color} {trend}")

def show_portfolio_analysis():
    """Portfolio analysis page"""
    st.title("📈 Portfolio Analysis")
    st.markdown("---")
    
    analysis_result = load_analysis_data()
    
    if not analysis_result:
        st.warning("⚠️ No analysis data files found!")
        st.info("""
        Run the analysis agent to generate portfolio analysis:
        ```bash
        cd agents
        python analysis_agent.py
        ```
        """)
        return
    
    analysis_data = analysis_result['data']
    
    st.success(f"✅ Using: {analysis_result['filename']}")
    st.caption(f"Last updated: {analysis_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    if 'summary' in analysis_data:
        summary = analysis_data['summary']
        
        # Overview metrics
        st.subheader("📊 Portfolio Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total AUM", f"${summary.get('total_aum', 0):,.0f}")
            st.metric("Total Stocks", summary.get('total_stocks', 0))
        
        with col2:
            st.metric("Mean Return", f"{summary.get('mean_return', 0):.2f}%")
            st.metric("Volatility", f"{summary.get('volatility', 0):.2f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{summary.get('sharpe_ratio', 0):.2f}")
            st.metric("Concentration Risk", summary.get('concentration_risk', 'Unknown'))
        
        st.markdown("---")
        
        # Earnings
        st.subheader("💰 Earnings Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            beats = summary.get('earnings_beats', 0)
            st.metric("Earnings Beats", beats, delta=f"+{beats}")
        
        with col2:
            misses = summary.get('earnings_misses', 0)
            st.metric("Earnings Misses", misses, delta=f"-{misses}")
        
        with col3:
            total = beats + misses
            if total > 0:
                beat_rate = (beats / total) * 100
                st.metric("Beat Rate", f"{beat_rate:.1f}%")
        
        st.markdown("---")
        
        # Recommendation
        st.subheader("🎯 Recommendation")
        st.info(summary.get('recommendation', 'No recommendation available'))

def show_ai_assistant():
    """AI Assistant chatbot page"""
    st.title("🤖 AI Assistant")
    st.markdown("---")
    
    st.markdown("""
    Ask me anything about your portfolio! I can help with:
    - 📊 Market performance and trends
    - 💭 Sentiment analysis
    - 📈 Stock recommendations
    - 💰 Earnings and fundamentals
    - 🌏 Regional comparisons
    - 📉 Risk assessment
    """)
    
    st.markdown("---")
    
    # Initialize agents
    with st.spinner("Initializing AI Agent..."):
        if not initialize_language_agent():
            st.error("❌ Failed to initialize Language Agent. Please check your GEMINI_API_KEY in .env file.")
            return
        
        initialize_retriever_agent()  # Optional
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Query input
    query = st.chat_input("Ask me about your portfolio...")
    
    if query:
        # Add user message to history
        st.session_state.chat_history.append({'role': 'user', 'content': query})
        
        # Process query
        with st.spinner("Thinking..."):
            try:
                # Load all agent data
                original_dir = os.getcwd()
                os.chdir(AGENTS_DIR)
                
                try:
                    all_data = st.session_state.language_agent.load_all_agent_data()
                    
                    # Generate response
                    result = st.session_state.language_agent.answer_with_rag(
                        query,
                        st.session_state.retriever_agent,
                        all_data
                    )
                    
                    if result['success']:
                        response = result['answer']
                    else:
                        response = f"I encountered an error: {result.get('error', 'Unknown error')}"
                
                finally:
                    os.chdir(original_dir)
                
                # Add assistant response to history
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                
                # Rerun to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

def show_voice_assistant():
    """Voice Assistant page with working audio integration"""
    st.title("🎤 Voice Assistant")
    st.markdown("---")
    
    # Check dependencies
    if not AUDIO_RECORDER_AVAILABLE:
        st.error("❌ Audio recorder not available!")
        st.info("""
        **Install required package:**
        ```bash
        pip install audio-recorder-streamlit
        ```
        """)
        return
    
    if not VOICE_AVAILABLE:
        st.error("❌ Speech recognition not available!")
        st.info("""
        **Install required packages:**
        ```bash
        pip install SpeechRecognition gtts
        ```
        """)
        return
    
    # Initialize agents
    with st.spinner("Initializing AI Agent..."):
        if not initialize_language_agent():
            st.error("❌ Failed to initialize Language Agent")
            return
        
        initialize_retriever_agent()
    
    st.success("✅ Voice Assistant Ready!")
    
    st.markdown("---")
    
    # Instructions
    st.info("""
    **How to use:**
    1. Click the microphone button below to start recording
    2. Speak your question clearly
    3. Click stop when finished
    4. Wait for the response (text + audio)
    
    **Example questions:**
    - "What is my portfolio performance?"
    - "Give me a morning brief"
    - "Which stocks should I invest in?"
    - "Show me sentiment analysis for East Asia"
    """)
    
    st.markdown("---")
    
    # Audio recorder
    st.subheader("🎤 Record Your Question")
    
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
    )
    
    # Process audio
    if audio_bytes and audio_bytes != st.session_state.last_audio:
        st.session_state.last_audio = audio_bytes
        st.session_state.processing_audio = True
        
        # Show processing status
        status_placeholder = st.empty()
        status_placeholder.markdown('<div class="voice-status processing">🔄 Processing audio...</div>', unsafe_allow_html=True)
        
        # Transcribe audio
        with st.spinner("Converting speech to text..."):
            transcript, error = transcribe_audio(audio_bytes)
        
        if error:
            status_placeholder.markdown(f'<div class="voice-status listening">❌ {error}</div>', unsafe_allow_html=True)
            st.session_state.processing_audio = False
        elif transcript:
            # Display transcript
            st.markdown("---")
            st.subheader("📝 Your Question:")
            st.info(transcript)
            
            status_placeholder.markdown('<div class="voice-status processing">🤔 Generating response...</div>', unsafe_allow_html=True)
            
            # Process query
            with st.spinner("Analyzing your question..."):
                response, error = process_voice_query(transcript)
            
            if error:
                st.error(f"❌ Error: {error}")
                status_placeholder.empty()
            elif response:
                # Display response
                st.markdown("---")
                st.subheader("💬 Assistant's Response:")
                st.success(response)
                
                # Generate audio response
                status_placeholder.markdown('<div class="voice-status speaking">🔊 Generating audio...</div>', unsafe_allow_html=True)
                
                with st.spinner("Converting text to speech..."):
                    audio_path = text_to_speech(response)
                
                if audio_path:
                    # Display audio player
                    st.markdown("---")
                    st.subheader("🔊 Listen to Response:")
                    
                    audio_html = get_audio_player_html(audio_path)
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                    status_placeholder.markdown('<div class="voice-status speaking">✅ Complete!</div>', unsafe_allow_html=True)
                    
                    # Cleanup
                    time.sleep(1)
                    try:
                        os.unlink(audio_path)
                    except:
                        pass
                else:
                    status_placeholder.markdown('<div class="voice-status listening">⚠️ Audio generation failed</div>', unsafe_allow_html=True)
            
            st.session_state.processing_audio = False
        else:
            status_placeholder.markdown('<div class="voice-status listening">❌ No speech detected</div>', unsafe_allow_html=True)
            st.session_state.processing_audio = False
    
    st.markdown("---")
    
    # Alternative text input
    st.subheader("⌨️ Or Type Your Question:")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        text_query = st.text_input("Enter your question:", key="voice_text_input", placeholder="What's my portfolio performance?")
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        submit_button = st.button("🔊 Ask", type="primary")
    
    if submit_button and text_query:
        # Display question
        st.markdown("---")
        st.subheader("📝 Your Question:")
        st.info(text_query)
        
        # Process query
        with st.spinner("Processing..."):
            response, error = process_voice_query(text_query)
        
        if error:
            st.error(f"❌ Error: {error}")
        elif response:
            # Display response
            st.markdown("---")
            st.subheader("💬 Assistant's Response:")
            st.success(response)
            
            # Generate audio
            with st.spinner("Generating audio..."):
                audio_path = text_to_speech(response)
            
            if audio_path:
                st.markdown("---")
                st.subheader("🔊 Listen to Response:")
                
                audio_html = get_audio_player_html(audio_path)
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Cleanup
                time.sleep(1)
                try:
                    os.unlink(audio_path)
                except:
                    pass

def show_settings():
    """Settings page"""
    st.title("⚙️ Settings")
    st.markdown("---")
    
    st.subheader("📁 Configuration")
    st.code(f"Project Root: {PROJECT_ROOT}")
    st.code(f"Agents Directory: {AGENTS_DIR}")
    
    st.markdown("---")
    
    # Check dependencies
    st.subheader("📦 Dependencies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Dependencies:**")
        st.write("✅ Streamlit" if 'streamlit' in sys.modules else "❌ Streamlit")
        st.write("✅ Pandas" if 'pandas' in sys.modules else "❌ Pandas")
        st.write("✅ Plotly" if 'plotly' in sys.modules else "❌ Plotly")
        st.write("✅ Python-dotenv" if 'dotenv' in sys.modules else "❌ Python-dotenv")
    
    with col2:
        st.markdown("**Voice Dependencies:**")
        st.write("✅ Audio Recorder" if AUDIO_RECORDER_AVAILABLE else "❌ Audio Recorder")
        st.write("✅ Speech Recognition" if VOICE_AVAILABLE else "❌ Speech Recognition")
        st.write("✅ gTTS" if VOICE_AVAILABLE else "❌ gTTS")
    
    st.markdown("---")
    
    st.subheader("📂 Available Data Files")
    
    # Search for files
    api_files = list(AGENTS_DIR.glob('multi_region_results*.json'))
    sentiment_files = list(AGENTS_DIR.glob('regional_sentiment*.json'))
    analysis_files = list(AGENTS_DIR.glob('morning_brief*.json'))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Market Data Files**")
        if api_files:
            for f in sorted(api_files)[-3:]:
                st.text(f"✅ {f.name}")
        else:
            st.text("❌ No files found")
    
    with col2:
        st.markdown("**Sentiment Data Files**")
        if sentiment_files:
            for f in sorted(sentiment_files)[-3:]:
                st.text(f"✅ {f.name}")
        else:
            st.text("❌ No files found")
    
    with col3:
        st.markdown("**Analysis Data Files**")
        if analysis_files:
            for f in sorted(analysis_files)[-3:]:
                st.text(f"✅ {f.name}")
        else:
            st.text("❌ No files found")
    
    st.markdown("---")
    
    st.subheader("🔄 Cache Management")
    
    if st.button("🗑️ Clear Cache", type="primary"):
        st.cache_data.clear()
        st.success("✅ Cache cleared! Please refresh the page.")
    
    st.markdown("---")
    
    st.subheader("📝 Installation Instructions")
    
    with st.expander("Install Voice Dependencies"):
        st.code("""
# Install audio recorder for Streamlit
pip install audio-recorder-streamlit

# Install speech recognition
pip install SpeechRecognition

# Install text-to-speech
pip install gTTS

# Install audio processing (if needed)
pip install pydub
        """)
    
    with st.expander("Install All Project Dependencies"):
        st.code("""
# Core dependencies
pip install streamlit pandas plotly python-dotenv

# Voice dependencies
pip install audio-recorder-streamlit SpeechRecognition gTTS

# Agent dependencies
pip install yfinance requests beautifulsoup4 feedparser
pip install google-generativeai
pip install sentence-transformers faiss-cpu
pip install langchain

# Optional
pip install pygame  # For standalone voice agent
        """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.title("📊 Financial Assistant")
        st.markdown("---")
        
        # Check for data files
        api_files = list(AGENTS_DIR.glob('multi_region_results*.json'))
        sentiment_files = list(AGENTS_DIR.glob('regional_sentiment*.json'))
        analysis_files = list(AGENTS_DIR.glob('morning_brief*.json'))
        
        st.subheader("📂 Data Files")
        st.write("Market:", "✅" if api_files else "❌")
        st.write("Sentiment:", "✅" if sentiment_files else "❌")
        st.write("Analysis:", "✅" if analysis_files else "❌")
        
        st.markdown("---")
        
        # Voice status
        st.subheader("🎤 Voice Status")
        if AUDIO_RECORDER_AVAILABLE and VOICE_AVAILABLE:
            st.success("✅ Voice Enabled")
        else:
            st.error("❌ Voice Disabled")
            if not AUDIO_RECORDER_AVAILABLE:
                st.caption("Missing: audio-recorder-streamlit")
            if not VOICE_AVAILABLE:
                st.caption("Missing: SpeechRecognition, gTTS")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Market Data", "💭 Sentiment Analysis", 
             "📈 Portfolio Analysis", "🤖 AI Assistant", "🎤 Voice Assistant", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Info
        st.caption("Financial Assistant v2.0")
        st.caption("Multi-Agent Portfolio System")
    
    # Main content - Route to correct page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Market Data":
        show_market_data()
    elif page == "💭 Sentiment Analysis":
        show_sentiment_analysis()
    elif page == "📈 Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "🤖 AI Assistant":
        show_ai_assistant()
    elif page == "🎤 Voice Assistant":
        show_voice_assistant()
    elif page == "⚙️ Settings":
        show_settings()

if __name__ == "__main__":
    main()