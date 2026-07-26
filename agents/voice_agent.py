"""
Voice Agent - FIXED VERSION with Better Query Processing
Fixes: Speech recognition errors, query validation, better fallback handling
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple
import tempfile
import re

# Speech Recognition & Text-to-Speech
import speech_recognition as sr
from gtts import gTTS
import pygame

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import language agent for processing
try:
    from language_agent import EnhancedLanguageAgent
    from retriever_agent import FinancialRetrieverAgent
except ImportError:
    print("⚠️  Language agent not found. Voice agent will work in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize pygame for audio playback
pygame.mixer.init()

# ============================================================================
# ENHANCED VOICE AGENT CLASS (FIXED)
# ============================================================================

class EnhancedVoiceAgent:
    """
    FIXED Voice-enabled portfolio assistant with better query processing
    """
    
    def __init__(self, use_language_agent: bool = True):
        """Initialize Enhanced Voice Agent"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust recognizer settings for better accuracy
        self.recognizer.energy_threshold = 4000  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Seconds of silence to consider end of phrase
        
        # Calibrate for ambient noise
        print("🎤 Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        
        # Language agent integration
        self.use_language_agent = use_language_agent
        self.language_agent = None
        self.retriever = None
        
        if use_language_agent:
            try:
                print("🤖 Loading Language Agent...")
                self.language_agent = EnhancedLanguageAgent(max_requests=2)
                
                print("📚 Loading Retriever Agent...")
                self.retriever = FinancialRetrieverAgent()
                if self.retriever.load_index():
                    print("✅ Vector index loaded")
                else:
                    print("⚠️  No vector index found")
                    self.retriever = None
                    
            except Exception as e:
                logger.warning(f"Could not load agents: {e}")
                self.use_language_agent = False
        
        # Load all agent data
        self.all_data = self._load_all_data() if self.use_language_agent else {}
        
        # Response history
        self.conversation_history = []
        
        # Portfolio keywords for validation
        self.portfolio_keywords = {
            'stock', 'stocks', 'portfolio', 'market', 'invest', 'investment',
            'trading', 'buy', 'sell', 'price', 'earnings', 'sentiment',
            'region', 'asia', 'tech', 'technology', 'performance', 'analysis',
            'risk', 'return', 'allocation', 'holding', 'holdings', 'brief',
            'summary', 'overview', 'update', 'morning', 'report', 'tsmc',
            'samsung', 'alibaba', 'tencent', 'infosys', 'tcs', 'wipro'
        }
        
        logger.info("✅ Enhanced Voice Agent initialized")
    
    def _load_all_data(self) -> Dict:
        """Load data from all agents"""
        if self.language_agent:
            return self.language_agent.load_all_agent_data()
        return {}
    
    # ========================================================================
    # IMPROVED SPEECH-TO-TEXT
    # ========================================================================
    
    def listen(self, timeout: int = 10, phrase_time_limit: int = 15) -> Optional[str]:
        """
        Listen to microphone and convert speech to text with improved accuracy
        """
        print("\n🎤 Listening... (Speak clearly)")
        
        try:
            with self.microphone as source:
                # Adjust for current noise level
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🔄 Processing speech...")
            
            # Try Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Recognized: \"{text}\"")
                
                # Validate recognition quality
                confidence, is_valid = self._validate_recognition(text)
                
                if not is_valid:
                    print(f"⚠️  Low confidence recognition ({confidence:.0%})")
                    print("💡 Tip: Try speaking more clearly or rephrasing")
                    
                    confirm = input("\nIs this correct? (y/n): ")
                    if confirm.lower() != 'y':
                        print("Let's try again...")
                        return None
                
                return text
                
            except sr.UnknownValueError:
                print("❌ Could not understand audio")
                print("💡 Tips:")
                print("   - Speak more clearly")
                print("   - Reduce background noise")
                print("   - Move closer to microphone")
                return None
                
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                return None
                    
        except sr.WaitTimeoutError:
            print("⏱️  No speech detected (timeout)")
            return None
            
        except Exception as e:
            logger.error(f"Error listening: {e}")
            return None
    
    def _validate_recognition(self, text: str) -> Tuple[float, bool]:
        """
        Validate recognition quality based on keywords
        
        Returns:
            (confidence_score, is_valid)
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Check if query contains portfolio-related keywords
        matches = words.intersection(self.portfolio_keywords)
        
        if matches:
            confidence = min(1.0, len(matches) / 3)  # Higher confidence with more keywords
            return confidence, True
        
        # Check for common misrecognitions
        suspicious_words = {'desktop', 'laptop', 'computer', 'device'}
        if words.intersection(suspicious_words):
            return 0.3, False  # Low confidence
        
        # Unknown query - could be valid
        return 0.5, True
    
    # ========================================================================
    # IMPROVED TEXT-TO-SPEECH
    # ========================================================================
    
    def speak(self, text: str, lang: str = 'en', slow: bool = False):
        """Convert text to speech and play it"""
        print(f"\n🔊 Speaking: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        
        try:
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)
            
            # Play audio using pygame
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Cleanup
            pygame.mixer.music.unload()
            os.remove(temp_file)
            
            print("✅ Speech completed")
            
        except Exception as e:
            logger.error(f"Error speaking: {e}")
            print(f"❌ Could not generate speech: {e}")
    
    # ========================================================================
    # IMPROVED QUERY PROCESSING
    # ========================================================================
    
    def process_query(self, query: str) -> str:
        """
        Process query with better logic and fallback handling
        """
        logger.info(f"Processing query: '{query}'")
        
        query_lower = query.lower()
        
        # 1. Exit commands (highest priority)
        if any(word in query_lower for word in ['bye', 'goodbye', 'exit', 'quit', 'stop']):
            return "Goodbye! Have a great trading day!"
        
        # 2. Greeting (only if clearly a greeting)
        greeting_phrases = ['hello', 'hi there', 'hey there', 'good morning', 'good afternoon']
        if any(phrase in query_lower for phrase in greeting_phrases):
            return "Hello! I'm your portfolio voice assistant. Ask me about your stocks, market performance, or portfolio analysis."
        
        # 3. Check if query is portfolio-related
        words = set(re.findall(r'\b\w+\b', query_lower))
        portfolio_related = bool(words.intersection(self.portfolio_keywords))
        
        if not portfolio_related:
            return (
                "I'm not sure I understood that correctly. "
                "I can help you with:\n"
                "- Portfolio overview and performance\n"
                "- Stock analysis and recommendations\n"
                "- Market sentiment and trends\n"
                "- Regional performance comparison\n"
                "- Risk assessment\n"
                "Could you please rephrase your question?"
            )
        
        # 4. Process with Language Agent
        if self.use_language_agent and self.language_agent:
            try:
                # Morning brief keywords
                brief_keywords = ['brief', 'summary', 'overview', 'morning', 'update', 'report']
                
                if any(keyword in query_lower for keyword in brief_keywords):
                    print("📊 Generating comprehensive brief...")
                    result = self.language_agent.generate_comprehensive_brief(self.all_data)
                    
                    if result['success']:
                        return result['brief_text']
                    else:
                        return f"I encountered an error generating the brief: {result.get('error', 'Unknown error')}"
                
                # Specific queries (use RAG)
                else:
                    print("🔍 Searching portfolio data...")
                    result = self.language_agent.answer_with_rag(
                        query, 
                        self.retriever, 
                        self.all_data
                    )
                    
                    if result['success']:
                        return result['answer']
                    else:
                        return f"I couldn't find an answer: {result.get('error', 'Unknown error')}"
                        
            except Exception as e:
                logger.error(f"Error processing with Language Agent: {e}")
                return f"I encountered an error: {str(e)}"
        
        # 5. Fallback
        else:
            return (
                "I'm running in standalone mode. "
                "Please ensure the Language Agent is properly configured for full functionality."
            )
    
    # ========================================================================
    # CONVERSATION LOOP
    # ========================================================================
    
    def voice_conversation_loop(self):
        """Main conversation loop with improved error handling"""
        print("\n" + "="*70)
        print("🎙️  VOICE AGENT - INTERACTIVE MODE")
        print("="*70)
        print("\nVoice Assistant Ready!")
        print("- Ask about portfolio, stocks, or market analysis")
        print("- Say 'goodbye' or 'exit' to quit")
        print("\n💡 Tips for best results:")
        print("   - Speak clearly and at normal pace")
        print("   - Use portfolio-related keywords")
        print("   - Reduce background noise")
        print("\n" + "="*70 + "\n")
        
        consecutive_failures = 0
        max_failures = 3
        
        while True:
            try:
                # Listen for input
                query = self.listen(timeout=30)
                
                if not query:
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        print(f"\n⚠️  Multiple recognition failures ({max_failures})")
                        print("Would you like to:")
                        print("1. Continue with voice")
                        print("2. Switch to text input")
                        print("3. Exit")
                        
                        choice = input("\nChoice (1/2/3): ").strip()
                        
                        if choice == '2':
                            self._text_input_mode()
                            consecutive_failures = 0
                            continue
                        elif choice == '3':
                            self.speak("Goodbye! Have a great trading day!")
                            break
                        else:
                            consecutive_failures = 0
                            continue
                    else:
                        print("⚠️  Please try again.\n")
                        continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Check for exit
                if any(word in query.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
                    response = "Goodbye! Have a great trading day!"
                    self.speak(response)
                    break
                
                # Process query
                response = self.process_query(query)
                
                # Save to history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'response': response
                })
                
                # Speak response
                self.speak(response)
                
                print("\n" + "-"*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                self.speak("Session terminated. Goodbye!")
                break
                
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                print(f"❌ Error: {e}\n")
    
    def _text_input_mode(self):
        """Fallback text input mode"""
        print("\n📝 Switched to text input mode")
        print("Type your query (or 'voice' to switch back, 'quit' to exit):\n")
        
        while True:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'voice':
                print("\n🎤 Switching back to voice mode...\n")
                break
            
            if query.lower() in ['quit', 'exit', 'goodbye']:
                self.speak("Goodbye! Have a great trading day!")
                return 'exit'
            
            # Process query
            response = self.process_query(query)
            
            # Save to history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'input_mode': 'text'
            })
            
            print(f"\nAssistant: {response}\n")
            
            # Ask if user wants to hear it
            hear = input("Speak this response? (y/n): ").strip().lower()
            if hear == 'y':
                self.speak(response)
            
            print()
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def save_conversation_history(self, filename: Optional[str] = None):
        """Save conversation history to file"""
        if not self.conversation_history:
            print("No conversation history to save")
            return
        
        if filename is None:
            filename = f"voice_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_interactions': len(self.conversation_history),
                    'conversations': self.conversation_history
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Conversation history saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
    
    def test_microphone(self) -> bool:
        """Test if microphone is working"""
        print("\n🎤 Testing microphone...")
        print("Say: 'What is my portfolio performance?'")
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Microphone working! Detected: '{text}'")
                
                confidence, is_valid = self._validate_recognition(text)
                print(f"   Recognition confidence: {confidence:.0%}")
                print(f"   Portfolio-related: {'Yes' if is_valid else 'No'}")
                
                return True
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False
    
    def test_speaker(self) -> bool:
        """Test if speaker/audio output is working"""
        print("\n🔊 Testing speaker...")
        try:
            self.speak("Audio output test. Can you hear me clearly?")
            return True
        except Exception as e:
            print(f"❌ Speaker test failed: {e}")
            return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎙️  ENHANCED VOICE AGENT - PORTFOLIO ASSISTANT (FIXED)")
    print("Improved Speech Recognition & Query Processing")
    print("="*70 + "\n")
    
    # Initialize Enhanced Voice Agent
    try:
        agent = EnhancedVoiceAgent(use_language_agent=True)
    except Exception as e:
        print(f"❌ Failed to initialize Voice Agent: {e}")
        return
    
    # Test audio systems
    print("\n📋 Running Audio System Tests...")
    print("-" * 70)
    
    mic_ok = agent.test_microphone()
    speaker_ok = agent.test_speaker()
    
    if not mic_ok or not speaker_ok:
        print("\n⚠️  Audio system issues detected.")
        proceed = input("\nContinue anyway? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    print("\n" + "="*70)
    print("🚀 STARTING VOICE CONVERSATION")
    print("="*70)
    
    # Start conversation loop
    try:
        agent.voice_conversation_loop()
        
        # Save conversation history
        if agent.conversation_history:
            agent.save_conversation_history()
        
        print("\n✅ Voice Agent session completed!")
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()