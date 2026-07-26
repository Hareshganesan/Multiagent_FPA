"""
Orchestrator Agent - FULLY FIXED VERSION
Correctly handles file paths across all workflows
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import time
from enum import Enum

# Configure paths
ORCHESTRATOR_DIR = Path(__file__).parent
PROJECT_ROOT = ORCHESTRATOR_DIR.parent
AGENTS_DIR = PROJECT_ROOT / 'agents'

# Add agents directory to Python path
sys.path.insert(0, str(AGENTS_DIR))

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# WORKFLOW DEFINITIONS
# ============================================================================

class WorkflowType(Enum):
    """Available workflow types"""
    FULL_PIPELINE = "full_pipeline"
    MORNING_BRIEF = "morning_brief"
    VOICE_QUERY = "voice_query"
    QUICK_ANALYSIS = "quick_analysis"
    SENTIMENT_CHECK = "sentiment_check"

# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class FinancialAssistantOrchestrator:
    """Master orchestrator for all financial assistant agents"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.project_root = PROJECT_ROOT
        self.agents_dir = AGENTS_DIR
        self.execution_history = []
        
        # Agent availability
        self.agents_available = {
            'api_agent': False,
            'scraping_agent': False,
            'retriever_agent': False,
            'analysis_agent': False,
            'language_agent': False,
            'voice_agent': False
        }
        
        # Check availability
        self._check_agent_availability()
        
        logger.info(f"✅ Orchestrator initialized")
    
    def _check_agent_availability(self):
        """Check which agents are available"""
        print("\n🔍 Checking agent availability...")
        print(f"Agents directory: {self.agents_dir}")
        print("-" * 70)
        
        # Change to agents directory for imports
        original_cwd = os.getcwd()
        os.chdir(self.agents_dir)
        
        try:
            # API Agent
            try:
                import api_agent
                self.agents_available['api_agent'] = True
                print(f"✅ API Agent: Available ({api_agent.__file__})")
            except ImportError as e:
                print(f"❌ API Agent: Not found - {e}")
            
            # Scraping Agent
            try:
                import scraping_agent
                self.agents_available['scraping_agent'] = True
                print(f"✅ Scraping Agent: Available ({scraping_agent.__file__})")
            except ImportError as e:
                print(f"❌ Scraping Agent: Not found - {e}")
            
            # Retriever Agent
            try:
                from retriever_agent import FinancialRetrieverAgent
                self.agents_available['retriever_agent'] = True
                print(f"✅ Retriever Agent: Available")
            except ImportError as e:
                print(f"❌ Retriever Agent: Not found - {e}")
            
            # Analysis Agent
            try:
                from analysis_agent import FinancialAnalysisAgent
                self.agents_available['analysis_agent'] = True
                print(f"✅ Analysis Agent: Available")
            except ImportError as e:
                print(f"❌ Analysis Agent: Not found - {e}")
            
            # Language Agent
            try:
                from language_agent import EnhancedLanguageAgent
                self.agents_available['language_agent'] = True
                print(f"✅ Language Agent: Available")
            except ImportError as e:
                print(f"❌ Language Agent: Not found - {e}")
            
            # Voice Agent
            try:
                import voice_agent
                self.agents_available['voice_agent'] = True
                print(f"✅ Voice Agent: Available")
            except ImportError as e:
                print(f"❌ Voice Agent: Not found - {e}")
        
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
        print("-" * 70)
        available_count = sum(self.agents_available.values())
        print(f"\n📊 Total Agents Available: {available_count}/6\n")
    
    # ========================================================================
    # WORKFLOW EXECUTION
    # ========================================================================
    
    def execute_workflow(self, workflow_type: WorkflowType, 
                        params: Optional[Dict] = None) -> Dict:
        """Execute a predefined workflow"""
        logger.info(f"Executing workflow: {workflow_type.value}")
        
        workflow_start = time.time()
        
        execution_record = {
            'workflow_type': workflow_type.value,
            'start_time': datetime.now().isoformat(),
            'params': params or {},
            'results': {},
            'status': 'running'
        }
        
        try:
            if workflow_type == WorkflowType.FULL_PIPELINE:
                result = self._execute_full_pipeline(params)
            elif workflow_type == WorkflowType.MORNING_BRIEF:
                result = self._execute_morning_brief(params)
            elif workflow_type == WorkflowType.VOICE_QUERY:
                result = self._execute_voice_query(params)
            elif workflow_type == WorkflowType.QUICK_ANALYSIS:
                result = self._execute_quick_analysis(params)
            elif workflow_type == WorkflowType.SENTIMENT_CHECK:
                result = self._execute_sentiment_check(params)
            else:
                result = {'error': f'Unknown workflow type: {workflow_type}'}
            
            execution_record['results'] = result
            execution_record['status'] = 'success' if 'error' not in result else 'failed'
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution_record['status'] = 'failed'
            result = {'error': str(e)}
        
        workflow_end = time.time()
        execution_record['end_time'] = datetime.now().isoformat()
        execution_record['duration_seconds'] = round(workflow_end - workflow_start, 2)
        
        self.execution_history.append(execution_record)
        return result
    
    # ========================================================================
    # WORKFLOW IMPLEMENTATIONS
    # ========================================================================
    
    def _execute_full_pipeline(self, params: Optional[Dict]) -> Dict:
        """Execute complete pipeline"""
        print("\n" + "="*80)
        print("🚀 FULL PIPELINE EXECUTION")
        print("="*80 + "\n")
        
        results = {'pipeline': 'full', 'steps_completed': 0, 'total_steps': 5, 'step_results': {}}
        
        # Change to agents directory
        original_dir = os.getcwd()
        os.chdir(self.agents_dir)
        
        try:
            # Step 1: API Agent
            print("📊 Step 1/5: Fetching market data...")
            if self.agents_available['api_agent']:
                try:
                    import api_agent
                    api_results = api_agent.analyze_regional_performance(force_fresh=True)
                    
                    output_file = f"multi_region_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(output_file, 'w') as f:
                        json.dump(api_results, f, indent=2, default=str)
                    
                    results['step_results']['api_agent'] = {'status': 'success', 'output_file': output_file}
                    results['steps_completed'] += 1
                    print(f"✅ Market data fetched: {output_file}\n")
                except Exception as e:
                    results['step_results']['api_agent'] = {'status': 'failed', 'error': str(e)}
            else:
                results['step_results']['api_agent'] = {'status': 'skipped'}
            
            # Step 2: Scraping Agent
            print("🔍 Step 2/5: Analyzing sentiment...")
            if self.agents_available['scraping_agent']:
                try:
                    from scraping_agent import EnhancedMultiRegionScrapingAgent
                    
                    scraper = EnhancedMultiRegionScrapingAgent(use_firecrawl=True)
                    regions = params.get('regions', ['East Asia', 'South Asia', 'Southeast Asia']) if params else ['East Asia', 'South Asia', 'Southeast Asia']
                    
                    all_sentiment = {'timestamp': datetime.now().isoformat(), 'regions': {}}
                    
                    for region in regions:
                        print(f"  Analyzing {region}...")
                        sentiment = scraper.analyze_regional_sentiment(region)
                        all_sentiment['regions'][region] = sentiment
                        time.sleep(2)
                    
                    output_file = f"regional_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    scraper.save_results(all_sentiment, output_file)
                    
                    results['step_results']['scraping_agent'] = {'status': 'success', 'output_file': output_file}
                    results['steps_completed'] += 1
                    print(f"✅ Sentiment analysis complete: {output_file}\n")
                except Exception as e:
                    results['step_results']['scraping_agent'] = {'status': 'failed', 'error': str(e)}
            else:
                results['step_results']['scraping_agent'] = {'status': 'skipped'}
            
            # Step 3: Retriever Agent
            print("📚 Step 3/5: Building vector index...")
            if self.agents_available['retriever_agent']:
                try:
                    from retriever_agent import FinancialRetrieverAgent
                    
                    retriever = FinancialRetrieverAgent()
                    
                    api_file = self._find_latest_file('multi_region_results_*.json')
                    sentiment_file = self._find_latest_file('regional_sentiment_*.json')
                    
                    if api_file or sentiment_file:
                        stats = retriever.build_index_from_files(api_file or '', sentiment_file or '')
                        retriever.save_index()
                        
                        results['step_results']['retriever_agent'] = {'status': 'success', 'total_chunks': stats['total_chunks']}
                        results['steps_completed'] += 1
                        print(f"✅ Vector index built: {stats['total_chunks']} chunks\n")
                    else:
                        results['step_results']['retriever_agent'] = {'status': 'skipped', 'reason': 'no_data'}
                except Exception as e:
                    results['step_results']['retriever_agent'] = {'status': 'failed', 'error': str(e)}
            else:
                results['step_results']['retriever_agent'] = {'status': 'skipped'}
            
            # Step 4: Analysis Agent
            print("📈 Step 4/5: Running quantitative analysis...")
            if self.agents_available['analysis_agent']:
                try:
                    from analysis_agent import FinancialAnalysisAgent
                    from retriever_agent import FinancialRetrieverAgent
                    
                    retriever = None
                    if (self.agents_dir / 'vector_store' / 'faiss_index.bin').exists():
                        retriever = FinancialRetrieverAgent()
                        retriever.load_index()
                    
                    analyst = FinancialAnalysisAgent(retriever)
                    brief = analyst.generate_morning_brief()
                    
                    output_file = f"morning_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    analyst.save_brief(brief, output_file)
                    
                    results['step_results']['analysis_agent'] = {'status': 'success', 'output_file': output_file}
                    results['steps_completed'] += 1
                    print(f"✅ Analysis complete: {output_file}\n")
                except Exception as e:
                    results['step_results']['analysis_agent'] = {'status': 'failed', 'error': str(e)}
            else:
                results['step_results']['analysis_agent'] = {'status': 'skipped'}
            
            # Step 5: Language Agent
            print("🤖 Step 5/5: Generating language report...")
            if self.agents_available['language_agent']:
                try:
                    from language_agent import EnhancedLanguageAgent
                    
                    lang_agent = EnhancedLanguageAgent(max_requests=2)
                    all_data = lang_agent.load_all_agent_data()
                    
                    report = lang_agent.generate_comprehensive_brief(all_data)
                    
                    if report['success']:
                        output_file = f"language_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        lang_agent.save_response(report, output_file)
                        
                        results['step_results']['language_agent'] = {'status': 'success', 'output_file': output_file}
                        results['steps_completed'] += 1
                        print(f"✅ Language report generated: {output_file}\n")
                    else:
                        results['step_results']['language_agent'] = {'status': 'failed', 'error': report.get('error')}
                except Exception as e:
                    results['step_results']['language_agent'] = {'status': 'failed', 'error': str(e)}
            else:
                results['step_results']['language_agent'] = {'status': 'skipped'}
        
        finally:
            os.chdir(original_dir)
        
        print("="*80)
        print(f"✅ PIPELINE COMPLETE: {results['steps_completed']}/{results['total_steps']} steps")
        print("="*80 + "\n")
        
        return results
    
    def _execute_morning_brief(self, params: Optional[Dict]) -> Dict:
        """Generate morning brief"""
        print("\n🌅 Generating Morning Brief...\n")
        
        # CRITICAL FIX: Look in agents directory for files
        api_file = self._find_latest_file('multi_region_results_*.json', max_age_hours=24)
        sentiment_file = self._find_latest_file('regional_sentiment_*.json', max_age_hours=24)
        
        if not api_file or not sentiment_file:
            print("📊 No recent data found. Running full pipeline first...\n")
            self._execute_full_pipeline(params)
            # After pipeline, look again
            api_file = self._find_latest_file('multi_region_results_*.json')
            sentiment_file = self._find_latest_file('regional_sentiment_*.json')
        
        print("✅ Using recent data\n")
        
        if self.agents_available['language_agent']:
            # Change to agents directory
            original_dir = os.getcwd()
            os.chdir(self.agents_dir)
            
            try:
                from language_agent import EnhancedLanguageAgent
                
                lang_agent = EnhancedLanguageAgent(max_requests=2)
                all_data = lang_agent.load_all_agent_data()
                
                report = lang_agent.generate_comprehensive_brief(all_data)
                
                if report['success']:
                    print("\n" + "="*80)
                    print("🌅 MORNING BRIEF")
                    print("="*80 + "\n")
                    print(report['brief_text'])
                    print("\n" + "="*80 + "\n")
                    
                    return {'brief': report['brief_text'], 'status': 'success'}
                else:
                    return {'error': report.get('error'), 'status': 'failed'}
            finally:
                os.chdir(original_dir)
        else:
            return {'error': 'Language agent not available', 'status': 'failed'}
    
    def _execute_voice_query(self, params: Optional[Dict]) -> Dict:
        """Execute voice query"""
        print("\n🎤 Voice Query Mode...\n")
        
        if not self.agents_available['voice_agent']:
            return {'error': 'Voice agent not available'}
        
        # Change to agents directory
        original_dir = os.getcwd()
        os.chdir(self.agents_dir)
        
        try:
            from voice_agent import EnhancedVoiceAgent
            
            agent = EnhancedVoiceAgent(use_language_agent=True)
            query = params.get('query') if params else None
            
            if not query:
                query = agent.listen()
            
            if query:
                response = agent.process_query(query)
                agent.speak(response)
                return {'query': query, 'response': response, 'status': 'success'}
            else:
                return {'error': 'No query received', 'status': 'failed'}
        finally:
            os.chdir(original_dir)
    
    def _execute_quick_analysis(self, params: Optional[Dict]) -> Dict:
        """Quick analysis"""
        print("\n⚡ Quick Analysis Mode...\n")
        
        results = {'analysis_type': 'quick', 'steps': {}}
        
        original_dir = os.getcwd()
        os.chdir(self.agents_dir)
        
        try:
            if self.agents_available['api_agent']:
                import api_agent
                api_results = api_agent.analyze_regional_performance(force_fresh=True)
                
                output_file = f"quick_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(api_results, f, indent=2, default=str)
                
                results['steps']['api'] = {'status': 'success', 'file': output_file}
            
            if self.agents_available['analysis_agent']:
                from analysis_agent import FinancialAnalysisAgent
                
                analyst = FinancialAnalysisAgent()
                brief = analyst.generate_morning_brief()
                
                output_file = f"quick_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                analyst.save_brief(brief, output_file)
                
                results['steps']['analysis'] = {'status': 'success', 'file': output_file}
                analyst.print_morning_brief(brief)
        finally:
            os.chdir(original_dir)
        
        return results
    
    def _execute_sentiment_check(self, params: Optional[Dict]) -> Dict:
        """Sentiment check"""
        print("\n💭 Sentiment Check Mode...\n")
        
        if not self.agents_available['scraping_agent']:
            return {'error': 'Scraping agent not available'}
        
        original_dir = os.getcwd()
        os.chdir(self.agents_dir)
        
        try:
            from scraping_agent import EnhancedMultiRegionScrapingAgent
            
            scraper = EnhancedMultiRegionScrapingAgent(use_firecrawl=True)
            
            symbol = params.get('symbol') if params else None
            region = params.get('region') if params else None
            
            if symbol:
                result = scraper.analyze_single_stock(symbol, max_articles=5)
            elif region:
                result = scraper.analyze_regional_sentiment(region)
            else:
                regions = ['East Asia', 'South Asia', 'Southeast Asia']
                result = {'regions': {}}
                for r in regions:
                    result['regions'][r] = scraper.analyze_regional_sentiment(r)
                    time.sleep(2)
            
            return result
        finally:
            os.chdir(original_dir)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _find_latest_file(self, pattern: str, max_age_hours: Optional[int] = None) -> Optional[str]:
        """Find latest file in agents directory"""
        files = sorted(self.agents_dir.glob(pattern))
        
        if not files:
            return None
        
        latest = files[-1]
        
        if max_age_hours:
            file_time = datetime.fromtimestamp(latest.stat().st_mtime)
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                return None
        
        return str(latest)
    
    def get_execution_summary(self) -> Dict:
        """Get execution summary"""
        return {
            'total_executions': len(self.execution_history),
            'successful': sum(1 for e in self.execution_history if e['status'] == 'success'),
            'failed': sum(1 for e in self.execution_history if e['status'] == 'failed'),
            'recent_executions': self.execution_history[-5:],
            'agents_available': self.agents_available
        }
    
    def save_execution_log(self, filename: Optional[str] = None):
        """Save execution log"""
        if filename is None:
            filename = f"orchestrator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save in orchestrator directory
        log_path = self.project_root / 'orchestrator' / filename
        
        try:
            with open(log_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'summary': self.get_execution_summary(),
                    'execution_history': self.execution_history
                }, f, indent=2, default=str)
            
            print(f"✅ Execution log saved: {log_path}")
        except Exception as e:
            logger.error(f"Failed to save log: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🎯 FINANCIAL ASSISTANT ORCHESTRATOR (FULLY FIXED)")
    print("Multi-Agent Workflow Manager")
    print("="*80)
    
    orchestrator = FinancialAssistantOrchestrator()
    
    while True:
        print("\n" + "="*80)
        print("📋 AVAILABLE WORKFLOWS")
        print("="*80)
        print("\n1. 🚀 Full Pipeline (All Agents)")
        print("2. 🌅 Morning Brief (Quick)")
        print("3. 🎤 Voice Query")
        print("4. ⚡ Quick Analysis")
        print("5. 💭 Sentiment Check")
        print("6. 📊 Show Execution Summary")
        print("7. 💾 Save Execution Log")
        print("8. 🚪 Exit")
        
        choice = input("\nSelect workflow (1-8): ").strip()
        
        try:
            if choice == '1':
                regions = input("\nRegions (comma-separated, or Enter for default): ").strip()
                params = {'regions': [r.strip() for r in regions.split(',')]} if regions else None
                orchestrator.execute_workflow(WorkflowType.FULL_PIPELINE, params)
            
            elif choice == '2':
                orchestrator.execute_workflow(WorkflowType.MORNING_BRIEF)
            
            elif choice == '3':
                query = input("\nEnter query (or Enter for voice): ").strip()
                params = {'query': query} if query else None
                orchestrator.execute_workflow(WorkflowType.VOICE_QUERY, params)
            
            elif choice == '4':
                orchestrator.execute_workflow(WorkflowType.QUICK_ANALYSIS)
            
            elif choice == '5':
                print("\n1. Specific symbol")
                print("2. Specific region")
                print("3. All regions")
                
                opt = input("\nChoice (1-3): ").strip()
                params = {}
                
                if opt == '1':
                    symbol = input("Symbol (e.g., TSM): ").strip()
                    params['symbol'] = symbol
                elif opt == '2':
                    region = input("Region: ").strip()
                    params['region'] = region
                
                orchestrator.execute_workflow(WorkflowType.SENTIMENT_CHECK, params)
            
            elif choice == '6':
                summary = orchestrator.get_execution_summary()
                print("\n" + "="*80)
                print("📊 EXECUTION SUMMARY")
                print("="*80)
                print(f"\nTotal: {summary['total_executions']}")
                print(f"Success: {summary['successful']}")
                print(f"Failed: {summary['failed']}")
                print(f"\nAgents: {sum(summary['agents_available'].values())}/6")
            
            elif choice == '7':
                orchestrator.save_execution_log()
            
            elif choice == '8':
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()