"""
Main system orchestrator for the competitive trading agents system.
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import signal
import sys
import os

from src.base_agent import BaseTradingAgent
from src.trading_agents import (
    ConservativeTradingAgent, AggressiveTradingAgent, BalancedTradingAgent,
    FractalAnalysisAgent, CandleRangeTheoryAgent, QuantitativePatternAgent
)
from src.ai_enhanced_agent import AIEnhancedTradingAgent
from forex_crypto_agents import (
    ForexSpecialistAgent, CryptoSpecialistAgent, MultiAssetArbitrageAgent
)
from src.hierarchy_manager import HierarchyManager
from src.data_sources import DataAggregator, MarketDataProvider, NewsProvider, SocialMediaProvider
from src.alpaca_integration import AlpacaTradingInterface
from src.real_alpaca_integration import RealAlpacaTradingInterface
from src.alpaca_http_integration import RealAlpacaHTTPInterface
from src.persistence import SQLitePersistence
from src.backtesting_engine import RealTimeBacktestingEngine
from src.sentiment_analyzer import RealTimeSentimentAnalyzer
from src.advanced_risk_manager import AdvancedRiskManager
from src.ml_enhancement import MLEnhancementEngine
from src.multi_timeframe_analyzer import MultiTimeFrameAnalyzer
from src.correlation_analyzer import CorrelationAnalyzer
from src.market_hours_optimizer import MarketHoursOptimizer
from src.performance_analytics import PerformanceAnalytics
from config.settings import SystemConfig, AlpacaConfig, DataSourceConfig, HierarchyConfig, AgentConfig, AgentType

class TradingSystemOrchestrator:
    """
    Main orchestrator for the competitive trading agents system.
    """
    
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.logger = logging.getLogger("TradingSystemOrchestrator")
        
        # Initialize components
        self.agents: Dict[str, BaseTradingAgent] = {}
        self.hierarchy_manager: Optional[HierarchyManager] = None
        self.data_aggregator: Optional[DataAggregator] = None
        self.alpaca_interface: Optional[RealAlpacaTradingInterface] = None
        
        # Advanced intelligence systems
        self.backtesting_engine: Optional[RealTimeBacktestingEngine] = None
        self.sentiment_analyzer: Optional[RealTimeSentimentAnalyzer] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.ml_engine: Optional[MLEnhancementEngine] = None
        self.multi_timeframe_analyzer: Optional[MultiTimeFrameAnalyzer] = None
        self.correlation_analyzer: Optional[CorrelationAnalyzer] = None
        self.market_hours_optimizer: Optional[MarketHoursOptimizer] = None
        self.performance_analytics: Optional[PerformanceAnalytics] = None
        
        # System state
        self.is_running = False
        self.cycle_count = 0
        self.start_time = None
        self.last_data_update = None
        
        # Performance tracking
        self.system_performance = {
            "total_cycles": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_return": 0.0,
            "system_uptime": 0.0
        }

        # Global profitable trade counter for suspension lifts
        self.profitable_trade_counter = 0

        # Persistence layer
        self.persistence: Optional[SQLitePersistence] = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Allow external log directory via env
        log_dir = os.getenv('LOG_DIR', 'logs')
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        # Rotating file handler
        try:
            from logging.handlers import RotatingFileHandler
            max_bytes = int(os.getenv('LOG_MAX_BYTES', '10000000'))  # 10MB default
            backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, 'trading_system.log'),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        except Exception:
            file_handler = logging.FileHandler(os.path.join(log_dir, 'trading_system.log'))

        logging.basicConfig(
            level=getattr(logging, self.system_config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                file_handler,
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing trading system...")
            
            # Initialize data sources
            await self._initialize_data_sources()
            
            # Initialize Alpaca interface
            await self._initialize_alpaca_interface()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize hierarchy manager
            await self._initialize_hierarchy_manager()

            # Initialize persistence (SQLite)
            await self._initialize_persistence()
            if self.hierarchy_manager and self.persistence:
                self.hierarchy_manager.attach_persistence(self.persistence)
            
            # Initialize advanced intelligence systems
            await self._initialize_advanced_systems()
            
            self.logger.info("Trading system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trading system: {e}")
            return False
    
    async def _initialize_data_sources(self):
        """Initialize data source providers."""
        try:
            data_config = DataSourceConfig()
            
            market_provider = MarketDataProvider(data_config)
            news_provider = NewsProvider(data_config)
            social_provider = SocialMediaProvider(data_config)
            
            self.data_aggregator = DataAggregator(market_provider, news_provider, social_provider)
            
            self.logger.info("Data sources initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {e}")
            raise
    
    async def _initialize_alpaca_interface(self):
        """Initialize Alpaca trading interface."""
        try:
            alpaca_config = AlpacaConfig()

            # Try official SDK first
            self.alpaca_interface = RealAlpacaTradingInterface(alpaca_config, self.system_config)
            success = await self.alpaca_interface.initialize()
            if not success:
                # Try HTTP fallback for real paper trading
                self.logger.warning("Falling back to Alpaca HTTP client for paper trading...")
                self.alpaca_interface = RealAlpacaHTTPInterface(alpaca_config, self.system_config)
                success = await self.alpaca_interface.initialize()
                if not success:
                    # Finally, use internal DEMO mode
                    self.logger.warning("Using internal DEMO trading interface.")
                    self.alpaca_interface = AlpacaTradingInterface(alpaca_config, self.system_config)
                    success = await self.alpaca_interface.initialize()
                    if not success:
                        raise Exception("Failed to initialize any Alpaca interface (sdk/http/demo)")

            self.logger.info("Alpaca interface initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca interface: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize trading agents."""
        try:
            for agent_config in self.system_config.agent_configs:
                if agent_config.agent_type == AgentType.CONSERVATIVE:
                    agent = ConservativeTradingAgent(agent_config)
                elif agent_config.agent_type == AgentType.AGGRESSIVE:
                    agent = AggressiveTradingAgent(agent_config)
                elif agent_config.agent_type == AgentType.BALANCED:
                    agent = BalancedTradingAgent(agent_config)
                elif agent_config.agent_type == AgentType.FRACTAL_ANALYSIS:
                    agent = FractalAnalysisAgent(agent_config)
                elif agent_config.agent_type == AgentType.CANDLE_RANGE_THEORY:
                    agent = CandleRangeTheoryAgent(agent_config)
                elif agent_config.agent_type == AgentType.QUANTITATIVE_PATTERN:
                    agent = QuantitativePatternAgent(agent_config)
                elif agent_config.agent_type == AgentType.FOREX_SPECIALIST:
                    agent = ForexSpecialistAgent(agent_config)
                elif agent_config.agent_type == AgentType.CRYPTO_SPECIALIST:
                    agent = CryptoSpecialistAgent(agent_config)
                elif agent_config.agent_type == AgentType.MULTI_ASSET_ARBITRAGE:
                    agent = MultiAssetArbitrageAgent(agent_config)
                elif agent_config.agent_type == AgentType.AI_ENHANCED:
                    # Get Perplexity API key from environment
                    perplexity_api_key = os.getenv('PERPLEXITY_API_KEY', '')
                    if not perplexity_api_key:
                        self.logger.warning("PERPLEXITY_API_KEY not found, AI-enhanced agent will use fallback mode")
                    agent = AIEnhancedTradingAgent(agent_config, perplexity_api_key, self.logger)
                else:
                    self.logger.warning(f"Unknown agent type: {agent_config.agent_type}")
                    continue
                
                self.agents[agent.agent_id] = agent
                self.logger.info(f"Initialized agent: {agent.agent_id} ({agent_config.agent_type.value})")
            
            self.logger.info(f"Initialized {len(self.agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}")
            raise
    
    async def _initialize_hierarchy_manager(self):
        """Initialize hierarchy manager."""
        try:
            hierarchy_config = HierarchyConfig()
            
            self.hierarchy_manager = HierarchyManager(hierarchy_config, self.system_config)
            
            # Register all agents with hierarchy manager
            for agent in self.agents.values():
                self.hierarchy_manager.register_agent(agent)
            
            self.logger.info("Hierarchy manager initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing hierarchy manager: {e}")
            raise

    async def _initialize_persistence(self):
        """Initialize SQLite persistence layer."""
        try:
            db_url = self.system_config.database_url or "sqlite:///trading_agents.db"
            self.persistence = SQLitePersistence(db_url)
            self.logger.info(f"Persistence initialized at {self.persistence.db_path}")
        except Exception as e:
            self.logger.warning(f"Could not initialize persistence: {e}")
    
    async def _initialize_advanced_systems(self):
        """Initialize advanced intelligence systems."""
        try:
            # Initialize backtesting engine
            self.backtesting_engine = RealTimeBacktestingEngine(
                self.data_aggregator.market_provider, self.logger
            )
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = RealTimeSentimentAnalyzer(self.logger)
            
            # Initialize risk manager
            self.risk_manager = AdvancedRiskManager(self.logger, initial_capital=100000)
            
            # Initialize ML enhancement engine
            self.ml_engine = MLEnhancementEngine(self.logger)
            
            # Initialize multi-timeframe analyzer
            self.multi_timeframe_analyzer = MultiTimeFrameAnalyzer(self.logger)
            
            # Initialize correlation analyzer
            self.correlation_analyzer = CorrelationAnalyzer(self.logger)
            
            # Initialize market hours optimizer
            self.market_hours_optimizer = MarketHoursOptimizer(self.logger)
            
            # Initialize performance analytics
            self.performance_analytics = PerformanceAnalytics(self.logger)
            
            # Initialize historical data for backtesting using actual trading symbols
            symbols = self.system_config.trading_symbols  # Use all symbols
            if symbols:
                await self.backtesting_engine.initialize_historical_data(symbols, days_back=30)
            
            self.logger.info("Advanced intelligence systems initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing advanced systems: {e}")
            raise
    
    async def run_system(self):
        """Run the main system loop."""
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info("Starting trading system...")
            
            # Start background optimization and sentiment analysis tasks
            background_tasks = []
            
            if self.backtesting_engine:
                # Get symbols safely
                if self.system_config.agent_configs:
                    if isinstance(self.system_config.agent_configs, dict):
                        symbols = list(self.system_config.agent_configs.keys())
                    else:
                        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTCUSD', 'ETHUSD']
                else:
                    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTCUSD', 'ETHUSD']
                
                if symbols:
                    optimization_task = asyncio.create_task(
                        self.backtesting_engine.run_continuous_optimization(
                            list(self.agents.values()), symbols, optimization_interval=3600
                        )
                    )
                    background_tasks.append(optimization_task)
            
            if self.sentiment_analyzer:
                # Get symbols safely
                if self.system_config.agent_configs:
                    if isinstance(self.system_config.agent_configs, dict):
                        symbols = list(self.system_config.agent_configs.keys())
                    else:
                        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTCUSD', 'ETHUSD']
                else:
                    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTCUSD', 'ETHUSD']
                
                if symbols:
                    sentiment_task = asyncio.create_task(
                        self.sentiment_analyzer.run_continuous_sentiment_analysis(
                            symbols, analysis_interval=300
                        )
                    )
                    background_tasks.append(sentiment_task)
            
            # Main system loop
            while self.is_running:
                try:
                    await self._run_cycle()
                    await asyncio.sleep(self.system_config.data_update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in system cycle: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
            
            # Cancel background tasks
            for task in background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            self.logger.error(f"Fatal error in system: {e}")
        finally:
            await self.shutdown()
    
    async def _run_cycle(self):
        """Run one complete system cycle."""
        cycle_start = datetime.now()
        self.cycle_count += 1
        
        self.logger.info(f"Starting cycle {self.cycle_count}")
        
        try:
            # Step 1: Collect market data
            market_data = await self._collect_market_data()
            if not market_data:
                self.logger.warning("No market data collected, skipping cycle")
                return
            
            # Step 2: Run agent cycles
            agent_results = await self._run_agent_cycles(market_data)
            
            # Step 3: Execute trades
            trade_results = await self._execute_trades(agent_results)
            
            # Step 4: Run hierarchy oversight
            hierarchy_results = await self._run_hierarchy_oversight()
            
            # Step 5: Update system performance
            await self._update_system_performance(agent_results, trade_results)
            
            # Step 6: Log cycle results
            await self._log_cycle_results(agent_results, trade_results, hierarchy_results)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"Cycle {self.cycle_count} completed in {cycle_duration:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in cycle {self.cycle_count}: {e}")
    
    async def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """Collect comprehensive market data."""
        try:
            if not self.data_aggregator:
                return None
            
            market_data = await self.data_aggregator.get_comprehensive_data(
                self.system_config.trading_symbols
            )
            
            self.last_data_update = datetime.now()
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            return None
    
    async def _run_agent_cycles(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run cycles for all agents with advanced intelligence."""
        agent_results = {}
        
        # Get sentiment analysis for all symbols
        sentiment_data = {}
        if self.sentiment_analyzer:
            try:
                # Extract symbols from the comprehensive data structure
                symbols = market_data.get('symbols', [])
                if not symbols:
                    # Fallback to price_data keys if symbols not available
                    symbols = list(market_data.get('price_data', {}).keys())
                sentiment_summary = await self.sentiment_analyzer.get_market_sentiment_summary(symbols)
                sentiment_data = sentiment_summary.get('symbol_sentiments', {})
                self.logger.info(f"Sentiment analysis complete: {len(sentiment_data)} symbols analyzed")
            except Exception as e:
                self.logger.error(f"Error in sentiment analysis: {e}")
        
        # Run agent cycles in parallel, skipping suspended agents
        tasks = []
        for agent_id, agent in self.agents.items():
            if self.hierarchy_manager and agent_id in getattr(self.hierarchy_manager, 'suspended', {}):
                # Agent is in simple suspension; it reflects/learns but does not trade
                self.logger.info(f"Agent {agent_id} is suspended; skipping trading decision this cycle")
                # Trigger reflection/learning path without trading
                task = asyncio.create_task(self._run_suspended_agent_cycle(agent, market_data, sentiment_data))
            else:
                task = asyncio.create_task(self._run_enhanced_agent_cycle(agent, market_data, sentiment_data))
            tasks.append((agent_id, task))
        
        # DEBUG: Log market data structure
        self.logger.info(f"Market data keys: {list(market_data.keys())}")
        if 'price_data' in market_data:
            self.logger.info(f"Price data symbols: {list(market_data['price_data'].keys())}")
            for symbol, data in market_data['price_data'].items():
                if data:
                    self.logger.info(f"Symbol {symbol}: {len(data)} bars, latest price: {data[-1].get('close', 'N/A')}")
        
        # Wait for all agents to complete
        for agent_id, task in tasks:
            try:
                result = await task
                agent_results[agent_id] = result
            except Exception as e:
                self.logger.error(f"Error in agent {agent_id} cycle: {e}")
                agent_results[agent_id] = {"error": str(e)}
        
        return agent_results

    async def _run_enhanced_agent_cycle(self, agent: BaseTradingAgent, market_data: Dict[str, Any], 
                                      sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced agent cycle with sentiment analysis and risk management."""
        try:
            # Run backtesting optimization if available
            if self.backtesting_engine:
                try:
                    # Extract symbols from the comprehensive data structure
                    symbols = market_data.get('symbols', [])
                    if not symbols:
                        symbols = list(market_data.get('price_data', {}).keys())
                    optimization_result = await self.backtesting_engine.optimize_agent_strategy(agent, symbols)
                    if optimization_result:
                        self.logger.info(f"Optimization complete for {agent.agent_id}: Score {optimization_result.get('optimization_score', 0):.3f}")
                except Exception as e:
                    self.logger.error(f"Error in backtesting optimization for {agent.agent_id}: {e}")
            
            # Convert comprehensive data to agent-expected format
            # The agents expect price_data to be a dictionary where each symbol maps to a list of price bars
            agent_market_data = {
                'price_data': market_data.get('price_data', {}),
                'technical_indicators': market_data.get('technical_indicators', {}),
                'news_data': market_data.get('news_data', {}),
                'news_sentiment': market_data.get('news_sentiment', {}),
                'social_sentiment': market_data.get('social_sentiment', {}),
                'youtube_insights': market_data.get('youtube_insights', {}),
                'market_overview': market_data.get('market_overview', {})
            }
            
            # Run standard agent cycle
            result = await agent.run_cycle(agent_market_data)
            
            # DEBUG: Log agent results
            self.logger.info(f"Agent {agent.agent_id} cycle result: {result}")
            
            # Enhance decisions with sentiment and risk management
            if result and 'decisions' in result:
                enhanced_decisions = []
                for decision in result['decisions']:
                    if decision:
                        # Handle both dictionary and object formats
                        if isinstance(decision, dict):
                            symbol = decision.get('symbol')
                            action = decision.get('action')
                            quantity = decision.get('quantity')
                            confidence = decision.get('confidence', 0.5)
                        else:
                            symbol = getattr(decision, 'symbol', None)
                            action = getattr(decision, 'action', None)
                            quantity = getattr(decision, 'quantity', None)
                            confidence = getattr(decision, 'confidence', 0.5)
                        
                        if not symbol or not action or not quantity:
                            continue
                        
                        # Get sentiment for this symbol
                        symbol_sentiment = sentiment_data.get(symbol)
                        if symbol_sentiment:
                            # Adjust decision confidence based on sentiment
                            sentiment_confidence = symbol_sentiment.confidence
                            sentiment_impact = symbol_sentiment.overall_sentiment
                            
                            # Enhance decision with sentiment data
                            if isinstance(decision, dict):
                                decision['sentiment_confidence'] = sentiment_confidence
                                decision['sentiment_impact'] = sentiment_impact
                                
                                # Adjust quantity based on sentiment confidence
                                if sentiment_confidence > 0.7:
                                    decision['quantity'] *= (1 + sentiment_confidence * 0.1)  # 10% boost for high confidence
                            else:
                                decision.sentiment_confidence = sentiment_confidence
                                decision.sentiment_impact = sentiment_impact
                                
                                # Adjust quantity based on sentiment confidence
                                if sentiment_confidence > 0.7:
                                    decision.quantity *= (1 + sentiment_confidence * 0.1)  # 10% boost for high confidence
                        
                        # Validate trade with risk management
                        if self.risk_manager:
                            current_price = market_data.get(symbol, {}).get('current_price', 100)
                            if not current_price or current_price <= 0:
                                # Try to get price from price_data
                                price_data = market_data.get('price_data', {}).get(symbol, [])
                                if price_data and len(price_data) > 0:
                                    current_price = price_data[-1].get('close', 100)
                                else:
                                    current_price = 100  # Fallback price
                            
                            is_valid, reason = await self.risk_manager.validate_trade(
                                symbol, action, quantity, current_price, market_data
                            )
                            
                            if not is_valid:
                                self.logger.warning(f"🔍 DEBUG: Trade rejected for {agent.agent_id}: {reason}")
                                continue  # Skip this decision
                            else:
                                self.logger.info(f"🔍 DEBUG: Trade validated for {agent.agent_id}: {symbol} {action} {quantity}")
                            
                            # Get optimal position sizing
                            position_sizing = await self.risk_manager.calculate_optimal_position_size(
                                symbol, 
                                market_data.get(symbol, {}).get('current_price', 100),
                                market_data, 
                                confidence,
                                0.02  # Default expected return
                            )
                            
                            # Update decision with optimal sizing
                            if position_sizing and position_sizing.recommended_size > 0:
                                if isinstance(decision, dict):
                                    decision['quantity'] = min(quantity, position_sizing.recommended_size)
                                    decision['stop_loss'] = position_sizing.stop_loss
                                    decision['take_profit'] = position_sizing.take_profit
                                    decision['risk_reasoning'] = position_sizing.reasoning
                                else:
                                    decision.quantity = min(quantity, position_sizing.recommended_size)
                                    decision.stop_loss = position_sizing.stop_loss
                                    decision.take_profit = position_sizing.take_profit
                                    decision.risk_reasoning = position_sizing.reasoning
                        
                        enhanced_decisions.append(decision)
                
                result['decisions'] = enhanced_decisions
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced agent cycle for {agent.agent_id}: {e}")
            return {"error": str(e)}
    
    async def _run_suspended_agent_cycle(self, agent: BaseTradingAgent, market_data: Dict[str, Any], 
                                       sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a reduced cycle for suspended agents: analyze + reflect, no trade decisions."""
        result = {
            "agent_id": agent.agent_id,
            "timestamp": datetime.now(),
            "decisions": [],
            "reflections": [],
            "learning": []
        }
        try:
            # Encourage learning during suspension
            if agent.should_reflect():
                reflection = await agent.self_reflect()
                result["reflections"].append(reflection)
            
            # Run backtesting optimization even when suspended
            if self.backtesting_engine:
                try:
                    # Extract symbols from the comprehensive data structure
                    symbols = market_data.get('symbols', [])
                    if not symbols:
                        symbols = list(market_data.get('price_data', {}).keys())
                    optimization_result = await self.backtesting_engine.optimize_agent_strategy(agent, symbols)
                    if optimization_result:
                        result["learning"].append({
                            "type": "optimization",
                            "score": optimization_result.get('optimization_score', 0),
                            "timestamp": datetime.now()
                        })
                except Exception as e:
                    self.logger.error(f"Error in suspended agent optimization for {agent.agent_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Suspended cycle error for {agent.agent_id}: {e}")
        return result
    
    async def _execute_trades(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trades from agent decisions."""
        trade_results = {}
        
        if not self.alpaca_interface:
            self.logger.warning("Alpaca interface not available, skipping trade execution")
            return trade_results
        
        self.logger.info(f"🔍 DEBUG: Starting trade execution for {len(agent_results)} agents")
        total_decisions = sum(len(result.get("decisions", [])) for result in agent_results.values() if "error" not in result)
        self.logger.info(f"🔍 DEBUG: Total decisions to process: {total_decisions}")
        
        # DEBUG: Log each agent's results
        for agent_id, result in agent_results.items():
            if "error" not in result:
                decisions = result.get("decisions", [])
                self.logger.info(f"🔍 DEBUG: Agent {agent_id} returned {len(decisions)} decisions")
                for i, decision in enumerate(decisions):
                    self.logger.info(f"🔍 DEBUG: Decision {i+1}: {decision}")
            else:
                self.logger.warning(f"🔍 DEBUG: Agent {agent_id} had error: {result.get('error')}")
        
        for agent_id, result in agent_results.items():
            if "error" in result:
                continue
            
            decisions = result.get("decisions", [])
            agent_trade_results = []
            
            self.logger.info(f"🔍 DEBUG: Agent {agent_id} has {len(decisions)} decisions")
            
            for i, decision in enumerate(decisions):
                self.logger.info(f"🔍 DEBUG: Processing decision {i+1}/{len(decisions)} for {agent_id}: {decision}")
                try:
                    # Extract decision details (handle both dict and object formats)
                    if isinstance(decision, dict):
                        symbol = decision.get('symbol')
                        action = decision.get('action')
                        quantity = decision.get('quantity')
                        price = decision.get('price', 100)
                        confidence = decision.get('confidence', 0.5)
                        reasoning = decision.get('reasoning', '')
                    else:
                        symbol = getattr(decision, 'symbol', None)
                        action = getattr(decision, 'action', None)
                        quantity = getattr(decision, 'quantity', None)
                        price = getattr(decision, 'price', 100)
                        confidence = getattr(decision, 'confidence', 0.5)
                        reasoning = getattr(decision, 'reasoning', '')
                    
                    if not symbol or not action or not quantity:
                        self.logger.warning(f"Invalid decision format for {agent_id}: missing symbol/action/quantity")
                        continue
                    
                    # For paper trading, be more aggressive - only skip if explicitly configured to respect market hours
                    try:
                        if hasattr(self.alpaca_interface, 'is_market_open_for_symbol'):
                            is_market_open = await self.alpaca_interface.is_market_open_for_symbol(symbol)
                            
                            # In paper trading mode, be more lenient with market hours
                            if not is_market_open and self.system_config.trading_mode.value == "paper":
                                # Allow after-hours trading for paper mode
                                self.logger.info(f"Market closed for {symbol}, but allowing after-hours paper trading")
                            elif not is_market_open:
                                self.logger.info(f"Market closed for {symbol}, skipping")
                                continue
                    except Exception as e:
                        self.logger.debug(f"Market hours check failed: {e}, proceeding with trade")

                    # Execute trade through Alpaca
                    self.logger.info(f"🔍 DEBUG: Executing trade for {agent_id}: {symbol} {action} {quantity}")
                    try:
                        trade_result = await self.alpaca_interface.execute_trade_decision(decision)
                        self.logger.info(f"🔍 DEBUG: Trade result for {agent_id}: {trade_result}")
                    except Exception as e:
                        self.logger.error(f"🔍 DEBUG: Error executing trade for {agent_id}: {e}")
                        trade_result = None
                    
                    if trade_result:
                        agent_trade_results.append(trade_result)
                        
                        # Update risk management system
                        if self.risk_manager:
                            try:
                                await self.risk_manager.update_portfolio(symbol, action, quantity, price)
                            except Exception as e:
                                self.logger.error(f"Error updating risk management: {e}")
                        
                        # Update system performance
                        status = (trade_result.get("status") or "").lower()
                        if status in ("filled", "partially_filled", "accepted", "new", "submitted"):
                            self.system_performance["successful_trades"] += 1
                            self.profitable_trade_counter += 1
                            # Notify hierarchy for suspension lift progress
                            if self.hierarchy_manager:
                                # Treat any successful accepted trade as profitable for milestone progress
                                self.hierarchy_manager.record_trade_outcome(agent_id, 1.0, self.profitable_trade_counter)
                        else:
                            self.system_performance["failed_trades"] += 1
                            if self.hierarchy_manager:
                                self.hierarchy_manager.record_trade_outcome(agent_id, -1.0, self.profitable_trade_counter)
                    
                        # Persist trade
                        if self.persistence:
                            # normalize decision to dict
                            decision_dict = decision if isinstance(decision, dict) else {
                                "timestamp": getattr(decision, 'timestamp', datetime.now()),
                                "symbol": symbol,
                                "action": action,
                                "quantity": quantity,
                                "price": price,
                                "confidence": confidence,
                                "reasoning": reasoning,
                                "agent_id": agent_id,
                            }
                            self.persistence.log_trade(agent_id, decision_dict, trade_result)

                except Exception as e:
                    self.logger.error(f"Error executing trade for agent {agent_id}: {e}")
                    self.system_performance["failed_trades"] += 1
                    if self.hierarchy_manager:
                        self.hierarchy_manager.record_trade_outcome(agent_id, -1.0, self.profitable_trade_counter)
            
            trade_results[agent_id] = agent_trade_results
        
        return trade_results
    
    async def _run_hierarchy_oversight(self) -> Dict[str, Any]:
        """Run hierarchy manager oversight cycle."""
        try:
            if not self.hierarchy_manager:
                return {}
            
            hierarchy_results = await self.hierarchy_manager.run_oversight_cycle()
            return hierarchy_results
            
        except Exception as e:
            self.logger.error(f"Error in hierarchy oversight: {e}")
            return {"error": str(e)}
    
    async def _update_system_performance(self, agent_results: Dict[str, Any], trade_results: Dict[str, Any]):
        """Update system performance metrics."""
        try:
            self.system_performance["total_cycles"] = self.cycle_count
            
            # Calculate total return from all agents
            total_return = 0.0
            for agent in self.agents.values():
                metrics = agent.calculate_performance_metrics()
                total_return += metrics.total_return
                # Persist metrics
                if self.persistence:
                    self.persistence.log_agent_metrics(self.cycle_count, agent.agent_id, metrics.__dict__)
            
            self.system_performance["total_return"] = total_return
            
            # Calculate system uptime
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.system_performance["system_uptime"] = uptime
            
        except Exception as e:
            self.logger.error(f"Error updating system performance: {e}")
    
    async def _log_cycle_results(self, agent_results: Dict[str, Any], trade_results: Dict[str, Any], hierarchy_results: Dict[str, Any]):
        """Log cycle results for monitoring."""
        try:
            # Log summary
            self.logger.info(f"Cycle {self.cycle_count} Summary:")
            self.logger.info(f"  Agents: {len(agent_results)}")
            self.logger.info(f"  Trades executed: {sum(len(trades) for trades in trade_results.values())}")
            self.logger.info(f"  System return: {self.system_performance['total_return']:.4f}")
            
            # Log agent performance
            for agent_id, result in agent_results.items():
                if "error" not in result:
                    decisions = len(result.get("decisions", []))
                    reflections = len(result.get("reflections", []))
                    self.logger.info(f"  Agent {agent_id}: {decisions} decisions, {reflections} reflections")
            
            # Log hierarchy results
            if hierarchy_results and "evaluation" in hierarchy_results:
                evaluation = hierarchy_results["evaluation"]
                rankings = evaluation.get("rankings", {})
                if rankings:
                    best_agent = min(rankings.items(), key=lambda x: x[1])[0]
                    self.logger.info(f"  Best performing agent: {best_agent}")
            
        except Exception as e:
            self.logger.error(f"Error logging cycle results: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "is_running": self.is_running,
                "cycle_count": self.cycle_count,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "last_data_update": self.last_data_update.isoformat() if self.last_data_update else None,
                "system_performance": self.system_performance,
                "agents": {}
            }
            
            # Get agent statuses
            for agent_id, agent in self.agents.items():
                status["agents"][agent_id] = {
                    "agent_type": agent.agent_type.value,
                    "status": "active",
                    "performance_metrics": agent.calculate_performance_metrics().__dict__,
                    "trade_count": len(agent.trade_history)
                }
            
            # Get hierarchy status
            if self.hierarchy_manager:
                status["hierarchy"] = await self.hierarchy_manager.get_system_status()
            
            # Get portfolio status
            if self.alpaca_interface:
                status["portfolio"] = await self.alpaca_interface.get_portfolio_performance()
            
            # Get advanced systems status
            if self.backtesting_engine:
                status["backtesting"] = self.backtesting_engine.get_optimization_summary()
            
            if self.sentiment_analyzer:
                status["sentiment"] = {
                    "cache_size": len(self.sentiment_analyzer.get_sentiment_cache()),
                    "latest_analysis": datetime.now().isoformat()
                }
            
            if self.risk_manager:
                status["risk_management"] = await self.risk_manager.get_portfolio_risk_summary()
            
            if self.ml_engine:
                status["ml_models"] = self.ml_engine.get_model_status()
            
            if self.multi_timeframe_analyzer:
                status["multi_timeframe"] = self.multi_timeframe_analyzer.get_analysis_summary()
            
            if self.correlation_analyzer:
                status["correlation"] = self.correlation_analyzer.get_correlation_summary()
            
            if self.market_hours_optimizer:
                status["market_hours"] = self.market_hours_optimizer.get_session_performance()
            
            if self.performance_analytics:
                status["performance"] = self.performance_analytics.get_performance_summary()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        try:
            self.logger.info("Shutting down trading system...")
            
            self.is_running = False
            
            # Close all positions if in live trading mode
            if (self.system_config.trading_mode.value == "live" and 
                self.alpaca_interface):
                self.logger.info("Closing all positions...")
                await self.alpaca_interface.close_all_positions()
            
            # Log final performance
            self.logger.info(f"Final system performance: {self.system_performance}")
            
            self.logger.info("Trading system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point for the trading system."""
    # Load configuration
    system_config = SystemConfig()
    
    # Create and initialize orchestrator
    orchestrator = TradingSystemOrchestrator(system_config)
    
    # Initialize system
    success = await orchestrator.initialize()
    if not success:
        print("Failed to initialize trading system")
        return
    
    # Run system
    try:
        await orchestrator.run_system()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
