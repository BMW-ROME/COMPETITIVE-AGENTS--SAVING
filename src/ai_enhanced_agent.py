"""
AI-Enhanced Trading Agent with Perplexity Intelligence and Random Forest ML
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from src.base_agent import BaseTradingAgent, TradeDecision
from src.hybrid_ai_system import HybridAISystem
from src.free_intelligence_system import FreeIntelligenceSystem

class AIEnhancedTradingAgent(BaseTradingAgent):
    """AI-Enhanced Trading Agent with Perplexity Intelligence and Random Forest ML"""
    
    def __init__(self, config, perplexity_api_key: str, logger: logging.Logger):
        super().__init__(config)
        # Always use free system for now, with fallback to hybrid if API key is valid
        if perplexity_api_key and len(perplexity_api_key) > 10:
            try:
                self.hybrid_ai = HybridAISystem(perplexity_api_key, logger)
                self.free_intelligence = FreeIntelligenceSystem(logger)  # Fallback
                self.logger.info(f"AI-Enhanced agent {self.agent_id} initialized with Perplexity API")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Perplexity, using free system: {e}")
                self.hybrid_ai = None
                self.free_intelligence = FreeIntelligenceSystem(logger)
        else:
            self.hybrid_ai = None
            self.free_intelligence = FreeIntelligenceSystem(logger)
            self.logger.info(f"AI-Enhanced agent {self.agent_id} initialized with free intelligence system")
        
        self.ai_confidence_threshold = 0.6
        self.risk_tolerance = "medium"
        self.ai_insights_cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def __aenter__(self):
        if self.hybrid_ai:
            await self.hybrid_ai.__aenter__()
        else:
            await self.free_intelligence.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.hybrid_ai:
            await self.hybrid_ai.__aexit__(exc_type, exc_val, exc_tb)
        else:
            await self.free_intelligence.__aexit__(exc_type, exc_val, exc_tb)
    
    async def make_trading_decision(self, market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Make AI-enhanced trading decisions"""
        try:
            # Get symbols from market data
            symbols = self._extract_symbols(market_data)
            if not symbols:
                return None
            
            # Get AI-enhanced analysis
            if self.hybrid_ai:
                ai_analysis = await self._get_ai_analysis(symbols, market_data)
            else:
                ai_analysis = await self._get_free_ai_analysis(symbols, market_data)
            if not ai_analysis:
                return None
            
            # Find best trading opportunity
            best_opportunity = self._find_best_opportunity(ai_analysis, market_data)
            if not best_opportunity:
                return None
            
            # Create trading decision
            decision = await self._create_ai_decision(best_opportunity, market_data)
            
            if decision:
                self.logger.info(f"AI-Enhanced agent {self.agent_id} executing {decision.action} {decision.quantity} {decision.symbol}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in AI-enhanced trading decision: {e}")
            return None
    
    async def _get_free_ai_analysis(self, symbols: List[str], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get free AI analysis using free intelligence system"""
        try:
            # Get free market intelligence
            intelligence = await self.free_intelligence.get_market_intelligence(symbols)
            
            # Get market sentiment
            market_sentiment = await self.free_intelligence.get_market_sentiment()
            
            # Create analysis structure similar to hybrid AI
            symbol_analysis = {}
            for symbol in symbols:
                symbol_data = intelligence.get(symbol, {})
                symbol_analysis[symbol] = {
                    'combined_signal': (symbol_data.get('sentiment_score', 50) - 50) / 50,  # -1 to 1
                    'risk_level': 'low' if symbol_data.get('sentiment_score', 50) > 70 else 'high' if symbol_data.get('sentiment_score', 50) < 30 else 'medium',
                    'recommendation': {
                        'action': 'BUY' if symbol_data.get('sentiment_score', 50) > 60 else 'SELL' if symbol_data.get('sentiment_score', 50) < 40 else 'HOLD',
                        'confidence': abs(symbol_data.get('sentiment_score', 50) - 50) / 50,
                        'reasoning': f"Free AI analysis: {symbol_data.get('outlook', 'Neutral outlook')}"
                    },
                    'ai_confidence': abs(symbol_data.get('sentiment_score', 50) - 50) / 50
                }
            
            return {
                'symbol_analysis': symbol_analysis,
                'market_sentiment': market_sentiment,
                'ai_confidence': 0.7,  # Good confidence for free system
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting free AI analysis: {e}")
            return None
    
    def _extract_symbols(self, market_data: Dict[str, Any]) -> List[str]:
        """Extract trading symbols from market data"""
        try:
            symbols = []
            
            # Try different data structures
            if 'symbols' in market_data:
                symbols.extend(market_data['symbols'])
            elif 'price_data' in market_data:
                symbols.extend(market_data['price_data'].keys())
            elif 'market_data' in market_data:
                symbols.extend(market_data['market_data'].keys())
            
            return list(set(symbols))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error extracting symbols: {e}")
            return []
    
    async def _get_ai_analysis(self, symbols: List[str], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get AI-enhanced market analysis"""
        try:
            # Check cache first
            cache_key = f"ai_analysis_{hash(str(symbols))}"
            if cache_key in self.ai_insights_cache:
                cached_data, timestamp = self.ai_insights_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                    return cached_data
            
            # Get fresh AI analysis
            ai_analysis = await self.hybrid_ai.get_enhanced_market_analysis(symbols, market_data)
            
            # Cache the result
            self.ai_insights_cache[cache_key] = (ai_analysis, datetime.now())
            
            return ai_analysis
            
        except Exception as e:
            self.logger.error(f"Error getting AI analysis: {e}")
            return None
    
    def _find_best_opportunity(self, ai_analysis: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best trading opportunity from AI analysis"""
        try:
            symbol_analysis = ai_analysis.get('symbol_analysis', {})
            if not symbol_analysis:
                return None
            
            best_opportunity = None
            best_score = -float('inf')
            
            for symbol, analysis in symbol_analysis.items():
                # Calculate opportunity score
                score = self._calculate_opportunity_score(analysis, market_data)
                
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_opportunity = {
                        'symbol': symbol,
                        'analysis': analysis,
                        'score': score
                    }
            
            return best_opportunity
            
        except Exception as e:
            self.logger.error(f"Error finding best opportunity: {e}")
            return None
    
    def _calculate_opportunity_score(self, analysis: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate opportunity score for a symbol"""
        try:
            # Base signal strength
            signal_strength = analysis.get('combined_signal', 0)
            
            # AI confidence
            ai_confidence = analysis.get('ai_confidence', 0)
            
            # Risk adjustment
            risk_level = analysis.get('risk_level', 'medium')
            risk_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.6}.get(risk_level, 1.0)
            
            # Recommendation confidence
            recommendation = analysis.get('recommendation', {})
            rec_confidence = recommendation.get('confidence', 0)
            
            # Calculate weighted score
            score = (
                abs(signal_strength) * 0.4 +
                ai_confidence * 0.3 +
                rec_confidence * 0.3
            ) * risk_multiplier
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            return 0
    
    async def _create_ai_decision(self, opportunity: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[TradeDecision]:
        """Create trading decision from AI opportunity"""
        try:
            symbol = opportunity['symbol']
            analysis = opportunity['analysis']
            recommendation = analysis.get('recommendation', {})
            
            # Get current price
            current_price = self._get_current_price(symbol, market_data)
            if not current_price:
                return None
            
            # Determine action and quantity
            action = recommendation.get('action', 'HOLD')
            if action == 'HOLD':
                return None
            
            # Calculate quantity based on AI confidence and risk
            confidence = recommendation.get('confidence', 0.5)
            risk_level = analysis.get('risk_level', 'medium')
            
            # Base quantity calculation
            base_quantity = self._calculate_ai_quantity(confidence, risk_level, current_price)
            
            # Create decision
            decision = TradeDecision(
                symbol=symbol,
                action=action,
                quantity=base_quantity,
                price=current_price,
                confidence=confidence,
                reasoning=recommendation.get('reasoning', 'AI-enhanced decision'),
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                metadata={
                    'ai_models_used': recommendation.get('ai_models_used', []),
                    'signal_strength': analysis.get('combined_signal', 0),
                    'risk_level': risk_level,
                    'perplexity_insights': analysis.get('perplexity_insights', {}),
                    'ml_predictions': analysis.get('ml_predictions', {})
                }
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error creating AI decision: {e}")
            return None
    
    def _get_current_price(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # Try different data structures
            if 'price_data' in market_data and symbol in market_data['price_data']:
                price_data = market_data['price_data'][symbol]
                if price_data and len(price_data) > 0:
                    return float(price_data[-1].get('close', 0))
            
            if 'market_data' in market_data and symbol in market_data['market_data']:
                market_data_symbol = market_data['market_data'][symbol]
                if 'current_price' in market_data_symbol:
                    return float(market_data_symbol['current_price'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _calculate_ai_quantity(self, confidence: float, risk_level: str, price: float) -> float:
        """Calculate quantity based on AI confidence and risk"""
        try:
            # Base quantity (percentage of portfolio)
            base_percentage = 0.1  # 10% of portfolio
            
            # Adjust for confidence
            confidence_multiplier = confidence
            
            # Adjust for risk level
            risk_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.6}.get(risk_level, 1.0)
            
            # Calculate final percentage
            final_percentage = base_percentage * confidence_multiplier * risk_multiplier
            
            # Convert to quantity (assuming $10,000 portfolio)
            portfolio_value = 10000
            quantity = (portfolio_value * final_percentage) / price
            
            # Round to reasonable precision
            return round(quantity, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating AI quantity: {e}")
            return 1.0
    
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI-enhanced market data analysis"""
        try:
            symbols = self._extract_symbols(market_data)
            if not symbols:
                return {'error': 'No symbols found'}
            
            # Get AI analysis
            ai_analysis = await self._get_ai_analysis(symbols, market_data)
            if not ai_analysis:
                return {'error': 'AI analysis failed'}
            
            # Extract key insights
            insights = {
                'ai_confidence': ai_analysis.get('ai_confidence', 0),
                'market_sentiment': ai_analysis.get('market_sentiment', {}),
                'symbol_insights': {},
                'recommendations': []
            }
            
            # Process symbol analysis
            symbol_analysis = ai_analysis.get('symbol_analysis', {})
            for symbol, analysis in symbol_analysis.items():
                insights['symbol_insights'][symbol] = {
                    'signal': analysis.get('combined_signal', 0),
                    'risk': analysis.get('risk_level', 'medium'),
                    'recommendation': analysis.get('recommendation', {}),
                    'confidence': analysis.get('ai_confidence', 0)
                }
                
                # Add to recommendations
                rec = analysis.get('recommendation', {})
                if rec.get('action') != 'HOLD':
                    insights['recommendations'].append({
                        'symbol': symbol,
                        'action': rec.get('action'),
                        'confidence': rec.get('confidence', 0),
                        'reasoning': rec.get('reasoning', '')
                    })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error in AI market analysis: {e}")
            return {'error': str(e)}
    
    async def update_strategy(self, performance_data: Dict[str, Any]) -> None:
        """Update AI strategy based on performance"""
        try:
            # Analyze performance
            performance_score = performance_data.get('total_return', 0)
            win_rate = performance_data.get('win_rate', 0.5)
            
            # Adjust AI confidence threshold based on performance
            if performance_score > 0.1 and win_rate > 0.6:
                # Good performance - be more aggressive
                self.ai_confidence_threshold = max(0.5, self.ai_confidence_threshold - 0.05)
                self.logger.info(f"AI agent {self.agent_id} reducing confidence threshold to {self.ai_confidence_threshold}")
            elif performance_score < -0.05 or win_rate < 0.4:
                # Poor performance - be more conservative
                self.ai_confidence_threshold = min(0.8, self.ai_confidence_threshold + 0.05)
                self.logger.info(f"AI agent {self.agent_id} increasing confidence threshold to {self.ai_confidence_threshold}")
            
            # Adjust risk tolerance
            if performance_score > 0.15:
                self.risk_tolerance = "low"  # Take more risk
            elif performance_score < -0.1:
                self.risk_tolerance = "high"  # Reduce risk
            
        except Exception as e:
            self.logger.error(f"Error updating AI strategy: {e}")
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get AI system status"""
        try:
            return {
                'agent_id': self.agent_id,
                'ai_confidence_threshold': self.ai_confidence_threshold,
                'risk_tolerance': self.risk_tolerance,
                'cache_size': len(self.ai_insights_cache),
                'hybrid_ai_status': self.hybrid_ai.get_system_status()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting AI status: {e}")
            return {}
