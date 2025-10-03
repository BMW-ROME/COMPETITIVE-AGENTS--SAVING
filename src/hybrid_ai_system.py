"""
Hybrid AI System combining Perplexity Intelligence with Random Forest ML
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import json

from src.perplexity_intelligence import PerplexityIntelligence
from src.random_forest_ml import RandomForestTradingML

class HybridAISystem:
    """Hybrid AI system combining Perplexity intelligence with Random Forest ML"""
    
    def __init__(self, perplexity_api_key: str, logger: logging.Logger):
        self.logger = logger
        self.perplexity = PerplexityIntelligence(perplexity_api_key, logger)
        self.random_forest = RandomForestTradingML(logger)
        self.ai_cache = {}
        self.cache_duration = 300  # 5 minutes
        self.ensemble_weights = {
            'perplexity': 0.4,
            'random_forest': 0.4,
            'technical': 0.2
        }
        
    async def __aenter__(self):
        await self.perplexity.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.perplexity.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_enhanced_market_analysis(self, symbols: List[str], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive AI-enhanced market analysis"""
        try:
            self.logger.info(f"Getting enhanced AI analysis for {len(symbols)} symbols")
            
            # Get Perplexity intelligence
            perplexity_intelligence = await self.perplexity.get_market_intelligence(symbols)
            
            # Get Random Forest predictions
            rf_predictions = {}
            for symbol in symbols:
                rf_predictions[symbol] = {
                    'price_movement': self.random_forest.predict_price_movement(symbol, market_data),
                    'direction': self.random_forest.predict_direction(symbol, market_data)
                }
            
            # Combine insights
            enhanced_analysis = {}
            for symbol in symbols:
                enhanced_analysis[symbol] = await self._combine_ai_insights(
                    symbol, 
                    perplexity_intelligence.get(symbol, {}),
                    rf_predictions.get(symbol, {}),
                    market_data
                )
            
            # Get overall market sentiment
            market_sentiment = await self.perplexity.get_market_sentiment()
            
            return {
                'symbol_analysis': enhanced_analysis,
                'market_sentiment': market_sentiment,
                'ai_confidence': self._calculate_overall_confidence(enhanced_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced market analysis: {e}")
            return {}
    
    async def _combine_ai_insights(self, symbol: str, perplexity_data: Dict, rf_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """Combine Perplexity and Random Forest insights"""
        try:
            # Extract key metrics
            perplexity_sentiment = perplexity_data.get('sentiment_score', 50) / 100
            perplexity_outlook = perplexity_data.get('outlook', 'neutral')
            
            rf_price_pred = rf_data.get('price_movement', {}).get('prediction', 0)
            rf_price_conf = rf_data.get('price_movement', {}).get('confidence', 0)
            rf_direction = rf_data.get('direction', {}).get('direction', 'neutral')
            rf_direction_conf = rf_data.get('direction', {}).get('confidence', 0)
            
            # Calculate combined signals
            combined_signal = self._calculate_combined_signal(
                perplexity_sentiment, rf_price_pred, rf_direction, rf_price_conf, rf_direction_conf
            )
            
            # Risk assessment
            risk_level = self._assess_risk_level(perplexity_data, rf_data, market_data)
            
            # Trading recommendation
            recommendation = self._generate_trading_recommendation(
                combined_signal, risk_level, perplexity_data, rf_data
            )
            
            return {
                'symbol': symbol,
                'combined_signal': combined_signal,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'perplexity_insights': {
                    'sentiment': perplexity_sentiment,
                    'outlook': perplexity_outlook,
                    'news_summary': perplexity_data.get('news_summary', ''),
                    'catalysts': perplexity_data.get('catalysts', ''),
                    'risk_factors': perplexity_data.get('risk_factors', '')
                },
                'ml_predictions': {
                    'price_movement': rf_price_pred,
                    'direction': rf_direction,
                    'confidence': max(rf_price_conf, rf_direction_conf),
                    'probabilities': rf_data.get('direction', {}).get('probabilities', {})
                },
                'ai_confidence': self._calculate_ai_confidence(perplexity_data, rf_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error combining AI insights for {symbol}: {e}")
            return self._get_fallback_analysis(symbol)
    
    def _calculate_combined_signal(self, perplexity_sentiment: float, rf_price_pred: float, 
                                 rf_direction: str, rf_price_conf: float, rf_direction_conf: float) -> float:
        """Calculate combined AI signal"""
        try:
            # Normalize signals
            sentiment_signal = (perplexity_sentiment - 0.5) * 2  # -1 to 1
            
            # Price movement signal
            price_signal = np.tanh(rf_price_pred * 10)  # Normalize to -1 to 1
            
            # Direction signal
            direction_signal = 1 if rf_direction == 'up' else -1 if rf_direction == 'down' else 0
            
            # Weighted combination
            combined = (
                self.ensemble_weights['perplexity'] * sentiment_signal +
                self.ensemble_weights['random_forest'] * (price_signal + direction_signal) / 2
            )
            
            # Apply confidence weighting
            avg_confidence = (rf_price_conf + rf_direction_conf) / 2
            confidence_weighted = combined * avg_confidence
            
            return max(-1, min(1, confidence_weighted))
            
        except Exception as e:
            self.logger.error(f"Error calculating combined signal: {e}")
            return 0
    
    def _assess_risk_level(self, perplexity_data: Dict, rf_data: Dict, market_data: Dict) -> str:
        """Assess risk level based on AI insights"""
        try:
            risk_factors = []
            
            # Perplexity risk factors
            if 'risk_factors' in perplexity_data:
                risk_factors.append(perplexity_data['risk_factors'])
            
            # ML confidence levels
            rf_price_conf = rf_data.get('price_movement', {}).get('confidence', 0)
            rf_direction_conf = rf_data.get('direction', {}).get('confidence', 0)
            
            if rf_price_conf < 0.3 or rf_direction_conf < 0.3:
                risk_factors.append("Low ML confidence")
            
            # Market volatility
            volatility = market_data.get('volatility', {}).get('current_volatility', 0.2)
            if volatility > 0.3:
                risk_factors.append("High volatility")
            
            # Determine risk level
            if len(risk_factors) >= 3:
                return "high"
            elif len(risk_factors) >= 1:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error assessing risk level: {e}")
            return "medium"
    
    def _generate_trading_recommendation(self, combined_signal: float, risk_level: str, 
                                       perplexity_data: Dict, rf_data: Dict) -> Dict[str, Any]:
        """Generate trading recommendation based on AI analysis"""
        try:
            # Determine action based on signal strength
            if combined_signal > 0.3:
                action = "BUY"
                confidence = min(0.9, abs(combined_signal))
            elif combined_signal < -0.3:
                action = "SELL"
                confidence = min(0.9, abs(combined_signal))
            else:
                action = "HOLD"
                confidence = 0.5
            
            # Adjust for risk level
            if risk_level == "high":
                confidence *= 0.7
                if action != "HOLD":
                    action = "HOLD"  # High risk = hold
            elif risk_level == "medium":
                confidence *= 0.85
            
            # Generate reasoning
            reasoning_parts = []
            
            # Perplexity reasoning
            if perplexity_data.get('sentiment_score', 50) > 60:
                reasoning_parts.append("Positive sentiment")
            elif perplexity_data.get('sentiment_score', 50) < 40:
                reasoning_parts.append("Negative sentiment")
            
            # ML reasoning
            rf_direction = rf_data.get('direction', {}).get('direction', 'neutral')
            if rf_direction == 'up':
                reasoning_parts.append("ML predicts upward movement")
            elif rf_direction == 'down':
                reasoning_parts.append("ML predicts downward movement")
            
            # Risk reasoning
            if risk_level == "high":
                reasoning_parts.append("High risk environment")
            
            reasoning = f"AI recommendation: {', '.join(reasoning_parts)}"
            
            return {
                'action': action,
                'confidence': confidence,
                'signal_strength': abs(combined_signal),
                'reasoning': reasoning,
                'risk_level': risk_level,
                'ai_models_used': ['perplexity', 'random_forest']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading recommendation: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'signal_strength': 0,
                'reasoning': 'AI analysis unavailable',
                'risk_level': 'medium',
                'ai_models_used': []
            }
    
    def _calculate_ai_confidence(self, perplexity_data: Dict, rf_data: Dict) -> float:
        """Calculate overall AI confidence"""
        try:
            # Perplexity confidence (based on data availability)
            perplexity_conf = 0.5
            if perplexity_data.get('sentiment_score') is not None:
                perplexity_conf = 0.8
            
            # ML confidence
            rf_price_conf = rf_data.get('price_movement', {}).get('confidence', 0)
            rf_direction_conf = rf_data.get('direction', {}).get('confidence', 0)
            ml_conf = (rf_price_conf + rf_direction_conf) / 2
            
            # Combined confidence
            combined_conf = (perplexity_conf * 0.4 + ml_conf * 0.6)
            
            return max(0, min(1, combined_conf))
            
        except Exception as e:
            self.logger.error(f"Error calculating AI confidence: {e}")
            return 0.5
    
    def _calculate_overall_confidence(self, enhanced_analysis: Dict) -> float:
        """Calculate overall system confidence"""
        try:
            if not enhanced_analysis:
                return 0
            
            confidences = []
            for symbol, analysis in enhanced_analysis.items():
                if 'ai_confidence' in analysis:
                    confidences.append(analysis['ai_confidence'])
            
            return np.mean(confidences) if confidences else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall confidence: {e}")
            return 0
    
    def _get_fallback_analysis(self, symbol: str) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        return {
            'symbol': symbol,
            'combined_signal': 0,
            'risk_level': 'medium',
            'recommendation': {
                'action': 'HOLD',
                'confidence': 0.5,
                'signal_strength': 0,
                'reasoning': 'AI analysis unavailable',
                'risk_level': 'medium',
                'ai_models_used': []
            },
            'perplexity_insights': {},
            'ml_predictions': {},
            'ai_confidence': 0.5,
            'timestamp': datetime.now().isoformat()
        }
    
    async def train_ml_models(self, symbols: List[str], historical_data: Dict[str, List[Dict]]) -> bool:
        """Train Random Forest models for all symbols"""
        try:
            self.logger.info(f"Training ML models for {len(symbols)} symbols")
            
            success_count = 0
            
            for symbol in symbols:
                if symbol in historical_data and len(historical_data[symbol]) >= 50:
                    # Train price prediction model
                    price_success = self.random_forest.train_price_prediction_model(
                        symbol, historical_data[symbol]
                    )
                    
                    # Train direction prediction model
                    direction_success = self.random_forest.train_direction_prediction_model(
                        symbol, historical_data[symbol]
                    )
                    
                    if price_success or direction_success:
                        success_count += 1
                        self.logger.info(f"ML models trained for {symbol}")
                    else:
                        self.logger.warning(f"Failed to train ML models for {symbol}")
                else:
                    self.logger.warning(f"Insufficient data for {symbol}")
            
            self.logger.info(f"ML training complete: {success_count}/{len(symbols)} symbols")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the hybrid AI system"""
        try:
            return {
                'perplexity_available': bool(self.perplexity.api_key),
                'ml_models_trained': len(self.random_forest.models),
                'cache_size': len(self.ai_cache),
                'ensemble_weights': self.ensemble_weights,
                'ml_performance': self.random_forest.get_model_performance()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}
    
    async def get_sector_analysis(self, sectors: List[str]) -> Dict[str, Any]:
        """Get AI-enhanced sector analysis"""
        try:
            sector_analysis = await self.perplexity.get_sector_analysis(sectors)
            
            # Add ML insights if available
            for sector in sectors:
                if sector in sector_analysis:
                    # Could add ML-based sector predictions here
                    sector_analysis[sector]['ai_enhanced'] = True
                    sector_analysis[sector]['timestamp'] = datetime.now().isoformat()
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"Error getting sector analysis: {e}")
            return {}
