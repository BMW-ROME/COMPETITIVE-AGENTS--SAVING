"""
Test script for the AI-enhanced trading system
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AISystemTest")

async def test_perplexity_intelligence():
    """Test Perplexity intelligence system"""
    try:
        from src.perplexity_intelligence import PerplexityIntelligence
        
        # Get API key from environment
        api_key = os.getenv('PERPLEXITY_API_KEY', '')
        if not api_key:
            logger.warning("PERPLEXITY_API_KEY not found, using fallback mode")
        
        async with PerplexityIntelligence(api_key, logger) as perplexity:
            # Test market intelligence
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            intelligence = await perplexity.get_market_intelligence(symbols)
            
            logger.info(f"Perplexity Intelligence Test Results:")
            for symbol, data in intelligence.items():
                logger.info(f"  {symbol}: {data.get('sentiment', 'N/A')} sentiment")
            
            # Test market sentiment
            sentiment = await perplexity.get_market_sentiment()
            logger.info(f"Overall market sentiment: {sentiment.get('mood', 'N/A')}")
            
            return True
            
    except Exception as e:
        logger.error(f"Perplexity intelligence test failed: {e}")
        return False

async def test_random_forest_ml():
    """Test Random Forest ML system"""
    try:
        from src.random_forest_ml import RandomForestTradingML
        
        ml_system = RandomForestTradingML(logger)
        
        # Create mock historical data
        mock_data = []
        base_price = 100.0
        for i in range(100):
            price_change = 0.01 * (i % 10 - 5)  # Simple pattern
            base_price *= (1 + price_change)
            mock_data.append({
                'timestamp': datetime.now().isoformat(),
                'open': base_price * 0.999,
                'high': base_price * 1.001,
                'low': base_price * 0.998,
                'close': base_price,
                'volume': 1000 + i * 10
            })
        
        # Train models
        price_success = ml_system.train_price_prediction_model('TEST', mock_data)
        direction_success = ml_system.train_direction_prediction_model('TEST', mock_data)
        
        logger.info(f"Random Forest ML Test Results:")
        logger.info(f"  Price prediction model: {'✅' if price_success else '❌'}")
        logger.info(f"  Direction prediction model: {'✅' if direction_success else '❌'}")
        
        # Test predictions
        if price_success or direction_success:
            mock_market_data = {
                'price_data': {'TEST': mock_data[-20:]},
                'technical_indicators': {'rsi': 50, 'macd': 0},
                'sentiment': {'overall_sentiment': 0.6}
            }
            
            price_pred = ml_system.predict_price_movement('TEST', mock_market_data)
            direction_pred = ml_system.predict_direction('TEST', mock_market_data)
            
            logger.info(f"  Price prediction: {price_pred.get('prediction', 0):.4f}")
            logger.info(f"  Direction prediction: {direction_pred.get('direction', 'neutral')}")
        
        return price_success or direction_success
        
    except Exception as e:
        logger.error(f"Random Forest ML test failed: {e}")
        return False

async def test_hybrid_ai_system():
    """Test hybrid AI system"""
    try:
        from src.hybrid_ai_system import HybridAISystem
        
        api_key = os.getenv('PERPLEXITY_API_KEY', '')
        async with HybridAISystem(api_key, logger) as hybrid_ai:
            # Test enhanced market analysis
            symbols = ['AAPL', 'MSFT']
            mock_market_data = {
                'price_data': {
                    'AAPL': [{'close': 150.0, 'volume': 1000}],
                    'MSFT': [{'close': 300.0, 'volume': 1000}]
                },
                'technical_indicators': {'rsi': 50, 'macd': 0},
                'sentiment': {'overall_sentiment': 0.6}
            }
            
            analysis = await hybrid_ai.get_enhanced_market_analysis(symbols, mock_market_data)
            
            logger.info(f"Hybrid AI System Test Results:")
            logger.info(f"  Analysis completed: {'✅' if analysis else '❌'}")
            
            if analysis:
                symbol_analysis = analysis.get('symbol_analysis', {})
                for symbol, data in symbol_analysis.items():
                    signal = data.get('combined_signal', 0)
                    risk = data.get('risk_level', 'unknown')
                    logger.info(f"  {symbol}: Signal={signal:.3f}, Risk={risk}")
            
            return bool(analysis)
            
    except Exception as e:
        logger.error(f"Hybrid AI system test failed: {e}")
        return False

async def test_ai_enhanced_agent():
    """Test AI-enhanced trading agent"""
    try:
        from src.ai_enhanced_agent import AIEnhancedTradingAgent
        
        api_key = os.getenv('PERPLEXITY_API_KEY', '')
        agent = AIEnhancedTradingAgent('test_ai_agent', api_key, logger)
        
        async with agent:
            # Test market analysis
            mock_market_data = {
                'price_data': {
                    'AAPL': [{'close': 150.0, 'volume': 1000}],
                    'MSFT': [{'close': 300.0, 'volume': 1000}]
                },
                'technical_indicators': {'rsi': 50, 'macd': 0},
                'sentiment': {'overall_sentiment': 0.6}
            }
            
            analysis = await agent.analyze_market_data(mock_market_data)
            
            logger.info(f"AI-Enhanced Agent Test Results:")
            logger.info(f"  Analysis completed: {'✅' if analysis else '❌'}")
            
            if analysis and 'ai_confidence' in analysis:
                logger.info(f"  AI Confidence: {analysis['ai_confidence']:.3f}")
                logger.info(f"  Recommendations: {len(analysis.get('recommendations', []))}")
            
            return bool(analysis)
            
    except Exception as e:
        logger.error(f"AI-enhanced agent test failed: {e}")
        return False

async def main():
    """Run all AI system tests"""
    logger.info("🚀 Starting AI System Tests...")
    
    tests = [
        ("Perplexity Intelligence", test_perplexity_intelligence),
        ("Random Forest ML", test_random_forest_ml),
        ("Hybrid AI System", test_hybrid_ai_system),
        ("AI-Enhanced Agent", test_ai_enhanced_agent)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n📊 Test Summary:")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        logger.info(f"  {status} {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL AI SYSTEMS WORKING PERFECTLY!")
    elif passed > 0:
        logger.info("⚠️  Some AI systems working, check failed tests")
    else:
        logger.error("❌ All AI systems failed, check configuration")

if __name__ == "__main__":
    asyncio.run(main())



