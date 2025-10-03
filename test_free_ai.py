"""
Test the free AI system (no API keys required)
"""
import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FreeAITest")

async def test_free_intelligence():
    """Test the free intelligence system"""
    try:
        from src.free_intelligence_system import FreeIntelligenceSystem
        
        async with FreeIntelligenceSystem(logger) as free_ai:
            # Test market intelligence
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            intelligence = await free_ai.get_market_intelligence(symbols)
            
            logger.info(f"Free Intelligence Test Results:")
            for symbol, data in intelligence.items():
                sentiment = data.get('sentiment', 'N/A')
                score = data.get('sentiment_score', 50)
                logger.info(f"  {symbol}: {sentiment} (score: {score})")
            
            # Test market sentiment
            sentiment = await free_ai.get_market_sentiment()
            logger.info(f"Overall market sentiment: {sentiment.get('mood', 'N/A')}")
            
            # Test system status
            status = free_ai.get_system_status()
            logger.info(f"System status: {status}")
            
            return True
            
    except Exception as e:
        logger.error(f"Free intelligence test failed: {e}")
        return False

async def main():
    """Run the free AI test"""
    logger.info("🧪 Testing Free AI System (No API Keys Required)...")
    
    try:
        result = await test_free_intelligence()
        if result:
            logger.info("✅ Free AI System working perfectly!")
        else:
            logger.error("❌ Free AI System failed")
    except Exception as e:
        logger.error(f"❌ Test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())



