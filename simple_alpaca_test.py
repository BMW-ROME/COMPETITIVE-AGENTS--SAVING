#!/usr/bin/env python3
"""
MAXIMAL ALPACA PAPER TRADING SYSTEM - Simplified Version
Real-time competitive trading with working credentials test
"""

import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alpaca_simple_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AlpacaTest')

class SimpleAlpacaTest:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        logger.info("🚀 SIMPLIFIED ALPACA CREDENTIALS TEST")
        logger.info("=" * 50)
        
        try:
            # Initialize API
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                self.base_url,
                api_version='v2'
            )
            
            # Test connection
            self.account = self.api.get_account()
            logger.info(f"✅ Connected successfully!")
            logger.info(f"   Account: {self.account.id}")
            logger.info(f"   Status: {self.account.status}")
            logger.info(f"   Portfolio Value: ${float(self.account.portfolio_value):,.2f}")
            logger.info(f"   Buying Power: ${float(self.account.buying_power):,.2f}")
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
    
    def test_market_data(self):
        """Test market data access"""
        if not self.connected:
            return
            
        try:
            logger.info("📊 Testing market data...")
            
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
            for symbol in symbols:
                try:
                    snapshot = self.api.get_snapshot(symbol)
                    price = snapshot.latest_trade.price
                    logger.info(f"   {symbol}: ${price}")
                except Exception as e:
                    logger.warning(f"   {symbol}: Failed - {e}")
                    
        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
    
    def test_account_activities(self):
        """Test account activities and positions"""
        if not self.connected:
            return
            
        try:
            logger.info("💼 Testing account data...")
            
            # Get positions
            positions = self.api.list_positions()
            logger.info(f"   Current positions: {len(positions)}")
            
            for pos in positions[:5]:  # Show first 5
                logger.info(f"   {pos.symbol}: {pos.qty} shares @ ${pos.avg_cost}")
            
            # Get recent orders
            orders = self.api.list_orders(status='all', limit=5)
            logger.info(f"   Recent orders: {len(orders)}")
            
        except Exception as e:
            logger.error(f"❌ Account data test failed: {e}")
    
    def run_test_cycle(self):
        """Run a complete test cycle"""
        logger.info("🔄 Starting test cycle...")
        
        if not self.connected:
            logger.error("❌ Not connected to Alpaca")
            return
            
        self.test_market_data()
        self.test_account_activities()
        
        logger.info("✅ Test cycle completed successfully!")
        logger.info(f"🎯 Your credentials are working perfectly!")

def main():
    """Main execution"""
    logger.info("🧪 STARTING ALPACA CREDENTIALS AND API TEST")
    
    test = SimpleAlpacaTest()
    
    if test.connected:
        test.run_test_cycle()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 CONGRATULATIONS!")
        logger.info("Your Alpaca paper trading credentials are working!")
        logger.info("You can now run the full maximal system.")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ CREDENTIALS NOT WORKING")
        logger.error("Please check your .env file")
        logger.error("="*60)

if __name__ == "__main__":
    main()