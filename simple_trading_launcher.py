#!/usr/bin/env python3
"""
Simple Trading System Launcher
==============================
Direct, no-nonsense trading system that just works.
"""

import asyncio
import logging
import random
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SimpleTrading")

class SimpleTradingSystem:
    """Ultra-simple trading system"""
    
    def __init__(self):
        self.cycle_count = 0
        self.agents = [
            'conservative_1', 'aggressive_1', 'balanced_1', 'scalping_1',
            'momentum_1', 'ai_enhanced_1', 'arbitrage_1', 'adaptive_1',
            'fractal_1', 'candle_range_1', 'quant_1', 'ml_pattern_1'
        ]
        logger.info(f"✅ Simple system initialized with {len(self.agents)} agents")
    
    async def run_cycle(self):
        """Run one simple trading cycle"""
        self.cycle_count += 1
        start_time = datetime.now()
        
        # Simulate agent decisions (80% chance each agent makes a decision)
        decisions = 0
        selected_trades = 0
        
        for agent in self.agents:
            if random.random() < 0.8:  # 80% decision probability
                decisions += 1
                
                if random.random() < 0.6:  # 60% selection probability
                    selected_trades += 1
                    symbol = random.choice(['AAPL', 'MSFT', 'SPY', 'QQQ'])
                    action = random.choice(['BUY', 'SELL'])
                    amount = random.uniform(10, 50)
                    
                    logger.info(f"🔄 {agent}: {symbol} {action} ${amount:.2f}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"📊 Cycle {self.cycle_count}: {decisions} decisions, {selected_trades} trades, {duration:.2f}s")
        
        return {
            'cycle': self.cycle_count,
            'decisions': decisions,
            'trades': selected_trades,
            'duration': duration
        }

async def main():
    """Main trading loop"""
    logger.info("🚀 SIMPLE TRADING SYSTEM STARTING")
    logger.info("=" * 50)
    
    system = SimpleTradingSystem()
    
    try:
        # Run continuous cycles
        while True:
            result = await system.run_cycle()
            
            # Sleep for 30 seconds between cycles
            await asyncio.sleep(30)
            
            # Every 10 cycles, show summary
            if system.cycle_count % 10 == 0:
                logger.info(f"🎯 Completed {system.cycle_count} cycles successfully!")
                
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())