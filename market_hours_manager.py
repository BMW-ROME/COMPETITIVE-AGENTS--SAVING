#!/usr/bin/env python3
"""
Market Hours Aware Trading System - Handles market schedules intelligently
"""
import datetime
import pytz
import time
from alpaca_trade_api.rest import REST
import os
from dotenv import load_dotenv

class MarketHoursManager:
    def __init__(self):
        load_dotenv()
        self.api = REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'), 
            'https://paper-api.alpaca.markets'
        )
        self.et_tz = pytz.timezone('US/Eastern')
        
    def get_market_status(self):
        """Get comprehensive market status"""
        try:
            clock = self.api.get_clock()
            utc_now = datetime.datetime.now(pytz.UTC)
            et_now = utc_now.astimezone(self.et_tz)
            
            status = {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'current_time_et': et_now,
                'current_time_utc': utc_now
            }
            
            if not clock.is_open:
                time_to_open = clock.next_open - utc_now
                status['hours_to_open'] = time_to_open.total_seconds() / 3600
                status['minutes_to_open'] = time_to_open.total_seconds() / 60
            else:
                time_to_close = clock.next_close - utc_now
                status['hours_to_close'] = time_to_close.total_seconds() / 3600
                status['minutes_to_close'] = time_to_close.total_seconds() / 60
            
            return status
            
        except Exception as e:
            print(f"Error getting market status: {e}")
            return None
    
    def is_trading_session(self):
        """Check if we're in a valid trading session"""
        status = self.get_market_status()
        if not status:
            return False, "Unable to determine market status"
        
        if status['is_open']:
            return True, "Market is open"
        else:
            hours_to_open = status.get('hours_to_open', 24)
            if hours_to_open < 12:
                return False, f"Market opens in {hours_to_open:.1f} hours"
            else:
                return False, "Market closed for extended period"
    
    def wait_for_market_open(self, max_wait_hours=12):
        """Wait for market to open with status updates"""
        status = self.get_market_status()
        if not status:
            return False
        
        if status['is_open']:
            print("✅ Market is already open!")
            return True
        
        hours_to_wait = status.get('hours_to_open', 24)
        if hours_to_wait > max_wait_hours:
            print(f"❌ Market opens in {hours_to_wait:.1f} hours (too long to wait)")
            return False
        
        print(f"⏰ Market closed. Waiting {hours_to_wait:.1f} hours for market open...")
        print(f"📅 Next market open: {status['next_open'].astimezone(self.et_tz).strftime('%Y-%m-%d %I:%M %p ET')}")
        
        # Wait in chunks and show progress
        total_seconds = hours_to_wait * 3600
        update_interval = min(300, total_seconds / 10)  # Update every 5 minutes or 10% of wait time
        
        while total_seconds > 0:
            if total_seconds > 300:
                print(f"⏳ {total_seconds/3600:.1f} hours remaining...")
                time.sleep(update_interval)
                total_seconds -= update_interval
            else:
                print(f"⏳ {total_seconds/60:.1f} minutes remaining...")
                time.sleep(60)
                total_seconds -= 60
        
        # Final check
        status = self.get_market_status()
        return status and status['is_open']

def main():
    """Main market hours checker and waiter"""
    mhm = MarketHoursManager()
    
    print("📊 MARKET HOURS MANAGER")
    print("=" * 40)
    
    status = mhm.get_market_status()
    if not status:
        print("❌ Cannot determine market status")
        return
    
    et_time = status['current_time_et']
    print(f"🕐 Current Time (ET): {et_time.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    
    if status['is_open']:
        print("🟢 MARKET IS OPEN")
        print(f"⏰ Market closes in {status.get('hours_to_close', 0):.1f} hours")
        print("🚀 Ready for live trading!")
    else:
        print("🔴 MARKET IS CLOSED")
        hours_to_open = status.get('hours_to_open', 0)
        print(f"⏰ Market opens in {hours_to_open:.1f} hours")
        print(f"📅 Next open: {status['next_open'].astimezone(mhm.et_tz).strftime('%Y-%m-%d %I:%M %p ET')}")
        
        if hours_to_open < 1:
            print("🕐 Market opening very soon!")
        elif hours_to_open < 12:
            print("💤 Consider waiting for market open")
            response = input("Wait for market open? (y/n): ").lower()
            if response == 'y':
                if mhm.wait_for_market_open():
                    print("🚀 Market is now open! Ready to trade!")
                else:
                    print("❌ Failed to wait for market open")
        else:
            print("😴 Market won't open for a while. Consider running during market hours.")
    
    # Show optimal trading times
    print(f"\n⏰ OPTIMAL TRADING TIMES (ET):")
    print(f"   🌅 Market Open: 9:30 AM - 10:30 AM (High volatility)")
    print(f"   ☀️  Mid-Morning: 10:30 AM - 12:00 PM (Stable trends)")
    print(f"   🌇 Pre-Close: 3:00 PM - 4:00 PM (Volume spike)")

if __name__ == "__main__":
    main()