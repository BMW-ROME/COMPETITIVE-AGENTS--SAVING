#!/usr/bin/env python3
"""
Quick test to verify Alpaca Paper Trading credentials
"""

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

# Get credentials
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

print("🧪 TESTING ALPACA PAPER TRADING CREDENTIALS")
print("=" * 50)
print(f"API Key: {api_key[:8]}..." if api_key else "❌ No API Key")
print(f"Secret Key: {'✅ Present' if secret_key else '❌ Missing'}")
print(f"Base URL: {base_url}")
print()

try:
    # Initialize API
    api = tradeapi.REST(
        api_key,
        secret_key,
        base_url,
        api_version='v2'
    )
    
    print("🔗 Attempting to connect to Alpaca...")
    
    # Test basic connection
    account = api.get_account()
    print(f"✅ CONNECTION SUCCESSFUL!")
    print(f"   Account ID: {account.id}")
    print(f"   Status: {account.status}")
    print(f"   Trading Blocked: {account.trading_blocked}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print()
    
    # Test market data
    print("📊 Testing market data access...")
    try:
        snapshot = api.get_snapshot('AAPL')
        print(f"✅ Market data working - AAPL: ${snapshot.latest_trade.price}")
    except Exception as e:
        print(f"⚠️ Market data issue: {e}")
    
    print()
    print("🎯 CREDENTIALS ARE WORKING PERFECTLY!")
    print("Your maximal system should work with these credentials.")
    
except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")
    print()
    if "unauthorized" in str(e).lower():
        print("💡 SOLUTION:")
        print("1. Double-check your ALPACA_SECRET_KEY in the .env file")
        print("2. Make sure you're using PAPER TRADING credentials")
        print("3. Verify your Alpaca account is active")
    else:
        print(f"💡 Error details: {e}")