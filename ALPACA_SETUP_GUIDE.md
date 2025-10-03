# Alpaca Paper Trading Setup Guide

## Step 1: Get Alpaca Paper Trading Account
1. Go to: https://app.alpaca.markets/paper/dashboard/overview
2. Sign up for a free paper trading account
3. No real money required - it's all simulated

## Step 2: Get Your API Keys
1. Go to: https://app.alpaca.markets/paper/dashboard/overview
2. Click on "API Keys" in the sidebar
3. Generate new API keys if you don't have them
4. Copy your API Key and Secret Key

## Step 3: Configure Environment Variables
Edit the .env file with your actual keys:

```bash
# Replace with your actual keys
ALPACA_API_KEY=PKTEST_your_actual_key_here
ALPACA_SECRET_KEY=your_actual_secret_here
```

## Step 4: Test the Connection
Run: python test_alpaca_connection.py

## Step 5: Restart Trading System
Run: python run_ultimate_system_simple_mcp.py

## What This Enables:
- ✅ Real market data from Alpaca
- ✅ Actual paper trading (simulated money)
- ✅ Real trade execution
- ✅ Portfolio tracking
- ✅ Performance metrics

## Safety Notes:
- Paper trading uses simulated money
- No real money is at risk
- Perfect for testing strategies
- Switch to live trading only when ready
