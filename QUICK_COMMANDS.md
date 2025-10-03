# 🐍 Quick Commands Reference

## ⚡ Most Essential Commands

### System Status
```bash
python test_imports.py                    # Test all imports
python test_trading_setup.py              # Test configuration
python -c "from config.settings import *; print('Config OK')"
```

### Run Demos
```bash
python example_run.py                     # Basic trading demo
python advanced_agents_demo.py            # Advanced agents competition
python multi_asset_demo.py                # FOREX + Crypto + Stocks
python multi_exchange_demo.py             # Multi-exchange trading
```

### Configuration Check
```bash
python -c "from config.settings import SystemConfig; s=SystemConfig(); print(f'Agents: {len(s.agent_configs)}, Symbols: {len(s.trading_symbols)}')"
python -c "from config.settings import AlpacaConfig; a=AlpacaConfig(); print('Exchanges:', a.exchanges)"
```

### Environment Setup
```bash
set ALPACA_API_KEY=your_key              # Windows
export ALPACA_API_KEY=your_key           # Linux/Mac
python -c "import os; print('API Key:', 'SET' if os.getenv('ALPACA_API_KEY') else 'NOT SET')"
```

### Agent Information
```bash
python -c "from config.settings import SystemConfig; s=SystemConfig(); print([a.agent_id for a in s.agent_configs])"
python -c "from config.settings import SystemConfig; s=SystemConfig(); print([a.agent_type.value for a in s.agent_configs])"
```

### Trading Symbols
```bash
python -c "from config.settings import SystemConfig; s=SystemConfig(); print('Total symbols:', len(s.trading_symbols))"
python -c "from config.settings import SystemConfig; s=SystemConfig(); forex=[x for x in s.trading_symbols if 'USD' in x and len(x)==6]; print('FOREX:', forex[:5])"
python -c "from config.settings import SystemConfig; s=SystemConfig(); crypto=[x for x in s.trading_symbols if x.endswith('USD') and x not in ['EURUSD','GBPUSD','AUDUSD','USDCAD','NZDUSD']]; print('Crypto:', crypto[:5])"
```

## 🎯 Agent Types Available
- `AgentType.CONSERVATIVE`
- `AgentType.AGGRESSIVE` 
- `AgentType.BALANCED`
- `AgentType.FRACTAL_ANALYSIS`
- `AgentType.CANDLE_RANGE_THEORY`
- `AgentType.QUANTITATIVE_PATTERN`
- `AgentType.FOREX_SPECIALIST`
- `AgentType.CRYPTO_SPECIALIST`
- `AgentType.MULTI_ASSET_ARBITRAGE`

## 🌍 Supported Exchanges
- **NASDAQ** - Technology stocks
- **NYSE** - Blue-chip stocks  
- **ARCA** - ETFs
- **BATS** - Alternative ETFs
- **IEX** - Investor-friendly
- **FOREX** - Currency pairs (24/7)
- **CRYPTO** - Cryptocurrencies (24/7)

## 📊 Asset Classes
- **Stocks**: 70 symbols (AAPL, GOOGL, MSFT, etc.)
- **FOREX**: 28 pairs (EURUSD, GBPUSD, USDJPY, etc.)
- **Crypto**: 49 assets (BTCUSD, ETHUSD, ADAUSD, etc.)

## 🛠️ Troubleshooting
```bash
python -c "try: from src.trading_agents import *; print('Trading agents OK'); except Exception as e: print('Error:', e)"
python -c "try: from forex_crypto_agents import *; print('FOREX/Crypto agents OK'); except Exception as e: print('Error:', e)"
python -c "import sys; print('Python executable:', sys.executable)"
```

## 📝 Logs
```bash
# Monitor logs in real-time
tail -f logs/trading_system.log

# Check log files
python -c "import os; print('Log files:', [f for f in os.listdir('logs') if f.endswith('.log')])"
```

## 🚀 Live Trading Setup
1. Get Alpaca live account: https://alpaca.markets/
2. Set API keys in environment variables
3. Change `base_url` to `https://api.alpaca.markets`
4. Start with small amounts
5. Monitor trades closely

## 💡 Tips
- Copy and paste any command to run it
- Use Ctrl+C to stop running demos
- Check `logs/trading_system.log` for detailed output
- Always test with paper trading first
- Start with small position sizes for live trading
