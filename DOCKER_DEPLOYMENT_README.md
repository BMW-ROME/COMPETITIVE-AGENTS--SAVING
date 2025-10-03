# 🐳 Docker Deployment Guide - Alpaca Paper Trading System

## 🚀 Quick Start

### Option 1: Windows Batch File (Easiest)
1. Double-click `launch-docker.bat`
2. Follow the prompts
3. System will build and start automatically!

### Option 2: PowerShell (Recommended)
1. Right-click `launch-docker.ps1` → "Run with PowerShell"
2. Or open PowerShell as Admin and run: `.\launch-docker.ps1`

### Option 3: Manual Commands

**Windows CMD:**
```cmd
REM Build the image
docker build -f Dockerfile.alpaca -t alpaca-paper-trading .

REM Start the system
docker-compose -f docker-compose-local.yml up -d

REM View live logs
docker logs -f alpaca-paper-trading
```

**PowerShell:**
```powershell
# Build and start
docker build -f Dockerfile.alpaca -t alpaca-paper-trading .
docker-compose -f docker-compose-local.yml up -d

# Monitor
docker logs -f alpaca-paper-trading
```

## 📊 What Gets Deployed

| Container | Purpose | Port | Status |
|-----------|---------|------|--------|
| **alpaca-paper-trading** | Main trading system | 8000 | Always running |
| **simulation-trading** | Background simulation | - | Always running |
| **trading-log-monitor** | Log aggregation | - | Always running |

## 🔍 Monitoring Commands

```cmd
REM Check system status
docker-compose -f docker-compose-local.yml ps

REM View Alpaca trading logs
docker logs -f alpaca-paper-trading

REM View trading statistics
docker logs -f trading-log-monitor

REM View recent trades only
docker logs alpaca-paper-trading | findstr "REAL PAPER TRADE"

REM Check portfolio status
docker logs alpaca-paper-trading | findstr "Portfolio Value"
```

## 📈 Expected Output

When running correctly, you should see:
```
✅ Alpaca Trade API loaded successfully
🏦 ALPACA PAPER ACCOUNT CONNECTED
💰 Buying Power: $29.05
📊 Portfolio Value: $89,645.13
🤖 12 competitive paper trading agents initialized
📈 Trading symbols: AAPL, MSFT, SPY, QQQ, TSLA, NVDA
🔄 PAPER TRADING CYCLE 1
📡 Retrieved real market data for 6 symbols
✅ REAL PAPER TRADE: paper_momentum_1 | NVDA BUY 0.0162
```

## 🛠️ Management Commands

```cmd
REM Stop the system
docker-compose -f docker-compose-local.yml down

REM Restart just Alpaca container
docker-compose -f docker-compose-local.yml restart alpaca-paper-trading

REM View container resource usage
docker stats

REM Access container shell
docker exec -it alpaca-paper-trading /bin/bash

REM Clean up everything
docker-compose -f docker-compose-local.yml down -v
docker rmi alpaca-paper-trading
```

## 📁 File Structure

```
competitive-trading-agents/
├── 🐳 Dockerfile.alpaca          # Alpaca trading image
├── 🐳 docker-compose-local.yml   # Local Docker compose
├── 🚀 launch-docker.bat          # Windows launcher
├── 🚀 launch-docker.ps1          # PowerShell launcher
├── 📄 DOCKER_COMMANDS.txt        # All Docker commands
├── 💰 alpaca_paper_trading.py    # Main trading system
├── 📊 continuous_trading_system.py # Simulation system
├── 🔧 .env                       # Your Alpaca credentials
└── 📝 logs/                      # Trading logs (mounted)
    ├── alpaca_paper_trading.log
    └── continuous_trading.log
```

## 🔐 Environment Setup

Your `.env` file should contain:
```env
ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## 🎯 Key Features

- ✅ **Real Alpaca Paper Trading**: Live API integration
- ✅ **12 Competitive Agents**: AI-driven trading decisions  
- ✅ **Live Market Data**: Real-time price feeds
- ✅ **Persistent Logs**: All trading activity logged
- ✅ **Auto-restart**: System recovers from failures
- ✅ **Resource Monitoring**: Built-in health checks
- ✅ **Easy Management**: Simple start/stop commands

## 🚨 Troubleshooting

**Docker not found:**
- Install Docker Desktop for Windows
- Make sure Docker Desktop is running

**Build fails:**
- Check internet connection
- Try: `docker system prune -a` then rebuild

**No trades executing:**
- Check your `.env` file has correct Alpaca credentials
- Verify Alpaca account is active: Check logs for "ALPACA PAPER ACCOUNT CONNECTED"

**Container stops:**
- View logs: `docker logs alpaca-paper-trading`
- Check system resources: `docker stats`

## 📞 Support

If you need help:
1. Check `docker logs alpaca-paper-trading` for errors
2. Run `docker-compose -f docker-compose-local.yml ps` to see container status
3. Review the log files in the `logs/` folder

## 🎉 Success Indicators

Your system is working when you see:
- ✅ "ALPACA PAPER ACCOUNT CONNECTED"
- ✅ "Portfolio Value: $XX,XXX.XX" 
- ✅ "REAL PAPER TRADE" messages
- ✅ "Retrieved real market data" every cycle
- ✅ Regular "PAPER TRADING CYCLE" messages

---

**🚀 Ready to deploy? Run `launch-docker.bat` and watch your competitive trading agents come alive!**