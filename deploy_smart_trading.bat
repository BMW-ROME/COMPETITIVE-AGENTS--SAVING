@echo off
echo ========================================
echo SMART COMPETITIVE TRADING DEPLOYMENT
echo ========================================

echo.
echo [1/5] Stopping old containers...
docker stop ultra-aggressive-trading smart-competitive-trading 2>nul
docker rm ultra-aggressive-trading smart-competitive-trading 2>nul

echo.
echo [2/5] Rebuilding trading image...
docker build -t competitive-trading-agents-real-trading .

echo.
echo [3/5] Starting Smart Competitive Trading System...
docker run -d --name smart-competitive-trading --network tyree-systems_tyree-network -e ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF -e ALPACA_SECRET_KEY=CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw -e TRADING_MODE=PAPER competitive-trading-agents-real-trading python run_smart_competitive_trading.py

echo.
echo [4/5] Waiting for system to initialize...
timeout /t 10 /nobreak >nul

echo.
echo [5/5] Showing live trading logs...
echo ========================================
echo SMART TRADING ACTIVITY
echo ========================================
docker logs -f smart-competitive-trading

