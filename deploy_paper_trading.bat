@echo off
echo ========================================
echo ULTRA AGGRESSIVE PAPER TRADING DEPLOYMENT
echo ========================================

echo.
echo [1/5] Stopping old containers...
docker stop real-competitive-trading ultra-aggressive-trading 2>nul
docker rm real-competitive-trading ultra-aggressive-trading 2>nul

echo.
echo [2/5] Rebuilding trading image...
docker build -t competitive-trading-agents-real-trading .

echo.
echo [3/5] Starting Ultra Aggressive Trading System...
docker run -d --name ultra-aggressive-trading --network tyree-systems_tyree-network -e ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF -e ALPACA_SECRET_KEY=CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw -e TRADING_MODE=PAPER competitive-trading-agents-real-trading python run_ultra_aggressive_trading.py

echo.
echo [4/5] Waiting for system to initialize...
timeout /t 10 /nobreak >nul

echo.
echo [5/5] Showing live trading logs...
echo ========================================
echo LIVE TRADING ACTIVITY
echo ========================================
docker logs -f ultra-aggressive-trading

