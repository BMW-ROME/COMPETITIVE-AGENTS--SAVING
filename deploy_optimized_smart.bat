@echo off
echo ========================================
echo OPTIMIZED SMART TRADING DEPLOYMENT
echo ========================================

echo.
echo [1/6] Stopping all existing trading containers...
docker stop smart-competitive-trading fixed-ultra-aggressive optimized-smart-trading optimized-ultra-aggressive 2>nul
docker rm smart-competitive-trading fixed-ultra-aggressive optimized-smart-trading optimized-ultra-aggressive 2>nul

echo.
echo [2/6] Cleaning up old images...
docker image prune -f

echo.
echo [3/6] Rebuilding optimized trading image...
docker build -t competitive-trading-agents-optimized .

echo.
echo [4/6] Starting Optimized Smart Trading System...
docker run -d --name optimized-smart-trading --network tyree-systems_tyree-network -e ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF -e ALPACA_SECRET_KEY=CuQWde4QtPHAtwuMfxaQhB8njmrcJDq0YK4Oz9Rw -e TRADING_MODE=PAPER competitive-trading-agents-optimized python run_optimized_smart_trading.py

echo.
echo [5/6] Waiting for system to initialize...
timeout /t 15 /nobreak >nul

echo.
echo [6/6] Showing live optimized trading logs...
echo ========================================
echo OPTIMIZED SMART TRADING ACTIVITY
echo ========================================
docker logs -f optimized-smart-trading

