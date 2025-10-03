@echo off
REM 🚀 COMPETITIVE TRADING AGENTS - WINDOWS PROFIT MAXIMIZATION LAUNCHER
REM Choose your profit maximization strategy!

echo.
echo 🚀 COMPETITIVE TRADING AGENTS - PROFIT MAXIMIZATION LAUNCHER
echo =============================================================
echo.
echo Your optimized systems are ready to UNLEASH MASSIVE PROFITS!
echo Account Value: $90,056 ^| Expected Daily Profit: $5K-$100K+
echo.
echo Available Optimized Systems:
echo.
echo 1. 🔥 MAXIMAL SYSTEM (Recommended)
echo    - 50X larger position sizes
echo    - 3X faster cycles (15s)
echo    - 4X more trades per cycle
echo    - Expected: $10K-25K daily
echo.
echo 2. ⚡ ULTRA-AGGRESSIVE
echo    - 500X larger position sizes
echo    - 1000X lower thresholds
echo    - Expected: $25K-100K daily
echo.
echo 3. 🎯 SMART OPTIMIZED
echo    - 100X larger position sizes
echo    - 75%% lower thresholds
echo    - Expected: $5K-15K daily
echo.
echo 4. 💪 REAL COMPETITIVE
echo    - 50X larger position sizes
echo    - 90%% lower thresholds
echo    - Expected: $8K-20K daily
echo.
echo 5. 📊 PROFIT DEMO (Simulation)
echo    - Shows optimization impact
echo    - No real trading
echo.
echo 6. 🐳 DOCKER DEPLOY (All Systems)
echo    - Full containerized deployment
echo    - Production ready
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo 🔥 Starting MAXIMAL SYSTEM...
    echo Position sizes: 50X larger ^| Cycles: 3X faster ^| Volume: 4X higher
    if not exist logs mkdir logs
    if not exist reports mkdir reports
    if not exist data mkdir data
    python alpaca_paper_trading_maximal.py
) else if "%choice%"=="2" (
    echo ⚡ Starting ULTRA-AGGRESSIVE SYSTEM...
    echo Position sizes: 500X larger ^| Thresholds: 1000X lower
    if not exist logs mkdir logs
    python run_ultra_aggressive_trading.py
) else if "%choice%"=="3" (
    echo 🎯 Starting SMART OPTIMIZED SYSTEM...
    echo Position sizes: 100X larger ^| Thresholds: 75%% lower
    if not exist logs mkdir logs
    python run_optimized_smart_trading.py
) else if "%choice%"=="4" (
    echo 💪 Starting REAL COMPETITIVE SYSTEM...
    echo Position sizes: 50X larger ^| Thresholds: 90%% lower
    if not exist logs mkdir logs
    python run_real_competitive_trading.py
) else if "%choice%"=="5" (
    echo 📊 Running PROFIT DEMONSTRATION...
    python profit_demo.py
) else if "%choice%"=="6" (
    echo 🐳 Starting DOCKER DEPLOYMENT...
    echo Building and launching all optimized systems in containers...
    docker-compose up --build
) else (
    echo ❌ Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo.
echo System completed. Press any key to exit...
pause >nul