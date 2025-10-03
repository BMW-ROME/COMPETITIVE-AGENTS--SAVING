@echo off
echo ========================================
echo 🚀 MAXIMAL ALPACA PAPER TRADING SYSTEM
echo ========================================
echo 💼 Advanced AI-Driven Multi-Agent Trading
echo 🧠 Machine Learning Enhanced Analytics  
echo 📊 Real-time Performance Dashboards
echo 🎯 12 Advanced Competitive Agents
echo ========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is running...
echo.

REM Navigate to script directory
cd /d "%~dp0"

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo 🔧 Creating maximal .env configuration...
    echo # MAXIMAL Alpaca Paper Trading Configuration > .env
    echo ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF >> .env
    echo ALPACA_SECRET_KEY=your_secret_key_here >> .env
    echo ALPACA_BASE_URL=https://paper-api.alpaca.markets >> .env
    echo. >> .env
    echo # Advanced Configuration >> .env
    echo TRADING_MODE=ALPACA_MAXIMAL >> .env
    echo MAX_POSITION_SIZE=10000 >> .env
    echo RISK_TOLERANCE=0.02 >> .env
    echo ML_ENABLED=true >> .env
    echo ANALYTICS_ENABLED=true >> .env
    echo.
    echo ⚠️  IMPORTANT: Edit .env with your real Alpaca secret key!
    echo.
    pause
)

REM Create comprehensive directory structure
echo 📁 Creating maximal directory structure...
if not exist "logs" mkdir logs
if not exist "data" mkdir data  
if not exist "reports" mkdir reports
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "backups" mkdir backups
if not exist "sql" mkdir sql

REM Create database initialization
echo 🗄️  Setting up database schema...
echo -- Maximal Trading Database Schema > sql\init_maximal.sql
echo CREATE TABLE IF NOT EXISTS trades ( >> sql\init_maximal.sql
echo   id SERIAL PRIMARY KEY, >> sql\init_maximal.sql
echo   agent_name VARCHAR(100), >> sql\init_maximal.sql
echo   symbol VARCHAR(10), >> sql\init_maximal.sql
echo   side VARCHAR(10), >> sql\init_maximal.sql
echo   quantity DECIMAL(10,4), >> sql\init_maximal.sql
echo   price DECIMAL(10,2), >> sql\init_maximal.sql
echo   strategy VARCHAR(50), >> sql\init_maximal.sql
echo   pnl DECIMAL(10,2), >> sql\init_maximal.sql
echo   confidence DECIMAL(5,3), >> sql\init_maximal.sql
echo   timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP >> sql\init_maximal.sql
echo ); >> sql\init_maximal.sql

echo 🔨 Building MAXIMAL Docker image with all advanced features...
echo    - Python 3.11 with full scientific stack
echo    - Alpaca Trade API + yfinance backup
echo    - Machine Learning libraries (scikit-learn, pandas, numpy)
echo    - Visualization tools (matplotlib, seaborn, plotly)
echo    - Real-time dashboard (Flask, Dash)
echo    - Database integration (PostgreSQL, Redis)
echo    - Advanced analytics and reporting
echo.

docker build -f Dockerfile.maximal -t alpaca-maximal-system . --no-cache

if %errorlevel% neq 0 (
    echo ❌ Build failed! Let's try with cache...
    docker build -f Dockerfile.maximal -t alpaca-maximal-system .
)

if %errorlevel% neq 0 (
    echo ❌ Build still failed!
    echo Checking system resources...
    docker system df
    echo.
    echo Try: docker system prune -a
    pause
    exit /b 1
)

echo.
echo 🚀 Starting MAXIMAL Alpaca Trading Infrastructure...
echo    📊 Main Trading System (Port 8000)
echo    📈 Analytics Dashboard (Port 8001) 
echo    🧠 ML Model API (Port 8002)
echo    📡 WebSocket Feeds (Port 8003)
echo    🗄️  PostgreSQL Database (Port 5432)
echo    💾 Redis Cache (Port 6379)
echo.

docker-compose -f docker-compose-maximal.yml up -d

if %errorlevel% neq 0 (
    echo ❌ Failed to start maximal system!
    echo Checking Docker Compose logs...
    docker-compose -f docker-compose-maximal.yml logs
    pause
    exit /b 1
)

echo.
echo ✅ MAXIMAL SYSTEM LAUNCHED SUCCESSFULLY!
echo.

REM Wait for containers to initialize
echo ⏳ Initializing advanced systems (30 seconds)...
timeout /t 30 >nul

echo 📊 SYSTEM STATUS:
docker-compose -f docker-compose-maximal.yml ps

echo.
echo 🌐 AVAILABLE SERVICES:
echo    📊 Main Dashboard:     http://localhost:8000/dashboard
echo    📈 System Stats:       http://localhost:8000/stats  
echo    ❤️  Health Check:       http://localhost:8000/health
echo    🗄️  Database:           localhost:5432 (maximal_trading/maximal_trader)
echo    💾 Redis Cache:        localhost:6379
echo.

echo 📋 MONITORING COMMANDS:
echo    docker logs -f alpaca-maximal-system           (Live trading logs)
echo    docker logs -f maximal-log-analytics           (System analytics)
echo    docker logs -f maximal-performance-monitor     (Performance stats)
echo    docker-compose -f docker-compose-maximal.yml down  (Stop system)
echo.

echo 🎯 What would you like to do?
echo [1] Watch live maximal trading logs
echo [2] View system analytics  
echo [3] Check performance monitor
echo [4] Open dashboard (requires browser)
echo [5] Show all container status
echo [6] Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo 📊 LIVE MAXIMAL TRADING LOGS
    echo Press Ctrl+C to exit
    echo.
    docker logs -f alpaca-maximal-system
) else if "%choice%"=="2" (
    echo.
    echo 📈 SYSTEM ANALYTICS
    docker logs maximal-log-analytics --tail 50
    echo.
    pause
) else if "%choice%"=="3" (
    echo.
    echo 📊 PERFORMANCE MONITOR
    docker logs maximal-performance-monitor --tail 30
    echo.
    pause
) else if "%choice%"=="4" (
    echo.
    echo 🌐 Opening dashboard...
    start http://localhost:8000/dashboard
    echo Dashboard should open in your browser
    echo.
    pause
) else if "%choice%"=="5" (
    echo.
    echo 📋 CONTAINER STATUS:
    docker ps --filter "name=maximal" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo.
    echo 📊 RESOURCE USAGE:
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    echo.
    pause
)

echo.
echo ========================================
echo 🎉 MAXIMAL ALPACA TRADING SYSTEM READY!
echo ========================================
echo 🚀 12 Advanced AI agents actively trading
echo 📊 Real-time dashboards and analytics
echo 🧠 Machine learning enhanced decisions  
echo 💼 Full-featured paper trading system
echo.
echo 💡 The system runs continuously in Docker
echo 🔍 Monitor progress via logs and dashboard
echo 📈 Reports saved to ./reports/ directory
echo.
echo Happy maximal trading! 🚀💰
pause