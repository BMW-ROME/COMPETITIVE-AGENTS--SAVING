@echo off
echo ====================================
echo ALPACA PAPER TRADING DOCKER LAUNCHER
echo ====================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is running...
echo.

REM Navigate to script directory
cd /d "%~dp0"

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "reports" mkdir reports

echo 🔨 Building Alpaca Paper Trading Docker image...
docker build -f Dockerfile.alpaca -t alpaca-paper-trading .

if %errorlevel% neq 0 (
    echo ERROR: Failed to build Docker image!
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Alpaca Paper Trading System...
docker-compose -f docker-compose-local.yml up -d

if %errorlevel% neq 0 (
    echo ERROR: Failed to start system!
    pause
    exit /b 1
)

echo.
echo ✅ System started successfully!
echo.
echo 📊 MONITORING OPTIONS:
echo 1. View live Alpaca logs: docker logs -f alpaca-paper-trading
echo 2. View system status: docker-compose -f docker-compose-local.yml ps
echo 3. View trading monitor: docker logs -f trading-log-monitor
echo.
echo 🛑 TO STOP: docker-compose -f docker-compose-local.yml down
echo.

REM Ask user what they want to do
echo Choose an option:
echo [1] View live Alpaca trading logs
echo [2] View system status
echo [3] View trading monitor
echo [4] Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 📈 Showing live Alpaca Paper Trading logs...
    echo Press Ctrl+C to exit log view
    echo.
    docker logs -f alpaca-paper-trading
) else if "%choice%"=="2" (
    echo.
    echo 📊 System Status:
    docker-compose -f docker-compose-local.yml ps
    echo.
    pause
) else if "%choice%"=="3" (
    echo.
    echo 📈 Trading Monitor Output:
    docker logs -f trading-log-monitor
) else (
    echo.
    echo System is running in background!
    echo Use the commands shown above to monitor.
)

echo.
echo Done!
pause