@echo off
echo ====================================
echo ALPACA PAPER TRADING - FIXED VERSION
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

echo 🔧 Building SIMPLIFIED Alpaca Docker image (no dependency conflicts)...
docker build -f Dockerfile.simple -t alpaca-paper-trading-simple .

if %errorlevel% neq 0 (
    echo ERROR: Failed to build Docker image!
    echo Try cleaning Docker: docker system prune -a
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Alpaca Paper Trading System (SIMPLIFIED)...
docker-compose -f docker-compose-simple.yml up -d

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
echo 2. View system status: docker-compose -f docker-compose-simple.yml ps
echo 3. View trading monitor: docker logs -f trading-log-monitor
echo 4. Check recent trades: docker logs alpaca-paper-trading | findstr "REAL PAPER TRADE"
echo.
echo 🛑 TO STOP: docker-compose -f docker-compose-simple.yml down
echo.

REM Auto-show logs after 10 seconds
echo Showing live logs in 10 seconds... (Press any key to skip)
timeout /t 10 /nobreak >nul 2>&1
if %errorlevel% equ 0 goto showlogs

REM If user pressed a key, show menu
echo Choose an option:
echo [1] View live Alpaca trading logs
echo [2] View system status
echo [3] View trading monitor
echo [4] Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto showlogs
if "%choice%"=="2" goto showstatus
if "%choice%"=="3" goto showmonitor
goto end

:showlogs
echo.
echo 📈 Showing live Alpaca Paper Trading logs...
echo Press Ctrl+C to exit log view
echo.
docker logs -f alpaca-paper-trading
goto end

:showstatus
echo.
echo 📊 System Status:
docker-compose -f docker-compose-simple.yml ps
echo.
echo Recent Trades:
docker logs alpaca-paper-trading --tail 10 | findstr "REAL PAPER TRADE" 2>nul || echo No trades found yet...
echo.
pause
goto end

:showmonitor
echo.
echo 📈 Trading Monitor Output:
docker logs -f trading-log-monitor
goto end

:end
echo.
echo System is running! Use these commands to monitor:
echo - docker logs -f alpaca-paper-trading
echo - docker-compose -f docker-compose-simple.yml ps
echo - docker-compose -f docker-compose-simple.yml down (to stop)
pause