@echo off
echo ====================================
echo ALPACA PAPER TRADING - FINAL FIX
echo ====================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is running...

REM Navigate to script directory
cd /d "%~dp0"

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo 🔧 Creating .env file...
    echo # Alpaca Paper Trading Credentials > .env
    echo ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF >> .env
    echo ALPACA_SECRET_KEY=your_secret_key_here >> .env
    echo ALPACA_BASE_URL=https://paper-api.alpaca.markets >> .env
    echo.
    echo ⚠️  IMPORTANT: Please edit the .env file with your real Alpaca secret key!
    echo    Your API key is already set, but you need to add your SECRET_KEY.
    echo.
    pause
)

REM Create directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data  
if not exist "reports" mkdir reports

echo 🔨 Building Docker image...
docker build -f Dockerfile.simple -t alpaca-paper-trading-simple .

if %errorlevel% neq 0 (
    echo ❌ Build failed! Trying to fix...
    docker system prune -f
    docker build -f Dockerfile.simple -t alpaca-paper-trading-simple .
)

if %errorlevel% neq 0 (
    echo ❌ ERROR: Build still failing!
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Alpaca Paper Trading System...
docker-compose -f docker-compose-simple.yml up -d

if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to start!
    pause
    exit /b 1
)

echo.
echo ✅ SUCCESS! System is running!
echo.

REM Wait a moment for container to start
echo Checking container status...
timeout /t 5 >nul

docker ps --filter "name=alpaca-paper-trading"

echo.
echo 📊 QUICK COMMANDS:
echo docker logs -f alpaca-paper-trading     (Live logs)
echo docker ps                               (Container status)
echo docker-compose -f docker-compose-simple.yml down  (Stop system)
echo.

REM Show recent logs
echo 📈 Recent activity:
docker logs --tail 20 alpaca-paper-trading 2>nul

echo.
echo 🎯 Would you like to:
echo [1] Watch live logs
echo [2] Exit
echo.

set /p choice="Choose (1-2): "

if "%choice%"=="1" (
    echo.
    echo 📊 Live logs starting... Press Ctrl+C to exit
    echo.
    docker logs -f alpaca-paper-trading
)

echo.
echo 🎉 Your Alpaca Paper Trading system is running in Docker!
pause