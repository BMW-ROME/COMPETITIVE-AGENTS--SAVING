@echo off
echo ====================================
echo MINIMAL ALPACA PAPER TRADING - FINAL
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
    echo # Minimal Alpaca Paper Trading Credentials > .env
    echo ALPACA_API_KEY=PKK43GTIACJNUPGZPCPF >> .env
    echo ALPACA_SECRET_KEY=your_secret_key_here >> .env
    echo ALPACA_BASE_URL=https://paper-api.alpaca.markets >> .env
    echo.
    echo ⚠️  IMPORTANT: Edit .env with your real Alpaca secret key!
    echo.
)

REM Create directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data  
if not exist "reports" mkdir reports

echo 🔨 Building MINIMAL Docker image (Python 3.11 + requests only)...
docker build -f Dockerfile.minimal -t alpaca-minimal .

if %errorlevel% neq 0 (
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Minimal Alpaca System...
docker-compose -f docker-compose-minimal.yml up -d

if %errorlevel% neq 0 (
    echo ❌ Failed to start!
    pause
    exit /b 1
)

echo.
echo ✅ SUCCESS! Minimal system running!
echo.

REM Wait for container to start
echo Checking containers...
timeout /t 5 >nul

docker ps --filter "name=alpaca-minimal"

echo.
echo 📊 COMMANDS:
echo docker logs -f alpaca-minimal          (Live logs)
echo docker logs minimal-log-monitor        (Trading summary)
echo docker-compose -f docker-compose-minimal.yml down  (Stop)
echo.

echo 🎯 Would you like to:
echo [1] Watch live logs
echo [2] Show recent activity
echo [3] Exit
echo.

set /p choice="Choose (1-3): "

if "%choice%"=="1" (
    echo.
    echo 📊 Live minimal logs... Press Ctrl+C to exit
    echo.
    docker logs -f alpaca-minimal
) else if "%choice%"=="2" (
    echo.
    echo 📈 Recent activity:
    docker logs --tail 20 alpaca-minimal 2>nul
    echo.
    pause
)

echo.
echo 🎉 Minimal Alpaca Paper Trading running in Docker!
echo 💡 This version uses only Python 3.11 + requests (no complex dependencies)
pause