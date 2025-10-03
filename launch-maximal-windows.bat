@echo off
echo 🚀 LAUNCHING MAXIMAL ALPACA DOCKER SYSTEM
echo ========================================

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker compose -f docker-compose-maximal.yml down >nul 2>&1

REM Clean up
echo 🧹 Cleaning up...
docker system prune -f >nul 2>&1

REM Build fresh images
echo 🔨 Building maximal Docker image...
docker compose -f docker-compose-maximal.yml build --no-cache

if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)

echo ✅ Build successful!

REM Start the services
echo 🚀 Starting maximal services...
docker compose -f docker-compose-maximal.yml up -d

if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    pause
    exit /b 1
)

echo.
echo 🎉 MAXIMAL SYSTEM DEPLOYED SUCCESSFULLY!
echo ========================================
echo 📊 Dashboard: http://localhost:8000
echo 📈 Analytics: http://localhost:8001
echo 🤖 ML API: http://localhost:8002
echo 📡 WebSocket: http://localhost:8003
echo.
echo 📋 View logs with:
echo    docker compose -f docker-compose-maximal.yml logs -f
echo.
echo ⏹️ Stop system with:
echo    docker compose -f docker-compose-maximal.yml down
echo.
pause