@echo off
echo ========================================
echo  Rebuilding Competitive Trading Agents
echo ========================================

echo.
echo Step 1: Stopping and cleaning everything...
docker-compose down -v
docker system prune -f

echo.
echo Step 2: Rebuilding all containers from scratch...
docker-compose build --no-cache

echo.
echo Step 3: Starting the system with paper trading...
docker-compose --profile paper up -d

echo.
echo Step 4: Waiting for system to initialize...
timeout /t 10 /nobreak

echo.
echo Step 5: Showing system logs...
docker-compose logs -f trading-system

echo.
echo ========================================
echo  System rebuild complete!
echo ========================================
pause
