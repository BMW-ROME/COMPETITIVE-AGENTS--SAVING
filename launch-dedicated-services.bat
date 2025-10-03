@echo off
echo ========================================
echo COMPETITIVE TRADING AGENTS - DEDICATED SERVICES
echo ========================================

echo.
echo Starting dedicated PostgreSQL and Redis containers...
echo.

REM Stop existing containers if running
echo Stopping existing containers...
docker stop competitive-trading-postgres competitive-trading-redis 2>nul
docker rm competitive-trading-postgres competitive-trading-redis 2>nul

echo.
echo Starting PostgreSQL container...
docker run -d ^
  --name competitive-trading-postgres ^
  --network competitive-trading-network ^
  -e POSTGRES_DB=competitive_trading_agents ^
  -e POSTGRES_USER=trading_user ^
  -e POSTGRES_PASSWORD=trading_password ^
  -p 5433:5432 ^
  -v competitive_postgres_data:/var/lib/postgresql/data ^
  -v %cd%\init.sql:/docker-entrypoint-initdb.d/init.sql ^
  --restart unless-stopped ^
  postgres:15-alpine

echo.
echo Starting Redis container...
docker run -d ^
  --name competitive-trading-redis ^
  --network competitive-trading-network ^
  -p 6380:6379 ^
  -v competitive_redis_data:/data ^
  --restart unless-stopped ^
  redis:7-alpine

echo.
echo Creating network if it doesn't exist...
docker network create competitive-trading-network 2>nul

echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo Checking container status...
docker ps --filter "name=competitive-trading"

echo.
echo ========================================
echo SERVICES STARTED SUCCESSFULLY!
echo ========================================
echo.
echo PostgreSQL: localhost:5433
echo Redis: localhost:6380
echo.
echo Database: competitive_trading_agents
echo User: trading_user
echo Password: trading_password
echo.
echo You can now run the trading system with:
echo docker-compose -f docker-compose-dedicated.yml up --build -d
echo.
pause

