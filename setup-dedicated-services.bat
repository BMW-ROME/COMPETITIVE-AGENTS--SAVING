@echo off
echo ========================================
echo COMPETITIVE TRADING AGENTS - SAFE SETUP
echo ========================================
echo.
echo This script will create dedicated PostgreSQL and Redis containers
echo for the Competitive Trading Agents project.
echo.

REM Check if Docker is running
echo Checking Docker status...
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo Docker is running.

REM Stop existing containers if they exist
echo.
echo Stopping existing containers...
docker stop competitive-trading-postgres competitive-trading-redis >nul 2>&1
docker rm competitive-trading-postgres competitive-trading-redis >nul 2>&1

REM Create network
echo.
echo Creating network...
docker network create competitive-trading-network >nul 2>&1
echo Network created.

REM Create PostgreSQL container
echo.
echo Creating PostgreSQL container...
docker run -d ^
  --name competitive-trading-postgres ^
  --network competitive-trading-network ^
  -e POSTGRES_DB=competitive_trading_agents ^
  -e POSTGRES_USER=trading_user ^
  -e POSTGRES_PASSWORD=trading_password ^
  -p 5433:5432 ^
  -v competitive_postgres_data:/var/lib/postgresql/data ^
  -v "%cd%\init.sql:/docker-entrypoint-initdb.d/init.sql" ^
  --restart unless-stopped ^
  postgres:15-alpine

if %errorlevel% neq 0 (
    echo ERROR: Failed to create PostgreSQL container
    pause
    exit /b 1
)
echo PostgreSQL container created.

REM Create Redis container
echo.
echo Creating Redis container...
docker run -d ^
  --name competitive-trading-redis ^
  --network competitive-trading-network ^
  -p 6380:6379 ^
  -v competitive_redis_data:/data ^
  --restart unless-stopped ^
  redis:7-alpine

if %errorlevel% neq 0 (
    echo ERROR: Failed to create Redis container
    pause
    exit /b 1
)
echo Redis container created.

REM Wait for services to start
echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Test connections
echo.
echo Testing connections...

REM Test PostgreSQL
echo Testing PostgreSQL...
docker exec competitive-trading-postgres psql -U trading_user -d competitive_trading_agents -c "SELECT version();" >nul 2>&1
if %errorlevel% equ 0 (
    echo PostgreSQL is running.
) else (
    echo PostgreSQL is starting up...
)

REM Test Redis
echo Testing Redis...
docker exec competitive-trading-redis redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo Redis is running.
) else (
    echo Redis is starting up...
)

REM Show status
echo.
echo Container status:
docker ps --filter "name=competitive-trading"

echo.
echo ========================================
echo SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo PostgreSQL: localhost:5433
echo Redis: localhost:6380
echo.
echo Database: competitive_trading_agents
echo User: trading_user
echo Password: trading_password
echo.
echo Next steps:
echo 1. Update your credentials in dedicated-config.env
echo 2. Run: docker-compose -f docker-compose-dedicated.yml up --build -d
echo.
pause

