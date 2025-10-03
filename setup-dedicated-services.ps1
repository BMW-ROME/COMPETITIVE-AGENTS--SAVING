# Competitive Trading Agents - Safe Setup Script
# ===============================================

Write-Host "========================================" -ForegroundColor Green
Write-Host "COMPETITIVE TRADING AGENTS - SAFE SETUP" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "This script will create dedicated PostgreSQL and Redis containers" -ForegroundColor Yellow
Write-Host "for the Competitive Trading Agents project." -ForegroundColor Yellow
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Cyan
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Stop existing containers if they exist
Write-Host ""
Write-Host "Stopping existing containers..." -ForegroundColor Cyan
docker stop competitive-trading-postgres competitive-trading-redis 2>$null
docker rm competitive-trading-postgres competitive-trading-redis 2>$null

# Create network
Write-Host ""
Write-Host "Creating network..." -ForegroundColor Cyan
docker network create competitive-trading-network 2>$null
Write-Host "✅ Network created" -ForegroundColor Green

# Create PostgreSQL container
Write-Host ""
Write-Host "Creating PostgreSQL container..." -ForegroundColor Cyan
docker run -d `
  --name competitive-trading-postgres `
  --network competitive-trading-network `
  -e POSTGRES_DB=competitive_trading_agents `
  -e POSTGRES_USER=trading_user `
  -e POSTGRES_PASSWORD=trading_password `
  -p 5433:5432 `
  -v competitive_postgres_data:/var/lib/postgresql/data `
  -v "${PWD}/init.sql:/docker-entrypoint-initdb.d/init.sql" `
  --restart unless-stopped `
  postgres:15-alpine

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PostgreSQL container created" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create PostgreSQL container" -ForegroundColor Red
    exit 1
}

# Create Redis container
Write-Host ""
Write-Host "Creating Redis container..." -ForegroundColor Cyan
docker run -d `
  --name competitive-trading-redis `
  --network competitive-trading-network `
  -p 6380:6379 `
  -v competitive_redis_data:/data `
  --restart unless-stopped `
  redis:7-alpine

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Redis container created" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create Redis container" -ForegroundColor Red
    exit 1
}

# Wait for services to start
Write-Host ""
Write-Host "Waiting for services to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Test connections
Write-Host ""
Write-Host "Testing connections..." -ForegroundColor Cyan

# Test PostgreSQL
Write-Host "Testing PostgreSQL..." -ForegroundColor Yellow
try {
    $pgTest = docker exec competitive-trading-postgres psql -U trading_user -d competitive_trading_agents -c "SELECT version();" 2>$null
    if ($pgTest) {
        Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️ PostgreSQL is starting up..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ PostgreSQL is still starting up..." -ForegroundColor Yellow
}

# Test Redis
Write-Host "Testing Redis..." -ForegroundColor Yellow
try {
    $redisTest = docker exec competitive-trading-redis redis-cli ping 2>$null
    if ($redisTest -eq "PONG") {
        Write-Host "✅ Redis is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Redis is starting up..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Redis is still starting up..." -ForegroundColor Yellow
}

# Show status
Write-Host ""
Write-Host "Container status:" -ForegroundColor Cyan
docker ps --filter "name=competitive-trading"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "PostgreSQL: localhost:5433" -ForegroundColor White
Write-Host "Redis: localhost:6380" -ForegroundColor White
Write-Host ""
Write-Host "Database: competitive_trading_agents" -ForegroundColor White
Write-Host "User: trading_user" -ForegroundColor White
Write-Host "Password: trading_password" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Update your credentials in dedicated-config.env" -ForegroundColor White
Write-Host "2. Run: docker-compose -f docker-compose-dedicated.yml up --build -d" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

