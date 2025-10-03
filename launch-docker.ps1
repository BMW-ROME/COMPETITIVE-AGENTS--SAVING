# Alpaca Paper Trading - PowerShell Docker Launcher
# Run this in PowerShell as Administrator

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "ALPACA PAPER TRADING DOCKER LAUNCHER" -ForegroundColor Cyan  
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running..." -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Docker is not running or not installed!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Navigate to script directory
Set-Location -Path $PSScriptRoot

# Create necessary directories
@("logs", "data", "reports") | ForEach-Object {
    if (!(Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}

Write-Host ""
Write-Host "🔨 Building Alpaca Paper Trading Docker image..." -ForegroundColor Yellow
docker build -f Dockerfile.alpaca -t alpaca-paper-trading .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Failed to build Docker image!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting Alpaca Paper Trading System..." -ForegroundColor Yellow
docker-compose -f docker-compose-local.yml up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Failed to start system!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "✅ System started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 MONITORING OPTIONS:" -ForegroundColor Cyan
Write-Host "1. View live Alpaca logs: docker logs -f alpaca-paper-trading"
Write-Host "2. View system status: docker-compose -f docker-compose-local.yml ps" 
Write-Host "3. View trading monitor: docker logs -f trading-log-monitor"
Write-Host ""
Write-Host "🛑 TO STOP: docker-compose -f docker-compose-local.yml down" -ForegroundColor Red
Write-Host ""

# Menu options
Write-Host "Choose an option:" -ForegroundColor Cyan
Write-Host "[1] View live Alpaca trading logs"
Write-Host "[2] View system status"
Write-Host "[3] View trading monitor" 
Write-Host "[4] Exit"
Write-Host ""

$choice = Read-Host "Enter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "📈 Showing live Alpaca Paper Trading logs..." -ForegroundColor Green
        Write-Host "Press Ctrl+C to exit log view" -ForegroundColor Yellow
        Write-Host ""
        docker logs -f alpaca-paper-trading
    }
    "2" {
        Write-Host ""
        Write-Host "📊 System Status:" -ForegroundColor Green
        docker-compose -f docker-compose-local.yml ps
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    "3" {
        Write-Host ""
        Write-Host "📈 Trading Monitor Output:" -ForegroundColor Green
        docker logs -f trading-log-monitor
    }
    default {
        Write-Host ""
        Write-Host "System is running in background!" -ForegroundColor Green
        Write-Host "Use the commands shown above to monitor." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Read-Host "Press Enter to exit"