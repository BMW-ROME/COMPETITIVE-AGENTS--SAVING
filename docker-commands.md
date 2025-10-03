# 🐳 Docker Commands Guide

## Quick Start Commands

### Paper Trading Only (Default)
```bash
# Start paper trading
docker-compose --profile paper up trading-system

# Start paper trading in background
docker-compose --profile paper up -d trading-system

# View paper trading logs
docker-compose --profile paper logs -f trading-system
```

### Live Trading (Requires Confirmation)
```bash
# Start live trading (with confirmation prompt)
docker-compose --profile live up trading-system-live

# Start live trading in background
docker-compose --profile live up -d trading-system-live

# View live trading logs
docker-compose --profile live logs -f trading-system-live
```

### Full System (Paper + Monitoring)
```bash
# Start paper trading with monitoring dashboard
docker-compose --profile paper up

# Start in background
docker-compose --profile paper up -d

# View all logs
docker-compose --profile paper logs -f
```

### Full System with Live Trading
```bash
# Start everything including live trading
docker-compose --profile live up

# Start in background
docker-compose --profile live up -d
```

## Management Commands

### Stop Services
```bash
# Stop paper trading
docker-compose --profile paper stop trading-system

# Stop live trading
docker-compose --profile live stop trading-system-live

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### View Status
```bash
# Check running containers
docker-compose ps

# Check container health
docker-compose ps --services --filter "health=healthy"
```

### Access Containers
```bash
# Access paper trading container
docker-compose --profile paper exec trading-system bash

# Access live trading container
docker-compose --profile live exec trading-system-live bash

# View logs in real-time
docker-compose --profile paper logs -f trading-system
docker-compose --profile live logs -f trading-system-live
```

## Monitoring

### Web Dashboard
- **Paper Trading**: http://localhost:8000
- **Live Trading**: http://localhost:8001
- **API Status**: http://localhost:8000/api/status

### Log Files
```bash
# View paper trading logs
tail -f logs/paper_trading.log

# View live trading logs
tail -f logs/live_trading.log

# View all logs
tail -f logs/*.log
```

## Environment Setup

### Paper Trading (.env file)
```bash
# Your existing .env file works for paper trading
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
```

### Live Trading (.env.live file)
```bash
# Create .env.live for live trading
ALPACA_API_KEY=your_live_api_key
ALPACA_SECRET_KEY=your_live_secret_key
```

## Safety Features

### Paper Trading Container
- ✅ Safe by default
- ✅ Uses paper API keys
- ✅ No real money at risk
- ✅ Can run continuously

### Live Trading Container
- ⚠️ Requires explicit profile activation
- ⚠️ Requires live API keys
- ⚠️ Real money at risk
- ⚠️ Confirmation prompt before starting
- ⚠️ Conservative risk settings

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose --profile paper logs trading-system

# Rebuild container
docker-compose build trading-system

# Force recreate
docker-compose --profile paper up --force-recreate trading-system
```

### API Keys Not Working
```bash
# Check environment variables
docker-compose --profile paper exec trading-system env | grep ALPACA

# Verify .env file
cat .env
```

### Database Issues
```bash
# Reset database
docker-compose down -v
docker-compose up postgres

# Check database connection
docker-compose exec postgres psql -U trader -d trading_agents
```

## Performance Monitoring

### Resource Usage
```bash
# Check container resource usage
docker stats

# Check specific container
docker stats trading-agents-system
docker stats trading-agents-live
```

### Health Checks
```bash
# Check health status
docker-compose ps

# Manual health check
curl http://localhost:8000/health
```

## Backup and Restore

### Backup Data
```bash
# Backup database
docker-compose exec postgres pg_dump -U trader trading_agents > backup.sql

# Backup logs
tar -czf logs-backup.tar.gz logs/
```

### Restore Data
```bash
# Restore database
docker-compose exec -T postgres psql -U trader trading_agents < backup.sql
```

## Development

### Rebuild After Code Changes
```bash
# Rebuild and restart
docker-compose --profile paper up --build trading-system

# Rebuild specific service
docker-compose build trading-system
docker-compose --profile paper up trading-system
```

### Debug Mode
```bash
# Run with debug logging
docker-compose --profile paper up trading-system --env-file .env.debug
```
