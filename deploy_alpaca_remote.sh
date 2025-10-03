#!/bin/bash
# Remote Alpaca Paper Trading Deployment Script
# ==============================================
# Deploys the Alpaca Paper Trading system with comprehensive logging

# Configuration
LOG_DIR="/workspaces/competitive-trading-agents/logs"
DATA_DIR="/workspaces/competitive-trading-agents/data"
REPORTS_DIR="/workspaces/competitive-trading-agents/reports"
ALPACA_LOG="$LOG_DIR/alpaca_paper_trading.log"
SYSTEM_LOG="$LOG_DIR/system_status.log"
PID_FILE="/tmp/alpaca_paper_trading.pid"

# Create directories
mkdir -p "$LOG_DIR" "$DATA_DIR" "$REPORTS_DIR"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SYSTEM_LOG"
}

# Function to start Alpaca Paper Trading
start_alpaca_trading() {
    log_with_timestamp "🚀 Starting Alpaca Paper Trading System..."
    cd /workspaces/competitive-trading-agents
    
    # Start the system in background with comprehensive logging
    nohup /workspaces/competitive-trading-agents/.venv/bin/python alpaca_paper_trading.py \
        > "$ALPACA_LOG" 2>&1 &
    
    ALPACA_PID=$!
    echo $ALPACA_PID > "$PID_FILE"
    
    log_with_timestamp "✅ Alpaca Paper Trading started with PID: $ALPACA_PID"
    log_with_timestamp "📝 Logs being written to: $ALPACA_LOG"
}

# Function to monitor the system
monitor_system() {
    log_with_timestamp "🔍 Starting system monitoring..."
    
    while true; do
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                log_with_timestamp "✅ Alpaca Paper Trading (PID: $PID) is running"
            else
                log_with_timestamp "❌ Alpaca Paper Trading process not found. Restarting..."
                start_alpaca_trading
            fi
        else
            log_with_timestamp "🆕 Starting Alpaca Paper Trading for first time..."
            start_alpaca_trading
        fi
        
        # Log recent activity
        if [ -f "$ALPACA_LOG" ]; then
            RECENT_TRADES=$(tail -n 50 "$ALPACA_LOG" | grep -c "FILLED\|EXECUTED" || echo "0")
            RECENT_DECISIONS=$(tail -n 50 "$ALPACA_LOG" | grep -c "AGENT DECISIONS" || echo "0")
            log_with_timestamp "📊 Recent activity: $RECENT_TRADES trades, $RECENT_DECISIONS decision cycles"
        fi
        
        # Wait 5 minutes before next check
        sleep 300
    done
}

# Function to show current status
show_status() {
    log_with_timestamp "📊 ALPACA PAPER TRADING STATUS"
    log_with_timestamp "==============================="
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            log_with_timestamp "✅ System Status: RUNNING (PID: $PID)"
        else
            log_with_timestamp "❌ System Status: STOPPED"
        fi
    else
        log_with_timestamp "❌ System Status: NOT STARTED"
    fi
    
    if [ -f "$ALPACA_LOG" ]; then
        TOTAL_LINES=$(wc -l < "$ALPACA_LOG")
        TOTAL_TRADES=$(grep -c "FILLED\|EXECUTED" "$ALPACA_LOG" || echo "0")
        TOTAL_CYCLES=$(grep -c "PAPER TRADING CYCLE" "$ALPACA_LOG" || echo "0")
        
        log_with_timestamp "📝 Log entries: $TOTAL_LINES"
        log_with_timestamp "💰 Total trades: $TOTAL_TRADES"
        log_with_timestamp "🔄 Trading cycles: $TOTAL_CYCLES"
        
        log_with_timestamp "📈 Recent activity (last 20 lines):"
        tail -n 20 "$ALPACA_LOG" | while IFS= read -r line; do
            log_with_timestamp "   $line"
        done
    fi
}

# Function to stop the system
stop_system() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            log_with_timestamp "🛑 Stopping Alpaca Paper Trading (PID: $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            log_with_timestamp "✅ System stopped successfully"
        else
            log_with_timestamp "❌ Process not running"
            rm -f "$PID_FILE"
        fi
    else
        log_with_timestamp "❌ No PID file found"
    fi
}

# Main execution
case "$1" in
    "start")
        start_alpaca_trading
        ;;
    "stop")
        stop_system
        ;;
    "restart")
        stop_system
        sleep 3
        start_alpaca_trading
        ;;
    "status")
        show_status
        ;;
    "monitor")
        monitor_system
        ;;
    "logs")
        if [ -f "$ALPACA_LOG" ]; then
            tail -f "$ALPACA_LOG"
        else
            echo "No log file found at $ALPACA_LOG"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|monitor|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the Alpaca Paper Trading system"
        echo "  stop    - Stop the system"
        echo "  restart - Restart the system"
        echo "  status  - Show current system status"
        echo "  monitor - Start continuous monitoring"
        echo "  logs    - Follow the live logs"
        exit 1
        ;;
esac