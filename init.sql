-- Initialize PostgreSQL database for trading system


-- Create tables for trading data
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity DECIMAL(15, 8) NOT NULL,
    price DECIMAL(15, 8) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence DECIMAL(5, 4),
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_return DECIMAL(10, 6),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    total_trades INTEGER,
    successful_trades INTEGER,
    failed_trades INTEGER
);

CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trading_type VARCHAR(20) DEFAULT 'paper'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_trades_agent_id ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_agent_id ON performance_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_trading_type ON system_logs(trading_type);
