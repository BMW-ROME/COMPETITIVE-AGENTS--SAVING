"""
Configuration settings for the competitive trading agents system.
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"

class AgentType(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"

@dataclass
class AlpacaConfig:
    """Alpaca trading platform configuration."""
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    data_url: str = "https://data.alpaca.markets"
    
@dataclass
class DataSourceConfig:
    """Configuration for various data sources."""
    # News APIs
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    
    # Social Media
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    
    # Market Data
    yahoo_finance_enabled: bool = True
    alpha_vantage_enabled: bool = True

@dataclass
class AgentConfig:
    """Configuration for individual trading agents."""
    agent_id: str
    agent_type: AgentType
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    risk_tolerance: float = 0.05  # 5% max loss per trade
    learning_rate: float = 0.01
    memory_size: int = 1000
    update_frequency: int = 60  # seconds

@dataclass
class HierarchyConfig:
    """Configuration for the hierarchy manager."""
    evaluation_interval: int = 300  # 5 minutes
    performance_metrics: List[str] = None
    communication_interval: int = 30  # seconds
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = [
                "sharpe_ratio",
                "max_drawdown", 
                "win_rate",
                "profit_factor",
                "total_return"
            ]

@dataclass
class SystemConfig:
    """Main system configuration."""
    # Trading
    trading_mode: TradingMode = TradingMode.PAPER
    trading_symbols: List[str] = None
    trading_hours: Dict[str, str] = None
    
    # Agents
    num_agents: int = 2
    agent_configs: List[AgentConfig] = None
    
    # Data
    data_update_interval: int = 30  # seconds
    sentiment_analysis_enabled: bool = True
    news_analysis_enabled: bool = True
    
    # System
    log_level: str = "INFO"
    database_url: str = "sqlite:///trading_agents.db"
    redis_url: str = "redis://localhost:6379"
    
    def __post_init__(self):
        if self.trading_symbols is None:
            self.trading_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        if self.trading_hours is None:
            self.trading_hours = {
                "start": "09:30",
                "end": "16:00",
                "timezone": "US/Eastern"
            }
        
        if self.agent_configs is None:
            self.agent_configs = [
                AgentConfig("agent_1", AgentType.CONSERVATIVE),
                AgentConfig("agent_2", AgentType.AGGRESSIVE)
            ]

# Global configuration instance
config = SystemConfig()
alpaca_config = AlpacaConfig()
data_config = DataSourceConfig()
hierarchy_config = HierarchyConfig()
