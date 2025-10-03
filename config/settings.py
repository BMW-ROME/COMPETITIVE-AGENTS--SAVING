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
    FRACTAL_ANALYSIS = "fractal_analysis"
    CANDLE_RANGE_THEORY = "candle_range_theory"
    QUANTITATIVE_PATTERN = "quantitative_pattern"
    FOREX_SPECIALIST = "forex_specialist"
    CRYPTO_SPECIALIST = "crypto_specialist"
    MULTI_ASSET_ARBITRAGE = "multi_asset_arbitrage"
    AI_ENHANCED = "ai_enhanced"

@dataclass
class AlpacaConfig:
    """Alpaca trading platform configuration."""
    api_key: str = ""
    secret_key: str = ""
    
    # 🔄 EASY SWITCHING: Comment/uncomment or override with ALPACA_BASE_URL env
    base_url: str = "https://paper-api.alpaca.markets"  # ✅ PAPER TRADING (SAFE)
    # base_url: str = "https://api.alpaca.markets"      # 🚨 LIVE TRADING (REAL MONEY)
    
    data_url: str = "https://data.alpaca.markets"
    
    # Multiple exchange support
    exchanges: Optional[List[str]] = None
    
    def __post_init__(self):
        # Load environment variables if not already set
        if not self.api_key:
            self.api_key = os.getenv("ALPACA_API_KEY", "")
        if not self.secret_key:
            self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        # Allow overriding URLs via environment
        self.base_url = os.getenv("ALPACA_BASE_URL", self.base_url)
        self.data_url = os.getenv("ALPACA_DATA_URL", self.data_url)
        # Default exchanges list when not provided
        if self.exchanges is None:
            self.exchanges = [
                # Stock Exchanges
                "NASDAQ",
                "NYSE",
                "ARCA",
                "BATS",
                "IEX",
                # FOREX
                "FOREX",
                # Crypto
                "CRYPTO",
            ]
        
# (exchanges default handled in AlpacaConfig.__post_init__)
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
    # Add FOREX and Crypto data providers
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    oanda_api_key: str = os.getenv("OANDA_API_KEY", "")
    coinbase_api_key: str = os.getenv("COINBASE_API_KEY", "")
    coinbase_api_secret: str = os.getenv("COINBASE_API_SECRET", "")
    forex_enabled: bool = True
    crypto_enabled: bool = True

    # Optional MCP servers (external tool adapters)
    hf_mcp_url: str = os.getenv("HF_MCP_URL", "")  # Hugging Face MCP endpoint base
    playwright_mcp_url: str = os.getenv("PLAYWRIGHT_MCP_URL", "")  # Playwright MCP endpoint base
    yt_mcp_url: str = os.getenv("YOUTUBE_MCP_URL", "")  # YouTube transcripts MCP endpoint base
    cockroach_dsn: str = os.getenv("COCKROACH_DSN", "")  # CockroachDB DSN if available

    # Default HF model choices (overridable via env)
    hf_model_fin_sentiment: str = os.getenv("HF_MODEL_FIN_SENTIMENT", "ProsusAI/finbert")
    hf_model_social_sentiment: str = os.getenv("HF_MODEL_SOCIAL_SENTIMENT", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    hf_model_zeroshot: str = os.getenv("HF_MODEL_ZEROSHOT", "facebook/bart-large-mnli")
    hf_model_summarizer: str = os.getenv("HF_MODEL_SUMMARIZER", "sshleifer/distilbart-cnn-12-6")
    hf_model_embeddings: str = os.getenv("HF_MODEL_EMBEDDINGS", "sentence-transformers/all-MiniLM-L6-v2")

    # Optional list of YouTube video IDs to fetch transcripts for
    youtube_video_ids: List[str] = None

    def __post_init__(self):
        if self.youtube_video_ids is None:
            raw = os.getenv("YOUTUBE_VIDEO_IDS", "").strip()
            self.youtube_video_ids = [v.strip() for v in raw.split(",") if v.strip()] if raw else []

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
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///trading_agents.db")
    redis_url: str = "redis://localhost:6379"
    data_dir: str = os.getenv("DATA_DIR", "/tmp/cta-data")
    
    def __post_init__(self):
        if self.trading_symbols is None:
            # Expanded symbol list across different asset classes
            self.trading_symbols = [
                # Technology
                "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "ADBE", "CRM",
                # Financial
                "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "V",
                # Healthcare
                "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
                # Consumer
                "KO", "PEP", "WMT", "PG", "JNJ", "HD", "MCD", "NKE", "SBUX", "TGT",
                # Energy
                "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "VLO", "PSX", "KMI",
                # Industrial
                "BA", "CAT", "GE", "HON", "MMM", "UPS", "FDX", "LMT", "RTX", "DE",
                # ETFs for diversification
                "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "BND", "TLT", "GLD", "SLV",
                # FOREX - Major Currency Pairs
                "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
                "EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "AUDJPY", "CADJPY", "NZDJPY",
                "EURAUD", "EURCAD", "GBPAUD", "GBPCAD", "AUDCAD", "AUDNZD", "CADCHF",
                # FOREX - Minor Currency Pairs
                "EURCHF", "GBPCHF", "AUDCHF", "NZDCHF", "EURNZD", "GBPNZD", "NZDCAD",
                # CRYPTO - Major Cryptocurrencies
                "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD", "DOTUSD", "MATICUSD", "AVAXUSD",
                "LINKUSD", "UNIUSD", "AAVEUSD", "SUSHIUSD", "CRVUSD", "COMPUSD", "MKRUSD",
                "YFIUSD", "SNXUSD", "BALUSD", "LRCUSD", "1INCHUSD", "ALPHAUSD", "BANDUSD",
                # CRYPTO - Altcoins
                "DOGEUSD", "SHIBUSD", "XRPUSD", "LTCUSD", "BCHUSD", "ETCUSD", "XLMUSD",
                "VETUSD", "TRXUSD", "EOSUSD", "NEOUSD", "IOTAUSD", "DASHUSD", "ZECUSD",
                # CRYPTO - DeFi Tokens
                "SUSHIUSD", "UNIUSD", "AAVEUSD", "COMPUSD", "MKRUSD", "YFIUSD", "SNXUSD",
                "BALUSD", "LRCUSD", "1INCHUSD", "ALPHAUSD", "BANDUSD", "KNCUSD", "RENUSD"
            ]
        
        if self.trading_hours is None:
            self.trading_hours = {
                "start": "09:30",
                "end": "16:00",
                "timezone": "US/Eastern"
            }

        # Ensure data dir exists
        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception:
            pass
        
        if self.agent_configs is None:
            self.agent_configs = [
                # Traditional Stock Agents
                AgentConfig("conservative_1", AgentType.CONSERVATIVE, initial_capital=25000.0),
                AgentConfig("aggressive_1", AgentType.AGGRESSIVE, initial_capital=25000.0),
                AgentConfig("balanced_1", AgentType.BALANCED, initial_capital=25000.0),
                # Advanced Technical Analysis Agents
                AgentConfig("fractal_1", AgentType.FRACTAL_ANALYSIS, initial_capital=25000.0),
                AgentConfig("candle_range_1", AgentType.CANDLE_RANGE_THEORY, initial_capital=25000.0),
                AgentConfig("quant_pattern_1", AgentType.QUANTITATIVE_PATTERN, initial_capital=25000.0),
                # FOREX Specialized Agents
                AgentConfig("forex_major_1", AgentType.FOREX_SPECIALIST, initial_capital=30000.0, 
                           risk_tolerance=0.03, max_position_size=0.15),  # Lower risk for FOREX
                AgentConfig("forex_minor_1", AgentType.FOREX_SPECIALIST, initial_capital=20000.0,
                           risk_tolerance=0.05, max_position_size=0.10),  # Higher risk for minor pairs
                # Crypto Specialized Agents
                AgentConfig("crypto_major_1", AgentType.CRYPTO_SPECIALIST, initial_capital=35000.0,
                           risk_tolerance=0.08, max_position_size=0.12),  # Higher risk for crypto
                AgentConfig("crypto_defi_1", AgentType.CRYPTO_SPECIALIST, initial_capital=20000.0,
                           risk_tolerance=0.12, max_position_size=0.08),  # Highest risk for DeFi
                # Multi-Asset Arbitrage Agent
                AgentConfig("multi_asset_arb_1", AgentType.MULTI_ASSET_ARBITRAGE, initial_capital=40000.0,
                           risk_tolerance=0.06, max_position_size=0.20),  # Balanced risk for arbitrage
                # AI-Enhanced Agent
                AgentConfig("ai_enhanced_1", AgentType.AI_ENHANCED, initial_capital=50000.0,
                           risk_tolerance=0.04, max_position_size=0.15)  # AI-powered with moderate risk
            ]

# Global configuration instance
config = SystemConfig()
alpaca_config = AlpacaConfig()
data_config = DataSourceConfig()
hierarchy_config = HierarchyConfig()
