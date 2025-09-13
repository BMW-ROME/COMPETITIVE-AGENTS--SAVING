# Competitive Trading Agents System

A sophisticated multi-agent trading system where AI agents compete for accuracy, featuring self-reflective learning, hierarchical oversight, and real-time market analysis. The system integrates with Alpaca's algorithmic trading platform and analyzes multiple data sources including market trends, news articles, and social media sentiment.

## 🚀 Key Features

### Competitive Multi-Agent Architecture
- **Conservative Agent**: Risk-averse trading with focus on steady returns
- **Aggressive Agent**: High-risk, high-reward strategies with momentum trading
- **Balanced Agent**: Hybrid approach combining conservative and aggressive strategies
- **Real-time Competition**: Agents continuously compete and learn from each other

### Self-Reflective Learning
- **Performance Analysis**: Agents analyze their own trading performance
- **Strategy Adaptation**: Dynamic adjustment of trading parameters based on results
- **Learning from Competitors**: Agents study and adapt successful strategies from rivals
- **Memory System**: Maintains learning history for continuous improvement

### Hierarchical Oversight
- **Performance Evaluation**: Comprehensive ranking and comparison system
- **Resource Allocation**: Dynamic resource distribution based on performance
- **Intervention System**: Automatic intervention for underperforming agents
- **Communication Network**: Real-time communication between agents and hierarchy

### Advanced Data Integration
- **Market Data**: Real-time price feeds from multiple sources
- **News Analysis**: Financial news sentiment analysis using NLP
- **Social Media**: Twitter and Reddit sentiment tracking
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more

### Alpaca Integration
- **Paper Trading**: Safe testing environment
- **Live Trading**: Production-ready execution
- **Risk Management**: Built-in position sizing and stop-loss mechanisms
- **Portfolio Tracking**: Real-time performance monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hierarchy Manager                        │
│  • Performance Evaluation  • Resource Allocation           │
│  • Competition Analysis    • Intervention System           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼────┐ ┌──────▼──────┐
│ Conservative │ │Balanced│ │ Aggressive   │
│    Agent     │ │ Agent  │ │    Agent     │
│              │ │        │ │              │
│ • Risk Mgmt  │ │• Hybrid│ │ • Momentum   │
│ • Steady     │ │• Adapt │ │ • Volatility │
│ • Stable     │ │• Learn │ │ • High Risk  │
└───────┬──────┘ └───┬────┘ └──────┬──────┘
        │            │             │
        └────────────┼─────────────┘
                     │
        ┌────────────▼────────────┐
        │    Data Aggregator      │
        │                         │
        │ • Market Data Provider  │
        │ • News Provider         │
        │ • Social Media Provider │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Alpaca Integration    │
        │                         │
        │ • Order Execution       │
        │ • Portfolio Management  │
        │ • Risk Controls         │
        └─────────────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Alpaca Trading Account (Paper or Live)
- API Keys for data sources (optional but recommended)

### Required API Keys
- **Alpaca**: Trading platform access
- **Alpha Vantage**: Market data and news
- **News API**: Financial news
- **Twitter API**: Social sentiment (optional)
- **Reddit API**: Social sentiment (optional)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd competitive-trading-agents
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
TWITTER_BEARER_TOKEN=your_twitter_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

5. **Create necessary directories**
```bash
mkdir -p logs data
```

## 🚀 Quick Start

### 1. Paper Trading Mode (Recommended for Testing)

```python
from src.system_orchestrator import TradingSystemOrchestrator
from config.settings import SystemConfig, TradingMode

# Configure for paper trading
config = SystemConfig()
config.trading_mode = TradingMode.PAPER

# Create and run system
orchestrator = TradingSystemOrchestrator(config)
await orchestrator.initialize()
await orchestrator.run_system()
```

### 2. Live Trading Mode (Use with Caution)

```python
# Configure for live trading
config = SystemConfig()
config.trading_mode = TradingMode.LIVE

# Update Alpaca configuration for live trading
from config.settings import AlpacaConfig
alpaca_config = AlpacaConfig()
alpaca_config.base_url = "https://api.alpaca.markets"  # Live trading URL
```

### 3. Custom Agent Configuration

```python
from config.settings import AgentConfig, AgentType

# Create custom agent configurations
custom_agents = [
    AgentConfig("conservative_1", AgentType.CONSERVATIVE, initial_capital=50000.0),
    AgentConfig("aggressive_1", AgentType.AGGRESSIVE, initial_capital=50000.0),
    AgentConfig("balanced_1", AgentType.BALANCED, initial_capital=50000.0)
]

config.agent_configs = custom_agents
```

## 📊 Monitoring and Analysis

### Real-time System Status
```python
# Get current system status
status = await orchestrator.get_system_status()
print(f"System running: {status['is_running']}")
print(f"Total cycles: {status['cycle_count']}")
print(f"Total return: {status['system_performance']['total_return']:.4f}")
```

### Agent Performance Comparison
```python
# Get agent rankings and performance
for agent_id, agent_data in status['agents'].items():
    metrics = agent_data['performance_metrics']
    print(f"Agent {agent_id}:")
    print(f"  Total Return: {metrics['total_return']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {metrics['win_rate']:.4f}")
```

### Portfolio Performance
```python
# Get portfolio details
portfolio = await alpaca_interface.get_portfolio_performance()
print(f"Portfolio Value: ${portfolio['portfolio_value']:,.2f}")
print(f"Unrealized P&L: ${portfolio['total_unrealized_pl']:,.2f}")
print(f"Number of Positions: {portfolio['num_positions']}")
```

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Integration Tests
```bash
pytest tests/test_trading_agents.py::TestSystemIntegration -v
```

### Performance Testing
```bash
# Test with historical data
python tests/performance_test.py --start-date 2023-01-01 --end-date 2023-12-31
```

## ⚙️ Configuration

### System Configuration (`config/settings.py`)

```python
@dataclass
class SystemConfig:
    # Trading settings
    trading_mode: TradingMode = TradingMode.PAPER
    trading_symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    # Agent settings
    num_agents: int = 2
    data_update_interval: int = 30  # seconds
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05   # 5% max daily loss
```

### Agent Configuration

```python
@dataclass
class AgentConfig:
    agent_id: str
    agent_type: AgentType
    initial_capital: float = 10000.0
    max_position_size: float = 0.1
    risk_tolerance: float = 0.05
    learning_rate: float = 0.01
```

## 🔧 Advanced Features

### Custom Trading Strategies

Create your own trading agent by extending `BaseTradingAgent`:

```python
class CustomTradingAgent(BaseTradingAgent):
    async def analyze_market_data(self, market_data):
        # Your custom analysis logic
        return analysis_result
    
    async def make_trading_decision(self, market_data):
        # Your custom decision logic
        return trade_decision
    
    async def update_strategy(self, performance_feedback):
        # Your custom learning logic
        pass
```

### Custom Data Sources

Add new data sources by extending the data providers:

```python
class CustomDataProvider:
    async def get_data(self, symbols):
        # Your custom data fetching logic
        return data
```

### Custom Risk Management

Implement custom risk management rules:

```python
class CustomRiskManager:
    def calculate_position_size(self, signal_strength, volatility):
        # Your custom position sizing logic
        return position_size
```

## 📈 Performance Optimization

### Best Practices

1. **Start with Paper Trading**: Always test strategies in paper mode first
2. **Monitor Performance**: Regularly check agent performance and rankings
3. **Risk Management**: Set appropriate position sizes and stop-losses
4. **Data Quality**: Ensure reliable data sources for better decisions
5. **Backtesting**: Test strategies with historical data before live deployment

### Optimization Tips

- **Agent Diversity**: Use different agent types for better market coverage
- **Data Frequency**: Balance data update frequency with system performance
- **Memory Management**: Monitor learning memory size to prevent memory issues
- **Error Handling**: Implement robust error handling for market disruptions

## 🚨 Risk Warnings

⚠️ **Important Disclaimers**:

1. **Trading Risks**: All trading involves risk of loss. Past performance does not guarantee future results.
2. **Paper Trading First**: Always test thoroughly in paper trading mode before live deployment.
3. **API Limits**: Be aware of API rate limits and costs for data sources.
4. **Market Hours**: The system is designed for US market hours. Adjust for other markets.
5. **Regulatory Compliance**: Ensure compliance with local trading regulations.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `docs/` directory for detailed documentation
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions in GitHub Discussions
- **Email**: Contact the development team for enterprise support

## 🔮 Roadmap

### Version 2.0
- [ ] Machine Learning integration with TensorFlow/PyTorch
- [ ] Advanced sentiment analysis with transformer models
- [ ] Multi-market support (crypto, forex, commodities)
- [ ] Web-based dashboard for monitoring
- [ ] Mobile app for system management

### Version 2.1
- [ ] Reinforcement learning agents
- [ ] Genetic algorithm optimization
- [ ] Cloud deployment with AWS/Azure
- [ ] Real-time alerting system
- [ ] Advanced backtesting framework

---

**Happy Trading! 🚀📈**

*Remember: The best trading system is one that you understand, trust, and can maintain. Start small, test thoroughly, and scale gradually.*
