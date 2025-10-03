"""
Market Hours Optimization and Time-Specific Trading Strategies
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, time
import pytz
from dataclasses import dataclass
from enum import Enum

class MarketSession(Enum):
    """Market trading sessions"""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MORNING_SESSION = "morning_session"
    LUNCH_SESSION = "lunch_session"
    AFTERNOON_SESSION = "afternoon_session"
    MARKET_CLOSE = "market_close"
    AFTER_HOURS = "after_hours"
    OVERNIGHT = "overnight"

class AssetType(Enum):
    """Asset types for market hours"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"

@dataclass
class MarketHours:
    """Market hours configuration"""
    asset_type: AssetType
    timezone: str
    open_time: time
    close_time: time
    pre_market_start: time
    after_hours_end: time
    lunch_start: Optional[time] = None
    lunch_end: Optional[time] = None

@dataclass
class SessionStrategy:
    """Strategy configuration for a market session"""
    session: MarketSession
    strategy_type: str
    risk_level: str
    position_sizing_multiplier: float
    max_trades_per_session: int
    preferred_assets: List[str]
    technical_indicators: List[str]
    sentiment_weight: float

@dataclass
class MarketHoursAnalysis:
    """Market hours analysis result"""
    current_session: MarketSession
    time_to_next_session: timedelta
    session_volatility: float
    session_volume: float
    recommended_strategy: SessionStrategy
    market_conditions: Dict[str, Any]
    trading_opportunities: List[str]

class MarketHoursOptimizer:
    """Market hours optimizer for time-specific trading strategies"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.market_hours = self._initialize_market_hours()
        self.session_strategies = self._initialize_session_strategies()
        self.session_performance = {}
        self.current_analysis = None
        
    def _initialize_market_hours(self) -> Dict[AssetType, MarketHours]:
        """Initialize market hours for different asset types"""
        return {
            AssetType.STOCK: MarketHours(
                asset_type=AssetType.STOCK,
                timezone="America/New_York",
                open_time=time(9, 30),
                close_time=time(16, 0),
                pre_market_start=time(4, 0),
                after_hours_end=time(20, 0),
                lunch_start=time(12, 0),
                lunch_end=time(13, 0)
            ),
            AssetType.CRYPTO: MarketHours(
                asset_type=AssetType.CRYPTO,
                timezone="UTC",
                open_time=time(0, 0),
                close_time=time(23, 59),
                pre_market_start=time(0, 0),
                after_hours_end=time(23, 59)
            ),
            AssetType.FOREX: MarketHours(
                asset_type=AssetType.FOREX,
                timezone="UTC",
                open_time=time(0, 0),
                close_time=time(23, 59),
                pre_market_start=time(0, 0),
                after_hours_end=time(23, 59)
            )
        }
    
    def _initialize_session_strategies(self) -> Dict[MarketSession, SessionStrategy]:
        """Initialize strategies for different market sessions"""
        return {
            MarketSession.PRE_MARKET: SessionStrategy(
                session=MarketSession.PRE_MARKET,
                strategy_type="gap_trading",
                risk_level="medium",
                position_sizing_multiplier=0.7,
                max_trades_per_session=3,
                preferred_assets=["AAPL", "TSLA", "NVDA", "MSFT"],
                technical_indicators=["gap_analysis", "volume_profile", "support_resistance"],
                sentiment_weight=0.8
            ),
            MarketSession.MARKET_OPEN: SessionStrategy(
                session=MarketSession.MARKET_OPEN,
                strategy_type="momentum_breakout",
                risk_level="high",
                position_sizing_multiplier=1.2,
                max_trades_per_session=5,
                preferred_assets=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
                technical_indicators=["opening_range", "volume_spike", "momentum"],
                sentiment_weight=0.6
            ),
            MarketSession.MORNING_SESSION: SessionStrategy(
                session=MarketSession.MORNING_SESSION,
                strategy_type="trend_following",
                risk_level="medium",
                position_sizing_multiplier=1.0,
                max_trades_per_session=4,
                preferred_assets=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                technical_indicators=["moving_averages", "macd", "rsi"],
                sentiment_weight=0.5
            ),
            MarketSession.LUNCH_SESSION: SessionStrategy(
                session=MarketSession.LUNCH_SESSION,
                strategy_type="mean_reversion",
                risk_level="low",
                position_sizing_multiplier=0.6,
                max_trades_per_session=2,
                preferred_assets=["SPY", "QQQ", "AAPL"],
                technical_indicators=["bollinger_bands", "rsi", "support_resistance"],
                sentiment_weight=0.3
            ),
            MarketSession.AFTERNOON_SESSION: SessionStrategy(
                session=MarketSession.AFTERNOON_SESSION,
                strategy_type="momentum_continuation",
                risk_level="medium",
                position_sizing_multiplier=1.0,
                max_trades_per_session=4,
                preferred_assets=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                technical_indicators=["trend_lines", "momentum", "volume"],
                sentiment_weight=0.5
            ),
            MarketSession.MARKET_CLOSE: SessionStrategy(
                session=MarketSession.MARKET_CLOSE,
                strategy_type="closing_auction",
                risk_level="high",
                position_sizing_multiplier=1.1,
                max_trades_per_session=3,
                preferred_assets=["SPY", "QQQ", "AAPL", "MSFT"],
                technical_indicators=["closing_volume", "end_of_day_momentum"],
                sentiment_weight=0.7
            ),
            MarketSession.AFTER_HOURS: SessionStrategy(
                session=MarketSession.AFTER_HOURS,
                strategy_type="news_trading",
                risk_level="high",
                position_sizing_multiplier=0.8,
                max_trades_per_session=2,
                preferred_assets=["AAPL", "TSLA", "NVDA", "MSFT"],
                technical_indicators=["news_sentiment", "after_hours_volume"],
                sentiment_weight=0.9
            ),
            MarketSession.OVERNIGHT: SessionStrategy(
                session=MarketSession.OVERNIGHT,
                strategy_type="swing_trading",
                risk_level="low",
                position_sizing_multiplier=0.5,
                max_trades_per_session=1,
                preferred_assets=["BTC", "ETH", "EURUSD", "GBPUSD"],
                technical_indicators=["overnight_gaps", "asian_session_analysis"],
                sentiment_weight=0.4
            )
        }
    
    async def analyze_market_hours(self, market_data: Dict[str, Any]) -> MarketHoursAnalysis:
        """Analyze current market hours and recommend strategies"""
        try:
            current_time = datetime.now(pytz.timezone("America/New_York"))
            
            # Determine current session
            current_session = await self._determine_current_session(current_time)
            
            # Calculate time to next session
            time_to_next = await self._calculate_time_to_next_session(current_time, current_session)
            
            # Analyze session conditions
            session_volatility = await self._analyze_session_volatility(market_data, current_session)
            session_volume = await self._analyze_session_volume(market_data, current_session)
            
            # Get recommended strategy
            recommended_strategy = self.session_strategies.get(current_session)
            
            # Analyze market conditions
            market_conditions = await self._analyze_market_conditions(market_data, current_session)
            
            # Identify trading opportunities
            trading_opportunities = await self._identify_trading_opportunities(
                market_data, current_session, recommended_strategy
            )
            
            analysis = MarketHoursAnalysis(
                current_session=current_session,
                time_to_next_session=time_to_next,
                session_volatility=session_volatility,
                session_volume=session_volume,
                recommended_strategy=recommended_strategy,
                market_conditions=market_conditions,
                trading_opportunities=trading_opportunities
            )
            
            self.current_analysis = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market hours: {e}")
            return None
    
    async def _determine_current_session(self, current_time: datetime) -> MarketSession:
        """Determine current market session"""
        try:
            current_time_only = current_time.time()
            weekday = current_time.weekday()
            
            # Check if it's a weekend
            if weekday >= 5:  # Saturday or Sunday
                return MarketSession.OVERNIGHT
            
            # Stock market sessions
            stock_hours = self.market_hours[AssetType.STOCK]
            
            if stock_hours.pre_market_start <= current_time_only < stock_hours.open_time:
                return MarketSession.PRE_MARKET
            elif current_time_only == stock_hours.open_time:
                return MarketSession.MARKET_OPEN
            elif stock_hours.open_time < current_time_only < stock_hours.lunch_start:
                return MarketSession.MORNING_SESSION
            elif stock_hours.lunch_start <= current_time_only < stock_hours.lunch_end:
                return MarketSession.LUNCH_SESSION
            elif stock_hours.lunch_end <= current_time_only < stock_hours.close_time:
                return MarketSession.AFTERNOON_SESSION
            elif current_time_only == stock_hours.close_time:
                return MarketSession.MARKET_CLOSE
            elif stock_hours.close_time < current_time_only < stock_hours.after_hours_end:
                return MarketSession.AFTER_HOURS
            else:
                return MarketSession.OVERNIGHT
                
        except Exception as e:
            self.logger.error(f"Error determining current session: {e}")
            return MarketSession.OVERNIGHT
    
    async def _calculate_time_to_next_session(self, current_time: datetime, current_session: MarketSession) -> timedelta:
        """Calculate time until next market session"""
        try:
            # Define session transitions
            session_transitions = {
                MarketSession.PRE_MARKET: (9, 30),  # Market open
                MarketSession.MARKET_OPEN: (10, 0),  # Morning session
                MarketSession.MORNING_SESSION: (12, 0),  # Lunch
                MarketSession.LUNCH_SESSION: (13, 0),  # Afternoon
                MarketSession.AFTERNOON_SESSION: (16, 0),  # Market close
                MarketSession.MARKET_CLOSE: (16, 1),  # After hours
                MarketSession.AFTER_HOURS: (20, 0),  # Overnight
                MarketSession.OVERNIGHT: (4, 0)  # Pre-market (next day)
            }
            
            next_session_time = session_transitions.get(current_session)
            if not next_session_time:
                return timedelta(hours=1)  # Default
            
            # Calculate next session datetime
            next_session_datetime = current_time.replace(
                hour=next_session_time[0],
                minute=next_session_time[1],
                second=0,
                microsecond=0
            )
            
            # If next session is tomorrow
            if next_session_datetime <= current_time:
                next_session_datetime += timedelta(days=1)
            
            return next_session_datetime - current_time
            
        except Exception as e:
            self.logger.error(f"Error calculating time to next session: {e}")
            return timedelta(hours=1)
    
    async def _analyze_session_volatility(self, market_data: Dict[str, Any], session: MarketSession) -> float:
        """Analyze volatility for current session"""
        try:
            # Get historical volatility data for this session
            # For now, return session-specific volatility estimates
            session_volatility_map = {
                MarketSession.PRE_MARKET: 0.15,
                MarketSession.MARKET_OPEN: 0.25,
                MarketSession.MORNING_SESSION: 0.20,
                MarketSession.LUNCH_SESSION: 0.10,
                MarketSession.AFTERNOON_SESSION: 0.18,
                MarketSession.MARKET_CLOSE: 0.22,
                MarketSession.AFTER_HOURS: 0.30,
                MarketSession.OVERNIGHT: 0.12
            }
            
            base_volatility = session_volatility_map.get(session, 0.15)
            
            # Adjust based on current market conditions
            if market_data:
                # Calculate current market volatility
                current_volatility = 0
                count = 0
                
                for symbol, data in market_data.items():
                    if 'bars' in data and len(data['bars']) > 5:
                        prices = [bar.close for bar in data['bars'][-10:]]
                        if len(prices) > 1:
                            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                            if returns:
                                current_volatility += np.std(returns)
                                count += 1
                
                if count > 0:
                    current_volatility /= count
                    # Blend base session volatility with current volatility
                    return (base_volatility * 0.7) + (current_volatility * 0.3)
            
            return base_volatility
            
        except Exception as e:
            self.logger.error(f"Error analyzing session volatility: {e}")
            return 0.15
    
    async def _analyze_session_volume(self, market_data: Dict[str, Any], session: MarketSession) -> float:
        """Analyze volume for current session"""
        try:
            # Get session-specific volume estimates
            session_volume_map = {
                MarketSession.PRE_MARKET: 0.3,
                MarketSession.MARKET_OPEN: 1.5,
                MarketSession.MORNING_SESSION: 1.0,
                MarketSession.LUNCH_SESSION: 0.4,
                MarketSession.AFTERNOON_SESSION: 0.8,
                MarketSession.MARKET_CLOSE: 1.2,
                MarketSession.AFTER_HOURS: 0.2,
                MarketSession.OVERNIGHT: 0.1
            }
            
            base_volume = session_volume_map.get(session, 0.5)
            
            # Adjust based on current volume
            if market_data:
                current_volume = 0
                count = 0
                
                for symbol, data in market_data.items():
                    if 'bars' in data and len(data['bars']) > 0:
                        recent_volume = data['bars'][-1].volume
                        avg_volume = np.mean([bar.volume for bar in data['bars'][-20:]]) if len(data['bars']) >= 20 else recent_volume
                        
                        if avg_volume > 0:
                            current_volume += recent_volume / avg_volume
                            count += 1
                
                if count > 0:
                    current_volume /= count
                    return (base_volume * 0.6) + (current_volume * 0.4)
            
            return base_volume
            
        except Exception as e:
            self.logger.error(f"Error analyzing session volume: {e}")
            return 0.5
    
    async def _analyze_market_conditions(self, market_data: Dict[str, Any], session: MarketSession) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            conditions = {
                'session': session.value,
                'market_sentiment': 'neutral',
                'trend_direction': 'sideways',
                'volatility_regime': 'normal',
                'liquidity_level': 'normal',
                'news_impact': 'low'
            }
            
            if not market_data:
                return conditions
            
            # Analyze market sentiment
            bullish_count = 0
            bearish_count = 0
            total_symbols = 0
            
            for symbol, data in market_data.items():
                if 'bars' in data and len(data['bars']) > 1:
                    current_price = data['bars'][-1].close
                    previous_price = data['bars'][-2].close
                    
                    if current_price > previous_price:
                        bullish_count += 1
                    elif current_price < previous_price:
                        bearish_count += 1
                    
                    total_symbols += 1
            
            if total_symbols > 0:
                bullish_ratio = bullish_count / total_symbols
                if bullish_ratio > 0.6:
                    conditions['market_sentiment'] = 'bullish'
                elif bullish_ratio < 0.4:
                    conditions['market_sentiment'] = 'bearish'
            
            # Analyze trend direction
            price_changes = []
            for symbol, data in market_data.items():
                if 'bars' in data and len(data['bars']) > 5:
                    prices = [bar.close for bar in data['bars'][-5:]]
                    if len(prices) > 1:
                        change = (prices[-1] - prices[0]) / prices[0]
                        price_changes.append(change)
            
            if price_changes:
                avg_change = np.mean(price_changes)
                if avg_change > 0.01:
                    conditions['trend_direction'] = 'upward'
                elif avg_change < -0.01:
                    conditions['trend_direction'] = 'downward'
            
            # Analyze volatility regime
            if price_changes:
                volatility = np.std(price_changes)
                if volatility > 0.02:
                    conditions['volatility_regime'] = 'high'
                elif volatility < 0.005:
                    conditions['volatility_regime'] = 'low'
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    async def _identify_trading_opportunities(self, market_data: Dict[str, Any], 
                                           session: MarketSession, 
                                           strategy: SessionStrategy) -> List[str]:
        """Identify trading opportunities for current session"""
        try:
            opportunities = []
            
            if not strategy or not market_data:
                return opportunities
            
            # Session-specific opportunity identification
            if session == MarketSession.PRE_MARKET:
                opportunities.extend(await self._identify_gap_opportunities(market_data))
            elif session == MarketSession.MARKET_OPEN:
                opportunities.extend(await self._identify_breakout_opportunities(market_data))
            elif session == MarketSession.MORNING_SESSION:
                opportunities.extend(await self._identify_trend_opportunities(market_data))
            elif session == MarketSession.LUNCH_SESSION:
                opportunities.extend(await self._identify_mean_reversion_opportunities(market_data))
            elif session == MarketSession.AFTERNOON_SESSION:
                opportunities.extend(await self._identify_momentum_opportunities(market_data))
            elif session == MarketSession.MARKET_CLOSE:
                opportunities.extend(await self._identify_closing_opportunities(market_data))
            elif session == MarketSession.AFTER_HOURS:
                opportunities.extend(await self._identify_news_opportunities(market_data))
            elif session == MarketSession.OVERNIGHT:
                opportunities.extend(await self._identify_swing_opportunities(market_data))
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying trading opportunities: {e}")
            return []
    
    async def _identify_gap_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify gap trading opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if 'bars' in data and len(data['bars']) > 2:
                current_bar = data['bars'][-1]
                previous_bar = data['bars'][-2]
                
                # Calculate gap
                gap = (current_bar.open - previous_bar.close) / previous_bar.close
                
                if abs(gap) > 0.02:  # 2% gap
                    direction = "up" if gap > 0 else "down"
                    opportunities.append(f"GAP_{direction.upper()}: {symbol} has {abs(gap)*100:.1f}% gap")
        
        return opportunities
    
    async def _identify_breakout_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify breakout opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if 'bars' in data and len(data['bars']) > 20:
                recent_bars = data['bars'][-20:]
                highs = [bar.high for bar in recent_bars]
                lows = [bar.low for bar in recent_bars]
                current_price = recent_bars[-1].close
                
                resistance = max(highs[:-1])  # Exclude current bar
                support = min(lows[:-1])
                
                if current_price > resistance:
                    opportunities.append(f"BREAKOUT_UP: {symbol} breaking above resistance at {resistance:.2f}")
                elif current_price < support:
                    opportunities.append(f"BREAKOUT_DOWN: {symbol} breaking below support at {support:.2f}")
        
        return opportunities
    
    async def _identify_trend_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify trend following opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if 'bars' in data and len(data['bars']) > 10:
                prices = [bar.close for bar in data['bars'][-10:]]
                
                # Simple trend analysis
                if len(prices) >= 5:
                    short_ma = np.mean(prices[-5:])
                    long_ma = np.mean(prices[-10:])
                    
                    if short_ma > long_ma * 1.01:  # 1% above
                        opportunities.append(f"TREND_UP: {symbol} showing upward trend")
                    elif short_ma < long_ma * 0.99:  # 1% below
                        opportunities.append(f"TREND_DOWN: {symbol} showing downward trend")
        
        return opportunities
    
    async def _identify_mean_reversion_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify mean reversion opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if 'bars' in data and len(data['bars']) > 20:
                prices = [bar.close for bar in data['bars'][-20:]]
                current_price = prices[-1]
                mean_price = np.mean(prices)
                std_price = np.std(prices)
                
                # Check for mean reversion signals
                if current_price > mean_price + 2 * std_price:
                    opportunities.append(f"MEAN_REVERSION_SELL: {symbol} overextended above mean")
                elif current_price < mean_price - 2 * std_price:
                    opportunities.append(f"MEAN_REVERSION_BUY: {symbol} oversold below mean")
        
        return opportunities
    
    async def _identify_momentum_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify momentum opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if 'bars' in data and len(data['bars']) > 5:
                prices = [bar.close for bar in data['bars'][-5:]]
                volumes = [bar.volume for bar in data['bars'][-5:]]
                
                # Check for momentum with volume
                price_momentum = (prices[-1] - prices[0]) / prices[0]
                volume_increase = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1
                
                if price_momentum > 0.01 and volume_increase > 1.5:
                    opportunities.append(f"MOMENTUM_UP: {symbol} showing strong momentum with volume")
                elif price_momentum < -0.01 and volume_increase > 1.5:
                    opportunities.append(f"MOMENTUM_DOWN: {symbol} showing strong downward momentum")
        
        return opportunities
    
    async def _identify_closing_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify closing session opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if 'bars' in data and len(data['bars']) > 1:
                current_bar = data['bars'][-1]
                previous_bar = data['bars'][-2]
                
                # Check for end-of-day momentum
                price_change = (current_bar.close - previous_bar.close) / previous_bar.close
                volume_ratio = current_bar.volume / previous_bar.volume if previous_bar.volume > 0 else 1
                
                if abs(price_change) > 0.005 and volume_ratio > 1.2:
                    direction = "up" if price_change > 0 else "down"
                    opportunities.append(f"CLOSING_{direction.upper()}: {symbol} showing end-of-day momentum")
        
        return opportunities
    
    async def _identify_news_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify news-based opportunities"""
        opportunities = []
        
        # This would integrate with news sentiment analysis
        # For now, return placeholder opportunities
        opportunities.append("NEWS_OPPORTUNITY: Monitor after-hours news for trading opportunities")
        
        return opportunities
    
    async def _identify_swing_opportunities(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify swing trading opportunities"""
        opportunities = []
        
        # Focus on crypto and forex for overnight trading
        crypto_forex_symbols = ['BTC', 'ETH', 'EURUSD', 'GBPUSD', 'USDJPY']
        
        for symbol in crypto_forex_symbols:
            if symbol in market_data:
                data = market_data[symbol]
                if 'bars' in data and len(data['bars']) > 10:
                    prices = [bar.close for bar in data['bars'][-10:]]
                    
                    # Check for swing patterns
                    if len(prices) >= 5:
                        recent_high = max(prices[-5:])
                        recent_low = min(prices[-5:])
                        current_price = prices[-1]
                        
                        if current_price > (recent_high + recent_low) / 2 * 1.01:
                            opportunities.append(f"SWING_UP: {symbol} showing overnight bullish pattern")
                        elif current_price < (recent_high + recent_low) / 2 * 0.99:
                            opportunities.append(f"SWING_DOWN: {symbol} showing overnight bearish pattern")
        
        return opportunities
    
    def get_session_performance(self) -> Dict[str, Any]:
        """Get performance metrics by session"""
        try:
            if not self.session_performance:
                return {}
            
            return {
                'session_performance': self.session_performance,
                'best_session': max(self.session_performance.items(), key=lambda x: x[1].get('profit', 0))[0] if self.session_performance else None,
                'worst_session': min(self.session_performance.items(), key=lambda x: x[1].get('profit', 0))[0] if self.session_performance else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session performance: {e}")
            return {}
    
    def update_session_performance(self, session: MarketSession, profit: float, trades: int):
        """Update performance metrics for a session"""
        try:
            if session not in self.session_performance:
                self.session_performance[session] = {
                    'total_profit': 0,
                    'total_trades': 0,
                    'avg_profit': 0,
                    'sessions_count': 0
                }
            
            perf = self.session_performance[session]
            perf['total_profit'] += profit
            perf['total_trades'] += trades
            perf['sessions_count'] += 1
            perf['avg_profit'] = perf['total_profit'] / perf['sessions_count']
            
        except Exception as e:
            self.logger.error(f"Error updating session performance: {e}")


