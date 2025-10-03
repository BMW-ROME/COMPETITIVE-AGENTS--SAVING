"""
Multi-timeframe Analysis Engine for Trading Agents
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class TimeFrame(Enum):
    """Trading timeframes"""
    MINUTE_1 = "1Min"
    MINUTE_5 = "5Min"
    MINUTE_15 = "15Min"
    HOUR_1 = "1Hour"
    HOUR_4 = "4Hour"
    DAY_1 = "1Day"

@dataclass
class TimeFrameData:
    """Data for a specific timeframe"""
    timeframe: TimeFrame
    bars: List[Any]
    indicators: Dict[str, float]
    trend: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1
    support: float
    resistance: float
    volume_profile: Dict[str, float]

@dataclass
class MultiTimeFrameAnalysis:
    """Multi-timeframe analysis result"""
    symbol: str
    timestamp: datetime
    timeframes: Dict[TimeFrame, TimeFrameData]
    consensus_trend: str
    trend_strength: float
    key_levels: Dict[str, float]
    trading_signals: List[str]
    confidence: float

class MultiTimeFrameAnalyzer:
    """Multi-timeframe analysis engine"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timeframes = [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.HOUR_1]
        self.analysis_cache = {}
        
    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[MultiTimeFrameAnalysis]:
        """Perform multi-timeframe analysis for a symbol"""
        try:
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                return None
            
            # Get data for all timeframes
            timeframe_data = {}
            for tf in self.timeframes:
                tf_data = await self._get_timeframe_data(symbol, tf, market_data)
                if tf_data:
                    timeframe_data[tf] = tf_data
            
            if not timeframe_data:
                return None
            
            # Analyze each timeframe
            for tf, data in timeframe_data.items():
                data.indicators = await self._calculate_indicators(data.bars)
                data.trend, data.strength = await self._analyze_trend(data.bars, data.indicators)
                data.support, data.resistance = await self._find_support_resistance(data.bars)
                data.volume_profile = await self._analyze_volume_profile(data.bars)
            
            # Generate consensus analysis
            consensus_trend, trend_strength = await self._generate_consensus(timeframe_data)
            key_levels = await self._identify_key_levels(timeframe_data)
            trading_signals = await self._generate_trading_signals(timeframe_data, consensus_trend)
            confidence = await self._calculate_confidence(timeframe_data, trading_signals)
            
            analysis = MultiTimeFrameAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframes=timeframe_data,
                consensus_trend=consensus_trend,
                trend_strength=trend_strength,
                key_levels=key_levels,
                trading_signals=trading_signals,
                confidence=confidence
            )
            
            # Cache analysis
            self.analysis_cache[symbol] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol} multi-timeframe: {e}")
            return None
    
    async def _get_timeframe_data(self, symbol: str, timeframe: TimeFrame, market_data: Dict[str, Any]) -> Optional[TimeFrameData]:
        """Get data for a specific timeframe"""
        try:
            symbol_data = market_data.get(symbol, {})
            if not symbol_data or 'bars' not in symbol_data:
                return None
            
            bars = symbol_data['bars']
            if len(bars) < 10:
                return None
            
            # For now, we'll use the same bars but simulate different timeframes
            # In production, you'd fetch actual multi-timeframe data
            if timeframe == TimeFrame.MINUTE_1:
                # Use every bar (1-minute data)
                tf_bars = bars[-100:]  # Last 100 minutes
            elif timeframe == TimeFrame.MINUTE_5:
                # Simulate 5-minute bars (every 5th bar)
                tf_bars = bars[-100:][::5]  # Every 5th bar
            elif timeframe == TimeFrame.MINUTE_15:
                # Simulate 15-minute bars (every 15th bar)
                tf_bars = bars[-100:][::15]  # Every 15th bar
            elif timeframe == TimeFrame.HOUR_1:
                # Simulate 1-hour bars (every 60th bar)
                tf_bars = bars[-100:][::60]  # Every 60th bar
            else:
                tf_bars = bars[-50:]  # Default
            
            if len(tf_bars) < 5:
                return None
            
            return TimeFrameData(
                timeframe=timeframe,
                bars=tf_bars,
                indicators={},
                trend='neutral',
                strength=0.0,
                support=0.0,
                resistance=0.0,
                volume_profile={}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting timeframe data for {symbol} {timeframe}: {e}")
            return None
    
    async def _calculate_indicators(self, bars: List[Any]) -> Dict[str, float]:
        """Calculate technical indicators for timeframe"""
        try:
            if len(bars) < 20:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])
            
            indicators = {}
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1] if not rs.isna().iloc[-1] else 50
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1] if not macd.isna().iloc[-1] else 0
            indicators['macd_signal'] = signal.iloc[-1] if not signal.isna().iloc[-1] else 0
            indicators['macd_histogram'] = (macd - signal).iloc[-1] if not (macd - signal).isna().iloc[-1] else 0
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1] if not sma_20.isna().iloc[-1] else df['close'].iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1] if not sma_20.isna().iloc[-1] else df['close'].iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1] if not sma_20.isna().iloc[-1] else df['close'].iloc[-1]
            
            # Moving Averages
            indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else df['close'].iloc[-1]
            indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else df['close'].iloc[-1]
            indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else df['volume'].iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            indicators['volatility'] = returns.std() * np.sqrt(24) if len(returns) > 1 else 0
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}
    
    async def _analyze_trend(self, bars: List[Any], indicators: Dict[str, float]) -> Tuple[str, float]:
        """Analyze trend direction and strength"""
        try:
            if not bars or not indicators:
                return 'neutral', 0.0
            
            current_price = bars[-1].close
            trend_signals = []
            strength_signals = []
            
            # Price vs Moving Averages
            if 'sma_20' in indicators and 'sma_50' in indicators:
                if current_price > indicators['sma_20'] > indicators['sma_50']:
                    trend_signals.append('bullish')
                    strength_signals.append(0.8)
                elif current_price < indicators['sma_20'] < indicators['sma_50']:
                    trend_signals.append('bearish')
                    strength_signals.append(0.8)
                else:
                    trend_signals.append('neutral')
                    strength_signals.append(0.3)
            
            # MACD
            if 'macd' in indicators and 'macd_signal' in indicators:
                if indicators['macd'] > indicators['macd_signal']:
                    trend_signals.append('bullish')
                    strength_signals.append(0.6)
                elif indicators['macd'] < indicators['macd_signal']:
                    trend_signals.append('bearish')
                    strength_signals.append(0.6)
                else:
                    trend_signals.append('neutral')
                    strength_signals.append(0.2)
            
            # RSI
            if 'rsi' in indicators:
                if indicators['rsi'] > 70:
                    trend_signals.append('bearish')  # Overbought
                    strength_signals.append(0.7)
                elif indicators['rsi'] < 30:
                    trend_signals.append('bullish')  # Oversold
                    strength_signals.append(0.7)
                elif 40 < indicators['rsi'] < 60:
                    trend_signals.append('neutral')
                    strength_signals.append(0.3)
                else:
                    trend_signals.append('neutral')
                    strength_signals.append(0.5)
            
            # Price momentum
            if len(bars) >= 5:
                price_change = (current_price - bars[-5].close) / bars[-5].close
                if price_change > 0.02:  # 2% up
                    trend_signals.append('bullish')
                    strength_signals.append(0.7)
                elif price_change < -0.02:  # 2% down
                    trend_signals.append('bearish')
                    strength_signals.append(0.7)
                else:
                    trend_signals.append('neutral')
                    strength_signals.append(0.4)
            
            # Determine consensus trend
            bullish_count = trend_signals.count('bullish')
            bearish_count = trend_signals.count('bearish')
            neutral_count = trend_signals.count('neutral')
            
            if bullish_count > bearish_count and bullish_count > neutral_count:
                trend = 'bullish'
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Calculate strength
            strength = np.mean(strength_signals) if strength_signals else 0.0
            
            return trend, strength
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return 'neutral', 0.0
    
    async def _find_support_resistance(self, bars: List[Any]) -> Tuple[float, float]:
        """Find support and resistance levels"""
        try:
            if len(bars) < 20:
                return 0.0, 0.0
            
            highs = [bar.high for bar in bars[-20:]]
            lows = [bar.low for bar in bars[-20:]]
            
            # Simple support/resistance calculation
            resistance = max(highs)
            support = min(lows)
            
            return support, resistance
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {e}")
            return 0.0, 0.0
    
    async def _analyze_volume_profile(self, bars: List[Any]) -> Dict[str, float]:
        """Analyze volume profile"""
        try:
            if not bars:
                return {}
            
            volumes = [bar.volume for bar in bars]
            avg_volume = np.mean(volumes)
            max_volume = max(volumes)
            min_volume = min(volumes)
            
            return {
                'avg_volume': avg_volume,
                'max_volume': max_volume,
                'min_volume': min_volume,
                'volume_trend': (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0,
                'volume_ratio': volumes[-1] / avg_volume if avg_volume > 0 else 1
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {e}")
            return {}
    
    async def _generate_consensus(self, timeframe_data: Dict[TimeFrame, TimeFrameData]) -> Tuple[str, float]:
        """Generate consensus trend from all timeframes"""
        try:
            trends = []
            strengths = []
            
            for tf, data in timeframe_data.items():
                trends.append(data.trend)
                strengths.append(data.strength)
            
            # Weight timeframes (longer timeframes have more weight)
            weights = {
                TimeFrame.MINUTE_1: 0.1,
                TimeFrame.MINUTE_5: 0.2,
                TimeFrame.MINUTE_15: 0.3,
                TimeFrame.HOUR_1: 0.4
            }
            
            weighted_trends = []
            weighted_strengths = []
            
            for tf, data in timeframe_data.items():
                weight = weights.get(tf, 0.25)
                weighted_trends.extend([data.trend] * int(weight * 10))
                weighted_strengths.append(data.strength * weight)
            
            # Determine consensus
            bullish_count = weighted_trends.count('bullish')
            bearish_count = weighted_trends.count('bearish')
            neutral_count = weighted_trends.count('neutral')
            
            if bullish_count > bearish_count and bullish_count > neutral_count:
                consensus_trend = 'bullish'
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                consensus_trend = 'bearish'
            else:
                consensus_trend = 'neutral'
            
            consensus_strength = np.mean(weighted_strengths) if weighted_strengths else 0.0
            
            return consensus_trend, consensus_strength
            
        except Exception as e:
            self.logger.error(f"Error generating consensus: {e}")
            return 'neutral', 0.0
    
    async def _identify_key_levels(self, timeframe_data: Dict[TimeFrame, TimeFrameData]) -> Dict[str, float]:
        """Identify key support and resistance levels"""
        try:
            key_levels = {}
            
            # Collect all support/resistance levels
            supports = []
            resistances = []
            
            for tf, data in timeframe_data.items():
                if data.support > 0:
                    supports.append(data.support)
                if data.resistance > 0:
                    resistances.append(data.resistance)
            
            if supports:
                key_levels['major_support'] = min(supports)
                key_levels['minor_support'] = np.percentile(supports, 25)
            
            if resistances:
                key_levels['major_resistance'] = max(resistances)
                key_levels['minor_resistance'] = np.percentile(resistances, 75)
            
            # Calculate pivot points
            if supports and resistances:
                key_levels['pivot'] = (min(supports) + max(resistances)) / 2
                key_levels['range'] = max(resistances) - min(supports)
            
            return key_levels
            
        except Exception as e:
            self.logger.error(f"Error identifying key levels: {e}")
            return {}
    
    async def _generate_trading_signals(self, timeframe_data: Dict[TimeFrame, TimeFrameData], consensus_trend: str) -> List[str]:
        """Generate trading signals based on multi-timeframe analysis"""
        try:
            signals = []
            
            # Trend alignment signals
            if consensus_trend == 'bullish':
                signals.append('TREND_BULLISH')
                
                # Check for pullback opportunities
                short_term_trend = timeframe_data.get(TimeFrame.MINUTE_1, TimeFrameData(
                    TimeFrame.MINUTE_1, [], {}, 'neutral', 0, 0, 0, {}
                )).trend
                
                if short_term_trend == 'bearish':
                    signals.append('PULLBACK_BUY_OPPORTUNITY')
            
            elif consensus_trend == 'bearish':
                signals.append('TREND_BEARISH')
                
                # Check for rally opportunities
                short_term_trend = timeframe_data.get(TimeFrame.MINUTE_1, TimeFrameData(
                    TimeFrame.MINUTE_1, [], {}, 'neutral', 0, 0, 0, {}
                )).trend
                
                if short_term_trend == 'bullish':
                    signals.append('RALLY_SELL_OPPORTUNITY')
            
            # Breakout signals
            for tf, data in timeframe_data.items():
                if data.indicators:
                    current_price = data.bars[-1].close if data.bars else 0
                    
                    # Breakout above resistance
                    if 'bb_upper' in data.indicators and current_price > data.indicators['bb_upper']:
                        signals.append(f'BREAKOUT_UP_{tf.value}')
                    
                    # Breakdown below support
                    if 'bb_lower' in data.indicators and current_price < data.indicators['bb_lower']:
                        signals.append(f'BREAKDOWN_DOWN_{tf.value}')
            
            # Volume confirmation
            for tf, data in timeframe_data.items():
                if data.volume_profile and data.volume_profile.get('volume_ratio', 1) > 1.5:
                    signals.append(f'HIGH_VOLUME_{tf.value}')
            
            # Divergence signals
            for tf, data in timeframe_data.items():
                if data.indicators and 'rsi' in data.indicators and 'macd' in data.indicators:
                    rsi = data.indicators['rsi']
                    macd = data.indicators['macd']
                    
                    # Bullish divergence
                    if rsi < 30 and macd > 0:
                        signals.append(f'BULLISH_DIVERGENCE_{tf.value}')
                    
                    # Bearish divergence
                    if rsi > 70 and macd < 0:
                        signals.append(f'BEARISH_DIVERGENCE_{tf.value}')
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return []
    
    async def _calculate_confidence(self, timeframe_data: Dict[TimeFrame, TimeFrameData], signals: List[str]) -> float:
        """Calculate confidence score for the analysis"""
        try:
            confidence_factors = []
            
            # Trend alignment across timeframes
            trends = [data.trend for data in timeframe_data.values()]
            trend_consensus = len(set(trends)) == 1  # All timeframes agree
            confidence_factors.append(0.3 if trend_consensus else 0.1)
            
            # Signal strength
            signal_count = len(signals)
            confidence_factors.append(min(0.3, signal_count * 0.05))
            
            # Volume confirmation
            volume_confirmed = any('HIGH_VOLUME' in signal for signal in signals)
            confidence_factors.append(0.2 if volume_confirmed else 0.1)
            
            # Indicator alignment
            indicator_alignment = 0
            for data in timeframe_data.values():
                if data.indicators:
                    # Check if RSI and MACD agree
                    if 'rsi' in data.indicators and 'macd' in data.indicators:
                        rsi = data.indicators['rsi']
                        macd = data.indicators['macd']
                        if (rsi > 50 and macd > 0) or (rsi < 50 and macd < 0):
                            indicator_alignment += 0.1
            
            confidence_factors.append(min(0.2, indicator_alignment))
            
            return min(1.0, sum(confidence_factors))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all multi-timeframe analyses"""
        try:
            if not self.analysis_cache:
                return {}
            
            summary = {
                'total_symbols': len(self.analysis_cache),
                'consensus_trends': {},
                'high_confidence_signals': [],
                'timestamp': datetime.now().isoformat()
            }
            
            for symbol, analysis in self.analysis_cache.items():
                # Count consensus trends
                trend = analysis.consensus_trend
                summary['consensus_trends'][trend] = summary['consensus_trends'].get(trend, 0) + 1
                
                # Collect high confidence signals
                if analysis.confidence > 0.7:
                    summary['high_confidence_signals'].append({
                        'symbol': symbol,
                        'trend': analysis.consensus_trend,
                        'confidence': analysis.confidence,
                        'signals': analysis.trading_signals
                    })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting analysis summary: {e}")
            return {}


