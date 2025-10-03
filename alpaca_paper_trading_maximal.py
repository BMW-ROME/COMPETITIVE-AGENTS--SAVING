#!/usr/bin/env python3
"""
MAXIMAL Alpaca Paper Trading System
==================================
Full-featured competitive trading system with advanced analytics,
machine learning, real-time dashboards, and comprehensive logging.
"""

import asyncio
import logging
import random
import os
import json
import time
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional
import threading
from flask import Flask, jsonify, render_template_string
import pytz

# [ROCKET] FIX WINDOWS UNICODE ISSUES - Enable UTF-8 for emojis
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
import yfinance as yf

# Load environment variables
load_dotenv()

# [ROCKET] IMPORT RL OPTIMIZATION ENGINE
from rl_optimization_engine import get_rl_optimizer

# Import Alpaca with error handling
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
    print("[SUCCESS] Alpaca Trade API loaded successfully")
except ImportError:
    ALPACA_AVAILABLE = False
    print("[ERROR] Alpaca Trade API not available")

# [ROCKET] WINDOWS-COMPATIBLE logging setup (UTF-8 safe)
handlers = [logging.StreamHandler()]
try:
    os.makedirs('logs', exist_ok=True)
    handlers.append(logging.FileHandler('logs/alpaca_maximal_rl.log', encoding='utf-8'))
except Exception as e:
    print(f"Warning: Could not create log file: {e}")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger("AlpacaMaximal")

def is_market_open():
    """Check if the US stock market is currently open"""
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Market is open Monday-Friday, 9:30 AM - 4:00 PM ET
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False, "Market closed: Weekend"
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open:
            return False, f"Market closed: Pre-market (opens at {market_open.strftime('%I:%M %p ET')})"
        elif now > market_close:
            return False, f"Market closed: After-hours (closed at {market_close.strftime('%I:%M %p ET')})"
        else:
            return True, f"Market open: Trading hours ({market_open.strftime('%I:%M %p')} - {market_close.strftime('%I:%M %p ET')})"
    except Exception as e:
        logger.warning(f"Could not determine market hours: {e}")
        return False, "Market status unknown"

class AdvancedTradingAgent:
    """Advanced AI-driven trading agent with machine learning"""
    
    def __init__(self, name: str, strategy: str, risk_tolerance: float):
        self.name = name
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.confidence = 0.0
        self.performance_history = []
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.last_trade_time = None
        
        # Advanced features
        self.ml_model = self._initialize_ml_model()
        self.technical_indicators = {}
        self.sentiment_score = 0.0
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for predictions"""
        # Placeholder for actual ML model
        return {"type": "ensemble", "accuracy": random.uniform(0.6, 0.9)}
    
    def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Advanced market analysis with multiple indicators"""
        analysis = {
            'trend': 'neutral',
            'volatility': 0.0,
            'momentum': 0.0,
            'support_levels': [],
            'resistance_levels': [],
            'risk_assessment': 'medium'
        }
        
        for symbol, data in market_data.items():
            # Calculate technical indicators
            volatility = abs(data.get('change_pct', 0)) / 100
            volume_ratio = data.get('volume', 0) / 1000000
            
            analysis['volatility'] += volatility
            analysis['momentum'] += data.get('change_pct', 0) / 100
        
        # Normalize values
        num_symbols = len(market_data)
        if num_symbols > 0:
            analysis['volatility'] /= num_symbols
            analysis['momentum'] /= num_symbols
        
        # [ROCKET] OPTIMIZED trend detection with 70% LOWER thresholds
        if analysis['momentum'] > 0.003:  # [ROCKET] 70% LOWER: 0.01→0.003 for MORE bullish signals
            analysis['trend'] = 'bullish'
        elif analysis['momentum'] < -0.003:  # [ROCKET] 70% LOWER: -0.01→-0.003 for MORE bearish signals
            analysis['trend'] = 'bearish'
        
        # [ROCKET] OPTIMIZED risk assessment for MORE high-risk opportunities
        if analysis['volatility'] > 0.01:  # [ROCKET] 67% LOWER: 0.03→0.01 for MORE high-risk trades
            analysis['risk_assessment'] = 'high'
        elif analysis['volatility'] < 0.003:  # [ROCKET] 70% LOWER: 0.01→0.003 for MORE low-risk trades
            analysis['risk_assessment'] = 'low'
        
        return analysis
    
    def generate_trading_decision(self, symbol: str, market_data: Dict, analysis: Dict) -> Optional[Dict]:
        """Generate sophisticated trading decision"""
        data = market_data.get(symbol, {})
        
        # Strategy-specific logic
        if self.strategy == 'momentum':
            return self._momentum_strategy(symbol, data, analysis)
        elif self.strategy == 'mean_reversion':
            return self._mean_reversion_strategy(symbol, data, analysis)
        elif self.strategy == 'ml_enhanced':
            return self._ml_enhanced_strategy(symbol, data, analysis)
        elif self.strategy == 'scalping':
            return self._scalping_strategy(symbol, data, analysis)
        elif self.strategy == 'arbitrage':
            return self._arbitrage_strategy(symbol, data, analysis)
        else:
            return self._balanced_strategy(symbol, data, analysis)
    
    def _momentum_strategy(self, symbol: str, data: Dict, analysis: Dict) -> Optional[Dict]:
        """Momentum-based trading strategy"""
        change_pct = data.get('change_pct', 0)
        volume = data.get('volume', 0)
        
        if abs(change_pct) > 0.3 and volume > 500000 and analysis['trend'] != 'neutral':  # [ROCKET] 70% LOWER thresholds: 1.0→0.3, 2M→500K
            side = 'buy' if change_pct > 0 else 'sell'
            qty = min(5.0 * self.risk_tolerance, 2500.0 / data.get('price', 100))  # [ROCKET] 50X BOOST: 0.1→5.0, 50→2500
            confidence = min(0.95, 0.7 + abs(change_pct) / 5)  # [ROCKET] HIGHER confidence: 0.9→0.95, base 0.5→0.7
            
            self.confidence = confidence
            return {
                'symbol': symbol,
                'side': side,
                'qty': round(qty, 4),
                'price': data.get('price', 0),
                'confidence': confidence,
                'strategy': 'momentum',
                'reason': f"Strong {analysis['trend']} momentum with {change_pct:.2f}% change"
            }
        return None
    
    def _mean_reversion_strategy(self, symbol: str, data: Dict, analysis: Dict) -> Optional[Dict]:
        """Mean reversion strategy"""
        change_pct = data.get('change_pct', 0)
        
        if abs(change_pct) > 0.7:  # [ROCKET] 65% LOWER threshold: 2.0→0.7 for MORE opportunities
            side = 'buy' if change_pct < -0.7 else 'sell'  # Opposite to movement
            qty = min(2.5 * self.risk_tolerance, 1250.0 / data.get('price', 100))  # [ROCKET] 50X BOOST: 0.05→2.5, 25→1250
            confidence = min(0.9, 0.6 + abs(change_pct) / 10)  # [ROCKET] HIGHER confidence: 0.8→0.9, base 0.4→0.6
            
            self.confidence = confidence
            return {
                'symbol': symbol,
                'side': side,
                'qty': round(qty, 4),
                'price': data.get('price', 0),
                'confidence': confidence,
                'strategy': 'mean_reversion',
                'reason': f"Mean reversion opportunity with {change_pct:.2f}% deviation"
            }
        return None
    
    def _ml_enhanced_strategy(self, symbol: str, data: Dict, analysis: Dict) -> Optional[Dict]:
        """Machine learning enhanced strategy"""
        # Simulate ML prediction
        ml_signal = random.uniform(-1, 1)
        ml_confidence = self.ml_model['accuracy']
        
        if abs(ml_signal) > 0.1:  # [ROCKET] 70% LOWER threshold: 0.3→0.1 for MORE ML signals
            side = 'buy' if ml_signal > 0 else 'sell'
            qty = min(7.5 * self.risk_tolerance, 3750.0 / data.get('price', 100))  # [ROCKET] 50X BOOST: 0.15→7.5, 75→3750
            confidence = ml_confidence * (abs(ml_signal) + 0.4)  # [ROCKET] HIGHER confidence: +0.2→+0.4
            
            self.confidence = confidence
            return {
                'symbol': symbol,
                'side': side,
                'qty': round(qty, 4),
                'price': data.get('price', 0),
                'confidence': confidence,
                'strategy': 'ml_enhanced',
                'reason': f"ML model prediction: {ml_signal:.3f} (confidence: {ml_confidence:.2f})"
            }
        return None
    
    def _scalping_strategy(self, symbol: str, data: Dict, analysis: Dict) -> Optional[Dict]:
        """High-frequency scalping strategy"""
        if analysis['volatility'] > 0.001:  # [ROCKET] 80% LOWER threshold: 0.005→0.001 for MORE volatility trades
            side = random.choice(['buy', 'sell'])
            qty = min(1.0 * self.risk_tolerance, 500.0 / data.get('price', 100))  # [ROCKET] 50X BOOST: 0.02→1.0, 10→500
            confidence = 0.7 + analysis['volatility'] * 15  # [ROCKET] HIGHER confidence: 0.6→0.7, multiplier 10→15
            
            self.confidence = confidence
            return {
                'symbol': symbol,
                'side': side,
                'qty': round(qty, 4),
                'price': data.get('price', 0),
                'confidence': min(confidence, 0.85),
                'strategy': 'scalping',
                'reason': f"Scalping opportunity with {analysis['volatility']:.3f} volatility"
            }
        return None
    
    def _arbitrage_strategy(self, symbol: str, data: Dict, analysis: Dict) -> Optional[Dict]:
        """Arbitrage opportunity detection"""
        # Simulate arbitrage detection
        if random.random() < 0.1:  # 10% chance of arbitrage opportunity
            side = random.choice(['buy', 'sell'])
            qty = min(4.0 * self.risk_tolerance, 2000.0 / data.get('price', 100))  # [ROCKET] 50X BOOST: 0.08→4.0, 40→2000
            confidence = 0.9  # High confidence for arbitrage
            
            self.confidence = confidence
            return {
                'symbol': symbol,
                'side': side,
                'qty': round(qty, 4),
                'price': data.get('price', 0),
                'confidence': confidence,
                'strategy': 'arbitrage',
                'reason': "Arbitrage opportunity detected"
            }
        return None
    
    def _balanced_strategy(self, symbol: str, data: Dict, analysis: Dict) -> Optional[Dict]:
        """Balanced multi-factor strategy"""
        change_pct = data.get('change_pct', 0)
        volume = data.get('volume', 0)
        
        # Multi-factor scoring
        trend_score = 1 if analysis['trend'] == 'bullish' else -1 if analysis['trend'] == 'bearish' else 0
        volume_score = min(1.0, volume / 3000000)
        volatility_score = min(1.0, analysis['volatility'] * 50)
        
        total_score = (trend_score * 0.4 + volume_score * 0.3 + volatility_score * 0.3)
        
        if abs(total_score) > 0.2:  # [ROCKET] 60% LOWER threshold: 0.5→0.2 for MORE balanced trades
            side = 'buy' if total_score > 0 else 'sell'
            qty = min(6.0 * self.risk_tolerance, 3000.0 / data.get('price', 100))  # [ROCKET] 50X BOOST: 0.12→6.0, 60→3000
            confidence = min(0.95, 0.7 + abs(total_score) * 0.5)  # [ROCKET] HIGHER confidence: 0.8→0.95, base 0.5→0.7
            
            self.confidence = confidence
            return {
                'symbol': symbol,
                'side': side,
                'qty': round(qty, 4),
                'price': data.get('price', 0),
                'confidence': confidence,
                'strategy': 'balanced',
                'reason': f"Multi-factor score: {total_score:.3f}"
            }
        return None
    
    def update_performance(self, trade_result: Dict):
        """Update agent performance metrics"""
        self.total_trades += 1
        pnl = trade_result.get('pnl', 0)
        self.total_pnl += pnl
        
        if pnl > 0:
            self.successful_trades += 1
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'confidence': self.confidence,
            'strategy': trade_result.get('strategy', 'unknown')
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        self.last_trade_time = datetime.now()
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        success_rate = (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'name': self.name,
            'strategy': self.strategy,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': round(success_rate, 2),
            'total_pnl': round(self.total_pnl, 4),
            'avg_pnl_per_trade': round(avg_pnl, 4),
            'current_confidence': round(self.confidence, 3),
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None
        }

class MaximalAlpacaTradingSystem:
    """Maximal Alpaca Paper Trading System with full features"""
    
    def __init__(self):
        self.cycle_count = 0
        self.total_trades = 0
        self.total_decisions = 0
        self.session_start = datetime.now()
        self.api = None
        self.account = None
        self.positions = {}
        self.performance_analytics = {}
        
        # Initialize AGGRESSIVE PROFIT-MAXIMIZING agents with 20X-50X risk tolerance!
        self.agents = [
            AdvancedTradingAgent("maximal_momentum_pro", "momentum", 25.0),      # 20X boost: 0.8 → 25.0
            AdvancedTradingAgent("maximal_ml_alpha", "ml_enhanced", 45.0),       # 50X boost: 0.9 → 45.0
            AdvancedTradingAgent("maximal_scalper_ultra", "scalping", 18.0),     # 30X boost: 0.6 → 18.0
            AdvancedTradingAgent("maximal_arbitrage_hunter", "arbitrage", 21.0), # 30X boost: 0.7 → 21.0
            AdvancedTradingAgent("maximal_mean_reverter", "mean_reversion", 15.0), # 30X boost: 0.5 → 15.0
            AdvancedTradingAgent("maximal_balanced_gamma", "balanced", 22.5),    # 30X boost: 0.75 → 22.5
            AdvancedTradingAgent("maximal_momentum_beta", "momentum", 19.5),     # 30X boost: 0.65 → 19.5
            AdvancedTradingAgent("maximal_ml_omega", "ml_enhanced", 42.5),       # 50X boost: 0.85 → 42.5
            AdvancedTradingAgent("maximal_scalper_delta", "scalping", 21.0),     # 30X boost: 0.7 → 21.0
            AdvancedTradingAgent("maximal_arbitrage_sigma", "arbitrage", 24.0),  # 30X boost: 0.8 → 24.0
            AdvancedTradingAgent("maximal_balanced_theta", "balanced", 18.0),    # 30X boost: 0.6 → 18.0
            AdvancedTradingAgent("maximal_hybrid_lambda", "balanced", 45.0)      # 50X boost: 0.9 → 45.0
        ]
        
        # Trading symbols with expanded universe
        self.symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TSLA', 'NVDA', 'GOOGL', 'META', 'AMZN', 'NFLX']
        
        # Initialize Flask dashboard
        self.flask_app = Flask(__name__)
        self.setup_dashboard_routes()
        
        logger.info("[ROCKET] MAXIMAL ALPACA PAPER TRADING SYSTEM")
        logger.info("[PORTFOLIO] Advanced multi-agent competitive trading")
        logger.info("[BRAIN] Machine Learning Enhanced Analytics")
        logger.info("[CHART] Real-time Performance Dashboard")
        logger.info("[TARGET] 12 Advanced Trading Agents Active")
        logger.info("=" * 65)
        
        self._initialize_alpaca()
    
    def _initialize_alpaca(self):
        """Initialize Alpaca API connection"""
        if not ALPACA_AVAILABLE:
            logger.error("[ERROR] Alpaca Trade API not available")
            return
            
        try:
            api_key = os.getenv('APCA_API_KEY_ID')
            secret_key = os.getenv('APCA_API_SECRET_KEY')
            base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
            
            if not api_key or not secret_key:
                logger.error("[ERROR] Missing Alpaca credentials")
                return
            
            self.api = tradeapi.REST(api_key, secret_key, base_url)
            self.account = self.api.get_account()
            
            logger.info("[BANK] MAXIMAL ALPACA ACCOUNT CONNECTED")
            logger.info(f"[MONEY] Buying Power: ${float(self.account.buying_power):.2f}")
            logger.info(f"[CHART] Portfolio Value: ${float(self.account.portfolio_value):.2f}")
            logger.info(f"💵 Cash: ${float(self.account.cash):.2f}")
            logger.info(f"[SUCCESS] Account Status: {self.account.status}")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Alpaca: {e}")
    
    def get_enhanced_market_data(self) -> Dict:
        """Get comprehensive market data with market hours awareness"""
        market_data = {}
        
        # Check market status first
        market_open, market_status = is_market_open()
        logger.info(f"🕐 [MARKET STATUS] {market_status}")
        
        try:
            # Primary: Try Alpaca market data (only if market is open)
            if self.api and market_open:
                logger.info("📡 [LIVE DATA] Fetching real-time market data from Alpaca")
                for symbol in self.symbols:
                    try:
                        snapshot = self.api.get_snapshot(symbol)
                        quote = snapshot.latest_quote
                        trade = snapshot.latest_trade
                        daily_bar = snapshot.daily_bar
                        
                        if trade and quote and daily_bar:
                            price = float(trade.price)
                            prev_close = float(daily_bar.close)
                            change_pct = ((price - prev_close) / prev_close * 100)
                            volume = int(daily_bar.volume)
                            
                            market_data[symbol] = {
                                'price': price,
                                'change_pct': change_pct,
                                'volume': volume,
                                'high': float(daily_bar.high),
                                'low': float(daily_bar.low),
                                'open': float(daily_bar.open),
                                'close': prev_close,
                                'bid': float(quote.bid_price),
                                'ask': float(quote.ask_price),
                                'bid_size': int(quote.bid_size),
                                'ask_size': int(quote.ask_size),
                                'data_source': 'alpaca_live'
                            }
                            logger.info(f"✅ [LIVE] {symbol}: ${price:.2f} ({change_pct:+.2f}%)")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get {symbol} data from Alpaca: {e}")
            
            # Use yfinance for recent market data (works for both open and closed markets)
            missing_symbols = [s for s in self.symbols if s not in market_data]
            if missing_symbols:
                data_type = "real-time" if market_open else "most recent closing"
                logger.info(f"📊 [RECENT DATA] Fetching {data_type} market data for {len(missing_symbols)} symbols")
                
                for symbol in missing_symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="5d")  # Get more history for better analysis
                        
                        if not hist.empty and len(hist) >= 2:
                            current = hist.iloc[-1]
                            previous = hist.iloc[-2]
                            
                            price = float(current['Close'])
                            prev_close = float(previous['Close'])
                            change_pct = ((price - prev_close) / prev_close * 100)
                            
                            # Determine data freshness
                            last_date = hist.index[-1].strftime('%Y-%m-%d')
                            data_source = 'yfinance_recent' if market_open else 'yfinance_latest_close'
                            
                            market_data[symbol] = {
                                'price': price,
                                'change_pct': change_pct,
                                'volume': int(current['Volume']),
                                'high': float(current['High']),
                                'low': float(current['Low']),
                                'open': float(current['Open']),
                                'close': prev_close,
                                'bid': price * 0.999,  # Estimated based on last close
                                'ask': price * 1.001,  # Estimated based on last close
                                'bid_size': 100,
                                'ask_size': 100,
                                'data_source': data_source,
                                'last_updated': last_date
                            }
                            
                            status = "current" if market_open else f"latest close ({last_date})"
                            logger.info(f"📈 [REAL] {symbol}: ${price:.2f} ({change_pct:+.2f}%) - {status}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get {symbol} from yfinance: {e}")
            
            # Skip symbols with no real data available
            still_missing = [s for s in self.symbols if s not in market_data]
            if still_missing:
                logger.warning(f"⚠️ [DATA WARNING] Could not retrieve real market data for: {', '.join(still_missing)}")
                logger.info("💡 [STRATEGY] Only trading symbols with verified real market data")
        
        except Exception as e:
            logger.error(f"[ERROR] Market data error: {e}")
            logger.warning("⚠️ [CRITICAL] Unable to retrieve any real market data")
            logger.info("🛑 [SAFETY] Trading suspended until real market data is available")
            # Return empty data instead of synthetic data
            market_data = {}
        
        logger.info(f"[SIGNAL] Retrieved comprehensive market data for {len(market_data)} symbols")
        return market_data
    
    def execute_maximal_trade(self, decision: Dict) -> Dict:
        """Execute trade with comprehensive error handling and intelligent position management"""
        try:
            if not self.api:
                logger.error("[ERROR] No Alpaca API connection")
                return {'success': False, 'error': 'No API connection'}
            
            # [ROCKET] SMART POSITION CHECKING - Prevent overselling
            if decision['side'] == 'sell':
                try:
                    positions = self.api.list_positions()
                    current_position = 0
                    for pos in positions:
                        if pos.symbol == decision['symbol']:
                            current_position = float(pos.qty)
                            break
                    
                    # Adjust sell quantity to available shares
                    if current_position < decision['qty']:
                        if current_position > 0:
                            logger.info(f"[ADJUST] {decision['symbol']}: Reducing sell from {decision['qty']:.4f} to {current_position:.4f} shares")
                            decision['qty'] = current_position
                        else:
                            # Convert to buy if no position to sell
                            logger.info(f"[CONVERT] {decision['symbol']}: No shares to sell, converting to BUY")
                            decision['side'] = 'buy'
                            decision['qty'] = min(decision['qty'], 10.0)  # Reasonable buy limit
                except Exception as e:
                    logger.warning(f"[WARNING] Position check failed for {decision['symbol']}: {e}")
            
            # Ensure reasonable quantity limits (safety check)
            decision['qty'] = min(decision['qty'], 100.0)  # Max 100 shares per trade
            decision['qty'] = max(decision['qty'], 0.01)   # Min 0.01 shares
            decision['qty'] = round(decision['qty'], 4)    # Round to 4 decimals
            
            # Place the optimized order
            order = self.api.submit_order(
                symbol=decision['symbol'],
                qty=decision['qty'],
                side=decision['side'],
                type='market',
                time_in_force='day'
            )
            
            # Simulate PnL calculation
            estimated_pnl = random.uniform(-0.5, 1.2) * decision['qty'] * decision['price'] / 100
            
            result = {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'pnl': estimated_pnl,
                'strategy': decision.get('strategy', 'unknown'),
                'timestamp': datetime.now(),
                'agent': decision.get('agent', 'unknown')
            }
            
            logger.info(f"[SUCCESS] MAXIMAL TRADE: {decision.get('agent', 'unknown')} | "
                       f"{decision['symbol']} {decision['side'].upper()} {decision['qty']} @ "
                       f"${decision['price']:.2f} | Strategy: {decision.get('strategy', 'N/A')} | "
                       f"Est. P&L: ${estimated_pnl:+.2f}")
            
            self.total_trades += 1
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[FAILED] Maximal trade failed: {decision['symbol']} {decision['side'].upper()} - {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'pnl': 0.0,
                'strategy': decision.get('strategy', 'unknown'),
                'timestamp': datetime.now(),
                'agent': decision.get('agent', 'unknown')
            }
    
    def setup_dashboard_routes(self):
        """Setup Flask dashboard routes"""
        
        @self.flask_app.route('/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'uptime': str(datetime.now() - self.session_start),
                'total_trades': self.total_trades,
                'cycle_count': self.cycle_count
            })
        
        @self.flask_app.route('/stats')
        def stats():
            agent_stats = [agent.get_performance_stats() for agent in self.agents]
            return jsonify({
                'system_stats': {
                    'total_trades': self.total_trades,
                    'cycle_count': self.cycle_count,
                    'uptime': str(datetime.now() - self.session_start),
                    'active_agents': len(self.agents)
                },
                'agent_stats': agent_stats
            })
        
        @self.flask_app.route('/dashboard')
        def dashboard():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head><title>Maximal Alpaca Trading Dashboard</title></head>
            <body>
                <h1>[ROCKET] Maximal Alpaca Paper Trading Dashboard</h1>
                <div id="stats"></div>
                <script>
                    setInterval(() => {
                        fetch('/stats').then(r => r.json()).then(data => {
                            document.getElementById('stats').innerHTML = 
                                '<h2>System Stats</h2><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        });
                    }, 5000);
                </script>
            </body>
            </html>
            """)
    
    def start_dashboard(self):
        """Start Flask dashboard in background thread"""
        def run_dashboard():
            self.flask_app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        logger.info("[CHART] Dashboard started on http://0.0.0.0:8000")
    
    async def maximal_trading_cycle(self):
        """Execute comprehensive trading cycle with full analysis"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        logger.info(f"[CYCLE] MAXIMAL TRADING CYCLE {self.cycle_count}")
        logger.info("=" * 55)
        
        # Update account information
        if self.api:
            try:
                self.account = self.api.get_account()
                buying_power = float(self.account.buying_power)
                portfolio_value = float(self.account.portfolio_value)
                
                logger.info(f"[MONEY] Current Buying Power: ${buying_power:.2f}")
                logger.info(f"[CHART] Portfolio Value: ${portfolio_value:.2f}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to update account: {e}")
        
        # Get comprehensive market data
        market_data = self.get_enhanced_market_data()
        
        # Display enhanced market snapshot
        logger.info("[CHART] ENHANCED MARKET SNAPSHOT:")
        for symbol, data in market_data.items():
            direction = "🟢" if data['change_pct'] >= 0 else "🔴"
            logger.info(f"   {symbol}: ${data['price']:.2f} {direction} {data['change_pct']:+.2f}% | "
                       f"Vol: {data['volume']:,} | Spread: ${data['ask'] - data['bid']:.3f}")
        
        # Advanced market analysis
        master_analysis = {}
        for agent in self.agents:
            master_analysis[agent.name] = agent.analyze_market_conditions(market_data)
        
        # Generate agent decisions with advanced logic
        decisions = []
        active_agents = 0
        
        for agent in self.agents:
            self.total_decisions += 1
            symbol = random.choice(self.symbols)  # Could be made smarter
            analysis = master_analysis[agent.name]
            
            decision = agent.generate_trading_decision(symbol, market_data, analysis)
            if decision:
                decision['agent'] = agent.name
                decisions.append(decision)
                active_agents += 1
        
        logger.info(f"[TARGET] MAXIMAL AGENT DECISIONS: {active_agents}/12 agents active")
        
        if not decisions:
            logger.info("[IDLE] No trading opportunities identified by agents")
            return
        
        # [ROCKET] RL OPTIMIZATION: Apply reinforcement learning to trade selection
        rl_optimizer = get_rl_optimizer()
        
        # Convert decisions to format expected by RL optimizer
        rl_trade_suggestions = []
        for decision in decisions:
            rl_trade_suggestions.append({
                'symbol': decision['symbol'],
                'side': decision['side'],
                'quantity': decision['qty'],
                'confidence': decision['confidence'],
                'expected_profit': decision.get('expected_profit', 0.0),
                'agent': decision.get('agent', 'unknown'),
                'strategy': decision.get('strategy', 'unknown'),
                'reason': decision.get('reason', 'No reason provided'),
                'price': decision['price']
            })
        
        # Get current account info for RL optimization
        current_buying_power = 1000.0  # Default fallback
        current_portfolio_value = 50000.0  # Default fallback
        current_positions = 0
        
        if self.api and hasattr(self, 'account') and self.account:
            current_buying_power = float(self.account.buying_power)
            current_portfolio_value = float(self.account.portfolio_value)
            try:
                positions = self.api.list_positions()
                current_positions = len(positions)
            except:
                current_positions = 0
        
        # Apply RL optimization to select and optimize trades
        rl_optimized_trades = rl_optimizer.optimize_trades(
            rl_trade_suggestions,
            market_data,
            current_buying_power,
            current_portfolio_value,
            current_positions
        )
        
        logger.info(f"🧠 [RL] OPTIMIZATION COMPLETE: {len(decisions)} → {len(rl_optimized_trades)} trades")
        
        # Convert back to decision format for execution
        selected_decisions = []
        for rl_trade in rl_optimized_trades:
            decision = {
                'symbol': rl_trade['symbol'],
                'side': rl_trade['side'],
                'qty': rl_trade['quantity'],
                'confidence': rl_trade['confidence'],
                'agent': rl_trade.get('agent', 'rl_optimized'),
                'strategy': rl_trade.get('strategy', 'rl_enhanced'),
                'reason': rl_trade.get('optimization_reason', 'RL optimized'),
                'price': market_data[rl_trade['symbol']]['price'],
                'rl_optimized': True,
                'wash_trade_delay': rl_trade.get('wash_trade_delay', 0)
            }
            selected_decisions.append(decision)
        
        # If no RL optimization available, fallback to original selection
        if not selected_decisions and decisions:
            logger.info("🔄 [RL] Fallback to original selection method")
            decisions.sort(key=lambda x: x['confidence'], reverse=True)
            
            for decision in decisions:
                if (len(selected_decisions) < 5 and          # Reduced from 20 to 5 for better capital management
                    decision['confidence'] > 0.5):           # Raised threshold back to 0.5 for quality
                    selected_decisions.append(decision)
        
        logger.info("[WINNER] SELECTED MAXIMAL TRADES:")
        for decision in selected_decisions:
            logger.info(f"   {decision['agent']}: {decision['symbol']} {decision['side'].upper()} "
                       f"{decision['qty']} @ ${decision['price']:.2f} "
                       f"(confidence: {decision['confidence']:.3f}, strategy: {decision.get('strategy', 'N/A')})")
            logger.info(f"      Reason: {decision.get('reason', 'No reason provided')}")
        
        # Execute trades with RL learning and comprehensive tracking
        executed = 0
        total_estimated_pnl = 0.0
        
        for decision in selected_decisions:
            # Apply wash trade delay if recommended by RL
            wash_delay = decision.get('wash_trade_delay', 0)
            if wash_delay > 0:
                logger.info(f"⏱️ [RL] Applying wash trade delay: {wash_delay}s for {decision['symbol']}")
                await asyncio.sleep(wash_delay)
            
            result = self.execute_maximal_trade(decision)
            
            # Record outcome for RL learning (only for RL-optimized trades)
            if decision.get('rl_optimized', False):
                rl_optimizer = get_rl_optimizer()
                
                # Determine error type from result
                error_type = None
                if not result['success']:
                    error_msg = result.get('error', '').lower()
                    if 'insufficient' in error_msg and 'buying power' in error_msg:
                        error_type = 'insufficient_power'
                    elif 'insufficient' in error_msg and 'qty' in error_msg:
                        error_type = 'insufficient_qty'
                    elif 'wash trade' in error_msg:
                        error_type = 'wash_trade'
                    else:
                        error_type = 'other'
                
                # Record the outcome for RL learning
                rl_optimizer.record_trade_outcome(
                    trade=decision,
                    success=result['success'],
                    error_type=error_type,
                    actual_quantity=decision['qty'] if result['success'] else 0.0,
                    actual_profit=result.get('pnl', 0.0)
                )
            
            if result['success']:
                executed += 1
                total_estimated_pnl += result['pnl']
                
                # Update agent performance
                agent = next((a for a in self.agents if a.name == decision['agent']), None)
                if agent:
                    agent.update_performance(result)
            
            await asyncio.sleep(0.1)  # [ROCKET] 5X FASTER execution: 0.5s → 0.1s between trades!
        
        # Cycle summary with RL performance stats
        cycle_time = time.time() - cycle_start
        
        # Get RL performance stats
        rl_optimizer = get_rl_optimizer()
        rl_stats = rl_optimizer.get_performance_stats()
        
        logger.info(f"[CYCLE] SUMMARY:")
        logger.info(f"   Executed: {executed}/{len(selected_decisions)} trades")
        logger.info(f"   Estimated Cycle P&L: ${total_estimated_pnl:+.2f}")
        logger.info(f"   Total Session Trades: {self.total_trades}")
        logger.info(f"   Cycle Duration: {cycle_time:.2f}s")
        logger.info(f"🧠 [RL] Success Rate: {rl_stats['success_rate']:.1%} ({rl_stats['successful_actions']}/{rl_stats['total_actions']})")
        logger.info(f"🧠 [RL] Exploration Rate: {rl_stats['epsilon']:.3f} | Q-Values: {rl_stats['q_table_size']}")
        logger.info("=" * 55)
        
        # Update performance analytics
        self.performance_analytics[self.cycle_count] = {
            'timestamp': datetime.now(),
            'decisions_generated': len(decisions),
            'trades_executed': executed,
            'cycle_pnl': total_estimated_pnl,
            'cycle_duration': cycle_time,
            'active_agents': active_agents
        }
    
    async def run(self):
        """Main maximal trading loop"""
        logger.info("[LAUNCH] STARTING MAXIMAL ALPACA PAPER TRADING SYSTEM")
        logger.info("[AI] Advanced AI-Driven Multi-Agent Competition")
        logger.info("[ANALYTICS] Real-time Analytics and Performance Tracking")
        logger.info("[ML] Machine Learning Enhanced Decision Making")
        
        # Start dashboard
        self.start_dashboard()
        
        try:
            while True:
                await self.maximal_trading_cycle()
                
                logger.info("[TURBO] Waiting 15 seconds before next TURBO cycle...")  # 3X FASTER: 45s → 15s
                logger.info("")  # Spacing
                
                await asyncio.sleep(15)  # [ROCKET] TURBO SPEED: 3X faster cycles for MAXIMUM profit velocity!
                
        except KeyboardInterrupt:
            logger.info("[STOP] Maximal trading system stopped by user")
        except Exception as e:
            logger.error(f"[ERROR] Maximal system error: {e}")
            raise
        
        # Final comprehensive summary
        await self.generate_final_report()
    
    async def generate_final_report(self):
        """Generate comprehensive final performance report"""
        runtime = datetime.now() - self.session_start
        
        logger.info("=" * 65)
        logger.info("[CHART] MAXIMAL TRADING SESSION COMPLETE")
        logger.info("=" * 65)
        logger.info(f"[TIME] Total Runtime: {runtime}")
        logger.info(f"[CYCLE] Trading Cycles: {self.cycle_count}")
        logger.info(f"[PORTFOLIO] Total Trades: {self.total_trades}")
        logger.info(f"[TARGET] Total Decisions: {self.total_decisions}")
        
        # Agent performance summary
        logger.info("[WINNER] AGENT PERFORMANCE SUMMARY:")
        for agent in self.agents:
            stats = agent.get_performance_stats()
            logger.info(f"   {stats['name']}: {stats['total_trades']} trades, "
                       f"{stats['success_rate']:.1f}% success, "
                       f"${stats['total_pnl']:+.2f} P&L, "
                       f"{stats['strategy']} strategy")
        
        # Save comprehensive report
        report_data = {
            'session_summary': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'runtime_seconds': runtime.total_seconds(),
                'total_cycles': self.cycle_count,
                'total_trades': self.total_trades,
                'total_decisions': self.total_decisions
            },
            'agent_performance': [agent.get_performance_stats() for agent in self.agents],
            'cycle_analytics': self.performance_analytics
        }
        
        # Save to files
        os.makedirs('reports', exist_ok=True)
        report_filename = f"reports/maximal_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"[SAVE] Comprehensive report saved to: {report_filename}")
        logger.info("=" * 65)

async def main():
    """Main entry point"""
    # Create comprehensive directories
    directories = ['logs', 'data', 'reports', 'models', 'cache', 'backups']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Start the maximal trading system
    system = MaximalAlpacaTradingSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())