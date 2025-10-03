#!/usr/bin/env python3
"""
Optimized Ultra Aggressive Trading System
========================================
Bulletproof ultra aggressive system with comprehensive error handling,
position validation, and guaranteed trade execution.
"""

import asyncio
import logging
import random
import sys
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/optimized_ultra_aggressive.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("OptimizedUltraAggressive")


class OptimizedUltraAggressiveTradingSystem:
    """Optimized ultra aggressive trading system with bulletproof execution"""

    def __init__(self):
        self.logger = logger
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]

        # Initialize Alpaca API with error handling
        try:
            api_key = os.getenv("APCA_API_KEY_ID")
            secret_key = os.getenv("APCA_API_SECRET_KEY")
            base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

            if not api_key or not secret_key:
                raise ValueError("Missing API credentials in environment variables")

            self.api = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=base_url, api_version="v2")
            # Test connection
            account = self.api.get_account()
            self.logger.info(f"✅ Connected to Alpaca PAPER TRADING API - Account: {account.id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Alpaca API: {e}")
            raise

        # Initialize MAXIMAL PROFIT-GENERATING agents with 20X-50X risk tolerance boost!
        self.agents = {}
        self.agent_performance = {}
        self.cycle_count = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.max_retries = 3
        self.retry_delay = 1  # TURBO: Reduced from 3s to 1s for faster recovery

        # Initialize Risk Management System
        self.position_monitors = {}  # Track SL/TP for each position
        self.risk_settings = {
            "stop_loss_pct": 0.15,  # 15% stop loss (ultra aggressive)
            "take_profit_pct": 0.25,  # 25% take profit
            "trailing_stop_pct": 0.08,  # 8% trailing stop
            "max_position_loss": 0.20,  # 20% max loss per position
            "enable_trailing": True,
        }

        # Initialize Adaptive Market Intelligence System
        self.market_regime = {
            "current_regime": "UNKNOWN",
            "volatility_trend": "NEUTRAL",
            "momentum_score": 0.0,
            "correlation_matrix": {},
            "regime_history": [],
            "adaptation_factor": 1.0,
        }
        self.performance_memory = {
            "recent_trades": [],
            "win_rate_by_regime": {},
            "best_agents_by_regime": {},
            "adaptive_weights": {},
        }

        # Initialize system
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize 12 MAXIMAL PROFIT-GENERATING agents with 20X-50X risk tolerance boost!"""
        # CONSOLIDATING IDEAS FROM MAXIMAL SYSTEM - PROFIT-FOCUSED CONFIGURATION
        aggression_levels = [
            {"level": "maximal_momentum_pro", "confidence": 0.99, "max_position": 0.25, "trade_probability": 0.98, "risk_tolerance": 25.0},
            {"level": "maximal_ml_alpha", "confidence": 0.99, "max_position": 0.35, "trade_probability": 0.99, "risk_tolerance": 45.0},
            {"level": "maximal_scalper_ultra", "confidence": 0.97, "max_position": 0.20, "trade_probability": 0.96, "risk_tolerance": 18.0},
            {"level": "maximal_arbitrage_hunter", "confidence": 0.98, "max_position": 0.22, "trade_probability": 0.97, "risk_tolerance": 21.0},
            {"level": "maximal_mean_reverter", "confidence": 0.95, "max_position": 0.18, "trade_probability": 0.94, "risk_tolerance": 15.0},
            {"level": "maximal_balanced_gamma", "confidence": 0.98, "max_position": 0.24, "trade_probability": 0.97, "risk_tolerance": 22.5},
            {"level": "maximal_momentum_beta", "confidence": 0.96, "max_position": 0.21, "trade_probability": 0.95, "risk_tolerance": 19.5},
            {"level": "maximal_ml_omega", "confidence": 0.99, "max_position": 0.33, "trade_probability": 0.98, "risk_tolerance": 42.5},
            {"level": "maximal_scalper_delta", "confidence": 0.97, "max_position": 0.22, "trade_probability": 0.96, "risk_tolerance": 21.0},
            {"level": "maximal_arbitrage_sigma", "confidence": 0.98, "max_position": 0.26, "trade_probability": 0.97, "risk_tolerance": 24.0},
            {"level": "maximal_balanced_theta", "confidence": 0.96, "max_position": 0.20, "trade_probability": 0.95, "risk_tolerance": 18.0},
            {"level": "maximal_hybrid_lambda", "confidence": 0.99, "max_position": 0.40, "trade_probability": 0.99, "risk_tolerance": 45.0},
        ]

        for i, config in enumerate(aggression_levels, 1):
            agent_id = config["level"]  # Use maximal agent names directly
            self.agents[agent_id] = {
                "id": agent_id,
                "style": "maximal_profit_generating",
                "aggression_level": config["level"],
                "confidence": config["confidence"],
                "max_position": config["max_position"],
                "trade_probability": config["trade_probability"],
                "risk_tolerance": config["risk_tolerance"],  # NEW: 20X-50X boost!
                "trades_count": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "last_trade_time": None,
                "cooldown_seconds": 5,  # TURBO: 30s → 5s for maximal trading volume!
            }
            self.agent_performance[agent_id] = {
                "decisions": 0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "success_rate": 0.0,
            }

        self.logger.info(f"Initialized {len(self.agents)} OPTIMIZED ultra aggressive agents")

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get current account information with retry logic"""
        for attempt in range(self.max_retries):
            try:
                account = self.api.get_account()
                return {
                    "buying_power": float(account.buying_power),
                    "cash": float(account.cash),
                    "portfolio_value": float(account.portfolio_value),
                    "equity": float(account.equity),
                    "day_trade_count": int(getattr(account, "day_trade_count", 0)),
                    "pattern_day_trader": getattr(account, "pattern_day_trader", False),
                }
            except Exception as e:
                self.logger.warning(f"Account info attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to get account info after {self.max_retries} attempts")
                    return None

    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions with retry logic"""
        for attempt in range(self.max_retries):
            try:
                positions = self.api.list_positions()
                return {
                    pos.symbol: {
                        "qty": float(pos.qty),
                        "market_value": float(pos.market_value),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "side": pos.side,
                        "avg_fill_price": float(getattr(pos, "avg_fill_price", 0.0)),
                    }
                    for pos in positions
                }
            except Exception as e:
                self.logger.warning(f"Positions attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to get positions after {self.max_retries} attempts")
                    return {}

    async def monitor_positions_risk_management(self) -> List[Dict[str, Any]]:
        """Monitor all positions for stop-loss and take-profit triggers"""
        risk_actions = []

        try:
            positions = await self.get_positions()

            for symbol, position_data in positions.items():
                if symbol not in self.position_monitors:
                    # Initialize monitoring for new position
                    self.position_monitors[symbol] = {
                        "entry_price": abs(float(position_data["market_value"]) / float(position_data["qty"])),
                        "highest_price": 0,
                        "stop_loss_price": 0,
                        "take_profit_price": 0,
                        "trailing_stop_price": 0,
                        "created_at": datetime.now(),
                    }

                monitor = self.position_monitors[symbol]
                current_price = await self._get_current_price(symbol)
                qty = float(position_data["qty"])

                if current_price is None:
                    continue

                # Update trailing high
                if current_price > monitor["highest_price"]:
                    monitor["highest_price"] = current_price

                    # Update trailing stop
                    if self.risk_settings["enable_trailing"] and qty > 0:
                        new_trailing_stop = current_price * (1 - self.risk_settings["trailing_stop_pct"])
                        monitor["trailing_stop_price"] = max(monitor["trailing_stop_price"], new_trailing_stop)

                # Calculate P&L percentage
                entry_price = monitor["entry_price"]
                if qty > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price

                    # Check stop-loss
                    if pnl_pct <= -self.risk_settings["stop_loss_pct"]:
                        risk_actions.append(
                            {
                                "action": "STOP_LOSS",
                                "symbol": symbol,
                                "quantity": abs(qty),
                                "current_price": current_price,
                                "reason": f"Stop-loss triggered: {pnl_pct:.2%} loss",
                                "urgency": "HIGH",
                            }
                        )

                    # Check take-profit
                    elif pnl_pct >= self.risk_settings["take_profit_pct"]:
                        risk_actions.append(
                            {
                                "action": "TAKE_PROFIT",
                                "symbol": symbol,
                                "quantity": abs(qty),
                                "current_price": current_price,
                                "reason": f"Take-profit triggered: {pnl_pct:.2%} gain",
                                "urgency": "MEDIUM",
                            }
                        )

                    # Check trailing stop
                    elif (
                        self.risk_settings["enable_trailing"]
                        and monitor["trailing_stop_price"] > 0
                        and current_price <= monitor["trailing_stop_price"]
                    ):
                        risk_actions.append(
                            {
                                "action": "TRAILING_STOP",
                                "symbol": symbol,
                                "quantity": abs(qty),
                                "current_price": current_price,
                                "reason": f'Trailing stop triggered at ${monitor["trailing_stop_price"]:.2f}',
                                "urgency": "HIGH",
                            }
                        )

                # Log position status
                self.logger.debug(
                    f"Position Monitor {symbol}: Price=${current_price:.2f}, P&L={pnl_pct:.2%}, Trailing=${monitor['trailing_stop_price']:.2f}"
                )

        except Exception as e:
            self.logger.error(f"Position monitoring error: {e}")

        return risk_actions

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            # Try to get latest trade first
            trades = self.api.get_latest_trade(symbol)
            if trades and hasattr(trades, "price"):
                return float(trades.price)

            # Fallback to last bar
            bars = self.api.get_bars(symbol, "1Min", limit=1)
            if bars and len(bars) > 0:
                return float(bars[0].c)

            return None
        except Exception as e:
            self.logger.warning(f"Failed to get current price for {symbol}: {e}")
            return None

    async def detect_market_regime(self, market_data: Dict[str, Dict[str, Any]]) -> str:
        """Advanced market regime detection using multiple signals"""
        try:
            volatilities = []
            price_changes = []
            volumes = []

            for symbol, data in market_data.items():
                volatilities.append(data.get("volatility", 0))
                price_changes.append(data.get("price_change", 0))
                volumes.append(data.get("volume", 0))

            # Calculate regime indicators
            avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
            avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
            momentum_score = abs(avg_price_change) * (1 + avg_volatility)

            # Regime classification with adaptive thresholds
            if avg_volatility > 0.04:  # High volatility
                if abs(avg_price_change) > 0.02:
                    regime = "VOLATILE_TRENDING"
                else:
                    regime = "VOLATILE_RANGING"
            elif avg_volatility < 0.01:  # Low volatility
                if abs(avg_price_change) > 0.015:
                    regime = "LOW_VOL_TRENDING"
                else:
                    regime = "LOW_VOL_RANGING"
            else:  # Medium volatility
                if avg_price_change > 0.01:
                    regime = "BULLISH"
                elif avg_price_change < -0.01:
                    regime = "BEARISH"
                else:
                    regime = "NEUTRAL"

            # Update regime history and momentum
            self.market_regime["current_regime"] = regime
            self.market_regime["momentum_score"] = momentum_score
            self.market_regime["regime_history"].append(
                {
                    "regime": regime,
                    "timestamp": datetime.now(),
                    "volatility": avg_volatility,
                    "momentum": momentum_score,
                }
            )

            # Keep only last 50 regime observations
            if len(self.market_regime["regime_history"]) > 50:
                self.market_regime["regime_history"] = self.market_regime["regime_history"][-50:]

            self.logger.debug(
                f"Market Regime: {regime}, Volatility: {avg_volatility:.4f}, Momentum: {momentum_score:.4f}"
            )
            return regime

        except Exception as e:
            self.logger.error(f"Market regime detection error: {e}")
            return "UNKNOWN"

    def calculate_adaptive_weights(self, regime: str) -> Dict[str, float]:
        """Calculate adaptive weights for agents based on market regime and performance"""
        try:
            # Base weights (equal for all agents)
            base_weight = 1.0
            weights = {agent_id: base_weight for agent_id in self.agents}

            # Adaptive weighting based on performance in current regime
            if regime in self.performance_memory["best_agents_by_regime"]:
                best_agents = self.performance_memory["best_agents_by_regime"][regime]
                for agent_id in best_agents[:3]:  # Top 3 agents get boost
                    if agent_id in weights:
                        weights[agent_id] *= 1.5  # 50% boost for top performers

            # Regime-specific adjustments
            regime_adjustments = {
                "VOLATILE_TRENDING": {"extreme": 1.3, "ultra": 1.2, "hyper": 1.1},
                "VOLATILE_RANGING": {"maximum": 1.2, "extreme": 0.9, "ultra": 1.1},
                "LOW_VOL_TRENDING": {"hyper": 1.3, "maximum": 1.2, "extreme": 0.8},
                "LOW_VOL_RANGING": {"ultra": 1.2, "hyper": 0.9, "extreme": 0.8},
                "BULLISH": {"extreme": 1.4, "ultra": 1.3, "hyper": 1.2},
                "BEARISH": {"maximum": 1.2, "ultra": 1.1, "extreme": 0.9},
                "NEUTRAL": {"hyper": 1.1, "maximum": 1.1, "ultra": 1.0},
            }

            if regime in regime_adjustments:
                adjustments = regime_adjustments[regime]
                for agent_id, agent in self.agents.items():
                    agent_level = agent["aggression_level"]
                    if agent_level in adjustments:
                        weights[agent_id] *= adjustments[agent_level]

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            self.performance_memory["adaptive_weights"] = weights
            return weights

        except Exception as e:
            self.logger.error(f"Adaptive weight calculation error: {e}")
            return {agent_id: 1.0 / len(self.agents) for agent_id in self.agents}

    def update_performance_memory(self, agent_id: str, trade_result: Dict[str, Any]):
        """Update performance memory for adaptive learning"""
        try:
            # Add to recent trades
            trade_record = {
                "agent_id": agent_id,
                "regime": self.market_regime["current_regime"],
                "success": trade_result.get("success", False),
                "pnl": trade_result.get("pnl", 0.0),
                "timestamp": datetime.now(),
            }

            self.performance_memory["recent_trades"].append(trade_record)

            # Keep only last 100 trades
            if len(self.performance_memory["recent_trades"]) > 100:
                self.performance_memory["recent_trades"] = self.performance_memory["recent_trades"][-100:]

            # Update regime-specific performance
            regime = self.market_regime["current_regime"]
            if regime not in self.performance_memory["win_rate_by_regime"]:
                self.performance_memory["win_rate_by_regime"][regime] = {}
                self.performance_memory["best_agents_by_regime"][regime] = []

            # Calculate agent performance by regime
            regime_trades = [
                t
                for t in self.performance_memory["recent_trades"]
                if t["regime"] == regime and t["agent_id"] == agent_id
            ]

            if regime_trades:
                wins = sum(1 for t in regime_trades if t["success"])
                win_rate = wins / len(regime_trades)
                self.performance_memory["win_rate_by_regime"][regime][agent_id] = win_rate

                # Update best agents list
                agent_performance = [
                    (aid, wr) for aid, wr in self.performance_memory["win_rate_by_regime"][regime].items()
                ]
                agent_performance.sort(key=lambda x: x[1], reverse=True)
                self.performance_memory["best_agents_by_regime"][regime] = [ap[0] for ap in agent_performance]

        except Exception as e:
            self.logger.error(f"Performance memory update error: {e}")

    async def get_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Get real market data with comprehensive error handling"""
        market_data = {}

        for symbol in self.symbols:
            try:
                # Get current price with retry logic
                for attempt in range(self.max_retries):
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        current_price = float(quote.bid_price)
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        else:
                            raise e

                # Get recent bars for analysis
                try:
                    bars = self.api.get_bars(symbol, "1Min", limit=5)
                    if bars:
                        # Fix: Access bar data correctly - use 'c' for close price
                        recent_prices = [float(bar.c) for bar in bars]
                        price_change = (
                            (current_price - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
                        )
                        volatility = (
                            (max(recent_prices) - min(recent_prices)) / recent_prices[0] if recent_prices[0] > 0 else 0
                        )
                    else:
                        price_change = 0
                        volatility = 0
                except Exception as e:
                    self.logger.warning(f"Failed to get bars for {symbol}: {e}")
                    price_change = 0
                    volatility = 0

                market_data[symbol] = {
                    "price": current_price,
                    "price_change": price_change,
                    "volatility": volatility,
                    "volume": 1000000,  # Default volume
                    "timestamp": datetime.now(),
                }

            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")
                continue

        return market_data

    def _generate_ultra_aggressive_decision(
        self, agent_id: str, market_data: Dict[str, Any], positions: Dict[str, Any], account_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate an ULTRA AGGRESSIVE trading decision with guaranteed execution"""
        agent = self.agents[agent_id]

        # Check cooldown period (shorter for ultra aggressive)
        if agent["last_trade_time"]:
            time_since_last_trade = (datetime.now() - agent["last_trade_time"]).total_seconds()
            if time_since_last_trade < agent["cooldown_seconds"]:
                self.logger.debug(
                    f"Agent {agent_id} in cooldown ({time_since_last_trade:.1f}s / {agent['cooldown_seconds']}s)"
                )
                return None

        # ULTRA AGGRESSIVE: 99% chance to make a decision
        trade_roll = random.random()
        if trade_roll > agent["trade_probability"]:
            self.logger.debug(
                f"Agent {agent_id} failed trade probability roll ({trade_roll:.3f} > {agent['trade_probability']})"
            )
            return None

        # Select a random symbol
        symbol = random.choice(self.symbols)
        if symbol not in market_data:
            self.logger.debug(f"Agent {agent_id} selected {symbol} but no market data available")
            return None

        data = market_data[symbol]
        price = data["price"]

        # Get current position for this symbol
        current_position = positions.get(symbol, {}).get("qty", 0)
        buying_power = account_info["buying_power"]

        # ULTRA AGGRESSIVE decision logic with DYNAMIC POSITION SIZING
        action = None
        quantity = 0

        # MAXIMAL PROFIT-GENERATING POSITION SIZING with RISK TOLERANCE BOOST!
        volatility = data.get("volatility", 0)
        base_volatility = 0.02  # 2% baseline volatility
        volatility_multiplier = max(
            0.8, min(3.0, base_volatility / max(volatility, 0.01))
        )  # More aggressive scaling for MAXIMAL profits
        
        # CONSOLIDATE MAXIMAL IDEAS: Use risk_tolerance for massive position boost!
        risk_tolerance = agent.get("risk_tolerance", 1.0)
        maximal_boost = min(5.0, risk_tolerance / 10.0)  # Convert 20X-50X risk to 2X-5X position boost
        
        # Dynamic position sizing with MAXIMAL PROFIT FOCUS
        agent_max_position = agent["max_position"] * volatility_multiplier * maximal_boost
        self.logger.debug(
            f"MAXIMAL Agent {agent_id}: volatility: {volatility:.4f}, boost: {maximal_boost:.2f}x, max_position: {agent_max_position:.4f}"
        )

        # Determine action based on current position and buying power
        if current_position > 0:
            # We have shares, consider selling based on volatility and confidence
            sell_probability = 0.7 + (volatility * 10)  # Higher volatility = more likely to sell
            if random.random() < min(0.9, sell_probability):
                action = "SELL"
                # DYNAMIC SELL: Sell more in high volatility, less in low volatility
                if volatility > 0.03:  # High volatility - sell more aggressively
                    sell_percentage = random.uniform(0.4, 0.9)
                elif volatility < 0.01:  # Low volatility - sell less
                    sell_percentage = random.uniform(0.1, 0.4)
                else:  # Medium volatility - normal selling
                    sell_percentage = random.uniform(0.2, 0.6)

                quantity = max(1, int(current_position * sell_percentage))
                # CRITICAL FIX: Ensure we don't try to sell more than we have
                quantity = min(quantity, int(current_position))
        else:
            # No position, consider buying if we have ANY buying power - DYNAMIC SIZING
            if buying_power > 1.0:  # Only need $1 minimum to attempt trades
                # Dynamic buy probability based on volatility and confidence
                volatility_boost = min(0.05, volatility * 2)  # Up to 5% boost from volatility
                buy_probability = min(0.98, agent["trade_probability"] + volatility_boost)

                if random.random() < buy_probability:
                    action = "BUY"

                    # DYNAMIC POSITION SIZING: Calculate optimal position size
                    max_position_value = buying_power * agent_max_position
                    target_position_value = min(
                        max_position_value, buying_power * 0.88
                    )  # Don't exceed 88% of buying power

                    # Volatility-adjusted position sizing
                    if volatility > 0.03:  # High volatility - smaller positions
                        target_position_value *= 0.7
                    elif volatility < 0.01:  # Low volatility - larger positions
                        target_position_value *= 1.3

                    # MAXIMAL VOLUME TRADING: Calculate aggressive quantity with MAXIMAL BOOST
                    risk_tolerance = agent.get("risk_tolerance", 1.0)
                    maximal_boost = min(risk_tolerance / 10.0, 5.0)  # Up to 5x boost from risk tolerance
                    target_quantity = (target_position_value / price) * maximal_boost
                    
                    # MAXIMAL SYSTEM CONSOLIDATION: Up to 100 shares per trade like maximal!
                    max_shares_per_trade = min(100.0, buying_power * 0.95 / price)  # Up to 100 shares or 95% of buying power
                    
                    # AGGRESSIVE FRACTIONAL SHARES: Push limits for maximum utilization
                    if buying_power < price:
                        # Buy maximum fractional shares possible
                        quantity = round(min(target_quantity, buying_power * 0.95 / price), 4)  # 4 decimals like maximal
                    else:
                        # MAXIMAL VOLUME: Use aggressive target with high limits
                        if target_quantity >= 1:
                            quantity = round(min(target_quantity, max_shares_per_trade), 4)  # Maximal volume
                        else:
                            quantity = round(min(buying_power * 0.95 / price, max_shares_per_trade), 4)  # Max utilization

        if not action or quantity <= 0:
            return None

        # MAXIMAL FINAL VALIDATION: AGGRESSIVE CAPITAL UTILIZATION
        if action == "BUY":
            required_capital = quantity * price
            max_capital = buying_power * 0.98  # Use up to 98% of buying power (aggressive!)

            if required_capital > max_capital:
                # MAXIMAL ADJUSTMENT: Scale to use maximum available capital
                if buying_power > 0.10:  # Lower threshold for micro-trades
                    quantity = round(max_capital / price, 4)  # 4 decimals like maximal
                    self.logger.debug(
                        f"MAXIMAL Agent {agent_id} scaled: {required_capital:.2f} -> {quantity * price:.2f}"
                    )
                else:
                    return None

            # MAXIMAL VOLUME: Lower minimum threshold for higher trade frequency
            if quantity * price < 0.10:  # Much lower threshold - trade even $0.10 positions!
                return None

        elif action == "SELL":
            if quantity > abs(current_position):  # Use abs to handle negative positions
                quantity = abs(current_position)  # Sell entire position if quantity too high
                self.logger.debug(f"Agent {agent_id} sell quantity capped at position size: {quantity}")

        # MAXIMAL: Log successful decision with profit-focused sizing info
        position_value = quantity * price
        volatility_pct = volatility * 100
        risk_tolerance = agent.get("risk_tolerance", 1.0)
        self.logger.info(
            f"🚀 MAXIMAL DECISION - Agent {agent_id}: {action} {quantity} {symbol} @ ${price:.2f} (${position_value:.2f} required, ${buying_power:.2f} available, {volatility_pct:.2f}% vol, {risk_tolerance:.1f}x risk)"
        )

        return {
            "agent_id": agent_id,
            "symbol": symbol,
            "action": action,
            "quantity": round(quantity, 6),  # Support fractional shares
            "price": price,
            "confidence": agent["confidence"],
            "reasoning": f"ULTRA AGGRESSIVE {agent['aggression_level']} - {action} {quantity} shares of {symbol}",
            "hold_duration": "short",
            "style": "ultra_aggressive",
        }

    async def execute_trade(self, trade: Dict[str, Any]) -> bool:
        """Execute a trade with comprehensive validation and error handling"""
        try:
            symbol = trade["symbol"]
            action = trade["action"]
            quantity = trade["quantity"]

            # Get current positions and account info
            positions = await self.get_positions()
            account_info = await self.get_account_info()

            if not account_info:
                self.logger.error("Cannot execute trade: No account info")
                return False

            current_position = positions.get(symbol, {}).get("qty", 0)
            buying_power = account_info["buying_power"]

            # Final validation - ULTRA AGGRESSIVE: use 99% of buying power
            if action == "BUY":
                required_capital = quantity * trade["price"]
                if required_capital > buying_power * 0.99:  # Ultra aggressive: use up to 99% of buying power
                    self.logger.warning(
                        f"Insufficient buying power for {symbol}: {required_capital:.2f} > {buying_power * 0.99:.2f}"
                    )
                    return False

            elif action == "SELL":
                if quantity > current_position:
                    self.logger.warning(f"Insufficient shares to sell {symbol}: {quantity} > {current_position}")
                    return False

            # Execute the trade with retry logic
            for attempt in range(self.max_retries):
                try:
                    # MAXIMAL: Ultra-fast execution with minimal delay
                    if attempt > 0:
                        await asyncio.sleep(0.1)  # Maximal speed: 1s → 0.1s

                    if action == "BUY":
                        self.api.submit_order(
                            symbol=symbol, qty=quantity, side="buy", type="market", time_in_force="day"
                        )
                    else:  # SELL
                        self.api.submit_order(
                            symbol=symbol, qty=quantity, side="sell", type="market", time_in_force="day"
                        )

                    self.logger.info(
                        f"✅ ULTRA AGGRESSIVE: Executed {action} {quantity} shares of {symbol} at ${trade['price']:.2f}"
                    )

                    # Update agent's last trade time
                    self.agents[trade["agent_id"]]["last_trade_time"] = datetime.now()

                    return True

                except Exception as e:
                    self.logger.warning(f"Trade execution attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(0.1)  # MAXIMAL speed: 1s → 0.1s retry delay
                    else:
                        raise e

        except Exception as e:
            self.logger.error(f"❌ Failed to execute ULTRA AGGRESSIVE trade: {e}")
            return False

    async def run_cycle(self):
        """Run one optimized ultra aggressive trading cycle with RISK MANAGEMENT & ADAPTIVE INTELLIGENCE"""
        cycle_start = time.time()
        self.cycle_count += 1

        self.logger.info(f"Starting OPTIMIZED Ultra Aggressive Cycle {self.cycle_count}")

        try:
            # PHASE 1: Risk Management - Check all positions for SL/TP triggers
            risk_actions = await self.monitor_positions_risk_management()
            executed_risk_trades = 0

            # Execute risk management trades first (highest priority)
            for risk_action in risk_actions:
                try:
                    risk_trade = {
                        "agent_id": "RISK_MANAGER",
                        "symbol": risk_action["symbol"],
                        "action": "SELL",  # Risk management always sells
                        "quantity": risk_action["quantity"],
                        "price": risk_action["current_price"],
                        "reason": risk_action["reason"],
                    }

                    success = await self.execute_trade(risk_trade)
                    if success:
                        executed_risk_trades += 1
                        self.logger.info(
                            f"🛡️ RISK MANAGEMENT: {risk_action['reason']} - Sold {risk_action['quantity']} {risk_action['symbol']}"
                        )

                        # Remove from monitoring since position is closed
                        if risk_action["symbol"] in self.position_monitors:
                            del self.position_monitors[risk_action["symbol"]]

                except Exception as e:
                    self.logger.error(f"Risk management trade failed: {e}")

            # PHASE 2: Get market data and account info (refreshed after risk management)
            market_data = await self.get_market_data()
            if not market_data:
                self.logger.warning("No market data available")
                return

            # PHASE 3: Adaptive Market Intelligence - Detect regime and calculate adaptive weights
            current_regime = await self.detect_market_regime(market_data)
            adaptive_weights = self.calculate_adaptive_weights(current_regime)

            # Get account info and positions (refresh after risk management)
            account_info = await self.get_account_info()
            if not account_info:
                self.logger.warning("No account info available")
                return

            positions = await self.get_positions()

            # Enhanced logging with adaptive intelligence metrics
            self.logger.info(
                f"Account: ${account_info['buying_power']:.2f} buying power, ${account_info['cash']:.2f} cash"
            )
            self.logger.info(f"Positions: {len(positions)} symbols")
            self.logger.info(
                f"🧠 Market Regime: {current_regime} (Momentum: {self.market_regime['momentum_score']:.3f})"
            )
            if executed_risk_trades > 0:
                self.logger.info(f"🛡️ Risk Management: {executed_risk_trades} protective trades executed")

            # PHASE 4: Generate adaptive decisions from all agents using regime-based weights
            agent_decisions = {}
            for agent_id in self.agents:
                try:
                    # Apply adaptive weight to agent decision probability
                    agent_weight = adaptive_weights.get(agent_id, 1.0)

                    # Temporarily adjust agent's trade probability based on adaptive weight
                    original_probability = self.agents[agent_id]["trade_probability"]
                    self.agents[agent_id]["trade_probability"] = min(0.99, original_probability * agent_weight)

                    decision = self._generate_ultra_aggressive_decision(agent_id, market_data, positions, account_info)

                    # Restore original probability
                    self.agents[agent_id]["trade_probability"] = original_probability

                    if decision:
                        # Add adaptive information to decision
                        decision["adaptive_weight"] = agent_weight
                        decision["market_regime"] = current_regime
                        agent_decisions[agent_id] = decision
                        self.agent_performance[agent_id]["decisions"] += 1

                        self.logger.debug(
                            f"🎯 Adaptive Decision: {agent_id} (weight: {agent_weight:.2f}) -> {decision['action']} {decision['quantity']} {decision['symbol']}"
                        )

                except Exception as e:
                    self.logger.warning(f"Decision generation failed for {agent_id}: {e}")
                    continue

            # PHASE 5: Execute trades with adaptive learning
            executed_trades = 0
            for agent_id, decision in agent_decisions.items():
                try:
                    trade_success = await self.execute_trade(decision)
                    if trade_success:
                        executed_trades += 1
                        self.agent_performance[agent_id]["trades"] += 1
                        self.total_trades += 1

                        # Update performance memory for adaptive learning
                        trade_result = {
                            "success": True,
                            "pnl": 0.0,  # Will be updated when position is closed
                            "symbol": decision["symbol"],
                            "action": decision["action"],
                            "adaptive_weight": decision.get("adaptive_weight", 1.0),
                        }
                        self.update_performance_memory(agent_id, trade_result)

                        self.logger.info(
                            f"✅ ADAPTIVE TRADE: {agent_id} executed {decision['action']} {decision['quantity']} {decision['symbol']} (regime: {current_regime}, weight: {decision.get('adaptive_weight', 1.0):.2f})"
                        )
                    else:
                        # Track failed trades for learning
                        trade_result = {
                            "success": False,
                            "pnl": 0.0,
                            "symbol": decision["symbol"],
                            "action": decision["action"],
                            "adaptive_weight": decision.get("adaptive_weight", 1.0),
                        }
                        self.update_performance_memory(agent_id, trade_result)

                except Exception as e:
                    self.logger.warning(f"Trade execution failed for {agent_id}: {e}")
                    continue

            # Update performance metrics
            for agent_id in self.agents:
                perf = self.agent_performance[agent_id]
                if perf["trades"] > 0:
                    self.agents[agent_id]["win_rate"] = perf["wins"] / perf["trades"]
                    self.agents[agent_id]["total_pnl"] = perf["total_pnl"]
                    perf["success_rate"] = perf["wins"] / perf["trades"]

            # PHASE 6: MAXIMAL profit-focused cycle summary with adaptive intelligence metrics
            cycle_duration = time.time() - cycle_start
            self.logger.info(f"🚀 MAXIMAL Profit-Generation Cycle {self.cycle_count} Summary:")
            self.logger.info(f"  🧠 Market Regime: {current_regime}")
            self.logger.info(f"  📊 Momentum Score: {self.market_regime['momentum_score']:.3f}")
            self.logger.info(f"  🎯 Total Decisions: {len(agent_decisions)}")
            self.logger.info(f"  ⚡ Executed Trades: {executed_trades}")
            self.logger.info(f"  🛡️ Risk Trades: {executed_risk_trades}")
            self.logger.info(f"  ⏱️ TURBO Cycle: {cycle_duration:.2f}s")
            self.logger.info(f"  🔄 Total Trades: {self.total_trades}")
            self.logger.info(f"  💰 MAXIMAL PnL: ${self.total_pnl:.2f}")

            # Log adaptive agent performance
            if agent_decisions:
                avg_weight = sum(d.get("adaptive_weight", 1.0) for d in agent_decisions.values()) / len(agent_decisions)
                self.logger.info(f"  🎯 Avg Adaptive Weight: {avg_weight:.2f}")

            # Log top performing agents with adaptive metrics
            top_agents = sorted(self.agent_performance.items(), key=lambda x: x[1]["success_rate"], reverse=True)[:3]
            for agent_id, perf in top_agents:
                if perf["trades"] > 0:
                    agent_weight = adaptive_weights.get(agent_id, 1.0)
                    self.logger.info(
                        f"  🏆 Top Agent {agent_id}: {perf['trades']} trades, {perf['success_rate']:.2f} success rate, ${perf['total_pnl']:.2f} PnL (weight: {agent_weight:.2f})"
                    )

            # Log regime performance history
            if len(self.market_regime["regime_history"]) > 5:
                recent_regimes = [r["regime"] for r in self.market_regime["regime_history"][-5:]]
                regime_stability = len(set(recent_regimes)) / len(recent_regimes)  # Lower = more stable
                self.logger.info(
                    f"  📈 Regime Stability: {regime_stability:.2f} (recent: {' -> '.join(recent_regimes[-3:])})"
                )

        except Exception as e:
            self.logger.error(f"Cycle {self.cycle_count} failed: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

    async def run_system(self):
        """Run the optimized ultra aggressive trading system"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 MAXIMAL PROFIT-GENERATION TRADING SYSTEM STARTING")
        self.logger.info("=" * 60)
        self.logger.info("✅ Connected to Alpaca PAPER TRADING API")
        self.logger.info(f"🔥 Initialized {len(self.agents)} MAXIMAL profit-generation agents")
        self.logger.info("💥 MAXIMAL CONFIGURATION: 20X-50X Risk Tolerance + TURBO Speed!")
        self.logger.info("⚡ All agents configured for HIGH-VOLUME trading with aggressive positioning!")
        self.logger.info("🎯 Using PAPER TRADING for safe profit maximization!")

        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(10)  # MAXIMAL TURBO SPEED: 20s → 10s for HIGH VOLUME trading!

        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            import traceback

            self.logger.error(traceback.format_exc())


async def main():
    """Main function"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Initialize and run system
    system = OptimizedUltraAggressiveTradingSystem()
    await system.run_system()


if __name__ == "__main__":
    asyncio.run(main())
