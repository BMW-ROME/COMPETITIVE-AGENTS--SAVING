"""
Real Alpaca trading platform integration.
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None
    APIError = Exception

from config.settings import AlpacaConfig, SystemConfig, TradingMode


class RealAlpacaTradingInterface:
    """Real interface for Alpaca trading platform."""

    def __init__(self, config: AlpacaConfig, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.logger = logging.getLogger("RealAlpacaTradingInterface")

        # Initialize Alpaca API
        self.api = None
        self.stream = None

        # Trading state
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Any] = {}
        self.account_info: Dict[str, Any] = {}

        # Risk management
        self.max_position_size = 0.05  # 5% of portfolio per position (reduced for safety)
        self.max_daily_loss = 0.02  # 2% max daily loss (reduced for safety)
        # Default SL/TP; allow overrides via env
        try:
            self.stop_loss_pct = float(os.getenv("ALPACA_STOP_LOSS_PCT", "0.03"))
        except Exception:
            self.stop_loss_pct = 0.03
        try:
            self.take_profit_pct = float(os.getenv("ALPACA_TAKE_PROFIT_PCT", "0.06"))
        except Exception:
            self.take_profit_pct = 0.06
        # Enable/disable bracket orders (BUY stock entries)
        self.use_bracket = os.getenv("ALPACA_USE_BRACKET", "true").lower() in ("1", "true", "yes", "on")
        # Allow attempting trades outside regular hours for stocks
        self.allow_after_hours = os.getenv("ALPACA_ALLOW_AFTER_HOURS", "true").lower() in ("1", "true", "yes", "on")

    def get_symbol_asset_class(self, symbol: str) -> str:
        """Best-effort asset class detection: stocks, crypto, forex."""
        s = symbol.upper()
        # Simple crypto detection
        crypto_set = {
            "BTCUSD",
            "ETHUSD",
            "ADAUSD",
            "SOLUSD",
            "DOTUSD",
            "MATICUSD",
            "AVAXUSD",
            "UNIUSD",
            "AAVEUSD",
            "SUSHIUSD",
            "CRVUSD",
            "COMPUSD",
            "MKRUSD",
            "YFIUSD",
            "DOGEUSD",
            "SHIBUSD",
            "XRPUSD",
            "LTCUSD",
            "BCHUSD",
            "ETCUSD",
            "XLMUSD",
        }
        if s in crypto_set or (s.endswith("USD") and len(s) > 6 and s not in {"USDTUSD", "USDCUSD"}):
            # Heuristic: many crypto pairs end with USD and are longer than 6
            return "crypto"
        # Forex detection: 6 letters, both sides likely in major currency list
        if len(s) == 6 and s.isalpha():
            majors = {"EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
            if s[:3] in majors and s[3:] in majors:
                return "forex"
        return "stocks"

    async def is_market_open_for_symbol(self, symbol: str) -> bool:
        """Check whether market is open for the given symbol."""
        asset_class = self.get_symbol_asset_class(symbol)
        if asset_class == "crypto":
            return True
        if asset_class == "forex":
            # Alpaca REST in this project isn't set up for forex; treat as closed to skip
            return False
        # Stocks
        try:
            if self.allow_after_hours:
                return True
            clock = self.api.get_clock()
            return bool(getattr(clock, "is_open", False))
        except Exception:
            return False

    async def initialize(self) -> bool:
        """Initialize the trading interface."""
        try:
            if not ALPACA_AVAILABLE:
                self.logger.error(
                    "alpaca-trade-api not installed. Install with: pip install alpaca-trade-api"
                )
                return False

            if not self.config.api_key or not self.config.secret_key:
                self.logger.error("Alpaca API keys not configured")
                return False

            # Initialize Alpaca API
            self.api = tradeapi.REST(
                self.config.api_key, self.config.secret_key, self.config.base_url, api_version="v2"
            )

            # Test connection
            account = self.api.get_account()
            self.account_info = {
                "account_number": account.id,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "day_trade_count": getattr(account, "day_trade_count", 0),
                "pattern_day_trader": getattr(account, "pattern_day_trader", False),
                "trading_blocked": getattr(account, "trading_blocked", False),
                "account_blocked": getattr(account, "account_blocked", False),
            }

            # Load existing positions
            await self._load_positions()

            # Determine trading mode
            mode = "PAPER TRADING" if "paper-api" in self.config.base_url else "LIVE TRADING"

            self.logger.info(f"Alpaca interface initialized ({mode})")
            self.logger.info(f"Account: {self.account_info['account_number']}")
            self.logger.info(
                f"Portfolio Value: ${self.account_info['portfolio_value']:,.2f}"
            )
            self.logger.info(f"Buying Power: ${self.account_info['buying_power']:,.2f}")
            self.logger.info(f"Cash: ${self.account_info['cash']:,.2f}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca interface: {e}")
            return False

    async def _load_positions(self):
        """Load current positions from Alpaca."""
        try:
            positions = self.api.list_positions()
            self.positions = {}

            for position in positions:
                self.positions[position.symbol] = {
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "side": position.side,
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "lastday_price": float(position.lastday_price),
                    "change_today": float(position.change_today),
                }

            self.logger.info(f"Loaded {len(self.positions)} positions")

        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")
            self.positions = {}

    async def get_market_data(
        self, symbol: str, timeframe: str = "1Min", limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """Get market data for a symbol."""
        try:
            # Get historical data
            start_time = datetime.now() - timedelta(days=1)
            end_time = datetime.now()

            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_time.strftime("%Y-%m-%d"),
                end=end_time.strftime("%Y-%m-%d"),
                limit=limit,
            )

            if not bars:
                return None

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    }
                )

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Dict]:
        """Place an order with Alpaca."""
        try:
            # Validate order
            if not self._validate_order(symbol, qty, side):
                return None

            order = None
            # Try bracket order for BUY stock entries if enabled
            try:
                asset_class = self.get_symbol_asset_class(symbol)
                if self.use_bracket and side.lower() == "buy" and asset_class == "stocks":
                    quote = self.api.get_latest_quote(symbol)
                    entry_price = None
                    if quote:
                        entry_price = float(getattr(quote, 'ask_price', 0) or 0) or float(getattr(quote, 'bid_price', 0) or 0)
                    if entry_price and entry_price > 0:
                        sl_price = round(entry_price * (1.0 - abs(self.stop_loss_pct)), 2)
                        tp_price = round(entry_price * (1.0 + abs(self.take_profit_pct)), 2)
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(qty),
                            side=side,
                            type=order_type,
                            time_in_force=time_in_force,
                            order_class="bracket",
                            take_profit={"limit_price": tp_price},
                            stop_loss={"stop_price": sl_price},
                            extended_hours=True
                        )
                        self.logger.info(f"Bracket order placed for {symbol} SL={sl_price} TP={tp_price}")
            except Exception as e:
                self.logger.warning(f"Bracket order attempt failed for {symbol}: {e}. Falling back to regular order.")
                order = None

            # Place regular order if bracket not used. Use limit orders after-hours for stocks to improve fills.
            if order is None:
                asset_class = self.get_symbol_asset_class(symbol)
                if asset_class == "stocks" and self.allow_after_hours and order_type == "market":
                    # Convert to limit order for after-hours; prefer quote, then trade with small buffer
                    px = 0.0
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        if quote:
                            px = float(getattr(quote, 'ask_price', 0) or 0) if side == 'buy' else float(getattr(quote, 'bid_price', 0) or 0)
                    except Exception:
                        px = 0.0
                    if not px or px <= 0:
                        try:
                            trade = getattr(self.api, 'get_latest_trade', None)
                            if callable(trade):
                                lt = trade(symbol)
                                last_price = float(getattr(lt, 'price', 0) or 0)
                                if last_price and last_price > 0:
                                    # Add small buffer to improve fill chances
                                    px = round(last_price * (1.002 if side == 'buy' else 0.998), 2)
                        except Exception:
                            px = 0.0
                    if px and px > 0:
                        try:
                            order = self.api.submit_order(
                                symbol=symbol,
                                qty=abs(qty),
                                side=side,
                                type="limit",
                                time_in_force=time_in_force,
                                limit_price=px,
                                extended_hours=True
                            )
                        except Exception:
                            order = None
                if order is None:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=abs(qty),
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force,
                        limit_price=limit_price,
                        stop_price=stop_price,
                        extended_hours=True
                    )

            order_info = {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "limit_price": float(order.limit_price) if getattr(order, 'limit_price', None) else None,
                "stop_price": float(order.stop_price) if getattr(order, 'stop_price', None) else None,
                "filled_avg_price": float(order.filled_avg_price) if getattr(order, 'filled_avg_price', None) else None
            }

            self.logger.info(f"Order placed: {order_info}")

            # Wait briefly for fill (paper may fill quickly during market hours)
            try:
                for _ in range(10):
                    o = self.api.get_order(order.id)
                    status = (o.status or "").lower()
                    if status in ("filled", "partially_filled"):
                        order_info["status"] = o.status
                        order_info["filled_at"] = o.filled_at
                        order_info["filled_avg_price"] = float(o.filled_avg_price) if getattr(o, 'filled_avg_price', None) else order_info.get("filled_avg_price")
                        # Update local positions snapshot
                        await self._load_positions()
                        break
                    await asyncio.sleep(0.5)
            except Exception as _:
                pass

            # Final refresh of positions regardless
            await self._load_positions()

            return order_info

        except APIError as e:
            self.logger.error(f"Alpaca API error placing order: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None

    def _validate_order(self, symbol: str, qty: float, side: str) -> bool:
        """Validate order parameters with resilient fallbacks, allowing after-hours stocks."""
        try:
            # Basic validation
            if not symbol or not side or float(qty) <= 0:
                self.logger.error(f"Invalid order params: {symbol} {qty} {side}")
                return False

            # For BUY orders, attempt light risk checks; skip strict price requirement if after-hours
            if side == "buy":
                current_price = 0.0
                try:
                    # Try latest trade first (more robust than quote for some assets)
                    trade = getattr(self.api, 'get_latest_trade', None)
                    if callable(trade):
                        lt = trade(symbol)
                        current_price = float(getattr(lt, 'price', 0) or 0)
                except Exception:
                    current_price = 0.0
                if current_price <= 0:
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        current_price = float(getattr(quote, 'ask_price', 0) or 0) or float(getattr(quote, 'bid_price', 0) or 0)
                    except Exception:
                        current_price = 0.0

                # If still no price and we allow after-hours, proceed without strict capital check
                if current_price > 0:
                    required_capital = abs(qty) * current_price
                    portfolio_value = float(self.account_info.get("portfolio_value", 0.0) or 0.0)
                    if portfolio_value > 0:
                        max_alloc = portfolio_value * float(self.max_position_size)
                        if required_capital > max_alloc:
                            self.logger.error(
                                f"Position too large for {symbol}. Required ${required_capital:,.2f} > max alloc ${max_alloc:,.2f} ({self.max_position_size*100:.0f}%)"
                            )
                            return False

                    try:
                        buying_power = float(self.account_info.get('buying_power', 0.0) or 0.0)
                        if buying_power <= 0:
                            # If no buying power, try to use cash as fallback
                            cash = float(self.account_info.get('cash', 0.0) or 0.0)
                            if cash > 0:
                                self.logger.warning(f"No buying power, using cash as fallback: ${cash:,.2f}")
                                if required_capital > cash:
                                    self.logger.error(f"Insufficient cash. Required: ${required_capital:,.2f}, Available: ${cash:,.2f}")
                                    return False
                            else:
                                self.logger.error(f"No buying power or cash available. Required: ${required_capital:,.2f}, Available: ${buying_power:,.2f}")
                                return False
                        elif required_capital > buying_power:
                            self.logger.error(f"Insufficient buying power. Required: ${required_capital:,.2f}, Available: ${buying_power:,.2f}")
                            return False
                    except Exception as e:
                        self.logger.warning(f"Error checking buying power: {e}")
                        # Allow trade to proceed if we can't check buying power
                        pass

            # For SELL orders, ensure sufficient position when we have it locally
            if side == "sell":
                try:
                    pos = self.positions.get(symbol)
                    if pos and abs(qty) > float(pos.get('qty', 0) or 0):
                        self.logger.error(
                            f"Insufficient shares. Requested: {abs(qty)}, Available: {pos.get('qty', 0)}"
                        )
                        return False
                except Exception:
                    pass

            return True

        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False

    async def get_portfolio_performance(self) -> Dict:
        """Get portfolio performance metrics."""
        try:
            # Update account info
            account = self.api.get_account()
            self.account_info.update({
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
            })

            # Calculate returns
            total_return = 0.0
            if self.positions:
                for symbol, position in self.positions.items():
                    total_return += position["unrealized_pl"]

            # Calculate return percentage
            return_pct = 0.0
            if self.account_info["portfolio_value"] > 0:
                return_pct = (total_return / self.account_info["portfolio_value"]) * 100

            return {
                "total_return": total_return,
                "return_percentage": return_pct,
                "portfolio_value": self.account_info["portfolio_value"],
                "cash": self.account_info["cash"],
                "buying_power": self.account_info["buying_power"],
                "positions": len(self.positions),
                "unrealized_pl": total_return,
            }

        except Exception as e:
            self.logger.error(f"Failed to get portfolio performance: {e}")
            return {
                "total_return": 0.0,
                "return_percentage": 0.0,
                "portfolio_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "positions": 0,
                "unrealized_pl": 0.0,
            }

    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        try:
            orders = self.api.list_orders(status="open")
            return [
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side,
                    "type": order.type,
                    "status": order.status,
                    "submitted_at": order.submitted_at,
                }
                for order in orders
            ]
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def execute_trade_decision(self, decision) -> Optional[Dict[str, Any]]:
        """Execute a trade decision from an agent. Returns order info dict or None."""
        try:
            # Extract decision parameters (handle both dict and object)
            if isinstance(decision, dict):
                symbol = decision.get("symbol")
                action = decision.get("action")
                quantity = decision.get("quantity")
                agent_id = decision.get("agent_id")
                confidence = decision.get("confidence")
                reasoning = decision.get("reasoning")
            else:
                symbol = decision.symbol
                action = decision.action
                quantity = decision.quantity
                agent_id = decision.agent_id
                confidence = decision.confidence
                reasoning = decision.reasoning

            # Determine order side
            side = "buy" if action.upper() == "BUY" else "sell"

            # Place order (bracket logic handled inside place_order)
            order_info = await self.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type="market",
                time_in_force="day",
            )

            if order_info:
                # Attach decision metadata
                order_info["decision"] = {
                    "agent_id": agent_id,
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }
                order_info["executed_at"] = datetime.now().isoformat()

                self.logger.info(
                    f"Trade executed for {agent_id}: {action} {quantity} {symbol}"
                )
                return order_info
            else:
                self.logger.error(
                    f"Failed to execute trade for {agent_id}: {action} {quantity} {symbol}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Error executing trade decision: {e}")
            return None

    async def shutdown(self):
        """Shutdown the trading interface."""
        try:
            if self.stream:
                self.stream.stop()
            self.logger.info("Alpaca interface shutdown")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
