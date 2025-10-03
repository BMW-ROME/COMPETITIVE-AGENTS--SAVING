"""
HTTP-based Alpaca Paper Trading integration.

Provides a fallback when the official alpaca-trade-api package is not
available (e.g., Python 3.12 compatibility issues). Uses REST endpoints
to perform essential actions needed by the orchestrator:
- initialize and fetch account
- load positions
- basic market-open checks
- place orders (with optional bracket orders)
- get portfolio performance
- close all positions
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
import requests

from config.settings import AlpacaConfig, SystemConfig


class RealAlpacaHTTPInterface:
    """Minimal HTTP client for Alpaca Paper Trading REST API."""

    def __init__(self, config: AlpacaConfig, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.logger = logging.getLogger("RealAlpacaHTTPInterface")

        # HTTP headers for Alpaca REST and Data APIs
        self._http_headers = {
            "APCA-API-KEY-ID": self.config.api_key or "",
            "APCA-API-SECRET-KEY": self.config.secret_key or "",
            "Content-Type": "application/json",
        }

        # State
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Any] = {}
        self.account_info: Dict[str, Any] = {}

        # Risk management defaults (match real integration defaults)
        self.max_position_size = 0.1
        self.max_daily_loss = 0.05
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.06
        self.use_bracket = True

    def _crypto_symbol_for_alpaca(self, symbol: str) -> str:
        s = (symbol or "").upper()
        if len(s) >= 6 and s.endswith("USD"):
            return f"{s[:-3]}/USD"
        return s

    def get_symbol_asset_class(self, symbol: str) -> str:
        s = (symbol or "").upper()
        # Crypto heuristic
        if s.endswith("USD") and len(s) > 6 and s not in {"USDTUSD", "USDCUSD"}:
            # Common crypto pairs
            return "crypto"
        # Forex heuristic
        if len(s) == 6 and s.isalpha():
            majors = {"EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
            if s[:3] in majors and s[3:] in majors:
                return "forex"
        return "stocks"

    async def is_market_open_for_symbol(self, symbol: str) -> bool:
        asset_class = self.get_symbol_asset_class(symbol)
        if asset_class == "crypto":
            return True
        if asset_class == "forex":
            # Not supported via this REST client in this project
            return False
        try:
            r = requests.get(f"{self.config.base_url}/v2/clock", headers=self._http_headers, timeout=10)
            if r.status_code == 200:
                return bool((r.json() or {}).get("is_open", False))
        except Exception:
            pass
        return False

    async def initialize(self) -> bool:
        try:
            if not (self.config.api_key and self.config.secret_key):
                self.logger.error("Alpaca API keys not configured")
                return False

            r = requests.get(f"{self.config.base_url}/v2/account", headers=self._http_headers, timeout=15)
            r.raise_for_status()
            account = r.json() or {}
            self.account_info = {
                "account_number": account.get("id", ""),
                "buying_power": float(account.get("buying_power", 0) or 0),
                "cash": float(account.get("cash", 0) or 0),
                "portfolio_value": float(account.get("portfolio_value", 0) or 0),
                "equity": float(account.get("equity", 0) or 0),
                "day_trade_count": int(account.get("day_trade_count", 0) or 0),
                "pattern_day_trader": bool(account.get("pattern_day_trader", False)),
                "trading_blocked": bool(account.get("trading_blocked", False)),
                "account_blocked": bool(account.get("account_blocked", False)),
            }

            await self._load_positions()
            mode = "PAPER TRADING" if "paper-api" in (self.config.base_url or "") else "LIVE TRADING"
            self.logger.info(f"Alpaca HTTP interface initialized ({mode})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca HTTP interface: {e}")
            return False

    async def _load_positions(self) -> None:
        try:
            r = requests.get(f"{self.config.base_url}/v2/positions", headers=self._http_headers, timeout=20)
            if r.status_code == 200:
                self.positions = {}
                for p in (r.json() or []):
                    sym = p.get("symbol")
                    if not sym:
                        continue
                    self.positions[sym] = {
                        "symbol": sym,
                        "qty": float(p.get("qty", 0) or 0),
                        "side": p.get("side", "long"),
                        "market_value": float(p.get("market_value", 0) or 0),
                        "cost_basis": float(p.get("cost_basis", 0) or 0),
                        "unrealized_pl": float(p.get("unrealized_pl", 0) or 0),
                        "unrealized_plpc": float(p.get("unrealized_plpc", 0) or 0),
                        "current_price": float(p.get("current_price", 0) or 0),
                        "lastday_price": float(p.get("lastday_price", 0) or 0),
                        "change_today": float(p.get("change_today", 0) or 0),
                    }
            self.logger.info(f"Loaded {len(self.positions)} positions (HTTP)")
        except Exception as e:
            self.logger.error(f"Failed to load positions (HTTP): {e}")
            self.positions = {}

    async def get_portfolio_performance(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.config.base_url}/v2/account", headers=self._http_headers, timeout=15)
            if r.status_code == 200:
                a = r.json() or {}
                self.account_info.update({
                    "buying_power": float(a.get("buying_power", 0) or 0),
                    "cash": float(a.get("cash", 0) or 0),
                    "portfolio_value": float(a.get("portfolio_value", 0) or 0),
                    "equity": float(a.get("equity", 0) or 0),
                })
            total_return = 0.0
            for pos in self.positions.values():
                total_return += float(pos.get("unrealized_pl", 0) or 0)
            return_pct = 0.0
            pv = float(self.account_info.get("portfolio_value", 0) or 0)
            if pv > 0:
                return_pct = (total_return / pv) * 100
            return {
                "total_return": total_return,
                "return_percentage": return_pct,
                "portfolio_value": self.account_info.get("portfolio_value", 0.0),
                "cash": self.account_info.get("cash", 0.0),
                "buying_power": self.account_info.get("buying_power", 0.0),
                "positions": len(self.positions),
                "unrealized_pl": total_return,
            }
        except Exception as e:
            self.logger.error(f"Failed to get portfolio performance (HTTP): {e}")
            return {
                "total_return": 0.0,
                "return_percentage": 0.0,
                "portfolio_value": 0.0,
                "cash": 0.0,
                "buying_power": 0.0,
                "positions": 0,
                "unrealized_pl": 0.0,
            }

    async def execute_trade_decision(self, decision: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(decision, dict):
                symbol = decision.get("symbol")
                action = decision.get("action")
                quantity = float(decision.get("quantity", 0) or 0)
                agent_id = decision.get("agent_id")
                confidence = decision.get("confidence")
                reasoning = decision.get("reasoning")
            else:
                symbol = decision.symbol
                action = decision.action
                quantity = float(decision.quantity or 0)
                agent_id = decision.agent_id
                confidence = decision.confidence
                reasoning = decision.reasoning

            if not symbol or not action or quantity <= 0:
                return None

            side = "buy" if str(action).upper() == "BUY" else "sell"
            order = await self._place_order(symbol, quantity, side)
            if not order:
                return None

            order["decision"] = {
                "agent_id": agent_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "confidence": confidence,
                "reasoning": reasoning,
            }
            self.logger.info(f"Trade executed (HTTP) for {agent_id}: {action} {quantity} {symbol}")
            return order
        except Exception as e:
            self.logger.error(f"Error executing trade decision (HTTP): {e}")
            return None

    async def _place_order_payload(self, symbol: str, qty: float, side: str) -> Optional[Dict[str, Any]]:
        # Build order payload; attempt bracket for BUY stocks
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": abs(float(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "extended_hours": True,
        }
        if self.use_bracket and side == "buy" and self.get_symbol_asset_class(symbol) == "stocks":
            price = await self._latest_price(symbol)
            if price and price > 0:
                sl = round(price * (1.0 - abs(self.stop_loss_pct)), 2)
                tp = round(price * (1.0 + abs(self.take_profit_pct)), 2)
                payload.update({
                    "order_class": "bracket",
                    "take_profit": {"limit_price": tp},
                    "stop_loss": {"stop_price": sl},
                })
        return payload

    async def _place_order(self, symbol: str, qty: float, side: str) -> Optional[Dict[str, Any]]:
        # Validate first
        if not await self._validate_order(symbol, qty, side):
            return None
        try:
            payload = await self._place_order_payload(symbol, qty, side)
            r = requests.post(f"{self.config.base_url}/v2/orders", headers=self._http_headers, json=payload, timeout=20)
            r.raise_for_status()
            od = r.json() or {}

            order_info = {
                "id": od.get("id"),
                "symbol": od.get("symbol"),
                "qty": float(od.get("qty", 0) or 0),
                "side": od.get("side"),
                "type": od.get("type"),
                "time_in_force": od.get("time_in_force"),
                "status": od.get("status"),
                "submitted_at": od.get("submitted_at"),
                "filled_at": od.get("filled_at"),
                "limit_price": float(od.get("limit_price", 0) or 0) if od.get("limit_price") else None,
                "stop_price": float(od.get("stop_price", 0) or 0) if od.get("stop_price") else None,
                "filled_avg_price": float(od.get("filled_avg_price", 0) or 0) if od.get("filled_avg_price") else None,
            }

            # Poll briefly for fill
            oid = order_info.get("id")
            if oid:
                for _ in range(10):
                    q = requests.get(f"{self.config.base_url}/v2/orders/{oid}", headers=self._http_headers, timeout=15)
                    if q.status_code == 200:
                        o2 = q.json() or {}
                        st = str(o2.get("status", "")).lower()
                        if st in ("filled", "partially_filled"):
                            order_info["status"] = o2.get("status")
                            order_info["filled_at"] = o2.get("filled_at")
                            fav = o2.get("filled_avg_price")
                            if fav is not None:
                                try:
                                    order_info["filled_avg_price"] = float(fav)
                                except Exception:
                                    pass
                            await self._load_positions()
                            break
                    await asyncio.sleep(0.5)

            await self._load_positions()
            return order_info
        except Exception as e:
            self.logger.error(f"HTTP order submit failed: {e}")
            return None

    async def _validate_order(self, symbol: str, qty: float, side: str) -> bool:
        try:
            if side == "buy":
                price = await self._latest_price(symbol)
                if not price or price <= 0:
                    self.logger.error(f"No quote available for {symbol}")
                    return False
                required = abs(qty) * price
                a = requests.get(f"{self.config.base_url}/v2/account", headers=self._http_headers, timeout=15)
                if a.status_code != 200:
                    return False
                bp = float((a.json() or {}).get("buying_power", 0) or 0)
                if required > bp:
                    self.logger.error(f"Insufficient buying power. Required: ${required:,.2f}, Available: ${bp:,.2f}")
                    return False
            elif side == "sell":
                if symbol not in self.positions:
                    await self._load_positions()
                if symbol not in self.positions:
                    self.logger.error(f"No position in {symbol} to sell")
                    return False
                position_data = self.positions.get(symbol, {})
                if isinstance(position_data, dict):
                    available_qty = float(position_data.get("qty", 0) or 0)
                else:
                    available_qty = float(position_data or 0)
                
                if abs(qty) > available_qty:
                    self.logger.error(
                        f"Insufficient shares. Requested: {abs(qty)}, Available: {available_qty}"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Order validation failed (HTTP): {e}")
            return False

    async def _latest_price(self, symbol: str) -> Optional[float]:
        """Fetch latest quote price for a symbol (stocks or crypto)."""
        try:
            asset_class = self.get_symbol_asset_class(symbol)
            if asset_class == "crypto":
                # Try v2 crypto quotes
                try:
                    api_symbol = self._crypto_symbol_for_alpaca(symbol)
                    r = requests.get(
                        f"{self.config.data_url}/v2/crypto/{api_symbol}/quotes/latest",
                        headers=self._http_headers,
                        timeout=10,
                    )
                    if r.status_code == 200:
                        q = (r.json() or {}).get("quote", {})
                        price = float(q.get("ap") or q.get("bp") or 0)
                        if price > 0:
                            return price
                except Exception:
                    pass
                # Fallback v1beta3 multi-symbol endpoint
                try:
                    api_symbol = self._crypto_symbol_for_alpaca(symbol)
                    r2 = requests.get(
                        f"{self.config.data_url}/v1beta3/crypto/us/quotes/latest",
                        headers=self._http_headers,
                        params={"symbols": api_symbol},
                        timeout=10,
                    )
                    if r2.status_code == 200:
                        payload = r2.json() or {}
                        quotes = payload.get("quotes") or payload.get("data") or {}
                        # quotes may be dict keyed by symbol
                        q = None
                        if isinstance(quotes, dict):
                            q = (quotes.get(api_symbol) or [{}])[0] if isinstance(quotes.get(api_symbol), list) else quotes.get(api_symbol)
                        elif isinstance(quotes, list) and quotes:
                            q = quotes[0]
                        if isinstance(q, dict):
                            price = float(q.get("ap") or q.get("bp") or 0)
                            if price > 0:
                                return price
                except Exception:
                    pass
                return None
            else:
                # Stocks quote (free plans typically require feed=iex)
                r = requests.get(
                    f"{self.config.data_url}/v2/stocks/{symbol}/quotes/latest",
                    headers=self._http_headers,
                    params={"feed": "iex"},
                    timeout=10,
                )
                if r.status_code == 200:
                    q = (r.json() or {}).get("quote", {})
                    price = float(q.get("ap") or q.get("bp") or 0)
                    return price if price > 0 else None
                return None
        except Exception:
            return None

    async def close_all_positions(self) -> List[Dict[str, Any]]:
        try:
            r = requests.delete(f"{self.config.base_url}/v2/positions", headers=self._http_headers, timeout=20)
            if r.status_code in (200, 207):  # 207 Multi-Status for batch results
                await self._load_positions()
                return r.json() if r.headers.get("Content-Type", "").startswith("application/json") else []
            return []
        except Exception as e:
            self.logger.error(f"Failed to close all positions (HTTP): {e}")
            return []


