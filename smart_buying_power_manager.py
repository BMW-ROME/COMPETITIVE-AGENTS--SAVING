#!/usr/bin/env python3
"""
Smart Buying Power Manager
Ensures system never depletes all buying power and manages funds intelligently
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

@dataclass
class BuyingPowerStatus:
    """Current buying power status"""
    total_buying_power: float
    available_for_trading: float
    reserved_amount: float
    last_updated: datetime
    is_sufficient: bool
    recommended_trade_size: float

class SmartBuyingPowerManager:
    """
    Intelligent buying power management system
    - Never depletes all funds
    - Keeps safety reserves
    - Adjusts trade sizes dynamically
    - Provides real-time status updates
    """
    
    def __init__(self, api, min_reserve_percent: float = 20.0, min_absolute_reserve: float = 100.0):
        self.api = api
        self.logger = logging.getLogger(f"{__name__}.BuyingPowerManager")
        
        # Safety parameters - configurable but conservative defaults
        self.min_reserve_percent = min_reserve_percent  # Never use more than 80% of funds
        self.min_absolute_reserve = min_absolute_reserve  # Always keep at least $100
        self.min_trade_amount = 10.0  # Minimum viable trade size
        self.max_single_trade_percent = 5.0  # Never risk more than 5% on a single trade
        
        # Refresh settings
        self.refresh_interval = timedelta(minutes=5)  # Update buying power every 5 minutes
        self.last_refresh = None
        self.current_status = None
        
        # Performance tracking
        self.refresh_count = 0
        self.error_count = 0
        
        self.logger.info("🏦 Smart Buying Power Manager initialized")
        self.logger.info(f"   Reserve: {min_reserve_percent}% + ${min_absolute_reserve}")
        self.logger.info(f"   Max single trade: {self.max_single_trade_percent}%")
    
    def get_current_status(self, force_refresh: bool = False) -> Optional[BuyingPowerStatus]:
        """Get current buying power status with smart caching"""
        
        # Check if refresh is needed
        needs_refresh = (
            force_refresh or 
            self.last_refresh is None or 
            datetime.now() - self.last_refresh > self.refresh_interval or
            self.current_status is None
        )
        
        if needs_refresh:
            self._refresh_buying_power()
        
        return self.current_status
    
    def _refresh_buying_power(self) -> None:
        """Refresh buying power from API"""
        try:
            self.refresh_count += 1
            account = self.api.get_account()
            
            # Get buying power - handle different account types
            total_bp = 0.0
            
            if hasattr(account, 'buying_power'):
                total_bp = float(account.buying_power)
            elif hasattr(account, 'cash'):
                total_bp = float(account.cash)
            else:
                # Try alternative attributes
                for attr in ['daytrading_buying_power', 'regt_buying_power']:
                    if hasattr(account, attr):
                        total_bp = float(getattr(account, attr))
                        break
            
            if total_bp <= 0:
                self.logger.warning(f"⚠️ Zero buying power detected: ${total_bp}")
            
            # Calculate reserves and available amounts
            percent_reserve = total_bp * (self.min_reserve_percent / 100.0)
            effective_reserve = max(percent_reserve, self.min_absolute_reserve)
            available_for_trading = max(0, total_bp - effective_reserve)
            
            # Determine if sufficient for trading
            is_sufficient = available_for_trading >= self.min_trade_amount
            
            # Calculate recommended trade size (conservative)
            if is_sufficient:
                recommended_trade_size = min(
                    available_for_trading * (self.max_single_trade_percent / 100.0),
                    available_for_trading / 10.0  # Never more than 10% of available funds
                )
                recommended_trade_size = max(recommended_trade_size, self.min_trade_amount)
            else:
                recommended_trade_size = 0.0
            
            # Update status
            self.current_status = BuyingPowerStatus(
                total_buying_power=total_bp,
                available_for_trading=available_for_trading,
                reserved_amount=effective_reserve,
                last_updated=datetime.now(),
                is_sufficient=is_sufficient,
                recommended_trade_size=recommended_trade_size
            )
            
            self.last_refresh = datetime.now()
            
            # Log status update
            self.logger.info(f"💰 Buying Power Update:")
            self.logger.info(f"   Total: ${total_bp:,.2f}")
            self.logger.info(f"   Available: ${available_for_trading:,.2f}")
            self.logger.info(f"   Reserved: ${effective_reserve:,.2f}")
            self.logger.info(f"   Status: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
            if is_sufficient:
                self.logger.info(f"   Recommended Trade Size: ${recommended_trade_size:,.2f}")
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Failed to refresh buying power: {e}")
            
            # Keep previous status if available, otherwise create emergency status
            if self.current_status is None:
                self.current_status = BuyingPowerStatus(
                    total_buying_power=0.0,
                    available_for_trading=0.0,
                    reserved_amount=0.0,
                    last_updated=datetime.now(),
                    is_sufficient=False,
                    recommended_trade_size=0.0
                )
    
    def can_afford_trade(self, trade_amount: float) -> tuple[bool, str]:
        """Check if a specific trade amount is affordable"""
        status = self.get_current_status()
        
        if not status:
            return False, "Unable to determine buying power"
        
        if not status.is_sufficient:
            return False, f"Insufficient funds: ${status.available_for_trading:.2f} available"
        
        if trade_amount > status.available_for_trading:
            return False, f"Trade amount ${trade_amount:.2f} exceeds available ${status.available_for_trading:.2f}"
        
        if trade_amount > status.recommended_trade_size * 2:  # Allow up to 2x recommended
            return False, f"Trade amount too large (max recommended: ${status.recommended_trade_size:.2f})"
        
        return True, "Trade approved"
    
    def get_safe_trade_amount(self, requested_amount: float) -> float:
        """Get a safe trade amount, potentially reducing the requested amount"""
        status = self.get_current_status()
        
        if not status or not status.is_sufficient:
            return 0.0
        
        # Return the minimum of: requested amount, available funds, recommended size
        return min(
            requested_amount,
            status.available_for_trading,
            status.recommended_trade_size
        )
    
    def should_stop_trading(self) -> tuple[bool, str]:
        """Determine if trading should be stopped due to low funds"""
        status = self.get_current_status()
        
        if not status:
            return True, "Unable to determine buying power status"
        
        if not status.is_sufficient:
            return True, f"Insufficient buying power: ${status.available_for_trading:.2f} available (need ${self.min_trade_amount})"
        
        # Additional safety check - if we're very close to reserves
        if status.available_for_trading < self.min_trade_amount * 2:
            return True, f"Approaching minimum reserves: ${status.available_for_trading:.2f} available"
        
        return False, "Trading can continue"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get manager performance statistics"""
        return {
            'refresh_count': self.refresh_count,
            'error_count': self.error_count,
            'error_rate': (self.error_count / max(1, self.refresh_count)) * 100,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'current_status_age_minutes': (
                (datetime.now() - self.current_status.last_updated).total_seconds() / 60
                if self.current_status else None
            )
        }
    
    def force_refresh_and_log(self) -> None:
        """Force refresh and log detailed status"""
        self.logger.info("🔄 Forcing buying power refresh...")
        status = self.get_current_status(force_refresh=True)
        
        if status:
            self.logger.info("📊 DETAILED BUYING POWER STATUS")
            self.logger.info(f"   🏦 Total Buying Power: ${status.total_buying_power:,.2f}")
            self.logger.info(f"   💰 Available for Trading: ${status.available_for_trading:,.2f}")
            self.logger.info(f"   🛡️ Reserved (Safety): ${status.reserved_amount:,.2f}")
            self.logger.info(f"   ✅ Sufficient for Trading: {status.is_sufficient}")
            self.logger.info(f"   📏 Recommended Trade Size: ${status.recommended_trade_size:,.2f}")
            self.logger.info(f"   ⏰ Last Updated: {status.last_updated.strftime('%H:%M:%S')}")
            
            # Performance stats
            stats = self.get_performance_stats()
            self.logger.info(f"   📈 Refresh Count: {stats['refresh_count']}")
            self.logger.info(f"   ❌ Error Rate: {stats['error_rate']:.1f}%")
        else:
            self.logger.error("❌ Unable to get buying power status")


def integrate_with_trading_system(trading_system, api):
    """
    Integration helper to add buying power management to existing trading systems
    """
    
    # Create the manager
    bp_manager = SmartBuyingPowerManager(api, min_reserve_percent=25.0)  # Keep 25% reserve
    
    # Add to trading system
    trading_system.buying_power_manager = bp_manager
    
    # Force initial refresh
    bp_manager.force_refresh_and_log()
    
    return bp_manager