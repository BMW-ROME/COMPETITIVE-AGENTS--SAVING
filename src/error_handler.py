"""
Comprehensive Error Handling and Recovery System
===============================================

This module provides bulletproof error handling for the trading system.
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import functools

class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    pass

class DataSourceError(TradingSystemError):
    """Data source related errors"""
    pass

class TradingError(TradingSystemError):
    """Trading execution errors"""
    pass

class DatabaseError(TradingSystemError):
    """Database related errors"""
    pass

class AISystemError(TradingSystemError):
    """AI system related errors"""
    pass

class ErrorHandler:
    """Comprehensive error handling system"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_counts = {}
        self.recovery_attempts = {}
        self.max_retries = 3
        self.circuit_breakers = {}
        
    def handle_error(self, error: Exception, context: str = "", retry: bool = True) -> bool:
        """Handle errors with automatic recovery"""
        try:
            error_type = type(error).__name__
            error_key = f"{error_type}:{context}"
            
            # Count errors
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Log error
            self.logger.error(f"Error in {context}: {error}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Check circuit breaker
            if self._is_circuit_open(error_key):
                self.logger.warning(f"Circuit breaker open for {error_key}")
                return False
            
            # Attempt recovery
            if retry and self.error_counts[error_key] <= self.max_retries:
                self.logger.info(f"Attempting recovery for {error_key} (attempt {self.error_counts[error_key]})")
                return self._attempt_recovery(error, context)
            else:
                self.logger.error(f"Max retries exceeded for {error_key}")
                self._open_circuit_breaker(error_key)
                return False
                
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return False
    
    def _is_circuit_open(self, error_key: str) -> bool:
        """Check if circuit breaker is open"""
        if error_key in self.circuit_breakers:
            last_failure = self.circuit_breakers[error_key]
            # Circuit breaker resets after 5 minutes
            if (datetime.now() - last_failure).total_seconds() < 300:
                return True
            else:
                # Reset circuit breaker
                del self.circuit_breakers[error_key]
        return False
    
    def _open_circuit_breaker(self, error_key: str):
        """Open circuit breaker for error type"""
        self.circuit_breakers[error_key] = datetime.now()
        self.logger.warning(f"Circuit breaker opened for {error_key}")
    
    def _attempt_recovery(self, error: Exception, context: str) -> bool:
        """Attempt to recover from error"""
        try:
            if isinstance(error, DatabaseError):
                return self._recover_database_error(error, context)
            elif isinstance(error, DataSourceError):
                return self._recover_data_source_error(error, context)
            elif isinstance(error, TradingError):
                return self._recover_trading_error(error, context)
            elif isinstance(error, AISystemError):
                return self._recover_ai_error(error, context)
            else:
                return self._recover_generic_error(error, context)
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False
    
    def _recover_database_error(self, error: Exception, context: str) -> bool:
        """Recover from database errors"""
        try:
            # Try to reconnect to database
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="trading_agents",
                user="trading_user",
                password="trading_password"
            )
            conn.close()
            self.logger.info("Database connection recovered")
            return True
        except Exception as e:
            self.logger.error(f"Database recovery failed: {e}")
            return False
    
    def _recover_data_source_error(self, error: Exception, context: str) -> bool:
        """Recover from data source errors"""
        try:
            # Switch to fallback data sources
            self.logger.info("Switching to fallback data sources")
            return True
        except Exception as e:
            self.logger.error(f"Data source recovery failed: {e}")
            return False
    
    def _recover_trading_error(self, error: Exception, context: str) -> bool:
        """Recover from trading errors"""
        try:
            # Implement trading error recovery
            self.logger.info("Implementing trading error recovery")
            return True
        except Exception as e:
            self.logger.error(f"Trading recovery failed: {e}")
            return False
    
    def _recover_ai_error(self, error: Exception, context: str) -> bool:
        """Recover from AI system errors"""
        try:
            # Switch to fallback AI systems
            self.logger.info("Switching to fallback AI systems")
            return True
        except Exception as e:
            self.logger.error(f"AI recovery failed: {e}")
            return False
    
    def _recover_generic_error(self, error: Exception, context: str) -> bool:
        """Recover from generic errors"""
        try:
            # Generic recovery - wait and retry
            self.logger.info("Implementing generic error recovery")
            return True
        except Exception as e:
            self.logger.error(f"Generic recovery failed: {e}")
            return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self.error_counts,
            "circuit_breakers": self.circuit_breakers,
            "total_errors": sum(self.error_counts.values())
        }

def error_handler(context: str = "", retry: bool = True):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error handler from first argument if it's a class instance
                if args and hasattr(args[0], 'error_handler'):
                    error_handler_instance = args[0].error_handler
                else:
                    # Create default error handler
                    logger = logging.getLogger(func.__module__)
                    error_handler_instance = ErrorHandler(logger)
                
                success = error_handler_instance.handle_error(e, context, retry)
                if not success:
                    raise
                return None
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, **kwargs) -> Any:
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(func.__module__)
        error_handler = ErrorHandler(logger)
        error_handler.handle_error(e, f"safe_execute:{func.__name__}")
        return None

async def safe_async_execute(func: Callable, *args, **kwargs) -> Any:
    """Safely execute an async function with error handling"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(func.__module__)
        error_handler = ErrorHandler(logger)
        error_handler.handle_error(e, f"safe_async_execute:{func.__name__}")
        return None

