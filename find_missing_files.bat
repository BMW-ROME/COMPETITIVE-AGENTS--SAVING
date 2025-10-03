@echo off
REM CRITICAL FILES TO SEARCH FOR IN 31K BACKUP
REM Run this in your backup directory to find what we need

echo SEARCHING FOR CRITICAL DYNAMIC/ADAPTIVE FILES...
echo ==================================================

echo.
echo DYNAMIC CONFIG FILES:
dir /s /b *dynamic*.py 2>nul

echo.
echo REAL-TIME ADAPTATION FILES:
dir /s /b *real_time*.py 2>nul
dir /s /b *realtime*.py 2>nul

echo.
echo ADVANCED CRYPTO FILES:
dir /s /b *crypto_advanced*.py 2>nul
dir /s /b *crypto_arbitrage*.py 2>nul
dir /s /b *crypto_market*.py 2>nul

echo.
echo MARKET ADAPTATION FILES:
dir /s /b *market_adapt*.py 2>nul
dir /s /b *adaptive_market*.py 2>nul

echo.
echo INTELLIGENT ROUTING FILES:
dir /s /b *routing*.py 2>nul
dir /s /b *smart_order*.py 2>nul

echo.
echo PORTFOLIO OPTIMIZATION FILES:
dir /s /b *portfolio_opt*.py 2>nul
dir /s /b *position_opt*.py 2>nul

echo.
echo MISSING IMPORT DEPENDENCIES:
findstr /s /i "dynamic_config_manager" *.py 2>nul
findstr /s /i "real_time" *.py 2>nul

echo.
echo ✅ SEARCH COMPLETE - Please check results above
pause