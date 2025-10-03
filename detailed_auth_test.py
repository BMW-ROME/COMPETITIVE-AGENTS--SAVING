#!/usr/bin/env python3
"""
Detailed Alpaca credentials diagnostic
"""

import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Get credentials
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')
base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

print("🔍 DETAILED ALPACA CREDENTIALS DIAGNOSTIC")
print("=" * 50)

# Check for invisible characters or length issues
print(f"API Key: '{api_key}'")
print(f"API Key length: {len(api_key) if api_key else 0}")
print(f"Secret Key length: {len(secret_key) if secret_key else 0}")
print(f"Secret Key (first 10 chars): '{secret_key[:10]}...' if secret_key else 'None'")

# Check for whitespace issues
if api_key:
    print(f"API Key has leading/trailing spaces: {api_key != api_key.strip()}")
if secret_key:
    print(f"Secret Key has leading/trailing spaces: {secret_key != secret_key.strip()}")

print(f"Base URL: {base_url}")
print()

# Test with raw HTTP request to get more detailed error
try:
    print("🌐 Testing with raw HTTP request...")
    
    headers = {
        'APCA-API-KEY-ID': api_key.strip() if api_key else '',
        'APCA-API-SECRET-KEY': secret_key.strip() if secret_key else '',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
    if response.status_code == 200:
        print("✅ CREDENTIALS WORK!")
    elif response.status_code == 401:
        print("❌ Unauthorized - Check your API credentials")
        print("💡 This could mean:")
        print("   - Wrong API key or secret key")
        print("   - Keys don't match each other")
        print("   - Account not activated")
        print("   - Using live keys for paper trading URL (or vice versa)")
    else:
        print(f"❓ Unexpected status code: {response.status_code}")
        
except Exception as e:
    print(f"❌ Connection error: {e}")

# Additional checks
print()
print("🧐 CREDENTIAL FORMAT ANALYSIS:")
if api_key:
    print(f"✅ API Key present: {len(api_key)} characters")
    print(f"   Starts with: {api_key[:3]}...")
    print(f"   All alphanumeric: {api_key.replace('_', '').isalnum()}")
else:
    print("❌ No API Key found")

if secret_key:
    print(f"✅ Secret Key present: {len(secret_key)} characters") 
    print(f"   Starts with: {secret_key[:3]}...")
    print(f"   All alphanumeric: {secret_key.replace('_', '').isalnum()}")
else:
    print("❌ No Secret Key found")

print()
print("💡 TROUBLESHOOTING TIPS:")
print("1. Make sure you're logged into the PAPER trading dashboard")
print("2. Regenerate your API keys if needed")
print("3. Copy keys exactly with no extra characters") 
print("4. Verify your account status is 'ACTIVE'")
print("5. Wait a few minutes after creating new keys")