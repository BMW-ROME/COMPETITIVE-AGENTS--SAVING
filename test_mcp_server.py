#!/usr/bin/env python3
"""
Test MCP Server
==============

Test the simple MCP server to ensure it works.
"""

import requests
import json
import time

def test_mcp_server():
    """Test the MCP server"""
    print("🧪 Testing Ultimate Trading System MCP Server...")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test status endpoint
        print("\n2. Testing status endpoint...")
        response = requests.get("http://localhost:8002/status", timeout=5)
        if response.status_code == 200:
            print("✅ Status check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
        
        # Test tools endpoint
        print("\n3. Testing tools endpoint...")
        response = requests.get("http://localhost:8002/tools", timeout=5)
        if response.status_code == 200:
            print("✅ Tools check passed")
            tools = response.json()
            print(f"   Available tools: {len(tools['tools'])}")
            for tool in tools['tools']:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print(f"❌ Tools check failed: {response.status_code}")
            return False
        
        # Test MCP chat endpoint
        print("\n4. Testing MCP chat endpoint...")
        test_request = {
            "prompt": "Get trading system status",
            "context": {"test": True},
            "request_type": "get_trading_status",
            "priority": 1
        }
        
        response = requests.post("http://localhost:8002/mcp/chat", json=test_request, timeout=10)
        if response.status_code == 200:
            print("✅ MCP chat test passed")
            result = response.json()
            print(f"   Response: {result['response'][:100]}...")
        else:
            print(f"❌ MCP chat test failed: {response.status_code}")
            return False
        
        print("\n🎉 All MCP server tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to MCP server")
        print("   Make sure the server is running: python simple_mcp_server.py")
        return False
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("\n✅ MCP Server is working correctly!")
        print("🎯 You can now configure Cursor to use this MCP server")
    else:
        print("\n❌ MCP Server tests failed")
        print("🔧 Please check the server configuration")
