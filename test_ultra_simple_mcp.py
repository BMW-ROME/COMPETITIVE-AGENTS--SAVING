#!/usr/bin/env python3
"""
Test Ultra Simple MCP Server
===========================

Test the ultra simple MCP server using only built-in Python libraries.
No external dependencies required!
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import time

def test_mcp_server():
    """Test the ultra simple MCP server"""
    print("🧪 Testing Ultra Simple MCP Server...")
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        try:
            with urllib.request.urlopen("http://localhost:8002/health", timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print("✅ Health check passed")
                    print(f"   Response: {data}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except urllib.error.URLError as e:
            print(f"❌ Could not connect to server: {e}")
            print("   Make sure the server is running: python ultra_simple_mcp_server.py")
            return False
        
        # Test status endpoint
        print("\n2. Testing status endpoint...")
        try:
            with urllib.request.urlopen("http://localhost:8002/status", timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print("✅ Status check passed")
                    print(f"   Response: {data}")
                else:
                    print(f"❌ Status check failed: {response.status}")
                    return False
        except urllib.error.URLError as e:
            print(f"❌ Status check failed: {e}")
            return False
        
        # Test tools endpoint
        print("\n3. Testing tools endpoint...")
        try:
            with urllib.request.urlopen("http://localhost:8002/tools", timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print("✅ Tools check passed")
                    print(f"   Available tools: {len(data['tools'])}")
                    for tool in data['tools']:
                        print(f"   - {tool['name']}: {tool['description']}")
                else:
                    print(f"❌ Tools check failed: {response.status}")
                    return False
        except urllib.error.URLError as e:
            print(f"❌ Tools check failed: {e}")
            return False
        
        # Test MCP chat endpoint
        print("\n4. Testing MCP chat endpoint...")
        test_request = {
            "prompt": "Get trading system status",
            "context": {"test": True},
            "request_type": "get_trading_status",
            "priority": 1
        }
        
        try:
            data = json.dumps(test_request).encode('utf-8')
            req = urllib.request.Request(
                "http://localhost:8002/mcp/chat",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode('utf-8'))
                    print("✅ MCP chat test passed")
                    print(f"   Response: {result['response'][:100]}...")
                else:
                    print(f"❌ MCP chat test failed: {response.status}")
                    return False
        except urllib.error.URLError as e:
            print(f"❌ MCP chat test failed: {e}")
            return False
        
        print("\n🎉 All MCP server tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Ultra Simple MCP Server Test")
    print("=" * 40)
    
    success = test_mcp_server()
    if success:
        print("\n✅ Ultra Simple MCP Server is working correctly!")
        print("🎯 You can now configure Cursor to use this MCP server")
        print("🔧 No external dependencies required!")
    else:
        print("\n❌ MCP Server tests failed")
        print("🔧 Please check the server configuration")
        print("💡 Make sure to run: python ultra_simple_mcp_server.py")






