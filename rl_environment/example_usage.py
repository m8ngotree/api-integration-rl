#!/usr/bin/env python3

import asyncio
import time
from typing import List, Dict, Any

from rl_environment.execution_environment import (
    SafeExecutionEnvironment, ExecutionEnvironmentConfig, ExecutionEnvironmentFactory,
    EnvironmentType, TestCase, TestSuite
)
from rl_environment.code_executor import SecurityPolicy, ExecutionStatus
from data_generation.api_schema_generator import APISchemaGenerator
from data_generation.endpoint_generator import EndpointGenerator


async def basic_subprocess_example():
    """Demonstrate basic subprocess code execution"""
    print("🔧 Basic Subprocess Execution Example")
    print("=" * 60)
    
    # Create basic environment
    env = ExecutionEnvironmentFactory.create_basic_environment()
    
    async with env.temporary_environment():
        # Test simple code execution
        simple_code = '''
print("Hello from safe execution environment!")
x = 2 + 2
print(f"2 + 2 = {x}")

# Test list comprehension
numbers = [i**2 for i in range(5)]
print(f"Squares: {numbers}")
'''
        
        print("\n📝 Executing simple Python code...")
        result = await env.execute_code(simple_code)
        
        print(f"✅ Status: {result.status.value}")
        print(f"⏱️  Execution time: {result.execution_time:.3f}s")
        print(f"💾 Memory usage: {result.memory_usage}MB")
        print(f"📤 Output:\n{result.stdout}")
        
        if result.stderr:
            print(f"⚠️  Errors:\n{result.stderr}")


async def security_violation_example():
    """Demonstrate security policy enforcement"""
    print("\n🔒 Security Policy Enforcement Example")
    print("=" * 60)
    
    # Create secure environment
    env = ExecutionEnvironmentFactory.create_secure_environment()
    
    async with env.temporary_environment():
        # Test blocked import
        malicious_code = '''
import os
import subprocess

# This should be blocked
os.system("echo 'This would be dangerous'")
'''
        
        print("\n🚫 Executing code with blocked imports...")
        result = await env.execute_code(malicious_code)
        
        print(f"✅ Status: {result.status.value}")
        if result.violations:
            print("🛡️  Security violations detected:")
            for violation in result.violations:
                print(f"   • {violation}")
        
        print(f"📤 Output:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️  Errors:\n{result.stderr}")


async def docker_execution_example():
    """Demonstrate Docker-based execution"""
    print("\n🐳 Docker Execution Example")
    print("=" * 60)
    
    try:
        # Create Docker environment
        env = ExecutionEnvironmentFactory.create_docker_environment()
        
        async with env.temporary_environment():
            # Test code with external libraries
            docker_code = '''
import requests
import json
from datetime import datetime

print("🐳 Running in Docker container!")
print(f"📅 Current time: {datetime.now()}")

# Test HTTP capabilities (should work in Docker)
try:
    response = requests.get("https://httpbin.org/json", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ HTTP request successful: {data.get('slideshow', {}).get('title', 'N/A')}")
    else:
        print(f"❌ HTTP request failed: {response.status_code}")
except Exception as e:
    print(f"⚠️  HTTP request error: {e}")

# Test memory and CPU intensive task
result = sum(i**2 for i in range(10000))
print(f"🧮 Computation result: {result}")
'''
            
            print("\n🐳 Executing code in Docker container...")
            result = await env.execute_code(docker_code, timeout=60)
            
            print(f"✅ Status: {result.status.value}")
            print(f"⏱️  Execution time: {result.execution_time:.3f}s")
            print(f"💾 Memory usage: {result.memory_usage}MB")
            print(f"🐳 Docker execution: {result.metadata.get('docker_execution', False)}")
            print(f"📤 Output:\n{result.stdout}")
            
            if result.stderr:
                print(f"⚠️  Errors:\n{result.stderr}")
    
    except Exception as e:
        print(f"❌ Docker execution failed: {e}")
        print("💡 Make sure Docker is installed and running")


async def api_testing_example():
    """Demonstrate API testing with mock server"""
    print("\n🌐 API Testing with Mock Server Example")
    print("=" * 60)
    
    try:
        # Create API testing environment
        env = ExecutionEnvironmentFactory.create_api_testing_environment(server_port=8003)
        
        async with env.temporary_environment():
            # Load some endpoints into the mock server
            if env.mock_server:
                # Generate and load a simple API
                from data_generation.api_schema_generator import APISchemaGenerator
                schema_gen = APISchemaGenerator(seed=42)
                api_spec = env.mock_server.load_user_management_api()
                print(f"📋 Loaded mock API with {len(api_spec['paths'])} endpoints")
            
            # Test API integration code
            api_test_code = '''
import requests
import json

# Get the API base URL
base_url = get_api_base_url()
print(f"🔗 API Base URL: {base_url}")

# Test server connection
if test_server_connection():
    print("✅ Mock server is available")
    
    # Test GET /users endpoint
    try:
        response = requests.get(f"{base_url}/users", timeout=10)
        print(f"📊 GET /users: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📄 Response data keys: {list(data.keys())}")
            if 'items' in data:
                print(f"👥 Found {len(data['items'])} users")
        
    except Exception as e:
        print(f"❌ API request failed: {e}")
    
    # Test POST /users endpoint
    try:
        new_user = {
            "username": "test_user",
            "email": "test@example.com",
            "full_name": "Test User"
        }
        
        response = requests.post(
            f"{base_url}/users",
            json=new_user,
            timeout=10
        )
        print(f"📝 POST /users: {response.status_code}")
        
        if response.status_code in [200, 201]:
            created_user = response.json()
            print(f"✅ Created user with ID: {created_user.get('id', 'unknown')}")
        
    except Exception as e:
        print(f"❌ POST request failed: {e}")

else:
    print("❌ Mock server not available")
'''
            
            print("\n🧪 Executing API integration test...")
            result = await env.execute_code(api_test_code, timeout=90)
            
            print(f"✅ Status: {result.status.value}")
            print(f"⏱️  Execution time: {result.execution_time:.3f}s")
            print(f"🌐 Server URL: {result.metadata.get('server_url', 'N/A')}")
            print(f"📤 Output:\n{result.stdout}")
            
            if result.stderr:
                print(f"⚠️  Errors:\n{result.stderr}")
    
    except Exception as e:
        print(f"❌ API testing failed: {e}")


async def test_suite_example():
    """Demonstrate comprehensive test suite execution"""
    print("\n🧪 Test Suite Execution Example")
    print("=" * 60)
    
    # Create test environment
    env = ExecutionEnvironmentFactory.create_api_testing_environment(server_port=8004)
    
    async with env.temporary_environment():
        # Load API into mock server
        if env.mock_server:
            env.mock_server.load_product_catalog_api()
            print("📋 Loaded product catalog API")
        
        # Create comprehensive test suite
        test_suite = TestSuite(
            name="API Integration Test Suite",
            setup_code='''
import requests
import json

base_url = get_api_base_url()
print(f"🔗 Testing API at: {base_url}")

# Verify server is available
assert test_server_connection(), "Mock server not available"
print("✅ Server connection verified")
''',
            test_cases=[
                TestCase(
                    name="test_get_products",
                    code='''
response = requests.get(f"{base_url}/products")
assert response.status_code == 200, f"Expected 200, got {response.status_code}"

data = response.json()
assert 'items' in data, "Response should contain 'items' key"
print(f"✅ Found {len(data['items'])} products")
''',
                    expected_status=ExecutionStatus.SUCCESS,
                    timeout=30
                ),
                
                TestCase(
                    name="test_create_product",
                    code='''
new_product = {
    "name": "Test Product",
    "description": "A test product",
    "price": 29.99,
    "category": "electronics",
    "stock_quantity": 100
}

response = requests.post(f"{base_url}/products", json=new_product)
assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}"

created_product = response.json()
assert created_product['name'] == new_product['name'], "Product name mismatch"
print(f"✅ Created product with ID: {created_product.get('id', 'unknown')}")
''',
                    expected_status=ExecutionStatus.SUCCESS,
                    timeout=30
                ),
                
                TestCase(
                    name="test_invalid_product",
                    code='''
# Test with invalid data
invalid_product = {
    "name": "",  # Invalid empty name
    "price": -10  # Invalid negative price
}

response = requests.post(f"{base_url}/products", json=invalid_product)
# This might succeed in mock server, but let's check the response
print(f"Response status: {response.status_code}")
print(f"Response data: {response.json() if response.status_code == 200 else 'Error'}")
''',
                    expected_status=ExecutionStatus.SUCCESS,
                    timeout=30
                ),
                
                TestCase(
                    name="test_get_product_by_id", 
                    code='''
# Test getting a specific product
response = requests.get(f"{base_url}/products/123")
print(f"GET /products/123: {response.status_code}")

if response.status_code == 200:
    product = response.json()
    assert 'id' in product, "Product should have an ID"
    assert product['id'] == 123, f"Expected ID 123, got {product.get('id')}"
    print(f"✅ Retrieved product: {product.get('name', 'Unknown')}")
else:
    print(f"Product not found or error: {response.status_code}")
''',
                    expected_status=ExecutionStatus.SUCCESS,
                    timeout=30
                )
            ],
            teardown_code='''
print("🧹 Test suite completed")
''',
            metadata={
                'api_type': 'product_catalog',
                'test_environment': 'mock_server'
            }
        )
        
        print("\n🏃 Running test suite...")
        results = await env.run_test_suite(test_suite)
        
        # Display results
        print(f"\n📊 Test Suite Results:")
        summary = results.get('__summary__', {})
        print(f"   • Total tests: {summary.get('total', 0)}")
        print(f"   • Passed: {summary.get('passed', 0)}")
        print(f"   • Failed: {summary.get('failed', 0)}")
        print(f"   • Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"   • Total execution time: {summary.get('total_execution_time', 0):.2f}s")
        
        # Show individual test results
        for test_name, result in results.items():
            if not test_name.startswith('__'):
                validation = result.metadata.get('validation', {})
                status_icon = "✅" if validation.get('overall_success', False) else "❌"
                print(f"   {status_icon} {test_name}: {result.status.value} ({result.execution_time:.2f}s)")


async def resource_limit_example():
    """Demonstrate resource limit enforcement"""
    print("\n⚡ Resource Limit Enforcement Example")  
    print("=" * 60)
    
    # Create environment with strict limits
    config = ExecutionEnvironmentConfig(
        environment_type=EnvironmentType.SUBPROCESS,
        timeout=10,
        max_memory_mb=32,  # Very low memory limit
        security_policy=SecurityPolicy(
            max_memory_mb=32,
            max_execution_time=10
        )
    )
    
    env = SafeExecutionEnvironment(config)
    
    async with env.temporary_environment():
        # Test memory-intensive code
        memory_intensive_code = '''
print("🧠 Testing memory limits...")

# Try to allocate a lot of memory
try:
    # This should trigger memory limit
    big_list = [i * "x" * 1000 for i in range(10000)]
    print(f"Allocated memory for {len(big_list)} items")
except MemoryError:
    print("❌ Memory allocation failed (expected)")
except Exception as e:
    print(f"⚠️  Other error: {e}")

print("✅ Memory test completed")
'''
        
        print("\n🧠 Testing memory limits...")
        result = await env.execute_code(memory_intensive_code)
        
        print(f"✅ Status: {result.status.value}")
        print(f"⏱️  Execution time: {result.execution_time:.3f}s")
        print(f"💾 Memory usage: {result.memory_usage}MB")
        print(f"📤 Output:\n{result.stdout}")
        
        # Test timeout
        timeout_code = '''
import time

print("⏰ Testing timeout limits...")
print("Starting long-running operation...")

# This should trigger timeout
for i in range(20):
    print(f"Step {i+1}/20")
    time.sleep(1)

print("✅ Operation completed")
'''
        
        print("\n⏰ Testing timeout limits...")
        result = await env.execute_code(timeout_code)
        
        print(f"✅ Status: {result.status.value}")
        print(f"⏱️  Execution time: {result.execution_time:.3f}s")
        print(f"📤 Output:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️  Errors:\n{result.stderr}")


async def batch_execution_example():
    """Demonstrate batch code execution"""
    print("\n📦 Batch Execution Example")
    print("=" * 60)
    
    env = ExecutionEnvironmentFactory.create_basic_environment()
    
    async with env.temporary_environment():
        # Create multiple code snippets to execute
        code_batches = [
            {
                'name': 'math_operations',
                'code': '''
import math
result = math.sqrt(144) + math.pi
print(f"Math result: {result:.2f}")
''',
                'timeout': 15
            },
            {
                'name': 'string_processing',
                'code': '''
text = "Hello, World!"
processed = text.upper().replace("WORLD", "PYTHON")
print(f"Processed text: {processed}")
''',
                'timeout': 15
            },
            {
                'name': 'list_operations',
                'code': '''
numbers = list(range(1, 11))
squares = [n**2 for n in numbers]
print(f"Squares: {squares}")
print(f"Sum of squares: {sum(squares)}")
''',
                'timeout': 15
            }
        ]
        
        print(f"\n📊 Executing {len(code_batches)} code batches...")
        
        # Execute batches sequentially
        for i, batch in enumerate(code_batches):
            print(f"\n🔄 Executing batch {i+1}: {batch['name']}")
            
            result = await env.execute_code(
                code=batch['code'],
                timeout=batch['timeout']
            )
            
            print(f"   Status: {result.status.value}")
            print(f"   Time: {result.execution_time:.3f}s")
            print(f"   Output: {result.stdout.strip()}")
        
        # Show environment statistics
        stats = env.get_execution_stats()
        print(f"\n📈 Environment Statistics:")
        print(f"   • Total executions: {stats['total_executions']}")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Average execution time: N/A")  # Would need to track this


async def main():
    """Run all execution environment examples"""
    print("🚀 Safe Code Execution Environment Demo")
    print("=" * 70)
    
    try:
        # Run all examples
        await basic_subprocess_example()
        await security_violation_example() 
        await docker_execution_example()
        await api_testing_example()
        await test_suite_example()
        await resource_limit_example()
        await batch_execution_example()
        
        print("\n" + "="*70)
        print("🎉 All examples completed successfully!")
        
        print("\n💡 Key Features Demonstrated:")
        print("   • Subprocess and Docker isolation")
        print("   • Security policy enforcement")
        print("   • Resource limit enforcement (memory, CPU, time)")
        print("   • Mock server integration for API testing")
        print("   • Comprehensive test suite execution")
        print("   • Batch processing capabilities")
        print("   • Detailed execution monitoring and statistics")
        
        print("\n🔧 Environment Types Available:")
        print("   • SUBPROCESS: Basic subprocess isolation")
        print("   • DOCKER: Full containerization with Docker")
        print("   • SUBPROCESS_WITH_SERVER: Subprocess + mock API server")
        print("   • DOCKER_WITH_SERVER: Docker + mock API server")
        
        print("\n🛡️  Security Features:")
        print("   • Import filtering and validation")
        print("   • Memory and CPU usage limits")
        print("   • Execution timeout enforcement")
        print("   • File system access restrictions")
        print("   • Network access controls")
        print("   • Subprocess execution blocking")
        
        print("\n🎯 Perfect for RL Training:")
        print("   • Safe code execution for agent-generated code")
        print("   • Comprehensive result capture and analysis")
        print("   • API testing capabilities with mock servers")
        print("   • Resource usage monitoring")
        print("   • Security violation detection")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())