#!/usr/bin/env python3

import time
import asyncio
import requests
from typing import Dict, Any

from .server_manager import ServerManager
from .schema_server import SchemaBasedMockServer
from ..data_generation.api_schema_generator import APISchemaGenerator


def basic_server_example():
    """Demonstrate basic mock server functionality"""
    print("🚀 Basic Mock Server Example")
    print("=" * 50)
    
    # Create server manager
    manager = ServerManager()
    
    try:
        # Create a server
        server = manager.create_server(
            server_name="demo_server",
            port=8001,
            title="Demo API",
            description="Basic demonstration server"
        )
        
        # Generate and load a random API
        print("\n📋 Generating random API specification...")
        api_spec = server.generate_and_load_random_api(
            title="Demo E-commerce API",
            num_endpoints=6
        )
        print(f"✅ Generated API with {len(api_spec['paths'])} endpoints")
        
        # Start the server
        print("\n🌐 Starting server...")
        manager.start_server("demo_server")
        time.sleep(2)  # Give server time to start
        
        # Test some endpoints
        base_url = server.get_server_url()
        print(f"🔗 Server running at: {base_url}")
        
        # Health check
        print("\n🏥 Health check:")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Server info
        print("\n📊 Server info:")
        response = requests.get(f"{base_url}/server-info")
        server_info = response.json()
        print(f"   Title: {server_info['title']}")
        print(f"   Endpoints: {server_info['registered_endpoints']}")
        
        # Test API endpoints
        print("\n🧪 Testing API endpoints:")
        
        # Test GET /users
        try:
            response = requests.get(f"{base_url}/users")
            print(f"   GET /users: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Found {len(data.get('items', []))} users")
        except Exception as e:
            print(f"   GET /users failed: {str(e)}")
        
        # Test GET /products
        try:
            response = requests.get(f"{base_url}/products?category=electronics")
            print(f"   GET /products: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Found {len(data.get('items', []))} products")
        except Exception as e:
            print(f"   GET /products failed: {str(e)}")
        
        # Test POST /users
        try:
            user_data = {
                "username": "test_user",
                "email": "test@example.com",
                "full_name": "Test User",
                "password": "secure123"
            }
            response = requests.post(f"{base_url}/users", json=user_data)
            print(f"   POST /users: {response.status_code}")
            if response.status_code == 201:
                created_user = response.json()
                print(f"   Created user ID: {created_user.get('id')}")
        except Exception as e:
            print(f"   POST /users failed: {str(e)}")
        
        print("\n✅ Basic example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in basic example: {str(e)}")
    finally:
        # Cleanup
        manager.shutdown_all_servers()


def advanced_server_example():
    """Demonstrate advanced server features"""
    print("\n🔬 Advanced Mock Server Example")
    print("=" * 50)
    
    manager = ServerManager()
    
    try:
        # Create server with schema generator
        schema_generator = APISchemaGenerator(seed=42)
        server = manager.create_server(
            server_name="advanced_server",
            port=8002,
            title="Advanced API",
            description="Advanced mock server with custom features",
            schema_generator=schema_generator
        )
        
        # Load user management API
        print("\n👥 Loading User Management API...")
        user_spec = server.load_user_management_api()
        print(f"✅ Loaded {len(user_spec['paths'])} user endpoints")
        
        # Add realistic delays and error simulation
        print("\n⏱️  Adding realistic delays and error simulation...")
        server.add_realistic_delay(min_ms=100, max_ms=800)
        server.simulate_error_responses(error_rate=0.15)
        
        # Start server
        manager.start_server("advanced_server")
        time.sleep(2)
        
        base_url = server.get_server_url()
        print(f"🔗 Advanced server running at: {base_url}")
        
        # Test with multiple requests to see delays and errors
        print("\n🧪 Testing with realistic delays and errors:")
        
        for i in range(5):
            try:
                start_time = time.time()
                response = requests.get(f"{base_url}/users", timeout=5)
                elapsed = (time.time() - start_time) * 1000
                
                print(f"   Request {i+1}: {response.status_code} ({elapsed:.0f}ms)")
                
                if response.status_code >= 400:
                    error_data = response.json()
                    print(f"      Error: {error_data.get('message', 'Unknown error')}")
                
            except requests.Timeout:
                print(f"   Request {i+1}: Timeout")
            except Exception as e:
                print(f"   Request {i+1}: Error - {str(e)}")
            
            time.sleep(0.5)  # Brief pause between requests
        
        print("\n📊 Server status:")
        status = manager.get_server_status("advanced_server")
        print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"   Endpoints: {status['endpoints_registered']}")
        
        print("\n✅ Advanced example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in advanced example: {str(e)}")
    finally:
        manager.shutdown_all_servers()


def scenario_based_example():
    """Demonstrate scenario-based testing"""
    print("\n🎭 Scenario-Based Testing Example")
    print("=" * 50)
    
    manager = ServerManager()
    
    try:
        # Generate training dataset
        schema_generator = APISchemaGenerator(seed=123)
        print("\n📚 Generating training dataset...")
        
        dataset = schema_generator.generate_training_dataset(
            num_scenarios=3,
            endpoints_per_scenario=4
        )
        print(f"✅ Generated {len(dataset)} training scenarios")
        
        # Create server and load dataset
        server = manager.create_server(
            server_name="scenario_server",
            port=8003,
            title="Scenario Testing Server",
            schema_generator=schema_generator
        )
        
        print("\n🔄 Loading training dataset...")
        server.load_training_dataset(dataset)
        
        manager.start_server("scenario_server")
        time.sleep(2)
        
        base_url = server.get_server_url()
        print(f"🔗 Scenario server running at: {base_url}")
        
        # Test each scenario
        for scenario_idx in range(len(dataset)):
            print(f"\n🎬 Testing Scenario {scenario_idx + 1}: {dataset[scenario_idx]['name']}")
            
            if scenario_idx > 0:
                # Switch to new scenario (requires server restart for this demo)
                manager.stop_server("scenario_server")
                time.sleep(1)
                server.switch_to_scenario(scenario_idx)
                manager.start_server("scenario_server")
                time.sleep(2)
            
            # Test a few endpoints
            scenario_info = server.get_current_schema_info()
            print(f"   📋 Loaded {scenario_info['endpoints_count']} endpoints")
            
            # Make some test requests
            for endpoint in scenario_info['endpoints'][:2]:  # Test first 2 endpoints
                path = endpoint['path']
                method = endpoint['method']
                
                try:
                    if method == 'GET':
                        # Handle path parameters by using a dummy ID
                        if '{' in path:
                            test_path = path.replace('{user_id}', '123').replace('{product_id}', '456')
                        else:
                            test_path = path
                        
                        response = requests.get(f"{base_url}{test_path}")
                        print(f"      {method} {path}: {response.status_code}")
                        
                except Exception as e:
                    print(f"      {method} {path}: Error - {str(e)}")
        
        print("\n✅ Scenario-based example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in scenario example: {str(e)}")
    finally:
        manager.shutdown_all_servers()


def cluster_example():
    """Demonstrate server clustering"""
    print("\n🏗️  Server Clustering Example")
    print("=" * 50)
    
    manager = ServerManager()
    
    try:
        # Create a cluster of servers
        print("\n🔧 Creating server cluster...")
        cluster_servers = manager.create_server_cluster(
            cluster_name="api_cluster",
            server_count=3,
            base_port=8010
        )
        print(f"✅ Created cluster with {len(cluster_servers)} servers")
        
        # Configure each server differently
        for i, server_name in enumerate(cluster_servers):
            server = manager.get_server(server_name)
            
            if i == 0:
                server.load_user_management_api()
                print(f"   🔧 {server_name}: Loaded User Management API")
            elif i == 1:
                server.load_product_catalog_api()
                print(f"   🔧 {server_name}: Loaded Product Catalog API")
            else:
                server.generate_and_load_random_api(num_endpoints=5)
                print(f"   🔧 {server_name}: Loaded Random API")
        
        # Start the cluster
        print("\n🚀 Starting cluster...")
        manager.start_cluster("api_cluster")
        time.sleep(3)
        
        # Check cluster status
        cluster_status = manager.get_cluster_status("api_cluster")
        print(f"\n📊 Cluster Status:")
        print(f"   Total servers: {cluster_status['total_servers']}")
        print(f"   Running servers: {cluster_status['running_servers']}")
        
        # Test each server in the cluster
        print("\n🧪 Testing cluster servers:")
        for server_info in cluster_status['servers']:
            if server_info['status'] == 'running':
                try:
                    response = requests.get(f"{server_info['url']}/health", timeout=2)
                    print(f"   ✅ {server_info['name']}: {response.status_code} ({server_info['endpoints_count']} endpoints)")
                except Exception as e:
                    print(f"   ❌ {server_info['name']}: Error - {str(e)}")
        
        # Health check all servers
        print("\n🏥 Health check results:")
        health_results = manager.health_check_all()
        print(f"   Healthy servers: {health_results['healthy_servers']}/{health_results['total_servers']}")
        
        print("\n✅ Cluster example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in cluster example: {str(e)}")
    finally:
        manager.shutdown_all_servers()


def main():
    """Run all examples"""
    print("🎯 FastAPI Mock Server Examples")
    print("=" * 60)
    
    try:
        # Run examples
        basic_server_example()
        time.sleep(1)
        
        advanced_server_example()
        time.sleep(1)
        
        scenario_based_example()
        time.sleep(1)
        
        cluster_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Tips for using the mock servers:")
        print("   - Use ServerManager for managing multiple servers")
        print("   - SchemaBasedMockServer for dynamic API generation") 
        print("   - Add realistic delays and errors for testing")
        print("   - Use scenarios for RL training datasets")
        print("   - Create clusters for load testing")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")


if __name__ == "__main__":
    main()