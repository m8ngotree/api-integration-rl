#!/usr/bin/env python3

import asyncio
import json
from typing import List, Dict, Any

from .reward_system import (
    BaseRewardEvaluator, BinaryRewardEvaluator, RewardConfig, 
    RewardComponent, APICallInfo
)
from .reward_integration import (
    CodeRewardIntegrator, EvaluationContext, EvaluationMode
)
from .execution_environment import (
    ExecutionEnvironmentFactory, TestCase, TestSuite
)
from .code_executor import ExecutionResult, ExecutionStatus
from ..data_generation.endpoint_generator import EndpointGenerator
from ..utilities.code_template_generator import (
    APIIntegrationTemplateGenerator, MissingComponent, DifficultyLevel
)


def print_reward_breakdown(breakdown, title="Reward Breakdown"):
    """Helper function to print reward breakdown nicely"""
    print(f"\n🏆 {title}")
    print("=" * 50)
    print(f"Total Reward: {breakdown.total_reward:.2f}")
    
    if breakdown.component_scores:
        print("\n📊 Component Scores:")
        for component, score in breakdown.component_scores.items():
            if score != 0:
                print(f"   • {component.value}: {score:.2f}")
    
    if breakdown.bonus_points:
        print(f"\n🎁 Bonus Points: {breakdown.bonus_points:.2f}")
    
    if breakdown.penalty_points:
        print(f"\n⚠️  Penalty Points: {breakdown.penalty_points:.2f}")
    
    if breakdown.multipliers:
        print(f"\n📈 Multipliers Applied:")
        for name, multiplier in breakdown.multipliers.items():
            print(f"   • {name}: {multiplier:.2f}x")
    
    if breakdown.explanation:
        print(f"\n💡 Explanation:")
        for explanation in breakdown.explanation:
            print(f"   • {explanation}")


async def basic_reward_evaluation_example():
    """Demonstrate basic reward evaluation"""
    print("🎯 Basic Reward Evaluation Example")
    print("=" * 60)
    
    # Create execution environment
    env = ExecutionEnvironmentFactory.create_api_testing_environment(server_port=8005)
    
    async with env.temporary_environment():
        # Load API into mock server
        if env.mock_server:
            env.mock_server.load_user_management_api()
            print("📋 Loaded user management API")
        
        # Create reward evaluator
        evaluator = BaseRewardEvaluator()
        
        # Test successful API integration code
        successful_code = '''
import requests

base_url = get_api_base_url()
print(f"Testing API at: {base_url}")

try:
    # Make a GET request to list users
    response = requests.get(f"{base_url}/users", timeout=10)
    response.raise_for_status()
    
    users = response.json()
    print(f"✅ Successfully retrieved {len(users.get('items', []))} users")
    
    # Create a new user
    new_user = {
        "username": "reward_test_user",
        "email": "test@reward.com",
        "full_name": "Reward Test User"
    }
    
    create_response = requests.post(f"{base_url}/users", json=new_user, timeout=10)
    create_response.raise_for_status()
    
    created_user = create_response.json()
    print(f"✅ Created user with ID: {created_user.get('id', 'unknown')}")
    
except requests.RequestException as e:
    print(f"❌ API request failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
'''
        
        print("\n🧪 Executing successful API integration code...")
        execution_result = await env.execute_code(successful_code, timeout=60)
        
        print(f"Execution Status: {execution_result.status.value}")
        print(f"Execution Time: {execution_result.execution_time:.2f}s")
        
        # Get expected endpoints
        endpoint_gen = EndpointGenerator()
        user_endpoints = endpoint_gen.get_user_endpoints()[:2]  # GET and POST
        
        # Evaluate the code
        reward_breakdown = evaluator.evaluate_code_execution(
            code=successful_code,
            execution_result=execution_result,
            expected_endpoints=user_endpoints
        )
        
        print_reward_breakdown(reward_breakdown, "Successful Code Evaluation")
        
        # Now test failing code
        failing_code = '''
import requests

base_url = get_api_base_url()
print(f"Testing API at: {base_url}")

# This will fail - no error handling
response = requests.get(f"{base_url}/nonexistent_endpoint")
data = response.json()  # This will likely fail
print(f"Data: {data}")
'''
        
        print("\n🧪 Executing failing API integration code...")
        failing_result = await env.execute_code(failing_code, timeout=30)
        
        # Evaluate the failing code
        failing_breakdown = evaluator.evaluate_code_execution(
            code=failing_code,
            execution_result=failing_result,
            expected_endpoints=user_endpoints
        )
        
        print_reward_breakdown(failing_breakdown, "Failing Code Evaluation")


async def binary_reward_example():
    """Demonstrate binary reward evaluation"""
    print("\n🔄 Binary Reward Evaluation Example")
    print("=" * 60)
    
    # Create execution environment
    env = ExecutionEnvironmentFactory.create_basic_environment()
    
    async with env.temporary_environment():
        # Create binary evaluator
        binary_evaluator = BinaryRewardEvaluator()
        
        # Test cases with different outcomes
        test_cases = [
            {
                'name': 'Working Code',
                'code': '''
import requests
print("Making API call...")
try:
    response = requests.get("https://httpbin.org/json", timeout=5)
    if response.status_code == 200:
        print("✅ API call successful")
    else:
        print(f"❌ API call failed: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")
''',
                'expected_binary': 1.0
            },
            {
                'name': 'Code with Syntax Error',
                'code': '''
import requests
print("Making API call..."
# Missing closing parenthesis - syntax error
response = requests.get("https://httpbin.org/json"
print(response.status_code)
''',
                'expected_binary': 0.0
            },
            {
                'name': 'Code with No API Calls',
                'code': '''
print("Hello World")
x = 2 + 2
print(f"2 + 2 = {x}")
''',
                'expected_binary': 0.0
            }
        ]
        
        print("\n📊 Binary evaluation results:")
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            
            # Execute code
            result = await env.execute_code(test_case['code'])
            
            # Get binary evaluation
            binary_score = binary_evaluator.evaluate_simple(
                test_case['code'], result
            )
            
            # Get detailed evaluation for comparison
            detailed_breakdown = binary_evaluator.evaluate_code_execution(
                test_case['code'], result
            )
            
            print(f"   Binary Score: {binary_score} (expected: {test_case['expected_binary']})")
            print(f"   Detailed Score: {detailed_breakdown.total_reward:.2f}")
            print(f"   Status: {result.status.value}")
            
            # Verify expectation
            if binary_score == test_case['expected_binary']:
                print("   ✅ Result matches expectation")
            else:
                print("   ❌ Result differs from expectation")


async def template_completion_evaluation_example():
    """Demonstrate evaluation of completed code templates"""
    print("\n📝 Template Completion Evaluation Example")
    print("=" * 60)
    
    # Create execution environment with server
    env = ExecutionEnvironmentFactory.create_api_testing_environment(server_port=8006)
    
    async with env.temporary_environment():
        # Load API into mock server
        if env.mock_server:
            env.mock_server.load_product_catalog_api()
            print("📋 Loaded product catalog API")
        
        # Generate a code template
        endpoint_gen = EndpointGenerator()
        product_endpoints = endpoint_gen.get_product_endpoints()[:2]
        
        template_gen = APIIntegrationTemplateGenerator(seed=42)
        template = template_gen.generate_template(
            endpoints=product_endpoints,
            missing_components=[
                MissingComponent.IMPORTS,
                MissingComponent.ERROR_HANDLING,
                MissingComponent.RESPONSE_PARSING
            ],
            difficulty=DifficultyLevel.INTERMEDIATE
        )
        
        print(f"📋 Generated template with {len(template.gaps)} gaps")
        
        # Simulate completed code (with gaps filled)
        completed_code = '''
import requests
import json

class ProductAPIClient:
    """API client for product management"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def list_items(self, page: int = 1, limit: int = 10):
        """Retrieve a list of items with pagination"""
        url = self.base_url + '/products'
        params = {'page': page, 'limit': limit}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f'API request failed: {e}')
        
        return response.json()

# Usage Example
if __name__ == '__main__':
    client = ProductAPIClient(
        base_url=get_api_base_url(),
        api_key='test-key'
    )
    
    try:
        result = client.list_items()
        print(f"✅ Retrieved products: {len(result.get('items', []))}")
    except Exception as e:
        print(f"❌ Error: {e}")
'''
        
        # Create reward integrator
        integrator = CodeRewardIntegrator(env, EvaluationMode.DETAILED)
        
        # Create evaluation context
        context = EvaluationContext(
            template=template,
            expected_endpoints=product_endpoints,
            difficulty_level="intermediate",
            time_limit=60,
            expected_api_calls=1
        )
        
        print("\n🔍 Evaluating completed template...")
        evaluation_result = await integrator.evaluate_code_completion(
            completed_code, context
        )
        
        print_reward_breakdown(evaluation_result.reward_breakdown, "Template Completion Evaluation")
        
        print(f"\n📈 Performance Metrics:")
        for metric, value in evaluation_result.performance_metrics.items():
            print(f"   • {metric}: {value}")
        
        print(f"\n✅ Gaps Completed: {len(evaluation_result.gaps_completed)}/{len(template.gaps)}")
        for gap in evaluation_result.gaps_completed:
            print(f"   • {gap.value}")
        
        if evaluation_result.suggestions:
            print(f"\n💡 Improvement Suggestions:")
            for suggestion in evaluation_result.suggestions:
                print(f"   • {suggestion}")


async def test_suite_evaluation_example():
    """Demonstrate evaluation using test suites"""
    print("\n🧪 Test Suite Evaluation Example")
    print("=" * 60)
    
    # Create execution environment
    env = ExecutionEnvironmentFactory.create_api_testing_environment(server_port=8007)
    
    async with env.temporary_environment():
        # Load API
        if env.mock_server:
            env.mock_server.load_user_management_api()
            print("📋 Loaded user management API")
        
        # Student's attempted solution
        student_code = '''
import requests

def get_users(base_url):
    """Get list of users"""
    response = requests.get(f"{base_url}/users")
    if response.status_code == 200:
        return response.json()
    else:
        return None

def create_user(base_url, user_data):
    """Create a new user"""
    response = requests.post(f"{base_url}/users", json=user_data)
    if response.status_code in [200, 201]:
        return response.json()
    else:
        return None

# Test the functions
base_url = get_api_base_url()
print(f"Testing with base URL: {base_url}")

users = get_users(base_url)
if users:
    print(f"✅ Retrieved {len(users.get('items', []))} users")
else:
    print("❌ Failed to retrieve users")

new_user = {
    "username": "test_student",
    "email": "student@test.com",
    "full_name": "Test Student"
}
created = create_user(base_url, new_user)
if created:
    print(f"✅ Created user: {created.get('username', 'unknown')}")
else:
    print("❌ Failed to create user")
'''
        
        # Create test suite
        test_suite = TestSuite(
            name="User Management API Test Suite",
            setup_code='''
# Test setup
base_url = get_api_base_url()
assert test_server_connection(), "Server must be available"
''',
            test_cases=[
                TestCase(
                    name="test_get_users_function_exists",
                    code='''
# Test that get_users function exists and works
assert 'get_users' in globals(), "get_users function not found"
result = get_users(base_url)
assert result is not None, "get_users should return data"
print("✅ get_users function works")
''',
                    expected_status=ExecutionStatus.SUCCESS
                ),
                
                TestCase(
                    name="test_create_user_function_exists",
                    code='''
# Test that create_user function exists and works
assert 'create_user' in globals(), "create_user function not found"
test_user = {"username": "test", "email": "test@example.com"}
result = create_user(base_url, test_user)
assert result is not None, "create_user should return data"
print("✅ create_user function works")
''',
                    expected_status=ExecutionStatus.SUCCESS
                ),
                
                TestCase(
                    name="test_api_calls_made",
                    code='''
# Verify API calls are actually made
import sys
from io import StringIO

# Capture output during function calls
old_stdout = sys.stdout
sys.stdout = captured_output = StringIO()

try:
    get_users(base_url)
    create_user(base_url, {"username": "test", "email": "test@example.com"})
finally:
    sys.stdout = old_stdout

output = captured_output.getvalue()
print("✅ API functions executed")
''',
                    expected_status=ExecutionStatus.SUCCESS
                )
            ],
            teardown_code='''
print("🧹 Test suite completed")
'''
        )
        
        # Create reward integrator
        integrator = CodeRewardIntegrator(env, EvaluationMode.DETAILED)
        
        # Create evaluation context
        endpoint_gen = EndpointGenerator()
        user_endpoints = endpoint_gen.get_user_endpoints()[:2]
        
        context = EvaluationContext(
            expected_endpoints=user_endpoints,
            difficulty_level="intermediate",
            time_limit=90,
            expected_api_calls=2
        )
        
        print("\n🏃 Running test suite evaluation...")
        evaluation_result = await integrator.evaluate_test_suite_completion(
            student_code, test_suite, context
        )
        
        print_reward_breakdown(evaluation_result.reward_breakdown, "Test Suite Evaluation")
        
        print(f"\n📊 Test Results:")
        test_metrics = evaluation_result.performance_metrics
        print(f"   • Tests Passed: {test_metrics.get('tests_passed', 0)}")
        print(f"   • Tests Failed: {test_metrics.get('tests_failed', 0)}")
        print(f"   • Success Rate: {test_metrics.get('test_success_rate', 0):.1%}")
        print(f"   • Total Test Time: {test_metrics.get('total_test_time', 0):.2f}s")


async def performance_tracking_example():
    """Demonstrate performance tracking over time"""
    print("\n📈 Performance Tracking Example")
    print("=" * 60)
    
    # Create execution environment
    env = ExecutionEnvironmentFactory.create_basic_environment()
    
    async with env.temporary_environment():
        # Create reward integrator
        integrator = CodeRewardIntegrator(env, EvaluationMode.DETAILED)
        
        # Simulate multiple code submissions with improving quality
        code_submissions = [
            # Submission 1: Basic, no error handling
            '''
import requests
response = requests.get("https://httpbin.org/json")
data = response.json()
print(data)
''',
            # Submission 2: Added error handling
            '''
import requests
try:
    response = requests.get("https://httpbin.org/json")
    response.raise_for_status()
    data = response.json()
    print(f"Success: {data}")
except requests.RequestException as e:
    print(f"Error: {e}")
''',
            # Submission 3: More comprehensive
            '''
import requests
import json

def fetch_data(url):
    """Fetch data from API with proper error handling"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print("Request timed out")
        return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Use the function
url = "https://httpbin.org/json"
data = fetch_data(url)
if data:
    print(f"✅ Successfully fetched data: {type(data)}")
else:
    print("❌ Failed to fetch data")
''',
            # Submission 4: Professional quality
            '''
import requests
import json
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClient:
    """Professional API client with comprehensive error handling"""
    
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def get(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make GET request with proper error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.info(f"Making GET request to {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Request successful: {response.status_code}")
            return data
            
        except requests.Timeout:
            logger.error(f"Request to {url} timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            return None

# Usage
client = APIClient("https://httpbin.org")
data = client.get("/json")
if data:
    print(f"✅ Data retrieved successfully")
else:
    print("❌ Failed to retrieve data")
'''
        ]
        
        print("\n🔄 Evaluating progressive code submissions...")
        
        context = EvaluationContext(
            difficulty_level="intermediate",
            time_limit=30,
            expected_api_calls=1
        )
        
        # Evaluate each submission
        for i, code in enumerate(code_submissions, 1):
            print(f"\n📝 Evaluating Submission {i}...")
            
            execution_result = await env.execute_code(code)
            evaluation_result = await integrator.evaluate_code_completion(code, context)
            
            print(f"   Reward: {evaluation_result.reward_breakdown.total_reward:.2f}")
            print(f"   Status: {execution_result.status.value}")
            print(f"   Time: {execution_result.execution_time:.2f}s")
        
        # Show performance summary
        print(f"\n📊 Performance Summary:")
        summary = integrator.get_performance_summary()
        
        print(f"   • Total Evaluations: {summary['total_evaluations']}")
        print(f"   • Average Reward: {summary['average_reward']:.2f}")
        print(f"   • Best Reward: {summary['best_reward']:.2f}")
        print(f"   • Worst Reward: {summary['worst_reward']:.2f}")
        print(f"   • Recent Trend: {summary['recent_trend']:.2f} (-1=declining, +1=improving)")
        
        if 'component_averages' in summary:
            print(f"\n📈 Component Averages:")
            for component, avg in summary['component_averages'].items():
                if avg > 0:
                    print(f"   • {component}: {avg:.2f}")


async def custom_validator_example():
    """Demonstrate custom validators"""
    print("\n🔧 Custom Validator Example")
    print("=" * 60)
    
    # Create execution environment
    env = ExecutionEnvironmentFactory.create_basic_environment()
    
    async with env.temporary_environment():
        # Create reward integrator
        integrator = CodeRewardIntegrator(env, EvaluationMode.DETAILED)
        
        # Create custom validators
        def check_authentication_validator(code, result, api_calls):
            """Check if code includes authentication"""
            auth_keywords = ['authorization', 'bearer', 'api_key', 'token']
            has_auth = any(keyword in code.lower() for keyword in auth_keywords)
            return {
                'has_authentication': has_auth,
                'score': 5.0 if has_auth else 0.0,
                'message': 'Authentication found' if has_auth else 'No authentication detected'
            }
        
        def check_logging_validator(code, result, api_calls):
            """Check if code includes proper logging"""
            has_logging = 'logging' in code.lower() or 'logger' in code.lower()
            return {
                'has_logging': has_logging,
                'score': 3.0 if has_logging else 0.0,
                'message': 'Logging implemented' if has_logging else 'No logging found'
            }
        
        def check_class_structure_validator(code, result, api_calls):
            """Check if code uses proper class structure"""
            import re
            class_count = len(re.findall(r'class\s+\w+', code))
            method_count = len(re.findall(r'def\s+\w+', code))
            
            return {
                'class_count': class_count,
                'method_count': method_count,
                'score': (class_count * 2.0) + (method_count * 0.5),
                'message': f'Found {class_count} classes and {method_count} methods'
            }
        
        # Test code with good structure
        test_code = '''
import requests
import logging

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    def get_data(self, url):
        logger.info(f"Fetching data from {url}")
        response = requests.get(url, headers=self.headers)
        return response.json()

client = APIClient('test-key')
data = client.get_data('https://httpbin.org/json')
print("Data retrieved")
'''
        
        # Create context with custom validators
        context = EvaluationContext(
            difficulty_level="advanced",
            time_limit=30,
            expected_api_calls=1,
            custom_validators=[
                integrator.create_custom_validator("auth_check", check_authentication_validator),
                integrator.create_custom_validator("logging_check", check_logging_validator),  
                integrator.create_custom_validator("structure_check", check_class_structure_validator)
            ]
        )
        
        print("\n🔍 Evaluating code with custom validators...")
        evaluation_result = await integrator.evaluate_code_completion(test_code, context)
        
        print_reward_breakdown(evaluation_result.reward_breakdown, "Custom Validator Evaluation")
        
        print(f"\n🔧 Custom Validation Results:")
        for validator_name, result in evaluation_result.validation_results.items():
            print(f"   • {validator_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"     - {key}: {value}")
            else:
                print(f"     - Result: {result}")


async def main():
    """Run all reward system examples"""
    print("🎯 Reward System Examples")
    print("=" * 70)
    
    try:
        # Run all examples
        await basic_reward_evaluation_example()
        await binary_reward_example()
        await template_completion_evaluation_example()
        await test_suite_evaluation_example()
        await performance_tracking_example()
        await custom_validator_example()
        
        print("\n" + "="*70)
        print("🎉 All reward system examples completed successfully!")
        
        print("\n💡 Key Features Demonstrated:")
        print("   • Binary (pass/fail) and detailed reward evaluation")
        print("   • API call detection and validation")
        print("   • Response handling assessment")
        print("   • Code quality analysis")
        print("   • Template completion tracking")
        print("   • Test suite integration")
        print("   • Performance metrics and trends")
        print("   • Custom validation functions")
        
        print("\n🎯 Reward Components:")
        for component in RewardComponent:
            print(f"   • {component.value}")
        
        print("\n📊 Evaluation Modes:")
        for mode in EvaluationMode:
            print(f"   • {mode.value}")
        
        print("\n🚀 Perfect for RL Training:")
        print("   • Comprehensive feedback for agent learning")
        print("   • Multiple difficulty levels and contexts")
        print("   • Performance tracking and trend analysis")
        print("   • Customizable validation and scoring")
        print("   • Integration with execution environments")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())