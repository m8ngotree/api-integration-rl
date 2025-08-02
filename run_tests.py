#!/usr/bin/env python3
"""
Test runner for API Integration RL Environment
Handles relative import issues and provides easy testing interface
"""

import sys
import asyncio
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_health_check():
    """Quick health check of all components"""
    print('🔍 API Integration RL Environment - Health Check')
    print('=' * 50)
    
    components = [
        ('Task Generator', 'rl_environment.task_generator', 'TaskGeneratorManager'),
        ('Mock Server', 'mock_servers.base_server', 'MockServer'),
        ('Code Templates', 'utilities.code_template_generator', 'APIIntegrationTemplateGenerator'),
        ('Execution Environment', 'rl_environment.execution_environment', 'SafeExecutionEnvironment'),
        ('Reward System', 'rl_environment.reward_system', 'BaseRewardEvaluator'),
        ('Data Generation', 'data_generation.data_generator', 'RandomDataGenerator'),
    ]
    
    for name, module, class_name in components:
        try:
            exec(f"from {module} import {class_name}")
            print(f'✅ {name}: OK')
        except Exception as e:
            print(f'❌ {name}: {e}')
    
    print('\n🎉 Health check complete!')

def test_task_generator():
    """Test task generator functionality"""
    print('\n🎯 Testing Task Generator')
    print('=' * 40)
    
    try:
        from rl_environment.task_generator import TaskGeneratorManager, TaskDifficulty
        
        manager = TaskGeneratorManager(seed=42)
        task = manager.generate_task(difficulty=TaskDifficulty.BEGINNER)
        
        print(f'✅ Generated task: {task.title}')
        print(f'   Type: {task.task_type.value}')
        print(f'   Difficulty: {task.difficulty.value}')
        print(f'   Estimated time: {task.estimated_time} minutes')
        print(f'   API endpoints: {len(task.api_documentation.endpoints)}')
        print(f'   Hints: {len(task.hints)}')
        
        # Test curriculum generation
        curriculum = manager.generate_progressive_curriculum(total_tasks=5)
        print(f'✅ Generated curriculum with {len(curriculum)} tasks')
        
        return True
    except Exception as e:
        print(f'❌ Task Generator test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_mock_server():
    """Test mock server functionality"""
    print('\n🌐 Testing Mock Server')
    print('=' * 40)
    
    try:
        from mock_servers.schema_server import SchemaBasedMockServer
        
        server = SchemaBasedMockServer()
        api_spec = server.load_user_management_api()
        
        print(f'✅ Mock server created successfully')
        print(f'   API title: {api_spec["info"]["title"]}')
        print(f'   Endpoints: {len(api_spec["paths"])}')
        
        return True
    except Exception as e:
        print(f'❌ Mock Server test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_code_templates():
    """Test code template generation"""
    print('\n🏗️ Testing Code Templates')
    print('=' * 40)
    
    try:
        from utilities.code_template_generator import APIIntegrationTemplateGenerator, DifficultyLevel, MissingComponent
        from data_generation.endpoint_generator import EndpointGenerator
        
        gen = APIIntegrationTemplateGenerator(seed=42)
        endpoints = EndpointGenerator().get_user_endpoints()[:2]
        missing_components = [MissingComponent.IMPORTS, MissingComponent.ERROR_HANDLING]
        template = gen.generate_template(endpoints, missing_components, difficulty=DifficultyLevel.BEGINNER)
        
        print(f'✅ Generated template with {len(template.gaps)} gaps')
        print(f'   Template length: {len(template.code)} characters')
        print(f'   Description: {template.description}')
        
        return True
    except Exception as e:
        print(f'❌ Code Templates test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_execution_environment():
    """Test safe code execution"""
    print('\n🔒 Testing Execution Environment')
    print('=' * 40)
    
    try:
        from rl_environment.execution_environment import ExecutionEnvironmentFactory
        
        env = ExecutionEnvironmentFactory.create_basic_environment()
        
        async with env.temporary_environment():
            # Test simple code execution
            result = await env.execute_code('print("Hello from safe execution!")')
            
            print(f'✅ Code execution successful')
            print(f'   Status: {result.status.value}')
            print(f'   Output: {result.stdout.strip()}')
            print(f'   Execution time: {result.execution_time:.3f}s')
        
        return True
    except Exception as e:
        print(f'❌ Execution Environment test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_reward_system():
    """Test reward system"""
    print('\n🏆 Testing Reward System')
    print('=' * 40)
    
    try:
        from rl_environment.reward_system import BaseRewardEvaluator
        from rl_environment.code_executor import ExecutionResult, ExecutionStatus
        
        evaluator = BaseRewardEvaluator()
        
        # Create a mock execution result
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            stdout='GET /users 200\nRetrieved 5 users',
            stderr='',
            execution_time=1.5,
            memory_usage=25.0
        )
        
        code = '''
import requests
response = requests.get("https://api.example.com/users")
print(f"GET /users {response.status_code}")
data = response.json()
print(f"Retrieved {len(data['items'])} users")
'''
        
        breakdown = evaluator.evaluate_code_execution(code, result)
        
        print(f'✅ Reward calculation successful')
        print(f'   Total reward: {breakdown.total_reward:.2f}')
        print(f'   Components evaluated: {len(breakdown.component_scores)}')
        
        return True
    except Exception as e:
        print(f'❌ Reward System test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_full_workflow():
    """Test complete workflow integration"""
    print('\n🎓 Testing Full Workflow')
    print('=' * 40)
    
    try:
        from rl_environment.task_generator import TaskGeneratorManager, TaskDifficulty
        from rl_environment.execution_environment import ExecutionEnvironmentFactory
        from rl_environment.reward_integration import CodeRewardIntegrator, EvaluationMode, EvaluationContext
        
        # 1. Generate a task
        manager = TaskGeneratorManager(seed=42)
        task = manager.generate_task(difficulty=TaskDifficulty.BEGINNER)
        print(f'✅ 1. Generated task: {task.title}')
        
        # 2. Set up execution environment
        env = ExecutionEnvironmentFactory.create_basic_environment()
        print('✅ 2. Created execution environment')
        
        # 3. Set up reward integration
        integrator = CodeRewardIntegrator(env, EvaluationMode.DETAILED)
        print('✅ 3. Created reward integrator')
        
        # 4. Test with simple code
        sample_code = '''
print("Testing API integration workflow")
import json
data = {"status": "success", "message": "API integration working"}
print(f"Result: {data['status']}")
'''
        
        async with env.temporary_environment():
            context = EvaluationContext(
                difficulty_level=task.difficulty.value,
                time_limit=30,
                expected_api_calls=0  # Simple test, no actual API calls
            )
            
            # 5. Execute and evaluate
            result = await integrator.evaluate_code_completion(sample_code, context)
            print(f'✅ 4. Executed and evaluated code')
            print(f'   Reward: {result.reward_breakdown.total_reward:.2f}')
            print(f'   Status: {result.execution_result.status.value}')
            print(f'   Execution time: {result.execution_result.execution_time:.3f}s')
        
        print('🎉 Complete workflow test successful!')
        return True
        
    except Exception as e:
        print(f'❌ Full workflow test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print('🚀 API Integration RL Environment - Test Suite')
    print('=' * 60)
    
    tests = [
        ('Health Check', test_health_check, False),
        ('Task Generator', test_task_generator, False),
        ('Mock Server', test_mock_server, False),
        ('Code Templates', test_code_templates, False),
        ('Execution Environment', test_execution_environment, True),
        ('Reward System', test_reward_system, False),
        ('Full Workflow', test_full_workflow, True),
    ]
    
    results = []
    
    for name, test_func, is_async in tests:
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f'❌ {name} test crashed: {e}')
            results.append((name, False))
    
    # Summary
    print('\n' + '=' * 60)
    print('📊 Test Results Summary')
    print('=' * 60)
    
    passed = 0
    for name, result in results:
        status = '✅ PASS' if result else '❌ FAIL'
        print(f'{status} {name}')
        if result:
            passed += 1
    
    print(f'\n🎯 Results: {passed}/{len(results)} tests passed')
    
    if passed == len(results):
        print('🎉 All tests passed! The API Integration RL Environment is ready to use!')
    else:
        print('⚠️  Some tests failed. Please check the error messages above.')

if __name__ == '__main__':
    asyncio.run(main())