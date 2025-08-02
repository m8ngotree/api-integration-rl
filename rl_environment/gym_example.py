"""
Example usage of the API Integration Gymnasium Environment

This script demonstrates how to use the APIIntegrationEnv for API integration learning tasks.
"""

import asyncio
import numpy as np
from gym_environment import APIIntegrationEnv, EnvironmentConfig
from rl_environment.task_generator import TaskDifficulty


async def basic_environment_demo():
    """Basic demonstration of the environment"""
    print("=== API Integration Environment Demo ===\n")
    
    # Create environment configuration
    config = EnvironmentConfig(
        max_episode_steps=20,
        task_difficulty=TaskDifficulty.BEGINNER,
        execution_timeout=15
    )
    
    # Create environment
    env = APIIntegrationEnv(config)
    
    try:
        # Reset environment
        print("Resetting environment...")
        observation, info = await env.reset(seed=42)
        
        print(f"Task: {info['task_title']}")
        print(f"Task ID: {info['task_id']}")
        print(f"Initial code length: {info['code_length']}")
        print(f"Hints available: {info['hints_available']}")
        print()
        
        # Run a few steps
        for step in range(5):
            print(f"--- Step {step + 1} ---")
            
            # Create a simple action (insert import statement)
            action = {
                'operation_type': 0,  # Insert
                'line_position': np.array([0]),  # At beginning
                'code_content': np.random.random(500),  # Random tokens (simplified)
                'execute_code': 1 if step % 2 == 0 else 0  # Execute every other step
            }
            
            # Take step
            observation, reward, terminated, truncated, info = await env.step(action)
            
            print(f"Reward: {reward:.3f}")
            print(f"Progress: {observation['completion_progress'][0]:.3f}")
            print(f"Current score: {observation['current_score'][0]:.3f}")
            print(f"Code length: {info['code_length']}")
            
            if 'execution_status' in info:
                print(f"Execution status: {info['execution_status']}")
                print(f"Execution time: {info['execution_time']:.3f}s")
            
            print()
            
            if terminated:
                print("✅ Task completed successfully!")
                break
            elif truncated:
                print("⏰ Episode truncated (max steps reached)")
                break
        
        print("Episode finished.")
        
    finally:
        # Clean up
        await env.close()


async def observation_space_demo():
    """Demonstrate the observation space structure"""
    print("=== Observation Space Demo ===\n")
    
    env = APIIntegrationEnv()
    
    try:
        observation, info = await env.reset()
        
        print("Observation space structure:")
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.size <= 10:
                    print(f"    values: {value}")
                else:
                    print(f"    sample values: {value[:5]}...")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        print("\nAction space structure:")
        print(f"  operation_type: Discrete(4) - {env.action_space['operation_type']}")
        print(f"  line_position: {env.action_space['line_position']}")
        print(f"  code_content: {env.action_space['code_content']}")
        print(f"  execute_code: Discrete(2) - {env.action_space['execute_code']}")
        
    finally:
        await env.close()


async def action_space_demo():
    """Demonstrate different types of actions"""
    print("=== Action Space Demo ===\n")
    
    env = APIIntegrationEnv(EnvironmentConfig(max_episode_steps=10))
    
    try:
        observation, info = await env.reset()
        print(f"Starting with task: {info['task_title']}\n")
        
        # Demonstrate different operation types
        actions = [
            {
                'operation_type': 0,  # Insert
                'line_position': np.array([0]),
                'code_content': np.random.random(500),
                'execute_code': 0
            },
            {
                'operation_type': 1,  # Replace
                'line_position': np.array([1]),
                'code_content': np.random.random(500),
                'execute_code': 1
            },
            {
                'operation_type': 3,  # Append
                'line_position': np.array([0]),  # Ignored for append
                'code_content': np.random.random(500),
                'execute_code': 1
            }
        ]
        
        operation_names = ['Insert', 'Replace', 'Append']
        
        for i, (action, op_name) in enumerate(zip(actions, operation_names)):
            print(f"Action {i+1}: {op_name}")
            observation, reward, terminated, truncated, info = await env.step(action)
            
            print(f"  Reward: {reward:.3f}")
            print(f"  New code length: {info['code_length']}")
            if 'execution_status' in info:
                print(f"  Execution: {info['execution_status']}")
            print()
            
            if terminated or truncated:
                break
                
    finally:
        await env.close()


if __name__ == "__main__":
    print("API Integration Gymnasium Environment Examples\n")
    
    # Run demos
    asyncio.run(basic_environment_demo())
    print("\n" + "="*50 + "\n")
    
    asyncio.run(observation_space_demo())
    print("\n" + "="*50 + "\n")
    
    asyncio.run(action_space_demo())