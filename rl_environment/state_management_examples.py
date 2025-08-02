"""
Examples and tests for the comprehensive state management system.

This script demonstrates state tracking, episode management, checkpointing,
and termination conditions in the API integration RL environment.
"""

import asyncio
import json
import time
from typing import Dict, List, Any

from rl_environment.state_management import (
    EnvironmentStateManager, EpisodeState, TerminationReason,
    CodeSnapshot, ExecutionSnapshot, APICallRecord, ProgressMetrics
)
from rl_environment.gym_environment import APIIntegrationEnv, EnvironmentConfig
from rl_environment.task_generator import TaskDifficulty
from rl_environment.code_executor import ExecutionResult, ExecutionStatus


def demonstrate_state_manager_basics():
    """Demonstrate basic state manager functionality"""
    print("=== State Manager Basics Demo ===\n")
    
    # Create state manager
    state_manager = EnvironmentStateManager(max_history_length=50)
    
    print("Initial state:")
    print(f"  Episode state: {state_manager.episode_state.value}")
    print(f"  Current step: {state_manager.current_step}")
    print(f"  Episode ID: {state_manager.episode_id}")
    print()
    
    # Start a new episode
    print("Starting new episode...")
    mock_task = type('MockTask', (), {
        'task_id': 'test_task_123',
        'title': 'Test API Integration Task',
        'difficulty': TaskDifficulty.INTERMEDIATE
    })()
    
    episode_id = state_manager.start_new_episode(
        task=mock_task,
        initial_code="# Initial code template\nimport requests\n",
        episode_config={'max_steps': 20}
    )
    
    print(f"  Episode ID: {episode_id}")
    print(f"  Episode state: {state_manager.episode_state.value}")
    print(f"  Initial code length: {len(state_manager.current_code)}")
    print(f"  Code history length: {len(state_manager.code_history)}")
    print()
    
    # Record some code changes
    print("Recording code changes...")
    code_versions = [
        "# Initial code template\nimport requests\n\nurl = 'https://api.example.com/users'",
        "# Initial code template\nimport requests\n\nurl = 'https://api.example.com/users'\nresponse = requests.get(url)",
        "# Initial code template\nimport requests\n\nurl = 'https://api.example.com/users'\nresponse = requests.get(url)\nif response.status_code == 200:\n    data = response.json()\n    print(data)"
    ]
    
    for i, code in enumerate(code_versions, 1):
        state_manager.record_code_change(code, [{'modification': f'step_{i}'}])
        state_manager.advance_step()
        
        print(f"  Step {i}: Code length = {len(code)}, Completion = {state_manager.progress_metrics.completion_percentage:.2f}")
    
    print()
    print("Final state summary:")
    summary = state_manager.get_state_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_execution_tracking():
    """Demonstrate execution history tracking"""
    print("=== Execution Tracking Demo ===\n")
    
    state_manager = EnvironmentStateManager()
    
    # Start episode
    mock_task = type('MockTask', (), {'task_id': 'exec_test', 'title': 'Execution Test'})()
    state_manager.start_new_episode(mock_task, "import requests")
    
    print("Recording execution events...")
    
    # Simulate different execution results
    executions = [
        # Successful execution
        ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            stdout="User data: {'id': 1, 'name': 'John Doe'}",
            stderr="",
            execution_time=0.5,
            metadata={'network_requests': [
                {'method': 'GET', 'url': 'https://api.example.com/users/1', 'status_code': 200, 'success': True}
            ]}
        ),
        # Failed execution
        ExecutionResult(
            status=ExecutionStatus.ERROR,
            stdout="",
            stderr="requests.exceptions.ConnectionError: Connection failed",
            execution_time=1.2
        ),
        # Timeout
        ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            stdout="",
            stderr="Execution timed out after 30 seconds",
            execution_time=30.0
        )
    ]
    
    for i, result in enumerate(executions, 1):
        state_manager.advance_step()
        
        # Record code change
        code = f"# Step {i} code\nimport requests\nresponse = requests.get('https://api.example.com/users/{i}')"
        state_manager.record_code_change(code)
        
        # Record execution
        execution_id = state_manager.record_execution(
            execution_result=result,
            code_hash=f"hash_{i}",
            api_calls=[{
                'method': 'GET',
                'url': f'https://api.example.com/users/{i}',
                'success': result.status == ExecutionStatus.SUCCESS,
                'duration': result.execution_time
            }] if result.status != ExecutionStatus.TIMEOUT else []
        )
        
        print(f"  Execution {i} ({result.status.value}): ID = {execution_id}")
        print(f"    Duration: {result.execution_time}s")
        print(f"    API calls recorded: {len(state_manager.api_call_history)}")
    
    print(f"\nExecution statistics:")
    stats = state_manager.episode_statistics
    print(f"  Total executions: {stats.total_executions}")
    print(f"  Successful: {stats.successful_executions}")
    print(f"  Failed: {stats.failed_executions}")
    print(f"  API calls: {stats.total_api_calls}")
    print(f"  Successful API calls: {stats.successful_api_calls}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_progress_tracking():
    """Demonstrate progress and milestone tracking"""
    print("=== Progress Tracking Demo ===\n")
    
    state_manager = EnvironmentStateManager()
    
    # Start episode
    mock_task = type('MockTask', (), {'task_id': 'progress_test', 'title': 'Progress Test'})()
    state_manager.start_new_episode(mock_task, "")
    
    print("Simulating code development progress...")
    
    # Simulate progressive code development
    code_stages = [
        ("Empty", ""),
        ("Imports", "import requests\nimport json"),
        ("Basic structure", "import requests\nimport json\n\ndef get_user_data():\n    pass"),
        ("API call", "import requests\nimport json\n\ndef get_user_data():\n    url = 'https://api.example.com/users'\n    response = requests.get(url)\n    return response"),
        ("Success execution", "import requests\nimport json\n\ndef get_user_data():\n    url = 'https://api.example.com/users'\n    response = requests.get(url)\n    return response"),
        ("Error handling", "import requests\nimport json\n\ndef get_user_data():\n    try:\n        url = 'https://api.example.com/users'\n        response = requests.get(url)\n        response.raise_for_status()\n        return response.json()\n    except requests.RequestException as e:\n        print(f'Error: {e}')\n        return None"),
    ]
    
    for i, (stage_name, code) in enumerate(code_stages):
        state_manager.advance_step()
        state_manager.record_code_change(code)
        
        # Simulate execution for later stages
        if i >= 4:  # Success execution stage
            result = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout="{'users': []}",
                stderr="",
                execution_time=0.3
            )
            state_manager.record_execution(result)
            
            # Record API call
            state_manager.record_api_call({
                'method': 'GET',
                'url': 'https://api.example.com/users',
                'success': True,
                'duration': 0.3
            })
        
        # Record some reward
        reward = 0.5 + (i * 0.3)  # Increasing reward
        score = i * 1.5
        state_manager.record_reward(reward, score)
        
        progress = state_manager.progress_metrics
        print(f"  {stage_name:15} | Progress: {progress.completion_percentage:.2f} | "
              f"Phase: {progress.current_phase:12} | "
              f"Milestones: {len(progress.milestones_achieved)}")
    
    print(f"\nFinal milestones achieved:")
    for milestone in state_manager.progress_metrics.milestones_achieved:
        print(f"  ✓ {milestone}")
    
    print(f"\nProgress phases completed:")
    for phase in state_manager.progress_metrics.phases_completed:
        print(f"  ✓ {phase}")
    
    print("\n" + "="*50 + "\n")


def demonstrate_termination_conditions():
    """Demonstrate episode termination conditions"""
    print("=== Termination Conditions Demo ===\n")
    
    # Test different termination scenarios
    scenarios = [
        ("Max steps", {'max_steps': 3}),
        ("Too many syntax errors", {'max_syntax_errors': 2}),
        ("Task completion", {'completion_score_threshold': 5.0}),
        ("No progress", {'no_progress_threshold': 3})
    ]
    
    for scenario_name, config in scenarios:
        print(f"--- {scenario_name} Scenario ---")
        
        state_manager = EnvironmentStateManager()
        state_manager.config.update(config)
        
        # Start episode
        mock_task = type('MockTask', (), {'task_id': f'term_test_{scenario_name.lower().replace(" ", "_")}', 'title': scenario_name})()
        state_manager.start_new_episode(mock_task, "import requests")
        
        # Simulate steps until termination
        step = 0
        while step < 10:  # Safety limit
            step += 1
            state_manager.advance_step()
            
            if scenario_name == "Too many syntax errors":
                # Introduce syntax errors
                bad_code = f"import requests\ndef bad_function(\n    return None  # Step {step}"
                state_manager.record_code_change(bad_code)
            elif scenario_name == "Task completion":
                # Increase score towards completion
                state_manager.record_reward(1.0, step * 2.0)
            elif scenario_name == "No progress":
                # Consistently negative rewards
                state_manager.record_reward(-0.6, 0.1)
            else:
                # Normal step
                state_manager.record_code_change(f"import requests\n# Step {step}")
                state_manager.record_reward(0.1, step * 0.5)
            
            # Check termination
            terminated, truncated, reason = state_manager.check_episode_termination()
            
            if terminated or truncated:
                state_manager.end_episode(reason)
                print(f"  Episode ended at step {step}")
                print(f"  Reason: {reason.value}")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
                break
        
        print()
    
    print("="*50 + "\n")


def demonstrate_checkpoints():
    """Demonstrate checkpoint functionality"""
    print("=== Checkpoint System Demo ===\n")
    
    state_manager = EnvironmentStateManager()
    
    # Start episode
    mock_task = type('MockTask', (), {'task_id': 'checkpoint_test', 'title': 'Checkpoint Test'})()
    state_manager.start_new_episode(mock_task, "import requests")
    
    print("Creating checkpoints during episode...")
    
    # Make some progress
    checkpoints = []
    for i in range(1, 4):
        state_manager.advance_step()
        code = f"import requests\n\n# Step {i}\nurl = 'https://api.example.com/step{i}'"
        state_manager.record_code_change(code)
        state_manager.record_reward(i * 0.3, i * 1.0)
        
        # Create checkpoint
        checkpoint_id = state_manager.create_checkpoint(f"step_{i}")
        checkpoints.append(checkpoint_id)
        print(f"  Step {i}: Created checkpoint {checkpoint_id}")
    
    print(f"\nCheckpoints created: {len(checkpoints)}")
    print(f"Available checkpoints: {list(state_manager.checkpoints.keys())}")
    
    # Save current state
    current_step = state_manager.current_step
    current_code = state_manager.current_code
    current_score = state_manager.current_score
    
    print(f"\nCurrent state before restore:")
    print(f"  Step: {current_step}")
    print(f"  Code length: {len(current_code)}")
    print(f"  Score: {current_score}")
    
    # Restore to earlier checkpoint
    restore_checkpoint = checkpoints[1]  # Step 2
    print(f"\nRestoring to checkpoint: {restore_checkpoint}")
    
    success = state_manager.restore_checkpoint(restore_checkpoint)
    print(f"Restore successful: {success}")
    
    if success:
        print(f"State after restore:")
        print(f"  Step: {state_manager.current_step}")
        print(f"  Code length: {len(state_manager.current_code)}")
        print(f"  Score: {state_manager.current_score}")
    
    print("\n" + "="*50 + "\n")


async def demonstrate_environment_integration():
    """Demonstrate state management integration with the environment"""
    print("=== Environment Integration Demo ===\n")
    
    # Create environment with custom state management config
    config = EnvironmentConfig(
        max_episode_steps=10,
        task_difficulty=TaskDifficulty.BEGINNER,
        max_syntax_errors=3,
        completion_score_threshold=6.0,
        auto_checkpoint_interval=3
    )
    
    env = APIIntegrationEnv(config)
    
    try:
        # Reset environment
        observation, info = await env.reset(seed=42)
        
        print(f"Episode started:")
        print(f"  Episode ID: {info['episode_id']}")
        print(f"  Task: {info['task_title']}")
        print(f"  Initial state: {info['episode_state']}")
        print()
        
        # Take several steps
        step = 0
        episode_active = True
        
        while episode_active and step < 8:
            step += 1
            print(f"--- Step {step} ---")
            
            # Sample action
            action = env.action_space.sample()
            
            # Take step
            observation, reward, terminated, truncated, info = await env.step(action)
            
            print(f"  Actions applied: {info.get('actions_applied', 0)}/{info.get('actions_total', 0)}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Episode state: {info['episode_state']}")
            print(f"  Progress: {info['progress_metrics']['completion_percentage']:.2f}")
            print(f"  Current phase: {info['progress_metrics']['current_phase']}")
            print(f"  Milestones: {info['progress_metrics']['milestones_achieved']}")
            
            if terminated or truncated:
                print(f"  Episode ended: {info['termination_reason']}")
                episode_active = False
            
            print()
        
        # Get final statistics
        print("Final episode statistics:")
        episode_stats = env.get_episode_statistics()
        
        stats = episode_stats['statistics']
        print(f"  Total steps: {stats.total_steps}")
        print(f"  Successful steps: {stats.successful_steps}")
        print(f"  Total executions: {stats.total_executions}")
        print(f"  Successful executions: {stats.successful_executions}")
        print(f"  Total reward: {stats.total_reward:.3f}")
        print(f"  Average reward: {stats.average_reward:.3f}")
        print(f"  Episode duration: {stats.total_duration:.2f}s")
        
        progress = episode_stats['progress_metrics']
        print(f"  Final completion: {progress.completion_percentage:.2f}")
        print(f"  Final phase: {progress.current_phase}")
        print(f"  Milestones achieved: {len(progress.milestones_achieved)}")
        
        # Export episode data
        print("\nExporting episode data...")
        episode_data = env.export_episode_data('dict')
        
        print(f"  Code history entries: {len(episode_data['code_history'])}")
        print(f"  Execution history entries: {len(episode_data['execution_history'])}")
        print(f"  API call history entries: {len(episode_data['api_call_history'])}")
        print(f"  Reward history entries: {len(episode_data['reward_history'])}")
        print(f"  Checkpoints created: {len(episode_data['checkpoints'])}")
        
    finally:
        await env.close()
    
    print("\n" + "="*50 + "\n")


def demonstrate_data_export():
    """Demonstrate episode data export functionality"""
    print("=== Data Export Demo ===\n")
    
    state_manager = EnvironmentStateManager()
    
    # Create a rich episode with various events
    mock_task = type('MockTask', (), {'task_id': 'export_test', 'title': 'Data Export Test'})()
    state_manager.start_new_episode(mock_task, "import requests")
    
    # Simulate a complete episode
    for step in range(1, 6):
        state_manager.advance_step()
        
        # Code progression
        code = f"import requests\n\ndef api_call_{step}():\n    response = requests.get('https://api.example.com/endpoint{step}')\n    return response.json()"
        state_manager.record_code_change(code)
        
        # Execution
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS if step % 2 == 0 else ExecutionStatus.ERROR,
            stdout=f"Output from step {step}" if step % 2 == 0 else "",
            stderr=f"Error in step {step}" if step % 2 == 1 else "",
            execution_time=0.1 * step
        )
        state_manager.record_execution(result)
        
        # API calls
        state_manager.record_api_call({
            'method': 'GET',
            'url': f'https://api.example.com/endpoint{step}',
            'success': step % 2 == 0,
            'duration': 0.1 * step
        })
        
        # Rewards
        state_manager.record_reward(0.5 if step % 2 == 0 else -0.2, step * 1.2)
        
        # Checkpoint every 2 steps
        if step % 2 == 0:
            state_manager.create_checkpoint(f"step_{step}")
    
    # End episode
    state_manager.end_episode(TerminationReason.TASK_COMPLETED)
    
    print("Episode completed. Exporting data...")
    
    # Export as dictionary
    episode_dict = state_manager.export_episode_data('dict')
    print(f"Dictionary export keys: {list(episode_dict.keys())}")
    
    # Export as JSON
    episode_json = state_manager.export_episode_data('json')
    print(f"JSON export length: {len(episode_json)} characters")
    
    # Show sample data structure
    print(f"\nSample data structure:")
    print(f"  Episode metadata: {list(episode_dict['episode_metadata'].keys())}")
    print(f"  Statistics: {len(episode_dict['statistics'])} fields")
    print(f"  Code snapshots: {len(episode_dict['code_history'])}")
    print(f"  Execution records: {len(episode_dict['execution_history'])}")
    print(f"  API call records: {len(episode_dict['api_call_history'])}")
    
    # Save to file (simulated)
    filename = f"episode_{state_manager.episode_id}.json"
    print(f"\nData can be saved to: {filename}")
    print(f"File size would be: ~{len(episode_json) / 1024:.1f} KB")
    
    print("\n" + "="*50 + "\n")


async def main():
    """Run all state management demonstrations"""
    print("Comprehensive State Management Examples\n")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_state_manager_basics()
    
    demonstrate_execution_tracking()
    
    demonstrate_progress_tracking()
    
    demonstrate_termination_conditions()
    
    demonstrate_checkpoints()
    
    await demonstrate_environment_integration()
    
    demonstrate_data_export()
    
    print("\nAll state management demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(main())