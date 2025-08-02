"""
Examples demonstrating the new structured action space for code modifications.

This script shows how to use the comprehensive action vocabulary and 
encoding/decoding system for API integration tasks.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any

from rl_environment.code_actions import (
    CodeActionVocabulary, CodeActionEncoder, CodeActionApplicator,
    ActionValidator, CodeAction, ActionType, CodePosition, ActionContext
)
from rl_environment.structured_action_space import StructuredActionSpace, ActionSpaceConfig
from rl_environment.gym_environment import APIIntegrationEnv, EnvironmentConfig
from rl_environment.task_generator import TaskDifficulty


def demonstrate_action_vocabulary():
    """Demonstrate the code action vocabulary"""
    print("=== Code Action Vocabulary Demo ===\n")
    
    vocab = CodeActionVocabulary()
    
    print(f"Total vocabulary size: {vocab.vocab_size}")
    print(f"Action types available: {len(list(ActionType))}")
    print()
    
    print("Import templates:")
    for category, templates in vocab.import_templates.items():
        print(f"  {category}: {len(templates)} templates")
        for template in templates[:3]:  # Show first 3
            print(f"    - {template}")
        if len(templates) > 3:
            print(f"    ... and {len(templates) - 3} more")
        print()
    
    print("Function call templates:")
    for category, templates in vocab.function_call_templates.items():
        print(f"  {category}: {len(templates)} templates")
        for template in templates[:2]:
            print(f"    - {template}")
        if len(templates) > 2:
            print(f"    ... and {len(templates) - 2} more")
        print()


def demonstrate_action_creation_and_encoding():
    """Demonstrate creating and encoding actions"""
    print("=== Action Creation and Encoding Demo ===\n")
    
    vocab = CodeActionVocabulary()
    encoder = CodeActionEncoder(vocab)
    
    # Create sample actions
    actions = [
        CodeAction(
            action_type=ActionType.ADD_IMPORT,
            template="import requests",
            parameters={},
            position=CodePosition(0, 0, 0),
            priority=0.9
        ),
        CodeAction(
            action_type=ActionType.ADD_FUNCTION_CALL,
            template="response = requests.get(url)",
            parameters={"url": "https://api.example.com/data"},
            position=CodePosition(5, 0, 0),
            priority=0.8
        ),
        CodeAction(
            action_type=ActionType.ADD_ERROR_HANDLING,
            template="try:\n    {code}\nexcept requests.RequestException as e:\n    print(f'Request failed: {e}')",
            parameters={"code": "response = requests.get(url)"},
            position=CodePosition(10, 0, 0),
            priority=0.7
        )
    ]
    
    for i, action in enumerate(actions):
        print(f"Action {i + 1}: {action.action_type.value}")
        print(f"  Template: {action.template}")
        print(f"  Parameters: {action.parameters}")
        print(f"  Position: Line {action.position.line}, Column {action.position.column}")
        print(f"  Priority: {action.priority}")
        
        # Encode action
        encoding = encoder.encode_action(action)
        print("  Encoded representation:")
        for key, value in encoding.items():
            if isinstance(value, np.ndarray):
                if value.size <= 5:
                    print(f"    {key}: {value}")
                else:
                    print(f"    {key}: shape={value.shape}, sample={value[:3]}")
            else:
                print(f"    {key}: {value}")
        
        # Decode back
        decoded_action = encoder.decode_action_encoding(encoding)
        print(f"  Decoded action type: {decoded_action.action_type.value}")
        print(f"  Decoded template: {decoded_action.template}")
        print()


async def demonstrate_action_application():
    """Demonstrate applying actions to code"""
    print("=== Action Application Demo ===\n")
    
    vocab = CodeActionVocabulary()
    applicator = CodeActionApplicator(vocab)
    validator = ActionValidator()
    
    # Starting code
    initial_code = '''# API Integration Script
# TODO: Add imports
# TODO: Add main logic
'''
    
    print("Initial code:")
    print(initial_code)
    print()
    
    # Define sequence of actions
    actions = [
        CodeAction(
            action_type=ActionType.ADD_IMPORT,
            template="import requests",
            position=CodePosition(1, 0, 0)
        ),
        CodeAction(
            action_type=ActionType.ADD_IMPORT,
            template="import json",
            position=CodePosition(2, 0, 0)
        ),
        CodeAction(
            action_type=ActionType.ADD_VARIABLE,
            template="url = '{endpoint}'",
            parameters={"endpoint": "https://api.example.com/users"},
            position=CodePosition(4, 0, 0)
        ),
        CodeAction(
            action_type=ActionType.ADD_FUNCTION_CALL,
            template="response = requests.get(url)",
            position=CodePosition(5, 0, 0)
        ),
        CodeAction(
            action_type=ActionType.ADD_CONDITIONAL,
            template="if response.status_code == 200:",
            position=CodePosition(6, 0, 0)
        ),
        CodeAction(
            action_type=ActionType.ADD_FUNCTION_CALL,
            template="    data = response.json()",
            position=CodePosition(7, 0, 1)
        ),
        CodeAction(
            action_type=ActionType.ADD_PRINT,
            template="    print(data)",
            position=CodePosition(8, 0, 1)
        )
    ]
    
    current_code = initial_code
    
    for i, action in enumerate(actions):
        print(f"Step {i + 1}: Applying {action.action_type.value}")
        
        # Create context
        context = ActionContext(current_code, action.position or CodePosition(0))
        
        # Validate action
        is_valid, errors = validator.validate_action(action, context)
        print(f"  Valid: {is_valid}")
        if errors:
            print(f"  Validation errors: {errors}")
        
        if is_valid:
            # Apply action
            try:
                new_code = applicator.apply_action(current_code, action, context)
                
                # Validate result
                result_valid, result_errors = validator.validate_result_code(new_code)
                
                if result_valid:
                    current_code = new_code
                    print("  ✅ Action applied successfully")
                else:
                    print(f"  ❌ Result validation failed: {result_errors}")
            
            except Exception as e:
                print(f"  ❌ Application failed: {e}")
        
        print()
    
    print("Final code:")
    print(current_code)
    print()


async def demonstrate_structured_action_space():
    """Demonstrate the structured action space"""
    print("=== Structured Action Space Demo ===\n")
    
    # Create action space with different configurations
    configs = [
        ("Single Action", ActionSpaceConfig(max_actions_per_step=1, enable_multi_actions=False)),
        ("Multi Action", ActionSpaceConfig(max_actions_per_step=3, enable_multi_actions=True)),
        ("Continuous", ActionSpaceConfig(action_selection_method="continuous"))
    ]
    
    for config_name, config in configs:
        print(f"--- {config_name} Configuration ---")
        
        action_space = StructuredActionSpace(config)
        gym_space = action_space.get_action_space()
        
        print(f"Action space type: {type(gym_space).__name__}")
        
        if hasattr(gym_space, 'spaces'):  # Dict space
            print("Action components:")
            for key, space in gym_space.spaces.items():
                print(f"  {key}: {space}")
        else:  # Box space
            print(f"Action shape: {gym_space.shape}")
            print(f"Action range: [{gym_space.low[0]:.2f}, {gym_space.high[0]:.2f}]")
        
        print(f"Vocabulary info: {action_space.get_vocabulary_info()}")
        print()


async def demonstrate_environment_integration():
    """Demonstrate using the action system in the environment"""
    print("=== Environment Integration Demo ===\n")
    
    # Create environment with different action configurations
    config = EnvironmentConfig(
        max_episode_steps=10,
        task_difficulty=TaskDifficulty.BEGINNER,
        max_actions_per_step=2,
        enable_multi_actions=True
    )
    
    env = APIIntegrationEnv(config)
    
    try:
        # Reset environment
        observation, info = await env.reset(seed=42)
        
        print(f"Task: {info['task_title']}")
        print(f"Action space: {type(env.action_space).__name__}")
        print(f"Action space components: {list(env.action_space.spaces.keys()) if hasattr(env.action_space, 'spaces') else 'Box space'}")
        print()
        
        # Sample and apply some actions
        for step in range(3):
            print(f"--- Step {step + 1} ---")
            
            # Sample action from action space
            action = env.action_space.sample()
            print(f"Sampled action keys: {list(action.keys()) if isinstance(action, dict) else 'Array action'}")
            
            # Take step
            observation, reward, terminated, truncated, info = await env.step(action)
            
            print(f"Actions applied: {info.get('actions_applied', 0)}/{info.get('actions_total', 0)}")
            print(f"Reward: {reward:.3f}")
            print(f"Code length: {info['code_length']}")
            print(f"Has imports: {observation['has_imports']}")
            print(f"Has functions: {observation['has_functions']}")
            print(f"Has API calls: {observation['has_api_calls']}")
            print(f"Completion progress: {observation['completion_progress'][0]:.3f}")
            
            if terminated:
                print("✅ Task completed!")
                break
            elif truncated:
                print("⏰ Episode truncated")
                break
            
            print()
        
        print("Episode finished.")
        
    finally:
        await env.close()


async def demonstrate_action_suggestions():
    """Demonstrate action suggestions based on code state"""
    print("=== Action Suggestions Demo ===\n")
    
    action_space = StructuredActionSpace()
    
    # Test different code states
    code_states = [
        ("Empty code", ""),
        ("Only comments", "# This is a comment\n# TODO: Add code"),
        ("With imports", "import requests\nimport json\n"),
        ("With imports and logic", "import requests\n\nurl = 'https://api.example.com'\nresponse = requests.get(url)"),
        ("Complete basic structure", """import requests
import json

def make_request():
    url = 'https://api.example.com'
    response = requests.get(url)
    return response.json()

result = make_request()
print(result)""")
    ]
    
    for state_name, code in code_states:
        print(f"--- {state_name} ---")
        print("Code:")
        if code:
            for i, line in enumerate(code.split('\n'), 1):
                print(f"  {i}: {line}")
        else:
            print("  (empty)")
        print()
        
        # Get suggestions
        suggestions = action_space.get_action_suggestions(code)
        
        print(f"Suggestions ({len(suggestions)}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion.action_type.value}")
            print(f"     Template: {suggestion.template}")
            print(f"     Priority: {suggestion.priority:.2f}")
            if suggestion.parameters:
                print(f"     Parameters: {suggestion.parameters}")
        
        if not suggestions:
            print("  No suggestions (code may be complete)")
        
        print()


async def main():
    """Run all demonstrations"""
    print("Structured Action Space for API Integration - Examples\n")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_action_vocabulary()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_action_creation_and_encoding()
    print("\n" + "=" * 60 + "\n")
    
    await demonstrate_action_application()
    print("\n" + "=" * 60 + "\n")
    
    await demonstrate_structured_action_space()
    print("\n" + "=" * 60 + "\n")
    
    await demonstrate_environment_integration()
    print("\n" + "=" * 60 + "\n")
    
    await demonstrate_action_suggestions()


if __name__ == "__main__":
    asyncio.run(main())