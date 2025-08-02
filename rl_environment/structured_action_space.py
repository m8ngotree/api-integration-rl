"""
Structured action space for the API Integration Gymnasium environment.

This module provides a high-level interface between the Gymnasium environment
and the detailed code action vocabulary system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from rl_environment.code_actions import (
    CodeActionVocabulary, CodeActionEncoder, CodeActionApplicator,
    ActionValidator, CodeAction, ActionType, CodePosition, ActionContext
)


@dataclass
class ActionSpaceConfig:
    """Configuration for the structured action space"""
    max_actions_per_step: int = 3
    enable_multi_actions: bool = True
    enable_no_op: bool = True
    action_selection_method: str = "discrete"  # "discrete" or "continuous"
    position_encoding: str = "relative"  # "absolute" or "relative"


class StructuredActionSpace:
    """
    High-level structured action space that converts between Gymnasium actions
    and semantic code modifications.
    """
    
    def __init__(self, config: Optional[ActionSpaceConfig] = None):
        self.config = config or ActionSpaceConfig()
        
        # Initialize core components
        self.vocabulary = CodeActionVocabulary()
        self.encoder = CodeActionEncoder(self.vocabulary)
        self.applicator = CodeActionApplicator(self.vocabulary)
        self.validator = ActionValidator()
        
        # Create Gymnasium action space
        self.gym_action_space = self._create_gym_action_space()
        
        # Action history for context
        self.action_history: List[CodeAction] = []
    
    def _create_gym_action_space(self) -> spaces.Space:
        """Create the Gymnasium-compatible action space"""
        if self.config.action_selection_method == "discrete":
            return self._create_discrete_action_space()
        else:
            return self._create_continuous_action_space()
    
    def _create_discrete_action_space(self) -> spaces.Dict:
        """Create discrete action space with semantic operations"""
        action_components = {
            # Primary action selection
            'primary_action': spaces.Discrete(len(ActionType) + (1 if self.config.enable_no_op else 0)),
            
            # Template selection within action type
            'template_selection': spaces.Discrete(self.vocabulary.vocab_size),
            
            # Position specification
            'position_line': spaces.Discrete(1000),  # Support up to 1000 lines
            'position_column': spaces.Discrete(100),  # Support up to 100 columns
            'indent_level': spaces.Discrete(10),  # Support up to 10 indent levels
            
            # Parameter specification (simplified)
            'param_count': spaces.Discrete(8),  # Up to 8 parameters
            'param_types': spaces.MultiDiscrete([8] * 8),  # Parameter types
            'param_values': spaces.Box(low=0, high=1, shape=(8, 16), dtype=np.float32),  # Parameter values
            
            # Action modifiers
            'priority': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'execute_after': spaces.Discrete(2),  # Whether to execute code after this action
            
            # Multi-action support
            'action_count': spaces.Discrete(self.config.max_actions_per_step) if self.config.enable_multi_actions else spaces.Discrete(1),
        }
        
        if self.config.enable_multi_actions:
            # Replicate action components for multiple actions
            multi_action_space = {}
            for i in range(self.config.max_actions_per_step):
                for key, space in action_components.items():
                    if key != 'action_count':  # Don't replicate this
                        multi_action_space[f'action_{i}_{key}'] = space
            multi_action_space['action_count'] = action_components['action_count']
            return spaces.Dict(multi_action_space)
        else:
            return spaces.Dict(action_components)
    
    def _create_continuous_action_space(self) -> spaces.Box:
        """Create continuous action space (for advanced RL algorithms)"""
        # Continuous representation of all action components
        action_dim = (
            len(ActionType) +  # Action type (one-hot)
            1 +  # Template selection (continuous index)
            3 +  # Position (line, column, indent)
            64 +  # Parameter encoding
            2   # Priority and execute_after
        )
        
        return spaces.Box(
            low=-1.0, high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
    
    def decode_gym_action(self, gym_action: Union[Dict, np.ndarray], current_code: str) -> List[CodeAction]:
        """Convert Gymnasium action to semantic CodeAction objects"""
        if isinstance(gym_action, dict):
            return self._decode_discrete_action(gym_action, current_code)
        else:
            return self._decode_continuous_action(gym_action, current_code)
    
    def _decode_discrete_action(self, gym_action: Dict[str, Any], current_code: str) -> List[CodeAction]:
        """Decode discrete action representation"""
        actions = []
        
        if self.config.enable_multi_actions:
            action_count = int(gym_action['action_count'])
            action_count = min(action_count, self.config.max_actions_per_step)
            
            for i in range(action_count):
                action = self._decode_single_discrete_action(gym_action, current_code, i)
                if action:
                    actions.append(action)
        else:
            action = self._decode_single_discrete_action(gym_action, current_code, 0)
            if action:
                actions.append(action)
        
        return actions
    
    def _decode_single_discrete_action(self, gym_action: Dict[str, Any], current_code: str, action_idx: int = 0) -> Optional[CodeAction]:
        """Decode a single discrete action"""
        prefix = f'action_{action_idx}_' if self.config.enable_multi_actions else ''
        
        # Extract primary action
        primary_action_id = int(gym_action[f'{prefix}primary_action'])
        
        # Handle no-op
        if self.config.enable_no_op and primary_action_id >= len(ActionType):
            return None
        
        if primary_action_id >= len(ActionType):
            primary_action_id = 0  # Fallback
        
        action_type = list(ActionType)[primary_action_id]
        
        # Extract template
        template_id = int(gym_action[f'{prefix}template_selection'])
        template = self.vocabulary.id_to_action.get(template_id, "pass")
        
        # Extract position
        position = CodePosition(
            line=int(gym_action[f'{prefix}position_line']),
            column=int(gym_action[f'{prefix}position_column']),
            indent_level=int(gym_action[f'{prefix}indent_level'])
        )
        
        # Extract and decode parameters
        parameters = self._decode_parameters(
            param_count=int(gym_action[f'{prefix}param_count']),
            param_types=gym_action[f'{prefix}param_types'],
            param_values=gym_action[f'{prefix}param_values']
        )
        
        # Extract modifiers
        priority = float(gym_action[f'{prefix}priority'][0])
        
        return CodeAction(
            action_type=action_type,
            template=template,
            parameters=parameters,
            position=position,
            priority=priority
        )
    
    def _decode_continuous_action(self, gym_action: np.ndarray, current_code: str) -> List[CodeAction]:
        """Decode continuous action representation"""
        offset = 0
        
        # Decode action type (from one-hot or softmax)
        action_type_probs = gym_action[offset:offset + len(ActionType)]
        action_type_id = np.argmax(action_type_probs)
        action_type = list(ActionType)[action_type_id]
        offset += len(ActionType)
        
        # Decode template selection
        template_selection = gym_action[offset]
        template_id = int((template_selection + 1) / 2 * self.vocabulary.vocab_size)
        template_id = np.clip(template_id, 0, self.vocabulary.vocab_size - 1)
        template = self.vocabulary.id_to_action.get(template_id, "pass")
        offset += 1
        
        # Decode position
        position_encoding = gym_action[offset:offset + 3]
        position = CodePosition(
            line=int((position_encoding[0] + 1) / 2 * 1000),
            column=int((position_encoding[1] + 1) / 2 * 100),
            indent_level=int((position_encoding[2] + 1) / 2 * 10)
        )
        offset += 3
        
        # Decode parameters
        param_encoding = gym_action[offset:offset + 64]
        parameters = self._decode_continuous_parameters(param_encoding)
        offset += 64
        
        # Decode modifiers
        priority = (gym_action[offset] + 1) / 2  # Convert from [-1,1] to [0,1]
        
        action = CodeAction(
            action_type=action_type,
            template=template,
            parameters=parameters,
            position=position,
            priority=priority
        )
        
        return [action]
    
    def _decode_parameters(self, param_count: int, param_types: np.ndarray, param_values: np.ndarray) -> Dict[str, Any]:
        """Decode action parameters from discrete representation"""
        parameters = {}
        
        param_count = min(param_count, 8)  # Safety limit
        
        for i in range(param_count):
            param_type = int(param_types[i])
            param_value_encoding = param_values[i]
            
            param_name = f"param_{i}"
            
            if param_type == 0:  # String
                # Convert encoding to string
                str_length = int(param_value_encoding[0] * 50) + 1
                parameters[param_name] = f"value_{hash(tuple(param_value_encoding)) % 1000}"
            elif param_type == 1:  # Integer
                parameters[param_name] = int(param_value_encoding[0] * 1000)
            elif param_type == 2:  # Float
                parameters[param_name] = float(param_value_encoding[0] * 100)
            elif param_type == 3:  # Boolean
                parameters[param_name] = param_value_encoding[0] > 0.5
            elif param_type == 4:  # URL/Endpoint
                parameters[param_name] = f"https://api.example.com/endpoint_{int(param_value_encoding[0] * 10)}"
            elif param_type == 5:  # Variable name
                var_names = ["data", "response", "result", "item", "value", "config", "params", "headers"]
                var_idx = int(param_value_encoding[0] * len(var_names))
                parameters[param_name] = var_names[min(var_idx, len(var_names) - 1)]
            elif param_type == 6:  # HTTP method
                methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
                method_idx = int(param_value_encoding[0] * len(methods))
                parameters[param_name] = methods[min(method_idx, len(methods) - 1)]
        
        return parameters
    
    def _decode_continuous_parameters(self, param_encoding: np.ndarray) -> Dict[str, Any]:
        """Decode parameters from continuous representation"""
        parameters = {}
        
        # Simple continuous parameter decoding
        for i in range(0, min(len(param_encoding), 64), 8):
            param_slice = param_encoding[i:i+8]
            
            if np.sum(np.abs(param_slice)) > 0.1:  # Parameter exists
                param_name = f"param_{i // 8}"
                
                # Determine parameter type and value
                if param_slice[0] > 0.5:  # String parameter
                    parameters[param_name] = f"string_value_{int(param_slice[1] * 100)}"
                elif param_slice[1] > 0.5:  # Numeric parameter
                    parameters[param_name] = param_slice[2] * 1000
                elif param_slice[2] > 0.5:  # Boolean parameter
                    parameters[param_name] = param_slice[3] > 0.0
        
        return parameters
    
    def apply_actions(self, current_code: str, actions: List[CodeAction]) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply a list of actions to code and return result with metadata"""
        modified_code = current_code
        action_results = []
        
        # Sort actions by priority (higher priority first)
        sorted_actions = sorted(actions, key=lambda a: a.priority, reverse=True)
        
        for action in sorted_actions:
            # Create context
            context = ActionContext(
                current_code=modified_code,
                position=action.position or CodePosition(0)
            )
            
            # Validate action
            is_valid, validation_errors = self.validator.validate_action(action, context)
            
            action_result = {
                'action': action.to_dict(),
                'valid': is_valid,
                'validation_errors': validation_errors,
                'applied': False,
                'result_length': len(modified_code)
            }
            
            if is_valid:
                try:
                    # Apply action
                    new_code = self.applicator.apply_action(modified_code, action, context)
                    
                    # Validate result
                    result_valid, result_errors = self.validator.validate_result_code(new_code)
                    
                    if result_valid:
                        modified_code = new_code
                        action_result['applied'] = True
                        action_result['result_length'] = len(modified_code)
                    else:
                        action_result['validation_errors'].extend(result_errors)
                
                except Exception as e:
                    action_result['validation_errors'].append(f"Application error: {str(e)}")
            
            action_results.append(action_result)
        
        # Update action history
        self.action_history.extend([a for a in sorted_actions if any(r['applied'] for r in action_results)])
        
        # Limit history size
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        return modified_code, action_results
    
    def get_action_suggestions(self, current_code: str, task_context: Optional[Dict[str, Any]] = None) -> List[CodeAction]:
        """Get suggested actions based on current code state"""
        suggestions = []
        
        lines = current_code.split('\n')
        
        # Analyze current code state
        has_imports = any(line.strip().startswith(('import ', 'from ')) for line in lines)
        has_main_logic = any('requests.' in line or 'http' in line.lower() for line in lines)
        has_error_handling = any('try:' in line or 'except' in line for line in lines)
        has_functions = any(line.strip().startswith('def ') for line in lines)
        
        # Suggest missing components
        if not has_imports:
            suggestions.append(CodeAction(
                action_type=ActionType.ADD_IMPORT,
                template="import requests",
                parameters={},
                position=CodePosition(0),
                priority=0.9
            ))
        
        if not has_main_logic and has_imports:
            suggestions.append(CodeAction(
                action_type=ActionType.ADD_FUNCTION_CALL,
                template="response = requests.get(url)",
                parameters={"url": "https://api.example.com/data"},
                position=CodePosition(len(lines)),
                priority=0.8
            ))
        
        if has_main_logic and not has_error_handling:
            suggestions.append(CodeAction(
                action_type=ActionType.ADD_ERROR_HANDLING,
                template="try:\n    {code}\nexcept requests.RequestException as e:\n    print(f'Request failed: {e}')",
                parameters={"code": "# API call code"},
                position=CodePosition(len(lines)),
                priority=0.7
            ))
        
        if not has_functions and len(lines) > 5:
            suggestions.append(CodeAction(
                action_type=ActionType.ADD_FUNCTION_DEF,
                template="def make_api_request(url, method='GET'):",
                parameters={},
                position=CodePosition(0),
                priority=0.6
            ))
        
        return suggestions
    
    def get_action_space(self) -> spaces.Space:
        """Get the Gymnasium action space"""
        return self.gym_action_space
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """Get information about the action vocabulary"""
        return {
            'vocab_size': self.vocabulary.vocab_size,
            'action_types': [t.value for t in ActionType],
            'total_templates': len(self.vocabulary.action_vocab),
            'import_templates': len(sum(self.vocabulary.import_templates.values(), [])),
            'function_templates': len(sum(self.vocabulary.function_call_templates.values(), [])),
            'error_handling_templates': len(sum(self.vocabulary.error_handling_templates.values(), [])),
        }
    
    def reset(self):
        """Reset the action space state"""
        self.action_history.clear()


# Convenience function for creating action spaces
def create_structured_action_space(
    max_actions_per_step: int = 1,
    enable_multi_actions: bool = False,
    action_selection_method: str = "discrete"
) -> StructuredActionSpace:
    """Create a structured action space with specified configuration"""
    config = ActionSpaceConfig(
        max_actions_per_step=max_actions_per_step,
        enable_multi_actions=enable_multi_actions,
        action_selection_method=action_selection_method
    )
    return StructuredActionSpace(config)