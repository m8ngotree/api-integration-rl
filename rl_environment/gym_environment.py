import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import json
import hashlib

from rl_environment.task_generator import LearningTask, TaskDifficulty
from rl_environment.execution_environment import SafeExecutionEnvironment, ExecutionEnvironmentConfig, EnvironmentType
from rl_environment.reward_system import RewardSystem, RewardConfig
from utilities.code_template_generator import CodeTemplateGenerator
from rl_environment.structured_action_space import StructuredActionSpace, ActionSpaceConfig
from rl_environment.code_actions import CodeAction


@dataclass
class EnvironmentConfig:
    """Configuration for the API integration environment"""
    max_code_length: int = 10000
    max_observation_tokens: int = 8000
    max_episode_steps: int = 50
    task_difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE
    execution_timeout: int = 30
    enable_hints: bool = True
    max_actions_per_step: int = 1
    enable_multi_actions: bool = False
    action_selection_method: str = "discrete"


class APIIntegrationEnv(gym.Env):
    """
    Gymnasium environment for API integration learning tasks.
    
    Observation Space:
        - API documentation (tokenized)
        - Current code state (tokenized)
        - Previous execution results
        - Task progress indicators
    
    Action Space:
        - Code modifications (tokenized operations)
        - Insert, replace, delete operations on code
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        
        self.config = config or EnvironmentConfig()
        
        # Initialize components
        self.execution_env = None
        self.reward_system = RewardSystem(RewardConfig())
        self.template_generator = CodeTemplateGenerator()
        
        # Initialize structured action space
        action_config = ActionSpaceConfig(
            max_actions_per_step=self.config.max_actions_per_step,
            enable_multi_actions=self.config.enable_multi_actions,
            action_selection_method=self.config.action_selection_method
        )
        self.structured_action_space = StructuredActionSpace(action_config)
        
        # Environment state
        self.current_task: Optional[LearningTask] = None
        self.current_code: str = ""
        self.step_count: int = 0
        self.episode_history: List[Dict[str, Any]] = []
        self.last_action_results: List[Dict[str, Any]] = []
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space (use structured action space)
        self.action_space = self.structured_action_space.get_action_space()
        
        # Initialize execution environment
        self._setup_execution_environment()
    
    def _create_observation_space(self) -> spaces.Dict:
        """Create the observation space for the environment"""
        return spaces.Dict({
            # API documentation as text tokens (simplified as Box for now)
            'api_docs': spaces.Box(
                low=0, high=1, 
                shape=(self.config.max_observation_tokens,), 
                dtype=np.float32
            ),
            
            # Current code state as text tokens
            'current_code': spaces.Box(
                low=0, high=1, 
                shape=(self.config.max_code_length,), 
                dtype=np.float32
            ),
            
            # Task metadata
            'task_difficulty': spaces.Discrete(4),  # BEGINNER, INTERMEDIATE, ADVANCED, EXPERT
            'step_count': spaces.Box(low=0, high=self.config.max_episode_steps, shape=(1,), dtype=np.int32),
            
            # Execution feedback
            'last_execution_success': spaces.Discrete(2),  # 0 = failed, 1 = success
            'syntax_errors': spaces.Discrete(2),  # 0 = no errors, 1 = has errors
            
            # Action feedback
            'last_action_applied': spaces.Discrete(2),  # 0 = not applied, 1 = applied
            'action_validation_errors': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            
            # Code analysis
            'has_imports': spaces.Discrete(2),
            'has_functions': spaces.Discrete(2),
            'has_error_handling': spaces.Discrete(2),
            'has_api_calls': spaces.Discrete(2),
            
            # Progress indicators
            'completion_progress': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_score': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        })
    
    
    def _setup_execution_environment(self):
        """Setup the code execution environment"""
        exec_config = ExecutionEnvironmentConfig(
            environment_type=EnvironmentType.SUBPROCESS_WITH_SERVER,
            timeout=self.config.execution_timeout,
            max_memory_mb=128,
            enable_networking=True
        )
        self.execution_env = SafeExecutionEnvironment(exec_config)
    
    async def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment with a new task"""
        super().reset(seed=seed)
        
        # Generate new task
        from rl_environment.task_generator import BasicGetRequestTaskGenerator
        task_generator = BasicGetRequestTaskGenerator(seed=seed)
        self.current_task = task_generator.generate_task(
            difficulty=self.config.task_difficulty
        )
        
        # Initialize code with starter template
        self.current_code = self.current_task.starter_code
        self.step_count = 0
        self.episode_history = []
        
        # Setup execution environment for new task
        if not self.execution_env:
            self._setup_execution_environment()
        await self.execution_env.initialize()
        
        # Reset structured action space
        self.structured_action_space.reset()
        self.last_action_results = []
        
        # Create initial observation
        observation = self._create_observation()
        info = self._create_info()
        
        return observation, info
    
    def _create_observation(self) -> Dict[str, Any]:
        """Create current observation from environment state"""
        if not self.current_task:
            raise RuntimeError("No current task - call reset() first")
        
        return {
            'api_docs': self._tokenize_api_docs(self.current_task.api_documentation),
            'current_code': self._tokenize_code(self.current_code),
            'task_difficulty': self._difficulty_to_int(self.current_task.difficulty),
            'step_count': np.array([self.step_count], dtype=np.int32),
            'last_execution_success': self._get_last_execution_success(),
            'syntax_errors': self._check_syntax_errors(),
            'last_action_applied': self._get_last_action_applied(),
            'action_validation_errors': self._get_action_validation_errors(),
            'has_imports': self._check_has_imports(),
            'has_functions': self._check_has_functions(),
            'has_error_handling': self._check_has_error_handling(),
            'has_api_calls': self._check_has_api_calls(),
            'completion_progress': np.array([self._calculate_progress()], dtype=np.float32),
            'current_score': np.array([self._get_current_score()], dtype=np.float32),
        }
    
    def _create_info(self) -> Dict[str, Any]:
        """Create info dictionary with additional environment details"""
        return {
            'task_id': self.current_task.task_id if self.current_task else None,
            'task_title': self.current_task.title if self.current_task else None,
            'step_count': self.step_count,
            'code_length': len(self.current_code),
            'hints_available': len(self.current_task.hints) if self.current_task else 0,
        }
    
    async def step(self, action: Union[Dict[str, Any], np.ndarray]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if not self.current_task:
            raise RuntimeError("No current task - call reset() first")
        
        self.step_count += 1
        
        # Decode gym action to semantic code actions
        code_actions = self.structured_action_space.decode_gym_action(action, self.current_code)
        
        # Apply code actions
        new_code, action_results = self.structured_action_space.apply_actions(self.current_code, code_actions)
        self.current_code = new_code
        self.last_action_results = action_results
        
        # Execute code if any action requested execution
        execution_result = None
        should_execute = any(
            result.get('action', {}).get('parameters', {}).get('execute_after', False) 
            for result in action_results
        ) or (len(code_actions) > 0 and self.step_count % 3 == 0)  # Execute periodically
        
        if should_execute:
            execution_result = await self._execute_current_code()
        
        # Calculate reward
        reward = self._calculate_reward_from_actions(code_actions, action_results, execution_result)
        
        # Check if episode is done
        terminated = self._check_task_completion()
        truncated = self.step_count >= self.config.max_episode_steps
        
        # Update episode history
        self.episode_history.append({
            'step': self.step_count,
            'actions': [action.to_dict() for action in code_actions],
            'action_results': action_results,
            'reward': reward,
            'execution_result': execution_result,
            'code_length': len(self.current_code)
        })
        
        # Create new observation
        observation = self._create_observation()
        info = self._create_info()
        
        # Add action and execution info
        info['actions_applied'] = sum(1 for result in action_results if result['applied'])
        info['actions_total'] = len(action_results)
        if execution_result:
            info['execution_status'] = execution_result.status.value
            info['execution_time'] = execution_result.execution_time
        
        return observation, reward, terminated, truncated, info
    
    def _apply_code_modification(self, action: Dict[str, Any]):
        """Apply code modification based on action"""
        operation_type = action['operation_type']
        line_position = int(action['line_position'][0])
        code_content = self._detokenize_code(action['code_content'])
        
        code_lines = self.current_code.split('\n')
        
        # Ensure line position is valid
        line_position = max(0, min(line_position, len(code_lines)))
        
        if operation_type == 0:  # Insert
            code_lines.insert(line_position, code_content)
        elif operation_type == 1:  # Replace
            if line_position < len(code_lines):
                code_lines[line_position] = code_content
        elif operation_type == 2:  # Delete
            if line_position < len(code_lines):
                del code_lines[line_position]
        elif operation_type == 3:  # Append
            code_lines.append(code_content)
        
        self.current_code = '\n'.join(code_lines)
        
        # Limit code length
        if len(self.current_code) > self.config.max_code_length:
            self.current_code = self.current_code[:self.config.max_code_length]
    
    async def _execute_current_code(self):
        """Execute the current code and return results"""
        try:
            result = await self.execution_env.execute_code(self.current_code)
            return result
        except Exception as e:
            # Return error result
            from rl_environment.code_executor import ExecutionResult, ExecutionStatus
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Execution error: {str(e)}"
            )
    
    def _calculate_reward(self, action: Dict[str, Any], execution_result) -> float:
        """Calculate reward for the current step"""
        if not self.current_task:
            return 0.0
        
        reward = 0.0
        
        # Basic step penalty to encourage efficiency
        reward -= 0.1
        
        # Reward for successful execution
        if execution_result:
            if execution_result.status.value == 'success':
                reward += 1.0
            elif execution_result.status.value == 'error':
                reward -= 0.5
        
        # Reward for progress towards completion
        progress = self._calculate_progress()
        if hasattr(self, '_last_progress'):
            progress_delta = progress - self._last_progress
            reward += progress_delta * 5.0
        self._last_progress = progress
        
        # Use reward system for detailed evaluation
        try:
            detailed_reward = self.reward_system.evaluate_code_completion(
                task=self.current_task,
                generated_code=self.current_code,
                execution_result=execution_result
            )
            reward += detailed_reward.total_score * 0.1
        except Exception:
            pass  # Fallback to basic reward if detailed evaluation fails
        
        return float(reward)
    
    def _calculate_reward_from_actions(self, actions: List[CodeAction], action_results: List[Dict[str, Any]], execution_result) -> float:
        """Calculate reward based on applied actions and their results"""
        reward = 0.0
        
        # Base penalty for taking a step
        reward -= 0.05
        
        # Reward for successfully applied actions
        applied_actions = sum(1 for result in action_results if result['applied'])
        failed_actions = len(action_results) - applied_actions
        
        reward += applied_actions * 0.2
        reward -= failed_actions * 0.1
        
        # Reward for execution success
        if execution_result:
            if execution_result.status.value == 'success':
                reward += 1.0
            elif execution_result.status.value == 'error':
                reward -= 0.3
        
        # Reward for progress
        progress = self._calculate_progress()
        if hasattr(self, '_last_progress'):
            progress_delta = progress - self._last_progress
            reward += progress_delta * 3.0
        self._last_progress = progress
        
        # Bonus for completing specific milestones
        if self._check_has_imports() and not hasattr(self, '_imports_bonus'):
            reward += 0.5
            self._imports_bonus = True
        
        if self._check_has_api_calls() and not hasattr(self, '_api_calls_bonus'):
            reward += 1.0
            self._api_calls_bonus = True
        
        if self._check_has_error_handling() and not hasattr(self, '_error_handling_bonus'):
            reward += 0.8
            self._error_handling_bonus = True
        
        # Use detailed reward system
        try:
            detailed_reward = self.reward_system.evaluate_code_completion(
                task=self.current_task,
                generated_code=self.current_code,
                execution_result=execution_result
            )
            reward += detailed_reward.total_score * 0.1
        except Exception:
            pass
        
        return float(reward)
    
    def _check_task_completion(self) -> bool:
        """Check if the current task is completed successfully"""
        if not self.current_task:
            return False
        
        try:
            # Use reward system to check completion
            detailed_reward = self.reward_system.evaluate_code_completion(
                task=self.current_task,
                generated_code=self.current_code,
                execution_result=None  # Will trigger execution internally
            )
            return detailed_reward.total_score >= self.current_task.success_criteria.minimum_score
        except Exception:
            return False
    
    def _calculate_progress(self) -> float:
        """Calculate current progress towards task completion (0.0 to 1.0)"""
        if not self.current_task:
            return 0.0
        
        progress = 0.0
        
        # Basic progress based on code length vs expected
        if self.current_code.strip():
            progress += 0.2
        
        # Check for required imports/patterns
        required_patterns = ['import requests', 'def ', 'http']
        for pattern in required_patterns:
            if pattern in self.current_code:
                progress += 0.1
        
        # Syntax check
        try:
            compile(self.current_code, '<string>', 'exec')
            progress += 0.3
        except SyntaxError:
            pass
        
        return min(progress, 1.0)
    
    def _get_current_score(self) -> float:
        """Get current score for the task"""
        if not self.current_task:
            return 0.0
        
        try:
            detailed_reward = self.reward_system.evaluate_code_completion(
                task=self.current_task,
                generated_code=self.current_code,
                execution_result=None
            )
            return detailed_reward.total_score
        except Exception:
            return 0.0
    
    # Helper methods for tokenization (simplified implementations)
    def _tokenize_api_docs(self, api_docs) -> np.ndarray:
        """Convert API documentation to tokenized representation"""
        # Simplified: convert to hash-based representation
        docs_text = json.dumps(api_docs.__dict__ if hasattr(api_docs, '__dict__') else str(api_docs))
        hash_val = int(hashlib.md5(docs_text.encode()).hexdigest(), 16)
        
        # Create a simple feature vector
        features = np.zeros(self.config.max_observation_tokens, dtype=np.float32)
        features[hash_val % self.config.max_observation_tokens] = 1.0
        features[:10] = np.random.random(10) * 0.1  # Add some variation
        
        return features
    
    def _tokenize_code(self, code: str) -> np.ndarray:
        """Convert code to tokenized representation"""
        # Simplified: create feature vector based on code characteristics
        features = np.zeros(self.config.max_code_length, dtype=np.float32)
        
        if code:
            # Basic features
            features[0] = min(len(code) / 1000.0, 1.0)  # Normalized length
            features[1] = code.count('\n') / 100.0  # Line count
            features[2] = code.count('def ') / 10.0  # Function count
            features[3] = code.count('import ') / 10.0  # Import count
            features[4] = code.count('requests.') / 10.0  # API call count
        
        return features
    
    def _detokenize_code(self, tokens: np.ndarray) -> str:
        """Convert tokenized representation back to code (simplified)"""
        # For now, return common code patterns based on token values
        if tokens[0] > 0.5:
            return "import requests"
        elif tokens[1] > 0.5:
            return "response = requests.get(url)"
        elif tokens[2] > 0.5:
            return "print(response.json())"
        else:
            return ""
    
    def _difficulty_to_int(self, difficulty: TaskDifficulty) -> int:
        """Convert difficulty enum to integer"""
        mapping = {
            TaskDifficulty.BEGINNER: 0,
            TaskDifficulty.INTERMEDIATE: 1,
            TaskDifficulty.ADVANCED: 2,
            TaskDifficulty.EXPERT: 3
        }
        return mapping.get(difficulty, 1)
    
    def _get_last_execution_success(self) -> int:
        """Get result of last execution (0 = failed, 1 = success)"""
        if not self.episode_history:
            return 0
        
        last_step = self.episode_history[-1]
        if 'execution_result' in last_step and last_step['execution_result']:
            return 1 if last_step['execution_result'].status.value == 'success' else 0
        return 0
    
    def _check_syntax_errors(self) -> int:
        """Check if current code has syntax errors (0 = no errors, 1 = has errors)"""
        try:
            compile(self.current_code, '<string>', 'exec')
            return 0
        except SyntaxError:
            return 1
    
    def _get_last_action_applied(self) -> int:
        """Get whether last action was successfully applied"""
        if not self.last_action_results:
            return 0
        return 1 if any(result['applied'] for result in self.last_action_results) else 0
    
    def _get_action_validation_errors(self) -> np.ndarray:
        """Get validation errors from last actions as feature vector"""
        errors = np.zeros(10, dtype=np.float32)
        if self.last_action_results:
            for i, result in enumerate(self.last_action_results[:10]):
                if result['validation_errors']:
                    errors[i] = 1.0
        return errors
    
    def _check_has_imports(self) -> int:
        """Check if code has import statements"""
        lines = self.current_code.split('\n')
        return 1 if any(line.strip().startswith(('import ', 'from ')) for line in lines) else 0
    
    def _check_has_functions(self) -> int:
        """Check if code has function definitions"""
        return 1 if 'def ' in self.current_code else 0
    
    def _check_has_error_handling(self) -> int:
        """Check if code has error handling"""
        return 1 if any(keyword in self.current_code for keyword in ['try:', 'except', 'raise', 'assert']) else 0
    
    def _check_has_api_calls(self) -> int:
        """Check if code has API calls"""
        api_patterns = ['requests.', 'httpx.', 'urllib.', 'http', '.get(', '.post(', '.put(', '.delete(']
        return 1 if any(pattern in self.current_code for pattern in api_patterns) else 0
    
    async def close(self):
        """Clean up environment resources"""
        if self.execution_env:
            await self.execution_env.cleanup()
    
    def render(self, mode='human'):
        """Render the environment state (optional)"""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Task: {self.current_task.title if self.current_task else 'None'}")
            print(f"Code length: {len(self.current_code)}")
            print(f"Progress: {self._calculate_progress():.2f}")
            print(f"Score: {self._get_current_score():.2f}")
            print("---")