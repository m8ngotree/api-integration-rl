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
from rl_environment.observation_processor import ObservationEncoder
from rl_environment.code_embeddings import CodeEmbeddingGenerator, CodeSemanticAnalyzer
from rl_environment.state_management import EnvironmentStateManager, EpisodeState, TerminationReason


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
    # Observation processing config
    max_code_tokens: int = 512
    max_api_tokens: int = 1024
    embedding_dim: int = 128
    enable_embeddings: bool = True
    enable_semantic_analysis: bool = True
    # State management config
    max_history_length: int = 1000
    auto_checkpoint_interval: int = 10
    completion_score_threshold: float = 8.0
    no_progress_threshold: int = 20
    episode_timeout: int = 3600  # 1 hour
    max_syntax_errors: int = 10
    max_execution_errors: int = 20


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
        
        # Initialize observation processing
        self.observation_encoder = ObservationEncoder(
            max_code_tokens=self.config.max_code_tokens,
            max_api_tokens=self.config.max_api_tokens,
            embedding_dim=self.config.embedding_dim
        )
        
        # Initialize embedding system
        if self.config.enable_embeddings:
            self.embedding_generator = CodeEmbeddingGenerator(self.config.embedding_dim)
        else:
            self.embedding_generator = None
        
        # Initialize semantic analyzer
        if self.config.enable_semantic_analysis:
            self.semantic_analyzer = CodeSemanticAnalyzer()
        else:
            self.semantic_analyzer = None
        
        # Initialize structured action space
        action_config = ActionSpaceConfig(
            max_actions_per_step=self.config.max_actions_per_step,
            enable_multi_actions=self.config.enable_multi_actions,
            action_selection_method=self.config.action_selection_method
        )
        self.structured_action_space = StructuredActionSpace(action_config)
        
        # Initialize state management
        self.state_manager = EnvironmentStateManager(
            max_history_length=self.config.max_history_length
        )
        
        # Configure state manager
        state_config = {
            'max_steps': self.config.max_episode_steps,
            'max_syntax_errors': self.config.max_syntax_errors,
            'max_execution_errors': self.config.max_execution_errors,
            'episode_timeout': self.config.episode_timeout,
            'no_progress_threshold': self.config.no_progress_threshold,
            'completion_score_threshold': self.config.completion_score_threshold,
            'auto_save_enabled': True
        }
        self.state_manager.config.update(state_config)
        self.state_manager.auto_checkpoint_interval = self.config.auto_checkpoint_interval
        
        # Environment state (delegated to state manager)
        self.current_task: Optional[LearningTask] = None
        self.last_action_results: List[Dict[str, Any]] = []
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space (use structured action space)
        self.action_space = self.structured_action_space.get_action_space()
        
        # Initialize execution environment
        self._setup_execution_environment()
    
    def _create_observation_space(self) -> spaces.Dict:
        """Create the observation space for the environment"""
        obs_space = {
            # Processed observations from encoder
            'api_schema': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.config.max_api_tokens,),
                dtype=np.float32
            ),
            'code_tokens': spaces.Box(
                low=0, high=self.observation_encoder.vocab_size,
                shape=(self.config.max_code_tokens,),
                dtype=np.int32
            ),
            'code_structure': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(128,),  # Fixed size from observation processor
                dtype=np.float32
            ),
            'code_features': spaces.Box(
                low=0, high=1,
                shape=(64,),  # Fixed size from observation processor
                dtype=np.float32
            ),
            'alignment_features': spaces.Box(
                low=0, high=1,
                shape=(32,),  # Fixed size from observation processor
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
            
            # Progress indicators
            'completion_progress': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_score': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        }
        
        # Add embeddings if enabled  
        if self.config.enable_embeddings:
            obs_space.update({
                'code_embedding': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.config.embedding_dim,),
                    dtype=np.float32
                ),
                'api_embedding': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.config.embedding_dim,),
                    dtype=np.float32
                ),
                'alignment_embedding': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.config.embedding_dim,),
                    dtype=np.float32
                )
            })
        
        # Add semantic features if enabled
        if self.config.enable_semantic_analysis:
            obs_space.update({
                'semantic_patterns': spaces.Box(
                    low=0, high=1,
                    shape=(20,),  # Max 20 pattern types
                    dtype=np.float32
                )
            })
        
        return spaces.Dict(obs_space)
    
    
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
        
        # End previous episode if it exists
        if self.state_manager.episode_state == EpisodeState.ACTIVE:
            self.state_manager.end_episode(TerminationReason.MANUAL_STOP)
        
        # Generate new task
        from rl_environment.task_generator import BasicGetRequestTaskGenerator
        task_generator = BasicGetRequestTaskGenerator(seed=seed)
        self.current_task = task_generator.generate_task(
            difficulty=self.config.task_difficulty
        )
        
        # Start new episode in state manager
        episode_config = options.get('episode_config', {}) if options else {}
        episode_id = self.state_manager.start_new_episode(
            task=self.current_task,
            initial_code=self.current_task.starter_code,
            episode_config=episode_config
        )
        
        # Setup execution environment for new task
        if not self.execution_env:
            self._setup_execution_environment()
        await self.execution_env.initialize()
        
        # Reset structured action space
        self.structured_action_space.reset()
        self.last_action_results = []
        
        # Reset milestone tracking flags
        self._reset_milestone_flags()
        
        # Create initial observation
        observation = self._create_observation()
        info = self._create_info()
        
        # Add state manager info
        info.update({
            'episode_id': episode_id,
            'episode_state': self.state_manager.episode_state.value,
            'state_summary': self.state_manager.get_state_summary()
        })
        
        return observation, info
    
    def _create_observation(self) -> Dict[str, Any]:
        """Create current observation from environment state"""
        if not self.current_task:
            raise RuntimeError("No current task - call reset() first")
        
        # Get current code from state manager
        current_code = self.state_manager.current_code
        
        # Create additional context for observation encoder
        additional_context = {
            'task_difficulty': self.current_task.difficulty.value,
            'step_count': self.state_manager.current_step,
            'completion_progress': self.state_manager.progress_metrics.completion_percentage,
            'last_execution_success': self._get_last_execution_success(),
            'syntax_errors': self._check_syntax_errors()
        }
        
        # Encode main observation using the observation encoder
        encoded_obs = self.observation_encoder.encode_observation(
            api_docs=self.current_task.api_documentation,
            current_code=current_code,
            additional_context=additional_context
        )
        
        # Start with encoded observations
        observation = {
            'api_schema': encoded_obs['api_schema'],
            'code_tokens': encoded_obs['code_tokens'],
            'code_structure': encoded_obs['code_structure'],
            'code_features': encoded_obs['code_features'],
            'alignment_features': encoded_obs['alignment_features'],
            
            # Task metadata
            'task_difficulty': self._difficulty_to_int(self.current_task.difficulty),
            'step_count': np.array([self.state_manager.current_step], dtype=np.int32),
            
            # Execution feedback
            'last_execution_success': self._get_last_execution_success(),
            'syntax_errors': self._check_syntax_errors(),
            
            # Action feedback
            'last_action_applied': self._get_last_action_applied(),
            'action_validation_errors': self._get_action_validation_errors(),
            
            # Progress indicators
            'completion_progress': np.array([self.state_manager.progress_metrics.completion_percentage], dtype=np.float32),
            'current_score': np.array([self.state_manager.current_score], dtype=np.float32),
        }
        
        # Add embeddings if enabled
        if self.config.enable_embeddings and self.embedding_generator:
            embeddings = self._generate_embeddings()
            observation.update({
                'code_embedding': embeddings['code_structure'],
                'api_embedding': embeddings['api_schema'],
                'alignment_embedding': embeddings['alignment']
            })
        
        # Add semantic patterns if enabled
        if self.config.enable_semantic_analysis and self.semantic_analyzer:
            semantic_features = self._extract_semantic_features()
            observation['semantic_patterns'] = semantic_features
        
        return observation
    
    def _create_info(self) -> Dict[str, Any]:
        """Create info dictionary with additional environment details"""
        return {
            'task_id': self.current_task.task_id if self.current_task else None,
            'task_title': self.current_task.title if self.current_task else None,
            'step_count': self.state_manager.current_step,
            'code_length': len(self.state_manager.current_code),
            'hints_available': len(self.current_task.hints) if self.current_task else 0,
            'episode_id': self.state_manager.episode_id,
            'episode_duration': time.time() - self.state_manager.episode_start_time if self.state_manager.episode_start_time else 0,
        }
    
    async def step(self, action: Union[Dict[str, Any], np.ndarray]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if not self.current_task:
            raise RuntimeError("No current task - call reset() first")
        
        if self.state_manager.episode_state != EpisodeState.ACTIVE:
            raise RuntimeError("Episode is not active - call reset() first")
        
        # Get current code from state manager
        current_code = self.state_manager.current_code
        
        # Decode gym action to semantic code actions
        code_actions = self.structured_action_space.decode_gym_action(action, current_code)
        
        # Record the action in state manager
        self.state_manager.record_step_action(action, [])  # Will update with results
        
        # Apply code actions
        new_code, action_results = self.structured_action_space.apply_actions(current_code, code_actions)
        self.last_action_results = action_results
        
        # Record code change in state manager
        modifications = []
        for i, result in enumerate(action_results):
            if result.get('applied', False):
                modifications.append({
                    'action_index': i,
                    'action_type': 'code_modification',
                    'success': True
                })
        
        self.state_manager.record_code_change(new_code, modifications)
        
        # Execute code if any action requested execution or periodically
        execution_result = None
        should_execute = any(
            result.get('action', {}).get('parameters', {}).get('execute_after', False) 
            for result in action_results
        ) or (len(code_actions) > 0 and self.state_manager.current_step % 3 == 0)  # Execute periodically
        
        if should_execute:
            execution_result = await self._execute_current_code()
            
            # Record execution in state manager
            if execution_result:
                # Extract API calls if any
                api_calls = self._extract_api_calls_from_execution(execution_result)
                code_hash = self.state_manager.code_history[-1].hash_code if self.state_manager.code_history else ""
                execution_id = self.state_manager.record_execution(execution_result, code_hash, api_calls)
        
        # Calculate reward
        current_score = self._get_current_score()
        reward = self._calculate_reward_from_actions(code_actions, action_results, execution_result)
        
        # Record reward in state manager
        self.state_manager.record_reward(reward, current_score)
        
        # Advance step in state manager
        self.state_manager.advance_step()
        
        # Check termination conditions using state manager
        terminated, truncated, termination_reason = self.state_manager.check_episode_termination()
        
        # End episode if terminated or truncated
        if terminated or truncated:
            self.state_manager.end_episode(termination_reason)
        
        # Create new observation
        observation = self._create_observation()
        info = self._create_info()
        
        # Add comprehensive state information
        info.update({
            'actions_applied': sum(1 for result in action_results if result['applied']),
            'actions_total': len(action_results),
            'episode_state': self.state_manager.episode_state.value,
            'state_summary': self.state_manager.get_state_summary(),
            'progress_metrics': {
                'completion_percentage': self.state_manager.progress_metrics.completion_percentage,
                'current_phase': self.state_manager.progress_metrics.current_phase,
                'milestones_achieved': len(self.state_manager.progress_metrics.milestones_achieved)
            }
        })
        
        if execution_result:
            info.update({
                'execution_status': execution_result.status.value,
                'execution_time': execution_result.execution_time
            })
        
        if terminated or truncated:
            info.update({
                'termination_reason': termination_reason.value if termination_reason else 'unknown',
                'episode_statistics': self.state_manager.episode_statistics
            })
        
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
            current_code = self.state_manager.current_code
            result = await self.execution_env.execute_code(current_code)
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
    
    def _generate_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for current state"""
        if not self.embedding_generator:
            return {}
        
        # Get code structure and API schema
        code_structure = self.observation_encoder.ast_analyzer.parse_code_structure(self.current_code)
        api_schema = self.observation_encoder.api_processor.process_api_documentation(
            self.current_task.api_documentation
        )
        
        # Get tokens
        tokens = self.observation_encoder.code_tokenizer.tokenize_code(self.current_code)
        
        # Get semantic patterns
        semantic_patterns = []
        if self.semantic_analyzer:
            semantic_patterns = self.semantic_analyzer.analyze_semantic_patterns(
                self.current_code, code_structure
            )
        
        # Generate combined embeddings
        embeddings = self.embedding_generator.create_combined_embedding(
            api_schema=api_schema,
            code_structure=code_structure,
            tokens=tokens,
            semantic_patterns=semantic_patterns
        )
        
        return embeddings
    
    def _extract_semantic_features(self) -> np.ndarray:
        """Extract semantic pattern features"""
        if not self.semantic_analyzer:
            return np.zeros(20, dtype=np.float32)
        
        # Get code structure
        code_structure = self.observation_encoder.ast_analyzer.parse_code_structure(self.current_code)
        
        # Analyze semantic patterns
        patterns = self.semantic_analyzer.analyze_semantic_patterns(self.current_code, code_structure)
        
        # Convert patterns to feature vector
        features = np.zeros(20, dtype=np.float32)
        
        # Map pattern types to feature indices
        pattern_type_mapping = {
            'authentication': 0,
            'http_request': 1,
            'data_processing': 2,
            'error_handling': 3,
            'pagination': 4,
            'url_construction': 5,
            'good_practices': 6,
            'anti_patterns': 7,
            'api_imports': 8,
            'async_programming': 9,
            'http_methods': 10,
            'request_response_flow': 11,
            'error_handling_flow': 12,
            'data_transformation': 13
        }
        
        # Fill feature vector based on detected patterns
        for pattern in patterns:
            pattern_idx = pattern_type_mapping.get(pattern.pattern_type)
            if pattern_idx is not None and pattern_idx < len(features):
                features[pattern_idx] = pattern.confidence
        
        return features
    
    def _reset_milestone_flags(self):
        """Reset milestone tracking flags for reward calculation"""
        if hasattr(self, '_imports_bonus'):
            delattr(self, '_imports_bonus')
        if hasattr(self, '_api_calls_bonus'):
            delattr(self, '_api_calls_bonus')
        if hasattr(self, '_error_handling_bonus'):
            delattr(self, '_error_handling_bonus')
        if hasattr(self, '_last_progress'):
            delattr(self, '_last_progress')
    
    def _extract_api_calls_from_execution(self, execution_result) -> List[Dict[str, Any]]:
        """Extract API calls from execution result"""
        api_calls = []
        
        # Check metadata for network requests
        if hasattr(execution_result, 'metadata') and execution_result.metadata:
            network_requests = execution_result.metadata.get('network_requests', [])
            for request in network_requests:
                api_calls.append({
                    'method': request.get('method', 'GET'),
                    'url': request.get('url', ''),
                    'headers': request.get('headers', {}),
                    'parameters': request.get('params', {}),
                    'response_status': request.get('status_code'),
                    'success': request.get('success', False),
                    'duration': request.get('duration', 0.0),
                    'error': request.get('error')
                })
        
        # Fallback: analyze stdout/stderr for API call patterns
        if not api_calls:
            output = execution_result.stdout + execution_result.stderr
            if any(pattern in output.lower() for pattern in ['http', 'api', 'request', 'response']):
                # Create a generic API call record
                api_calls.append({
                    'method': 'UNKNOWN',
                    'url': 'detected_in_output',
                    'success': execution_result.status.value == 'success',
                    'duration': execution_result.execution_time
                })
        
        return api_calls
    
    def _check_task_completion(self) -> bool:
        """Check if the current task is completed successfully"""
        # Delegate to state manager's termination conditions
        return self.state_manager.termination_conditions.get('task_completed', False)
    
    def _calculate_progress(self) -> float:
        """Calculate current progress (delegated to state manager)"""
        return self.state_manager.progress_metrics.completion_percentage
    
    def _get_current_score(self) -> float:
        """Get current score (delegated to state manager)"""
        return self.state_manager.current_score
    
    def _get_last_execution_success(self) -> int:
        """Get result of last execution from state manager"""
        if not self.state_manager.execution_history:
            return 0
        
        last_execution = self.state_manager.execution_history[-1]
        return 1 if last_execution.execution_result.status.value == 'success' else 0
    
    def _check_syntax_errors(self) -> int:
        """Check if current code has syntax errors"""
        current_code = self.state_manager.current_code
        try:
            compile(current_code, '<string>', 'exec')
            return 0
        except SyntaxError:
            return 1
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get detailed episode statistics"""
        return {
            'episode_id': self.state_manager.episode_id,
            'episode_state': self.state_manager.episode_state.value,
            'statistics': self.state_manager.episode_statistics,
            'progress_metrics': self.state_manager.progress_metrics,
            'state_summary': self.state_manager.get_state_summary()
        }
    
    def create_checkpoint(self, name: str = "manual") -> str:
        """Create a checkpoint of the current state"""
        return self.state_manager.create_checkpoint(name)
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint"""
        return self.state_manager.restore_checkpoint(checkpoint_id)
    
    def export_episode_data(self, format: str = 'json'):
        """Export complete episode data"""
        return self.state_manager.export_episode_data(format)
    
    def manual_terminate(self):
        """Manually terminate the current episode"""
        self.state_manager.manual_terminate()
    
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