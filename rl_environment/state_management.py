"""
Comprehensive state management for the API Integration RL environment.

This module handles tracking of code state, execution history, API call results,
episode progress, and provides robust reset/termination functionality.
"""

import time
import uuid
import json
import copy
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np

from rl_environment.code_actions import CodeAction
from rl_environment.code_executor import ExecutionResult, ExecutionStatus


class EpisodeState(Enum):
    """Possible states of an episode"""
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    PAUSED = "paused"


class TerminationReason(Enum):
    """Reasons for episode termination"""
    TASK_COMPLETED = "task_completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    SYNTAX_ERROR_LIMIT = "syntax_error_limit"
    EXECUTION_ERROR_LIMIT = "execution_error_limit"
    TIMEOUT = "timeout"
    MANUAL_STOP = "manual_stop"
    INVALID_STATE = "invalid_state"
    NO_PROGRESS = "no_progress"


@dataclass
class CodeSnapshot:
    """Represents a snapshot of the code at a specific point in time"""
    timestamp: float
    step_number: int
    code: str
    code_length: int
    line_count: int
    hash_code: str
    modifications_from_previous: List[Dict[str, Any]] = field(default_factory=list)
    syntax_valid: bool = True
    syntax_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.hash_code:
            import hashlib
            self.hash_code = hashlib.md5(self.code.encode()).hexdigest()


@dataclass
class ExecutionSnapshot:
    """Represents a code execution event"""
    execution_id: str
    timestamp: float
    step_number: int
    code_snapshot_hash: str
    execution_result: ExecutionResult
    duration: float
    memory_usage: Optional[int] = None
    api_calls_made: List[Dict[str, Any]] = field(default_factory=list)
    network_requests: List[Dict[str, Any]] = field(default_factory=list)
    output_captured: str = ""
    error_captured: str = ""
    
    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())


@dataclass
class APICallRecord:
    """Records an API call attempt"""
    call_id: str
    timestamp: float
    step_number: int
    execution_id: str
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_body: Any = None
    response_status: Optional[int] = None
    response_data: Any = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    duration: float = 0.0
    success: bool = False
    
    def __post_init__(self):
        if not self.call_id:
            self.call_id = str(uuid.uuid4())


@dataclass
class ProgressMetrics:
    """Tracks episode progress metrics"""
    completion_percentage: float = 0.0
    code_quality_score: float = 0.0
    api_integration_score: float = 0.0
    error_handling_score: float = 0.0
    total_score: float = 0.0
    milestones_achieved: List[str] = field(default_factory=list)
    current_phase: str = "initialization"
    phases_completed: List[str] = field(default_factory=list)
    time_spent_per_phase: Dict[str, float] = field(default_factory=dict)
    
    # Progress indicators
    has_imports: bool = False
    has_functions: bool = False
    has_api_calls: bool = False
    has_error_handling: bool = False
    has_successful_execution: bool = False
    has_successful_api_call: bool = False


@dataclass
class EpisodeStatistics:
    """Statistics for the current episode"""
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    syntax_errors: int = 0
    runtime_errors: int = 0
    total_api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    best_score: float = 0.0
    worst_score: float = 0.0
    
    # Time tracking
    episode_start_time: float = 0.0
    episode_end_time: Optional[float] = None
    total_duration: float = 0.0
    time_per_step: float = 0.0
    
    # Code metrics
    final_code_length: int = 0
    max_code_length: int = 0
    code_modifications: int = 0
    lines_added: int = 0
    lines_removed: int = 0


class EnvironmentStateManager:
    """Manages the complete state of the RL environment"""
    
    def __init__(self, max_history_length: int = 1000):
        self.max_history_length = max_history_length
        
        # Episode management
        self.episode_id: str = ""
        self.episode_state: EpisodeState = EpisodeState.NOT_STARTED
        self.episode_start_time: float = 0.0
        self.episode_statistics = EpisodeStatistics()
        
        # Current state
        self.current_step: int = 0
        self.current_task = None
        self.current_code: str = ""
        self.current_score: float = 0.0
        self.last_reward: float = 0.0
        
        # History tracking
        self.code_history: deque = deque(maxlen=max_history_length)
        self.execution_history: deque = deque(maxlen=max_history_length)
        self.api_call_history: deque = deque(maxlen=max_history_length)
        self.action_history: deque = deque(maxlen=max_history_length)
        self.reward_history: deque = deque(maxlen=max_history_length)
        
        # Progress tracking
        self.progress_metrics = ProgressMetrics()
        self.termination_conditions = self._initialize_termination_conditions()
        
        # State checkpoints
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.auto_checkpoint_interval: int = 10  # Steps
        
        # Configuration
        self.config = {
            'max_steps': 100,
            'max_syntax_errors': 10,
            'max_execution_errors': 20,
            'episode_timeout': 3600,  # 1 hour
            'no_progress_threshold': 20,  # Steps without progress
            'minimum_score_threshold': 1.0,
            'auto_save_enabled': True
        }
    
    def _initialize_termination_conditions(self) -> Dict[str, Any]:
        """Initialize episode termination conditions"""
        return {
            'max_steps_reached': False,
            'task_completed': False,
            'too_many_syntax_errors': False,
            'too_many_execution_errors': False,
            'episode_timeout': False,
            'no_progress_detected': False,
            'manual_termination': False
        }
    
    def start_new_episode(self, task: Any, initial_code: str = "", episode_config: Optional[Dict] = None) -> str:
        """Start a new episode with the given task"""
        # Generate new episode ID
        self.episode_id = f"episode_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Update configuration if provided
        if episode_config:
            self.config.update(episode_config)
        
        # Reset all state
        self._reset_state()
        
        # Initialize episode
        self.episode_state = EpisodeState.ACTIVE
        self.episode_start_time = time.time()
        self.current_task = task
        self.current_code = initial_code
        
        # Initialize statistics
        self.episode_statistics = EpisodeStatistics(episode_start_time=self.episode_start_time)
        
        # Create initial code snapshot
        initial_snapshot = CodeSnapshot(
            timestamp=self.episode_start_time,
            step_number=0,
            code=initial_code,
            code_length=len(initial_code),
            line_count=len(initial_code.split('\n')) if initial_code else 0,
            hash_code=""
        )
        self.code_history.append(initial_snapshot)
        
        # Initialize progress tracking
        self._update_progress_metrics()
        
        return self.episode_id
    
    def _reset_state(self):
        """Reset all state variables"""
        self.current_step = 0
        self.current_code = ""
        self.current_score = 0.0
        self.last_reward = 0.0
        
        # Clear histories
        self.code_history.clear()
        self.execution_history.clear()
        self.api_call_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        # Reset progress and termination conditions
        self.progress_metrics = ProgressMetrics()
        self.termination_conditions = self._initialize_termination_conditions()
        
        # Clear checkpoints
        self.checkpoints.clear()
    
    def record_step_action(self, action: Union[Dict, List[CodeAction]], action_results: List[Dict[str, Any]]):
        """Record an action taken in the environment"""
        step_data = {
            'step_number': self.current_step,
            'timestamp': time.time(),
            'action': action,
            'action_results': action_results,
            'code_before': self.current_code,
            'successful_actions': sum(1 for result in action_results if result.get('applied', False)),
            'total_actions': len(action_results)
        }
        
        self.action_history.append(step_data)
        
        # Update statistics
        if step_data['successful_actions'] > 0:
            self.episode_statistics.successful_steps += 1
        else:
            self.episode_statistics.failed_steps += 1
    
    def record_code_change(self, new_code: str, modifications: List[Dict[str, Any]] = None):
        """Record a change to the code"""
        current_time = time.time()
        
        # Detect syntax errors
        syntax_valid = True
        syntax_errors = []
        try:
            compile(new_code, '<string>', 'exec')
        except SyntaxError as e:
            syntax_valid = False
            syntax_errors = [str(e)]
            self.episode_statistics.syntax_errors += 1
        
        # Create code snapshot
        snapshot = CodeSnapshot(
            timestamp=current_time,
            step_number=self.current_step,
            code=new_code,
            code_length=len(new_code),
            line_count=len(new_code.split('\n')) if new_code else 0,
            hash_code="",
            modifications_from_previous=modifications or [],
            syntax_valid=syntax_valid,
            syntax_errors=syntax_errors
        )
        
        # Update current code
        old_code = self.current_code
        self.current_code = new_code
        
        # Track code metrics
        self._update_code_metrics(old_code, new_code)
        
        # Add to history
        self.code_history.append(snapshot)
        
        # Update progress
        self._update_progress_metrics()
    
    def record_execution(self, execution_result: ExecutionResult, code_hash: str = "", 
                        api_calls: List[Dict[str, Any]] = None) -> str:
        """Record a code execution"""
        execution_id = str(uuid.uuid4())
        current_time = time.time()
        
        # Extract network requests if available
        network_requests = []
        if hasattr(execution_result, 'metadata') and execution_result.metadata:
            network_requests = execution_result.metadata.get('network_requests', [])
        
        # Create execution snapshot
        snapshot = ExecutionSnapshot(
            execution_id=execution_id,
            timestamp=current_time,
            step_number=self.current_step,
            code_snapshot_hash=code_hash,
            execution_result=execution_result,
            duration=execution_result.execution_time,
            api_calls_made=api_calls or [],
            network_requests=network_requests,
            output_captured=execution_result.stdout,
            error_captured=execution_result.stderr
        )
        
        # Add to history
        self.execution_history.append(snapshot)
        
        # Update statistics
        self.episode_statistics.total_executions += 1
        if execution_result.status == ExecutionStatus.SUCCESS:
            self.episode_statistics.successful_executions += 1
            self.progress_metrics.has_successful_execution = True
        else:
            self.episode_statistics.failed_executions += 1
            if execution_result.status != ExecutionStatus.TIMEOUT:
                self.episode_statistics.runtime_errors += 1
        
        # Process API calls
        if api_calls:
            for call_data in api_calls:
                self.record_api_call(call_data, execution_id)
        
        return execution_id
    
    def record_api_call(self, call_data: Dict[str, Any], execution_id: str = ""):
        """Record an API call"""
        call_record = APICallRecord(
            call_id="",
            timestamp=time.time(),
            step_number=self.current_step,
            execution_id=execution_id,
            method=call_data.get('method', 'GET'),
            url=call_data.get('url', ''),
            headers=call_data.get('headers', {}),
            parameters=call_data.get('parameters', {}),
            request_body=call_data.get('request_body'),
            response_status=call_data.get('response_status'),
            response_data=call_data.get('response_data'),
            response_headers=call_data.get('response_headers', {}),
            error=call_data.get('error'),
            duration=call_data.get('duration', 0.0),
            success=call_data.get('success', False)
        )
        
        # Add to history
        self.api_call_history.append(call_record)
        
        # Update statistics
        self.episode_statistics.total_api_calls += 1
        if call_record.success:
            self.episode_statistics.successful_api_calls += 1
            self.progress_metrics.has_successful_api_call = True
        else:
            self.episode_statistics.failed_api_calls += 1
    
    def record_reward(self, reward: float, score: float):
        """Record a reward and score"""
        reward_data = {
            'step_number': self.current_step,
            'timestamp': time.time(),
            'reward': reward,
            'score': score,
            'cumulative_reward': self.episode_statistics.total_reward + reward
        }
        
        self.reward_history.append(reward_data)
        
        # Update current values
        self.last_reward = reward
        self.current_score = score
        
        # Update statistics
        self.episode_statistics.total_reward += reward
        self.episode_statistics.average_reward = (
            self.episode_statistics.total_reward / max(1, self.current_step)
        )
        self.episode_statistics.best_score = max(self.episode_statistics.best_score, score)
        if self.episode_statistics.worst_score == 0:
            self.episode_statistics.worst_score = score
        else:
            self.episode_statistics.worst_score = min(self.episode_statistics.worst_score, score)
        
        # Update progress metrics
        self.progress_metrics.total_score = score
    
    def advance_step(self):
        """Advance to the next step"""
        self.current_step += 1
        self.episode_statistics.total_steps += 1
        
        # Auto-checkpoint if enabled
        if (self.config.get('auto_save_enabled', True) and 
            self.current_step % self.auto_checkpoint_interval == 0):
            self.create_checkpoint(f"auto_step_{self.current_step}")
        
        # Check termination conditions
        self._check_termination_conditions()
    
    def _update_code_metrics(self, old_code: str, new_code: str):
        """Update code-related metrics"""
        old_lines = old_code.split('\n') if old_code else []
        new_lines = new_code.split('\n') if new_code else []
        
        lines_added = max(0, len(new_lines) - len(old_lines))
        lines_removed = max(0, len(old_lines) - len(new_lines))
        
        self.episode_statistics.lines_added += lines_added
        self.episode_statistics.lines_removed += lines_removed
        self.episode_statistics.code_modifications += 1
        self.episode_statistics.final_code_length = len(new_code)
        self.episode_statistics.max_code_length = max(
            self.episode_statistics.max_code_length, len(new_code)
        )
    
    def _update_progress_metrics(self):
        """Update progress metrics based on current code state"""
        code = self.current_code
        
        # Basic progress indicators
        self.progress_metrics.has_imports = 'import ' in code or 'from ' in code
        self.progress_metrics.has_functions = 'def ' in code
        self.progress_metrics.has_api_calls = any(
            pattern in code for pattern in ['requests.', 'httpx.', '.get(', '.post(']
        )
        self.progress_metrics.has_error_handling = any(
            pattern in code for pattern in ['try:', 'except', 'raise']
        )
        
        # Calculate completion percentage
        completion_factors = [
            self.progress_metrics.has_imports,
            self.progress_metrics.has_functions,
            self.progress_metrics.has_api_calls,
            self.progress_metrics.has_error_handling,
            self.progress_metrics.has_successful_execution,
            self.progress_metrics.has_successful_api_call
        ]
        
        self.progress_metrics.completion_percentage = sum(completion_factors) / len(completion_factors)
        
        # Update current phase
        if not self.progress_metrics.has_imports:
            self.progress_metrics.current_phase = "setup"
        elif not self.progress_metrics.has_api_calls:
            self.progress_metrics.current_phase = "implementation"
        elif not self.progress_metrics.has_successful_execution:
            self.progress_metrics.current_phase = "testing"
        elif not self.progress_metrics.has_error_handling:
            self.progress_metrics.current_phase = "error_handling"
        else:
            self.progress_metrics.current_phase = "completion"
        
        # Check for milestones
        self._check_milestones()
    
    def _check_milestones(self):
        """Check if any milestones have been achieved"""
        milestones = [
            ("first_import", self.progress_metrics.has_imports),
            ("first_function", self.progress_metrics.has_functions),
            ("first_api_call", self.progress_metrics.has_api_calls),
            ("first_execution", self.progress_metrics.has_successful_execution),
            ("first_successful_api", self.progress_metrics.has_successful_api_call),
            ("error_handling_added", self.progress_metrics.has_error_handling),
            ("50_percent_complete", self.progress_metrics.completion_percentage >= 0.5),
            ("80_percent_complete", self.progress_metrics.completion_percentage >= 0.8),
        ]
        
        for milestone_name, condition in milestones:
            if condition and milestone_name not in self.progress_metrics.milestones_achieved:
                self.progress_metrics.milestones_achieved.append(milestone_name)
    
    def _check_termination_conditions(self):
        """Check if any termination conditions are met"""
        # Max steps reached
        if self.current_step >= self.config['max_steps']:
            self.termination_conditions['max_steps_reached'] = True
        
        # Too many syntax errors
        if self.episode_statistics.syntax_errors >= self.config['max_syntax_errors']:
            self.termination_conditions['too_many_syntax_errors'] = True
        
        # Too many execution errors
        if self.episode_statistics.runtime_errors >= self.config['max_execution_errors']:
            self.termination_conditions['too_many_execution_errors'] = True
        
        # Episode timeout
        if time.time() - self.episode_start_time > self.config['episode_timeout']:
            self.termination_conditions['episode_timeout'] = True
        
        # No progress detection
        if self._detect_no_progress():
            self.termination_conditions['no_progress_detected'] = True
        
        # Task completion (high score threshold)
        if self.current_score >= self.config.get('completion_score_threshold', 8.0):
            self.termination_conditions['task_completed'] = True
    
    def _detect_no_progress(self) -> bool:
        """Detect if no meaningful progress is being made"""
        if len(self.reward_history) < self.config['no_progress_threshold']:
            return False
        
        # Check if rewards have been consistently low/negative
        recent_rewards = [r['reward'] for r in list(self.reward_history)[-self.config['no_progress_threshold']:]]
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
        
        # Check if completion percentage hasn't increased
        if self.progress_metrics.completion_percentage < 0.1 and self.current_step > 20:
            return True
        
        # Check if average reward is very negative
        if avg_recent_reward < -0.5:
            return True
        
        return False
    
    def check_episode_termination(self) -> Tuple[bool, bool, TerminationReason]:
        """Check if episode should terminate and return (terminated, truncated, reason)"""
        # Check for task completion (terminated = success)
        if self.termination_conditions['task_completed']:
            return True, False, TerminationReason.TASK_COMPLETED
        
        # Check for truncation conditions
        if self.termination_conditions['max_steps_reached']:
            return False, True, TerminationReason.MAX_STEPS_REACHED
        
        if self.termination_conditions['episode_timeout']:
            return False, True, TerminationReason.TIMEOUT
        
        if self.termination_conditions['too_many_syntax_errors']:
            return False, True, TerminationReason.SYNTAX_ERROR_LIMIT
        
        if self.termination_conditions['too_many_execution_errors']:
            return False, True, TerminationReason.EXECUTION_ERROR_LIMIT
        
        if self.termination_conditions['no_progress_detected']:
            return False, True, TerminationReason.NO_PROGRESS
        
        if self.termination_conditions['manual_termination']:
            return False, True, TerminationReason.MANUAL_STOP
        
        # Episode continues
        return False, False, None
    
    def end_episode(self, reason: TerminationReason):
        """End the current episode"""
        self.episode_statistics.episode_end_time = time.time()
        self.episode_statistics.total_duration = (
            self.episode_statistics.episode_end_time - self.episode_statistics.episode_start_time
        )
        self.episode_statistics.time_per_step = (
            self.episode_statistics.total_duration / max(1, self.current_step)
        )
        
        # Set final episode state
        if reason == TerminationReason.TASK_COMPLETED:
            self.episode_state = EpisodeState.COMPLETED
        elif reason in [TerminationReason.SYNTAX_ERROR_LIMIT, TerminationReason.EXECUTION_ERROR_LIMIT]:
            self.episode_state = EpisodeState.FAILED
        else:
            self.episode_state = EpisodeState.TRUNCATED
        
        # Create final checkpoint
        self.create_checkpoint("episode_end")
    
    def create_checkpoint(self, checkpoint_name: str) -> str:
        """Create a checkpoint of the current state"""
        checkpoint_id = f"{checkpoint_name}_{int(time.time())}"
        
        self.checkpoints[checkpoint_id] = {
            'episode_id': self.episode_id,
            'step_number': self.current_step,
            'timestamp': time.time(),
            'current_code': self.current_code,
            'current_score': self.current_score,
            'episode_statistics': asdict(self.episode_statistics),
            'progress_metrics': asdict(self.progress_metrics),
            'termination_conditions': self.termination_conditions.copy(),
            'recent_actions': list(self.action_history)[-10:],  # Last 10 actions
            'recent_rewards': list(self.reward_history)[-10:]   # Last 10 rewards
        }
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from a checkpoint"""
        if checkpoint_id not in self.checkpoints:
            return False
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        # Restore basic state
        self.episode_id = checkpoint['episode_id']
        self.current_step = checkpoint['step_number']
        self.current_code = checkpoint['current_code']
        self.current_score = checkpoint['current_score']
        
        # Restore complex objects
        self.episode_statistics = EpisodeStatistics(**checkpoint['episode_statistics'])
        self.progress_metrics = ProgressMetrics(**checkpoint['progress_metrics'])
        self.termination_conditions = checkpoint['termination_conditions']
        
        return True
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state"""
        return {
            'episode_id': self.episode_id,
            'episode_state': self.episode_state.value,
            'current_step': self.current_step,
            'current_score': self.current_score,
            'completion_percentage': self.progress_metrics.completion_percentage,
            'current_phase': self.progress_metrics.current_phase,
            'milestones_achieved': len(self.progress_metrics.milestones_achieved),
            'total_executions': self.episode_statistics.total_executions,
            'successful_executions': self.episode_statistics.successful_executions,
            'total_api_calls': self.episode_statistics.total_api_calls,
            'successful_api_calls': self.episode_statistics.successful_api_calls,
            'syntax_errors': self.episode_statistics.syntax_errors,
            'runtime_errors': self.episode_statistics.runtime_errors,
            'total_reward': self.episode_statistics.total_reward,
            'average_reward': self.episode_statistics.average_reward,
            'code_length': len(self.current_code),
            'termination_conditions': sum(self.termination_conditions.values()),
            'time_elapsed': time.time() - self.episode_start_time if self.episode_start_time else 0
        }
    
    def export_episode_data(self, format: str = 'json') -> Union[str, Dict]:
        """Export complete episode data"""
        data = {
            'episode_metadata': {
                'episode_id': self.episode_id,
                'episode_state': self.episode_state.value,
                'start_time': self.episode_start_time,
                'end_time': self.episode_statistics.episode_end_time,
                'total_duration': self.episode_statistics.total_duration
            },
            'statistics': asdict(self.episode_statistics),
            'progress_metrics': asdict(self.progress_metrics),
            'termination_conditions': self.termination_conditions,
            'code_history': [asdict(snapshot) for snapshot in self.code_history],
            'execution_history': [
                {**asdict(snapshot), 'execution_result': asdict(snapshot.execution_result)}
                for snapshot in self.execution_history
            ],
            'api_call_history': [asdict(call) for call in self.api_call_history],
            'action_history': list(self.action_history),
            'reward_history': list(self.reward_history),
            'checkpoints': list(self.checkpoints.keys())
        }
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def reset_for_new_episode(self):
        """Reset state manager for a completely new episode"""
        self.episode_state = EpisodeState.NOT_STARTED
        self._reset_state()
        
    def manual_terminate(self):
        """Manually terminate the current episode"""
        self.termination_conditions['manual_termination'] = True