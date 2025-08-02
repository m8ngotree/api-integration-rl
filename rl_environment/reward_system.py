import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from rl_environment.code_executor import ExecutionResult, ExecutionStatus
from data_generation.endpoint_generator import EndpointSpec, HTTPMethod


class RewardComponent(Enum):
    EXECUTION_SUCCESS = "execution_success"
    API_CALLS_MADE = "api_calls_made"
    CORRECT_HTTP_METHODS = "correct_http_methods"
    RESPONSE_HANDLING = "response_handling"
    ERROR_HANDLING = "error_handling"
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class APICallInfo:
    """Information about an API call detected in code execution"""
    method: str
    url: str
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    success: bool = False


@dataclass
class CodeAnalysis:
    """Analysis of code structure and patterns"""
    has_imports: bool = False
    has_error_handling: bool = False
    has_response_parsing: bool = False
    has_authentication: bool = False
    has_logging: bool = False
    function_count: int = 0
    class_count: int = 0
    lines_of_code: int = 0
    complexity_score: float = 0.0


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward calculation"""
    total_reward: float
    component_scores: Dict[RewardComponent, float] = field(default_factory=dict)
    bonus_points: float = 0.0
    penalty_points: float = 0.0
    multipliers: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Base rewards
    execution_success_reward: float = 10.0
    api_call_success_reward: float = 5.0
    correct_method_reward: float = 3.0
    response_handling_reward: float = 4.0
    error_handling_reward: float = 6.0
    
    # Bonus rewards
    performance_bonus_threshold: float = 2.0  # seconds
    performance_bonus: float = 2.0
    security_bonus: float = 3.0
    code_quality_bonus: float = 2.0
    
    # Penalties
    timeout_penalty: float = -5.0
    security_violation_penalty: float = -10.0
    error_penalty: float = -2.0
    
    # Multipliers
    multiple_endpoints_multiplier: float = 1.5
    authentication_multiplier: float = 1.2
    comprehensive_error_handling_multiplier: float = 1.3


class APICallTracker:
    """Tracks and validates API calls made during code execution"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns for detecting API calls in output
        self.api_call_patterns = [
            # requests library patterns
            r'requests\.(get|post|put|delete|patch)\([\'"]([^\'\"]+)[\'"]',
            r'response = requests\.(get|post|put|delete|patch)\(',
            r'(GET|POST|PUT|DELETE|PATCH)\s+([^\s]+)\s+(\d{3})',
            # httpx patterns
            r'httpx\.(get|post|put|delete|patch)\([\'"]([^\'\"]+)[\'"]',
            # urllib patterns
            r'urllib\.request\.urlopen\([\'"]([^\'\"]+)[\'"]'
        ]
        
        # Status code patterns
        self.status_patterns = [
            r'status_code[:\s=]+(\d{3})',
            r'Status:\s*(\d{3})',
            r'Response:\s*(\d{3})',
            r'HTTP/\d\.\d\s+(\d{3})'
        ]
        
        # Error patterns
        self.error_patterns = [
            r'requests\.exceptions\.(\w+)',
            r'HTTPError',
            r'ConnectionError',
            r'Timeout',
            r'RequestException'
        ]
    
    def analyze_execution_output(
        self,
        stdout: str,
        stderr: str,
        code: str
    ) -> List[APICallInfo]:
        """Analyze execution output to extract API call information"""
        
        api_calls = []
        
        # Analyze code for API call patterns
        code_calls = self._extract_calls_from_code(code)
        
        # Analyze output for execution evidence
        output_calls = self._extract_calls_from_output(stdout, stderr)
        
        # Combine and deduplicate
        all_calls = code_calls + output_calls
        api_calls = self._deduplicate_calls(all_calls)
        
        return api_calls
    
    def _extract_calls_from_code(self, code: str) -> List[APICallInfo]:
        """Extract API calls from static code analysis"""
        calls = []
        
        for pattern in self.api_call_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    method = match.group(1).upper()
                    url = match.group(2) if len(match.groups()) > 1 else "unknown"
                    
                    calls.append(APICallInfo(
                        method=method,
                        url=url,
                        success=False  # Will be updated based on execution
                    ))
        
        return calls
    
    def _extract_calls_from_output(self, stdout: str, stderr: str) -> List[APICallInfo]:
        """Extract API call results from execution output"""
        calls = []
        combined_output = stdout + "\n" + stderr
        
        # Look for HTTP method and URL patterns
        http_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+([^\s]+)(?:\s+(\d{3}))?'
        matches = re.finditer(http_pattern, combined_output, re.IGNORECASE)
        
        for match in matches:
            method = match.group(1).upper()
            url = match.group(2)
            status_code = int(match.group(3)) if match.group(3) else None
            
            # Determine success based on status code
            success = status_code is not None and 200 <= status_code < 400
            
            calls.append(APICallInfo(
                method=method,
                url=url,
                status_code=status_code,
                success=success
            ))
        
        # Look for status code patterns
        for pattern in self.status_patterns:
            matches = re.finditer(pattern, combined_output)
            for match in matches:
                status_code = int(match.group(1))
                # Create a generic call entry if we found status codes
                calls.append(APICallInfo(
                    method="UNKNOWN",
                    url="unknown",
                    status_code=status_code,
                    success=200 <= status_code < 400
                ))
        
        return calls
    
    def _deduplicate_calls(self, calls: List[APICallInfo]) -> List[APICallInfo]:
        """Remove duplicate API calls while preserving most complete information"""
        unique_calls = {}
        
        for call in calls:
            key = f"{call.method}:{call.url}"
            
            if key not in unique_calls:
                unique_calls[key] = call
            else:
                # Keep the call with more information
                existing = unique_calls[key]
                if call.status_code and not existing.status_code:
                    unique_calls[key] = call
                elif call.success and not existing.success:
                    unique_calls[key] = call
        
        return list(unique_calls.values())


class CodeQualityAnalyzer:
    """Analyzes code quality and structure"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_code(self, code: str) -> CodeAnalysis:
        """Perform comprehensive code analysis"""
        
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        analysis = CodeAnalysis(
            lines_of_code=len(non_empty_lines)
        )
        
        # Check for imports
        analysis.has_imports = any(
            line.strip().startswith(('import ', 'from '))
            for line in lines
        )
        
        # Check for error handling
        analysis.has_error_handling = any(
            keyword in code.lower()
            for keyword in ['try:', 'except:', 'raise', 'error', 'exception']
        )
        
        # Check for response parsing
        analysis.has_response_parsing = any(
            pattern in code.lower()
            for pattern in ['.json()', 'response.', 'data =', 'result =']
        )
        
        # Check for authentication
        analysis.has_authentication = any(
            pattern in code.lower()
            for pattern in ['authorization', 'bearer', 'api_key', 'token', 'auth']
        )
        
        # Check for logging
        analysis.has_logging = any(
            pattern in code.lower()
            for pattern in ['logging', 'print(', 'log.', 'logger.']
        )
        
        # Count functions and classes
        analysis.function_count = len(re.findall(r'def\s+\w+\s*\(', code))
        analysis.class_count = len(re.findall(r'class\s+\w+\s*[:\(]', code))
        
        # Calculate complexity score
        analysis.complexity_score = self._calculate_complexity(code)
        
        return analysis
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score"""
        complexity = 0.0
        
        # Cyclomatic complexity indicators
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code, re.IGNORECASE))
        
        # Normalize by lines of code
        lines_of_code = len([line for line in code.split('\n') if line.strip()])
        if lines_of_code > 0:
            complexity = complexity / lines_of_code
        
        return complexity


class BaseRewardEvaluator:
    """Base class for reward evaluation"""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.logger = logging.getLogger(__name__)
        self.call_tracker = APICallTracker()
        self.quality_analyzer = CodeQualityAnalyzer()
    
    def evaluate_code_execution(
        self,
        code: str,
        execution_result: ExecutionResult,
        expected_endpoints: Optional[List[EndpointSpec]] = None
    ) -> RewardBreakdown:
        """Evaluate code execution and return detailed reward breakdown"""
        
        reward_breakdown = RewardBreakdown(total_reward=0.0)
        
        try:
            # Basic execution success
            execution_reward = self._evaluate_execution_success(execution_result)
            reward_breakdown.component_scores[RewardComponent.EXECUTION_SUCCESS] = execution_reward
            
            # API call analysis
            api_calls = self.call_tracker.analyze_execution_output(
                execution_result.stdout,
                execution_result.stderr,
                code
            )
            
            api_reward = self._evaluate_api_calls(api_calls, expected_endpoints)
            reward_breakdown.component_scores[RewardComponent.API_CALLS_MADE] = api_reward
            
            # HTTP method correctness
            method_reward = self._evaluate_http_methods(api_calls, expected_endpoints)
            reward_breakdown.component_scores[RewardComponent.CORRECT_HTTP_METHODS] = method_reward
            
            # Response handling
            response_reward = self._evaluate_response_handling(code, execution_result)
            reward_breakdown.component_scores[RewardComponent.RESPONSE_HANDLING] = response_reward
            
            # Error handling
            error_reward = self._evaluate_error_handling(code, execution_result)
            reward_breakdown.component_scores[RewardComponent.ERROR_HANDLING] = error_reward
            
            # Code quality
            quality_reward = self._evaluate_code_quality(code)
            reward_breakdown.component_scores[RewardComponent.CODE_QUALITY] = quality_reward
            
            # Performance
            performance_reward = self._evaluate_performance(execution_result)
            reward_breakdown.component_scores[RewardComponent.PERFORMANCE] = performance_reward
            
            # Security
            security_reward = self._evaluate_security(execution_result)
            reward_breakdown.component_scores[RewardComponent.SECURITY] = security_reward
            
            # Apply bonuses and penalties
            self._apply_bonuses_and_penalties(reward_breakdown, api_calls, code, execution_result)
            
            # Calculate total reward
            reward_breakdown.total_reward = sum(reward_breakdown.component_scores.values())
            reward_breakdown.total_reward += reward_breakdown.bonus_points
            reward_breakdown.total_reward += reward_breakdown.penalty_points
            
            # Apply multipliers
            self._apply_multipliers(reward_breakdown, api_calls, code)
            
            # Generate explanation
            self._generate_explanation(reward_breakdown, api_calls, execution_result)
            
            return reward_breakdown
            
        except Exception as e:
            self.logger.error(f"Error evaluating code execution: {e}")
            return RewardBreakdown(
                total_reward=0.0,
                explanation=[f"Evaluation error: {str(e)}"]
            )
    
    def _evaluate_execution_success(self, result: ExecutionResult) -> float:
        """Evaluate basic execution success"""
        if result.status == ExecutionStatus.SUCCESS:
            return self.config.execution_success_reward
        elif result.status == ExecutionStatus.TIMEOUT:
            return self.config.timeout_penalty
        elif result.status == ExecutionStatus.SECURITY_VIOLATION:
            return self.config.security_violation_penalty
        else:
            return self.config.error_penalty
    
    def _evaluate_api_calls(
        self,
        api_calls: List[APICallInfo],
        expected_endpoints: Optional[List[EndpointSpec]]
    ) -> float:
        """Evaluate API calls made"""
        if not api_calls:
            return 0.0
        
        reward = 0.0
        successful_calls = [call for call in api_calls if call.success]
        
        # Base reward for successful calls
        reward += len(successful_calls) * self.config.api_call_success_reward
        
        # Bonus for making calls to expected endpoints
        if expected_endpoints:
            expected_paths = set(ep.path for ep in expected_endpoints)
            matched_calls = 0
            
            for call in successful_calls:
                for expected_path in expected_paths:
                    if expected_path in call.url or call.url in expected_path:
                        matched_calls += 1
                        break
            
            reward += matched_calls * 2.0  # Bonus for expected endpoint calls
        
        return reward
    
    def _evaluate_http_methods(
        self,
        api_calls: List[APICallInfo],
        expected_endpoints: Optional[List[EndpointSpec]]
    ) -> float:
        """Evaluate correctness of HTTP methods used"""
        if not api_calls or not expected_endpoints:
            return 0.0
        
        reward = 0.0
        expected_methods = {}
        
        # Build map of expected methods for paths
        for endpoint in expected_endpoints:
            path_key = endpoint.path.split('/')[-1]  # Use last part of path as key
            expected_methods[path_key] = endpoint.method.value
        
        # Check if calls use correct methods
        for call in api_calls:
            for path_key, expected_method in expected_methods.items():
                if path_key in call.url:
                    if call.method == expected_method:
                        reward += self.config.correct_method_reward
                    break
        
        return reward
    
    def _evaluate_response_handling(self, code: str, result: ExecutionResult) -> float:
        """Evaluate response handling in code"""
        reward = 0.0
        
        # Check for response parsing patterns
        response_patterns = [
            r'\.json\(\)',
            r'response\.text',
            r'response\.content',
            r'response\.status_code',
            r'response\.headers'
        ]
        
        for pattern in response_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                reward += self.config.response_handling_reward / len(response_patterns)
        
        # Check if response data is actually used in output
        if ('json' in result.stdout.lower() or 
            'data' in result.stdout.lower() or
            'response' in result.stdout.lower()):
            reward += 1.0
        
        return reward
    
    def _evaluate_error_handling(self, code: str, result: ExecutionResult) -> float:
        """Evaluate error handling implementation"""
        reward = 0.0
        
        # Check for error handling patterns
        error_patterns = [
            r'try\s*:',
            r'except\s+\w*Exception',
            r'except\s+requests\.',
            r'raise_for_status\(\)',
            r'if\s+response\.status_code'
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                reward += self.config.error_handling_reward / len(error_patterns)
        
        # Bonus if code executed without unhandled exceptions
        if result.status == ExecutionStatus.SUCCESS and not result.stderr:
            reward += 1.0
        
        return reward
    
    def _evaluate_code_quality(self, code: str) -> float:
        """Evaluate overall code quality"""
        analysis = self.quality_analyzer.analyze_code(code)
        reward = 0.0
        
        # Reward for good practices
        if analysis.has_imports:
            reward += 0.5
        if analysis.has_error_handling:
            reward += 1.0
        if analysis.has_response_parsing:
            reward += 1.0
        if analysis.has_authentication:
            reward += 0.5
        
        # Structure rewards
        if analysis.function_count > 0:
            reward += 0.5
        if analysis.class_count > 0:
            reward += 1.0
        
        # Complexity penalty/reward
        if 0.1 <= analysis.complexity_score <= 0.3:  # Good complexity
            reward += 0.5
        elif analysis.complexity_score > 0.5:  # Too complex
            reward -= 0.5
        
        return reward
    
    def _evaluate_performance(self, result: ExecutionResult) -> float:
        """Evaluate performance aspects"""
        reward = 0.0
        
        # Time-based performance
        if result.execution_time < self.config.performance_bonus_threshold:
            reward += self.config.performance_bonus
        elif result.execution_time > 10.0:  # Very slow
            reward -= 1.0
        
        # Memory efficiency
        if result.memory_usage < 50:  # MB
            reward += 1.0
        elif result.memory_usage > 100:
            reward -= 0.5
        
        return reward
    
    def _evaluate_security(self, result: ExecutionResult) -> float:
        """Evaluate security aspects"""
        reward = 0.0
        
        if result.status == ExecutionStatus.SECURITY_VIOLATION:
            reward += self.config.security_violation_penalty
        else:
            reward += self.config.security_bonus
        
        # No violations is good
        if not result.violations:
            reward += 1.0
        
        return reward
    
    def _apply_bonuses_and_penalties(
        self,
        breakdown: RewardBreakdown,
        api_calls: List[APICallInfo],
        code: str,
        result: ExecutionResult
    ):
        """Apply additional bonuses and penalties"""
        
        # Multiple successful API calls bonus
        successful_calls = [call for call in api_calls if call.success]
        if len(successful_calls) > 1:
            breakdown.bonus_points += 2.0
            breakdown.explanation.append(f"Bonus: {len(successful_calls)} successful API calls")
        
        # Comprehensive implementation bonus
        if (len(code.split('\n')) > 10 and 
            'try:' in code and 
            'requests.' in code):
            breakdown.bonus_points += self.config.code_quality_bonus
            breakdown.explanation.append("Bonus: Comprehensive implementation")
        
        # Timeout penalty
        if result.status == ExecutionStatus.TIMEOUT:
            breakdown.penalty_points += self.config.timeout_penalty
            breakdown.explanation.append("Penalty: Execution timeout")
    
    def _apply_multipliers(
        self,
        breakdown: RewardBreakdown,
        api_calls: List[APICallInfo],
        code: str
    ):
        """Apply reward multipliers"""
        
        multiplier = 1.0
        
        # Multiple endpoints multiplier
        unique_urls = set(call.url for call in api_calls)
        if len(unique_urls) > 1:
            multiplier *= self.config.multiple_endpoints_multiplier
            breakdown.multipliers['multiple_endpoints'] = self.config.multiple_endpoints_multiplier
        
        # Authentication multiplier
        if any(keyword in code.lower() for keyword in ['authorization', 'bearer', 'api_key']):
            multiplier *= self.config.authentication_multiplier
            breakdown.multipliers['authentication'] = self.config.authentication_multiplier
        
        # Error handling multiplier
        if 'try:' in code and 'except' in code:
            multiplier *= self.config.comprehensive_error_handling_multiplier
            breakdown.multipliers['error_handling'] = self.config.comprehensive_error_handling_multiplier
        
        # Apply multiplier to total
        if multiplier > 1.0:
            original_total = breakdown.total_reward
            breakdown.total_reward *= multiplier
            breakdown.explanation.append(
                f"Applied {multiplier:.1f}x multiplier (was {original_total:.1f}, now {breakdown.total_reward:.1f})"
            )
    
    def _generate_explanation(
        self,
        breakdown: RewardBreakdown,
        api_calls: List[APICallInfo],
        result: ExecutionResult
    ):
        """Generate human-readable explanation of reward calculation"""
        
        explanation = []
        
        # Component breakdown
        for component, score in breakdown.component_scores.items():
            if score != 0:
                explanation.append(f"{component.value}: {score:.1f} points")
        
        # API call summary
        if api_calls:
            successful_calls = [call for call in api_calls if call.success]
            explanation.append(
                f"API calls: {len(successful_calls)}/{len(api_calls)} successful"
            )
            
            for call in successful_calls[:3]:  # Show first 3
                explanation.append(f"  • {call.method} {call.url} -> {call.status_code}")
        
        # Execution summary
        explanation.append(f"Execution: {result.status.value} in {result.execution_time:.2f}s")
        
        breakdown.explanation.extend(explanation)


class BinaryRewardEvaluator(BaseRewardEvaluator):
    """Simple binary reward evaluator (works/doesn't work)"""
    
    def __init__(self):
        super().__init__(RewardConfig(
            execution_success_reward=1.0,
            api_call_success_reward=1.0,
            error_penalty=-1.0,
            timeout_penalty=-1.0,
            security_violation_penalty=-1.0
        ))
    
    def evaluate_simple(
        self,
        code: str,
        execution_result: ExecutionResult
    ) -> float:
        """Simple binary evaluation: 1.0 if works, 0.0 if doesn't"""
        
        if execution_result.status != ExecutionStatus.SUCCESS:
            return 0.0
        
        # Check if any API calls were made successfully
        api_calls = self.call_tracker.analyze_execution_output(
            execution_result.stdout,
            execution_result.stderr,
            code
        )
        
        successful_calls = [call for call in api_calls if call.success]
        
        return 1.0 if successful_calls else 0.0