import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .reward_system import (
    BaseRewardEvaluator, BinaryRewardEvaluator, RewardBreakdown, 
    RewardConfig, RewardComponent, APICallInfo
)
from .execution_environment import SafeExecutionEnvironment, TestCase, TestSuite
from .code_executor import ExecutionResult, ExecutionStatus
from ..data_generation.endpoint_generator import EndpointSpec
from ..utilities.code_template_generator import CodeTemplate, CodeGap, MissingComponent


class EvaluationMode(Enum):
    BINARY = "binary"              # Simple pass/fail
    DETAILED = "detailed"          # Comprehensive scoring
    PROGRESSIVE = "progressive"    # Difficulty-adjusted scoring
    COMPETITIVE = "competitive"    # Relative scoring against benchmarks


@dataclass
class EvaluationContext:
    """Context information for code evaluation"""
    template: Optional[CodeTemplate] = None
    expected_endpoints: List[EndpointSpec] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    time_limit: int = 30
    expected_api_calls: int = 1
    allow_partial_credit: bool = True
    custom_validators: List[Callable] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation result with rewards and analysis"""
    reward_breakdown: RewardBreakdown
    execution_result: ExecutionResult
    api_calls_detected: List[APICallInfo]
    gaps_completed: List[MissingComponent] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeRewardIntegrator:
    """
    Integrates reward evaluation with code execution environment
    for comprehensive RL training feedback
    """
    
    def __init__(
        self,
        execution_env: SafeExecutionEnvironment,
        evaluation_mode: EvaluationMode = EvaluationMode.DETAILED
    ):
        self.execution_env = execution_env
        self.evaluation_mode = evaluation_mode
        
        # Initialize evaluators
        self.binary_evaluator = BinaryRewardEvaluator()
        self.detailed_evaluator = BaseRewardEvaluator()
        self.progressive_evaluator = BaseRewardEvaluator(self._create_progressive_config())
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        self.performance_trends: Dict[str, List[float]] = {}
        
        # Benchmarks for competitive evaluation
        self.benchmarks: Dict[str, float] = {}
    
    def _create_progressive_config(self) -> RewardConfig:
        """Create configuration that adjusts rewards based on difficulty"""
        return RewardConfig(
            execution_success_reward=5.0,  # Lower base rewards
            api_call_success_reward=3.0,
            correct_method_reward=2.0,
            response_handling_reward=3.0,
            error_handling_reward=4.0,
            performance_bonus=1.0,
            code_quality_bonus=1.0
        )
    
    async def evaluate_code_completion(
        self,
        completed_code: str,
        context: EvaluationContext
    ) -> EvaluationResult:
        """
        Evaluate completed code against template and context requirements
        """
        
        # Execute the code
        execution_result = await self.execution_env.execute_code(
            code=completed_code,
            timeout=context.time_limit
        )
        
        # Choose evaluator based on mode
        evaluator = self._get_evaluator_for_mode()
        
        # Perform reward evaluation
        reward_breakdown = evaluator.evaluate_code_execution(
            code=completed_code,
            execution_result=execution_result,
            expected_endpoints=context.expected_endpoints
        )
        
        # Detect API calls
        api_calls = evaluator.call_tracker.analyze_execution_output(
            execution_result.stdout,
            execution_result.stderr,
            completed_code
        )
        
        # Analyze code completion if template provided
        gaps_completed = []
        if context.template:
            gaps_completed = self._analyze_gap_completion(
                completed_code,
                context.template
            )
        
        # Run custom validators
        validation_results = {}
        for validator in context.custom_validators:
            try:
                validator_name = validator.__name__
                validation_results[validator_name] = validator(
                    completed_code, execution_result, api_calls
                )
            except Exception as e:
                validation_results[validator.__name__] = f"Validator error: {e}"
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            execution_result, api_calls, context
        )
        
        # Generate suggestions for improvement
        suggestions = self._generate_improvement_suggestions(
            completed_code, execution_result, api_calls, context
        )
        
        # Apply context-specific adjustments
        self._apply_context_adjustments(
            reward_breakdown, context, gaps_completed, performance_metrics
        )
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            reward_breakdown=reward_breakdown,
            execution_result=execution_result,
            api_calls_detected=api_calls,
            gaps_completed=gaps_completed,
            validation_results=validation_results,
            performance_metrics=performance_metrics,
            suggestions=suggestions,
            metadata={
                'evaluation_mode': self.evaluation_mode.value,
                'context': {
                    'difficulty_level': context.difficulty_level,
                    'expected_endpoints': len(context.expected_endpoints),
                    'time_limit': context.time_limit
                },
                'timestamp': time.time()
            }
        )
        
        # Update history and trends
        self._update_evaluation_history(evaluation_result)
        
        return evaluation_result
    
    async def evaluate_test_suite_completion(
        self,
        completed_code: str,
        test_suite: TestSuite,
        context: EvaluationContext
    ) -> EvaluationResult:
        """
        Evaluate code by running it against a test suite
        """
        
        # Create enhanced test suite with the completed code
        enhanced_suite = TestSuite(
            name=f"Evaluation: {test_suite.name}",
            setup_code=f"{test_suite.setup_code}\n\n# User's completed code:\n{completed_code}",
            test_cases=test_suite.test_cases,
            teardown_code=test_suite.teardown_code
        )
        
        # Run the test suite
        test_results = await self.execution_env.run_test_suite(enhanced_suite)
        
        # Combine all execution results
        combined_stdout = ""
        combined_stderr = ""
        total_execution_time = 0.0
        overall_status = ExecutionStatus.SUCCESS
        
        for test_name, result in test_results.items():
            if not test_name.startswith('__'):
                combined_stdout += f"\n=== {test_name} ===\n{result.stdout}"
                combined_stderr += f"\n=== {test_name} ===\n{result.stderr}"
                total_execution_time += result.execution_time
                
                if result.status != ExecutionStatus.SUCCESS:
                    overall_status = result.status
        
        # Create combined execution result
        combined_result = ExecutionResult(
            status=overall_status,
            stdout=combined_stdout,
            stderr=combined_stderr,
            execution_time=total_execution_time,
            metadata={'test_results': test_results}
        )
        
        # Evaluate using standard process
        evaluation_result = await self.evaluate_code_completion(
            completed_code, context
        )
        
        # Enhance with test suite specific metrics
        summary = test_results.get('__summary__', {})
        evaluation_result.performance_metrics.update({
            'tests_passed': summary.get('passed', 0),
            'tests_failed': summary.get('failed', 0),
            'test_success_rate': summary.get('success_rate', 0.0),
            'total_test_time': summary.get('total_execution_time', 0.0)
        })
        
        # Add test suite bonus to reward
        if summary.get('success_rate', 0) > 0.8:  # 80% pass rate
            evaluation_result.reward_breakdown.bonus_points += 5.0
            evaluation_result.reward_breakdown.explanation.append(
                f"Test suite bonus: {summary.get('success_rate', 0):.1%} pass rate"
            )
        
        evaluation_result.reward_breakdown.total_reward += evaluation_result.reward_breakdown.bonus_points
        
        return evaluation_result
    
    def _get_evaluator_for_mode(self) -> BaseRewardEvaluator:
        """Get appropriate evaluator for current mode"""
        if self.evaluation_mode == EvaluationMode.BINARY:
            return self.binary_evaluator
        elif self.evaluation_mode == EvaluationMode.PROGRESSIVE:
            return self.progressive_evaluator
        else:
            return self.detailed_evaluator
    
    def _analyze_gap_completion(
        self,
        completed_code: str,
        template: CodeTemplate
    ) -> List[MissingComponent]:
        """Analyze which template gaps were successfully completed"""
        
        completed_gaps = []
        
        for gap in template.gaps:
            # Check if the gap placeholder was replaced
            if gap.placeholder not in completed_code:
                # Gap was replaced, now check if it was implemented correctly
                component_implemented = self._check_component_implementation(
                    completed_code, gap.component
                )
                
                if component_implemented:
                    completed_gaps.append(gap.component)
        
        return completed_gaps
    
    def _check_component_implementation(
        self,
        code: str,
        component: MissingComponent
    ) -> bool:
        """Check if a specific component was implemented correctly"""
        
        implementation_patterns = {
            MissingComponent.IMPORTS: [
                r'import\s+requests',
                r'import\s+json',
                r'from\s+typing'
            ],
            MissingComponent.AUTHENTICATION: [
                r'Authorization',
                r'Bearer',
                r'api_key',
                r'headers\[.*auth'
            ],
            MissingComponent.ERROR_HANDLING: [
                r'try\s*:',
                r'except\s+',
                r'raise_for_status'
            ],
            MissingComponent.HTTP_METHOD: [
                r'requests\.(get|post|put|delete)',
                r'response\s*=.*requests\.'
            ],
            MissingComponent.RESPONSE_PARSING: [
                r'\.json\(\)',
                r'response\.text',
                r'response\.content'
            ],
            MissingComponent.URL_CONSTRUCTION: [
                r'base_url\s*\+',
                r'f[\'\"]\{.*\}[\'\"]*',
                r'url\s*=.*format'
            ]
        }
        
        patterns = implementation_patterns.get(component, [])
        
        import re
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        
        return False
    
    def _calculate_performance_metrics(
        self,
        execution_result: ExecutionResult,
        api_calls: List[APICallInfo],
        context: EvaluationContext
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        metrics = {}
        
        # Basic execution metrics
        metrics['execution_time'] = execution_result.execution_time
        metrics['memory_usage_mb'] = execution_result.memory_usage
        
        # API call metrics
        successful_calls = [call for call in api_calls if call.success]
        metrics['api_calls_made'] = len(api_calls)
        metrics['api_calls_successful'] = len(successful_calls)
        metrics['api_success_rate'] = (
            len(successful_calls) / len(api_calls) if api_calls else 0.0
        )
        
        # Expected vs actual
        metrics['expected_calls_ratio'] = (
            len(successful_calls) / context.expected_api_calls
            if context.expected_api_calls > 0 else 1.0
        )
        
        # Time efficiency
        if context.time_limit > 0:
            metrics['time_efficiency'] = 1.0 - (execution_result.execution_time / context.time_limit)
        
        # Error rate
        metrics['has_errors'] = 1.0 if execution_result.stderr else 0.0
        
        return metrics
    
    def _generate_improvement_suggestions(
        self,
        code: str,
        execution_result: ExecutionResult,
        api_calls: List[APICallInfo],
        context: EvaluationContext
    ) -> List[str]:
        """Generate suggestions for code improvement"""
        
        suggestions = []
        
        # Execution-based suggestions
        if execution_result.status == ExecutionStatus.TIMEOUT:
            suggestions.append("Consider optimizing code for better performance")
        
        if execution_result.status == ExecutionStatus.ERROR:
            suggestions.append("Add proper error handling to prevent runtime errors")
        
        if execution_result.memory_usage > 100:
            suggestions.append("Consider memory optimization - current usage is high")
        
        # API call suggestions
        if not api_calls:
            suggestions.append("No API calls detected - ensure you're making HTTP requests")
        
        unsuccessful_calls = [call for call in api_calls if not call.success]
        if unsuccessful_calls:
            suggestions.append(f"{len(unsuccessful_calls)} API calls failed - check error handling")
        
        # Code structure suggestions
        if 'try:' not in code:
            suggestions.append("Add try-catch blocks for better error handling")
        
        if '.json()' not in code and 'json' in code:
            suggestions.append("Make sure to parse JSON responses properly")
        
        if len(context.expected_endpoints) > len(api_calls):
            suggestions.append(f"Expected {len(context.expected_endpoints)} endpoints, only found {len(api_calls)}")
        
        # Authentication suggestions
        auth_keywords = ['authorization', 'bearer', 'api_key', 'token']
        if not any(keyword in code.lower() for keyword in auth_keywords):
            suggestions.append("Consider adding authentication to your API requests")
        
        return suggestions
    
    def _apply_context_adjustments(
        self,
        reward_breakdown: RewardBreakdown,
        context: EvaluationContext,
        gaps_completed: List[MissingComponent],
        performance_metrics: Dict[str, float]
    ):
        """Apply context-specific reward adjustments"""
        
        # Difficulty-based adjustments
        difficulty_multipliers = {
            'beginner': 1.2,
            'intermediate': 1.0,
            'advanced': 0.8
        }
        
        difficulty_multiplier = difficulty_multipliers.get(context.difficulty_level, 1.0)
        if difficulty_multiplier != 1.0:
            reward_breakdown.total_reward *= difficulty_multiplier
            reward_breakdown.multipliers['difficulty'] = difficulty_multiplier
        
        # Gap completion bonus
        if gaps_completed and context.template:
            completion_rate = len(gaps_completed) / len(context.template.gaps)
            gap_bonus = completion_rate * 3.0
            reward_breakdown.bonus_points += gap_bonus
            reward_breakdown.explanation.append(
                f"Gap completion bonus: {completion_rate:.1%} ({len(gaps_completed)}/{len(context.template.gaps)})"
            )
        
        # Performance-based adjustments
        if performance_metrics.get('time_efficiency', 0) > 0.8:
            reward_breakdown.bonus_points += 2.0
            reward_breakdown.explanation.append("Performance bonus: Efficient execution")
        
        # Partial credit handling
        if context.allow_partial_credit and reward_breakdown.total_reward < 0:
            # Don't let total reward go too negative
            reward_breakdown.total_reward = max(reward_breakdown.total_reward, -5.0)
    
    def _update_evaluation_history(self, result: EvaluationResult):
        """Update evaluation history and performance trends"""
        
        self.evaluation_history.append(result)
        
        # Update trends
        reward = result.reward_breakdown.total_reward
        if 'total_reward' not in self.performance_trends:
            self.performance_trends['total_reward'] = []
        self.performance_trends['total_reward'].append(reward)
        
        # Track component trends
        for component, score in result.reward_breakdown.component_scores.items():
            trend_key = f"{component.value}_score"
            if trend_key not in self.performance_trends:
                self.performance_trends[trend_key] = []
            self.performance_trends[trend_key].append(score)
        
        # Keep only last 100 evaluations
        if len(self.evaluation_history) > 100:
            self.evaluation_history.pop(0)
        
        for trend_list in self.performance_trends.values():
            if len(trend_list) > 100:
                trend_list.pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance over time"""
        
        if not self.evaluation_history:
            return {}
        
        recent_rewards = self.performance_trends.get('total_reward', [])
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'average_reward': sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0,
            'best_reward': max(recent_rewards) if recent_rewards else 0,
            'worst_reward': min(recent_rewards) if recent_rewards else 0,
            'recent_trend': self._calculate_trend(recent_rewards[-10:]) if len(recent_rewards) >= 10 else 0
        }
        
        # Component averages
        component_averages = {}
        for component in RewardComponent:
            trend_key = f"{component.value}_score"
            if trend_key in self.performance_trends:
                scores = self.performance_trends[trend_key]
                component_averages[component.value] = sum(scores) / len(scores) if scores else 0
        
        summary['component_averages'] = component_averages
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, where 1 is improving)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope to -1 to 1 range
        max_possible_slope = max(values) - min(values)
        if max_possible_slope == 0:
            return 0.0
        
        normalized_slope = slope / max_possible_slope
        return max(-1.0, min(1.0, normalized_slope))
    
    def create_custom_validator(
        self,
        name: str,
        validation_func: Callable[[str, ExecutionResult, List[APICallInfo]], Any]
    ) -> Callable:
        """Create a custom validator function"""
        
        def validator(code: str, result: ExecutionResult, api_calls: List[APICallInfo]) -> Any:
            return validation_func(code, result, api_calls)
        
        validator.__name__ = name
        return validator
    
    def set_benchmark(self, benchmark_name: str, score: float):
        """Set a benchmark score for competitive evaluation"""
        self.benchmarks[benchmark_name] = score
    
    def compare_to_benchmark(self, result: EvaluationResult, benchmark_name: str) -> Dict[str, Any]:
        """Compare evaluation result to a benchmark"""
        
        if benchmark_name not in self.benchmarks:
            return {'error': f'Benchmark {benchmark_name} not found'}
        
        benchmark_score = self.benchmarks[benchmark_name]
        current_score = result.reward_breakdown.total_reward
        
        return {
            'benchmark_score': benchmark_score,
            'current_score': current_score,
            'difference': current_score - benchmark_score,
            'percentage_improvement': ((current_score - benchmark_score) / benchmark_score * 100) if benchmark_score != 0 else 0,
            'better_than_benchmark': current_score > benchmark_score
        }