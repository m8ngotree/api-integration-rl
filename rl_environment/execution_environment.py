import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

from .code_executor import (
    BaseCodeExecutor, SubprocessCodeExecutor, MockServerIntegratedExecutor,
    ExecutionResult, ExecutionStatus, SecurityPolicy, CodeExecutionManager
)
from .docker_executor import DockerCodeExecutor, DockerServerIntegratedExecutor
from ..mock_servers.server_manager import ServerManager
from ..mock_servers.schema_server import SchemaBasedMockServer


class EnvironmentType(Enum):
    SUBPROCESS = "subprocess"
    DOCKER = "docker"
    SUBPROCESS_WITH_SERVER = "subprocess_with_server"
    DOCKER_WITH_SERVER = "docker_with_server"


@dataclass
class ExecutionEnvironmentConfig:
    """Configuration for execution environment"""
    environment_type: EnvironmentType = EnvironmentType.SUBPROCESS
    security_policy: Optional[SecurityPolicy] = None
    timeout: int = 30
    max_memory_mb: int = 128
    enable_networking: bool = True
    mock_server_port: int = 8000
    docker_image: str = "python:3.11-slim"
    additional_packages: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    custom_setup_script: str = ""


@dataclass 
class TestCase:
    """Individual test case for code execution"""
    name: str
    code: str
    expected_output: Optional[str] = None
    expected_status: ExecutionStatus = ExecutionStatus.SUCCESS
    timeout: int = 30
    setup_code: str = ""
    teardown_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of test cases"""
    name: str
    test_cases: List[TestCase]
    setup_code: str = ""
    teardown_code: str = ""
    server_endpoints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafeExecutionEnvironment:
    """
    Comprehensive safe execution environment that supports multiple execution modes,
    mock server integration, and extensive security measures.
    """
    
    def __init__(self, config: ExecutionEnvironmentConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize security policy
        if not config.security_policy:
            self.config.security_policy = self._create_default_security_policy()
        
        # Initialize components
        self.execution_manager = CodeExecutionManager()
        self.server_manager = ServerManager()
        self.mock_server: Optional[SchemaBasedMockServer] = None
        
        self.executor: Optional[BaseCodeExecutor] = None
        self._execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timeouts': 0,
            'security_violations': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for execution environment"""
        logger = logging.getLogger(f"{__name__}.SafeExecutionEnvironment")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_default_security_policy(self) -> SecurityPolicy:
        """Create default security policy"""
        return SecurityPolicy(
            max_memory_mb=self.config.max_memory_mb,
            max_execution_time=self.config.timeout,
            allow_network=self.config.enable_networking,
            allow_subprocess=False
        )
    
    async def initialize(self):
        """Initialize the execution environment"""
        self.logger.info(f"Initializing execution environment: {self.config.environment_type.value}")
        
        try:
            # Initialize executor based on type
            if self.config.environment_type == EnvironmentType.SUBPROCESS:
                self.executor = self._create_subprocess_executor()
            
            elif self.config.environment_type == EnvironmentType.DOCKER:
                self.executor = self._create_docker_executor()
            
            elif self.config.environment_type == EnvironmentType.SUBPROCESS_WITH_SERVER:
                await self._setup_mock_server()
                self.executor = self._create_subprocess_server_executor()
            
            elif self.config.environment_type == EnvironmentType.DOCKER_WITH_SERVER:
                await self._setup_mock_server()
                self.executor = self._create_docker_server_executor()
            
            self.logger.info("Execution environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution environment: {e}")
            raise RuntimeError(f"Environment initialization failed: {e}")
    
    def _create_subprocess_executor(self) -> SubprocessCodeExecutor:
        """Create subprocess executor"""
        return SubprocessCodeExecutor(
            security_policy=self.config.security_policy
        )
    
    def _create_docker_executor(self) -> DockerCodeExecutor:
        """Create Docker executor"""
        executor = DockerCodeExecutor(
            image_name=self.config.docker_image,
            security_policy=self.config.security_policy
        )
        
        # Create custom image if additional packages specified
        if self.config.additional_packages:
            custom_image = executor.create_custom_image(
                base_image=self.config.docker_image,
                additional_packages=self.config.additional_packages,
                custom_setup=self.config.custom_setup_script
            )
            executor.execution_image_name = custom_image
        
        return executor
    
    async def _setup_mock_server(self):
        """Setup mock server for API testing"""
        try:
            self.mock_server = self.server_manager.create_server(
                server_name="execution_env_server",
                port=self.config.mock_server_port,
                title="Execution Environment Mock Server"
            )
            
            # Start server in background
            self.server_manager.start_server("execution_env_server", background=True)
            
            # Wait for server to be ready
            await asyncio.sleep(2)
            
            self.logger.info(f"Mock server started on port {self.config.mock_server_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup mock server: {e}")
            raise
    
    def _create_subprocess_server_executor(self) -> MockServerIntegratedExecutor:
        """Create subprocess executor with server integration"""
        server_url = f"http://localhost:{self.config.mock_server_port}"
        return MockServerIntegratedExecutor(
            mock_server_url=server_url,
            security_policy=self.config.security_policy
        )
    
    def _create_docker_server_executor(self) -> DockerServerIntegratedExecutor:
        """Create Docker executor with server integration"""
        server_url = f"http://host.docker.internal:{self.config.mock_server_port}"
        return DockerServerIntegratedExecutor(
            mock_server_url=server_url,
            image_name=self.config.docker_image,
            security_policy=self.config.security_policy
        )
    
    async def execute_code(
        self,
        code: str,
        timeout: Optional[int] = None,
        environment_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute code safely with comprehensive monitoring"""
        
        if not self.executor:
            raise RuntimeError("Execution environment not initialized")
        
        # Update stats
        self._execution_stats['total_executions'] += 1
        
        # Prepare environment variables
        env_vars = self.config.environment_variables.copy()
        if environment_vars:
            env_vars.update(environment_vars)
        
        # Execute code
        execution_timeout = timeout or self.config.timeout
        
        try:
            if hasattr(self.executor, 'execute_code_with_server'):
                result = self.executor.execute_code_with_server(
                    code=code,
                    timeout=execution_timeout
                )
            else:
                result = self.executor.execute_code(
                    code=code,
                    timeout=execution_timeout,
                    environment_vars=env_vars
                )
            
            # Update statistics
            self._update_execution_stats(result)
            
            # Add environment metadata
            result.metadata.update({
                'environment_type': self.config.environment_type.value,
                'execution_id': f"exec_{int(time.time())}",
                'total_executions': self._execution_stats['total_executions']
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            self._execution_stats['failed_executions'] += 1
            
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Execution environment error: {str(e)}"
            )
    
    async def run_test_case(self, test_case: TestCase) -> ExecutionResult:
        """Run a single test case"""
        
        # Prepare full code with setup and teardown
        full_code = ""
        
        if test_case.setup_code:
            full_code += f"# Setup code\n{test_case.setup_code}\n\n"
        
        full_code += f"# Test code\n{test_case.code}\n\n"
        
        if test_case.teardown_code:
            full_code += f"# Teardown code\n{test_case.teardown_code}\n"
        
        # Execute test case
        result = await self.execute_code(
            code=full_code,
            timeout=test_case.timeout
        )
        
        # Add test metadata
        result.metadata.update({
            'test_name': test_case.name,
            'expected_status': test_case.expected_status.value,
            'test_metadata': test_case.metadata
        })
        
        # Validate result against expectations
        validation_result = self._validate_test_result(test_case, result)
        result.metadata['validation'] = validation_result
        
        return result
    
    async def run_test_suite(self, test_suite: TestSuite) -> Dict[str, ExecutionResult]:
        """Run a complete test suite"""
        
        self.logger.info(f"Running test suite: {test_suite.name}")
        
        results = {}
        
        # Load server endpoints if provided
        if test_suite.server_endpoints and self.mock_server:
            for endpoint_spec in test_suite.server_endpoints:
                # Convert dict to EndpointSpec if needed
                pass  # Implementation depends on endpoint format
        
        # Run setup code if provided
        if test_suite.setup_code:
            setup_result = await self.execute_code(test_suite.setup_code)
            results['__setup__'] = setup_result
            
            if setup_result.status != ExecutionStatus.SUCCESS:
                self.logger.error("Test suite setup failed")
                return results
        
        # Run individual test cases
        for test_case in test_suite.test_cases:
            try:
                result = await self.run_test_case(test_case)
                results[test_case.name] = result
                
                self.logger.info(
                    f"Test {test_case.name}: {result.status.value} "
                    f"({result.execution_time:.2f}s)"
                )
                
            except Exception as e:
                self.logger.error(f"Test {test_case.name} failed with exception: {e}")
                results[test_case.name] = ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    stderr=f"Test execution error: {str(e)}"
                )
        
        # Run teardown code if provided
        if test_suite.teardown_code:
            teardown_result = await self.execute_code(test_suite.teardown_code)
            results['__teardown__'] = teardown_result
        
        # Generate summary
        summary = self._generate_test_summary(results)
        results['__summary__'] = summary
        
        self.logger.info(f"Test suite completed: {summary['passed']}/{summary['total']} tests passed")
        
        return results
    
    def _validate_test_result(self, test_case: TestCase, result: ExecutionResult) -> Dict[str, Any]:
        """Validate test result against expectations"""
        
        validation = {
            'status_match': result.status == test_case.expected_status,
            'output_match': None,
            'overall_success': False
        }
        
        # Check output if expected output is provided
        if test_case.expected_output is not None:
            validation['output_match'] = test_case.expected_output in result.stdout
        
        # Overall success
        validation['overall_success'] = (
            validation['status_match'] and
            (validation['output_match'] is None or validation['output_match'])
        )
        
        return validation
    
    def _generate_test_summary(self, results: Dict[str, ExecutionResult]) -> Dict[str, Any]:
        """Generate test suite summary"""
        
        # Filter out special results
        test_results = {
            name: result for name, result in results.items()
            if not name.startswith('__')
        }
        
        total = len(test_results)
        passed = sum(
            1 for result in test_results.values()
            if result.metadata.get('validation', {}).get('overall_success', False)
        )
        failed = total - passed
        
        avg_execution_time = sum(
            result.execution_time for result in test_results.values()
        ) / total if total > 0 else 0
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total if total > 0 else 0,
            'average_execution_time': avg_execution_time,
            'total_execution_time': sum(
                result.execution_time for result in test_results.values()
            )
        }
    
    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        
        if result.status == ExecutionStatus.SUCCESS:
            self._execution_stats['successful_executions'] += 1
        elif result.status == ExecutionStatus.TIMEOUT:
            self._execution_stats['timeouts'] += 1
        elif result.status == ExecutionStatus.SECURITY_VIOLATION:
            self._execution_stats['security_violations'] += 1
        else:
            self._execution_stats['failed_executions'] += 1
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self._execution_stats.copy()
        
        if stats['total_executions'] > 0:
            stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
            stats['failure_rate'] = stats['failed_executions'] / stats['total_executions']
            stats['timeout_rate'] = stats['timeouts'] / stats['total_executions']
            stats['violation_rate'] = stats['security_violations'] / stats['total_executions']
        else:
            stats.update({
                'success_rate': 0,
                'failure_rate': 0,
                'timeout_rate': 0,
                'violation_rate': 0
            })
        
        return stats
    
    def export_execution_log(self, filename: str):
        """Export execution log to file"""
        
        log_data = {
            'environment_config': {
                'type': self.config.environment_type.value,
                'timeout': self.config.timeout,
                'max_memory_mb': self.config.max_memory_mb,
                'enable_networking': self.config.enable_networking
            },
            'statistics': self.get_execution_stats(),
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"Execution log exported to: {filename}")
    
    async def cleanup(self):
        """Clean up all resources"""
        
        self.logger.info("Cleaning up execution environment")
        
        # Cleanup executor
        if self.executor:
            self.executor.cleanup()
        
        # Cleanup execution manager
        self.execution_manager.cleanup_all()
        
        # Cleanup mock server
        if self.mock_server:
            self.server_manager.shutdown_all_servers()
        
        self.logger.info("Cleanup completed")
    
    @asynccontextmanager
    async def temporary_environment(self):
        """Context manager for temporary execution environment"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()


class ExecutionEnvironmentFactory:
    """Factory for creating execution environments with common configurations"""
    
    @staticmethod
    def create_basic_environment() -> SafeExecutionEnvironment:
        """Create basic subprocess execution environment"""
        config = ExecutionEnvironmentConfig(
            environment_type=EnvironmentType.SUBPROCESS,
            timeout=30,
            max_memory_mb=64
        )
        return SafeExecutionEnvironment(config)
    
    @staticmethod
    def create_docker_environment() -> SafeExecutionEnvironment:
        """Create Docker-based execution environment"""
        config = ExecutionEnvironmentConfig(
            environment_type=EnvironmentType.DOCKER,
            timeout=45,
            max_memory_mb=128,
            docker_image="python:3.11-slim"
        )
        return SafeExecutionEnvironment(config)
    
    @staticmethod
    def create_api_testing_environment(server_port: int = 8001) -> SafeExecutionEnvironment:
        """Create environment optimized for API testing"""
        config = ExecutionEnvironmentConfig(
            environment_type=EnvironmentType.DOCKER_WITH_SERVER,
            timeout=60,
            max_memory_mb=256,
            enable_networking=True,
            mock_server_port=server_port,
            additional_packages=["requests", "httpx", "pytest"]
        )
        return SafeExecutionEnvironment(config)
    
    @staticmethod
    def create_secure_environment() -> SafeExecutionEnvironment:
        """Create high-security execution environment"""
        security_policy = SecurityPolicy(
            max_memory_mb=64,
            max_execution_time=15,
            allow_network=False,
            allow_subprocess=False,
            max_file_operations=0
        )
        
        config = ExecutionEnvironmentConfig(
            environment_type=EnvironmentType.DOCKER,
            security_policy=security_policy,
            timeout=20,
            max_memory_mb=64,
            enable_networking=False
        )
        
        return SafeExecutionEnvironment(config)
    
    @staticmethod
    def create_rl_training_environment() -> SafeExecutionEnvironment:
        """Create environment optimized for RL training"""
        config = ExecutionEnvironmentConfig(
            environment_type=EnvironmentType.SUBPROCESS_WITH_SERVER,
            timeout=45,
            max_memory_mb=128,
            enable_networking=True,
            mock_server_port=8002,
            environment_variables={
                'RL_TRAINING_MODE': 'true',
                'EXECUTION_ENV': 'training'
            }
        )
        return SafeExecutionEnvironment(config)