import os
import sys
import subprocess
import tempfile
import time
import signal
import shutil
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import psutil


class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"


class SecurityViolation(Exception):
    """Raised when code violates security policies"""
    pass


@dataclass
class ExecutionResult:
    """Result of code execution"""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    memory_usage: int = 0  # MB
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy for code execution"""
    allowed_imports: List[str] = field(default_factory=lambda: [
        "requests", "httpx", "json", "urllib", "http", 
        "typing", "dataclasses", "datetime", "time",
        "logging", "asyncio", "functools", "itertools",
        "collections", "re", "math", "random"
    ])
    blocked_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "shutil", "pathlib",
        "importlib", "exec", "eval", "compile", "open",
        "file", "__import__", "globals", "locals", "vars",
        "socket", "threading", "multiprocessing"
    ])
    max_file_operations: int = 0  # No file operations allowed by default
    allow_network: bool = True
    allow_subprocess: bool = False
    max_memory_mb: int = 128
    max_execution_time: int = 30


class CodeSecurityAnalyzer:
    """Analyzes code for security violations before execution"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
    
    def analyze_code(self, code: str) -> List[str]:
        """Analyze code for security violations"""
        violations = []
        
        # Check for blocked imports
        violations.extend(self._check_imports(code))
        
        # Check for dangerous functions
        violations.extend(self._check_dangerous_functions(code))
        
        # Check for file operations
        violations.extend(self._check_file_operations(code))
        
        # Check for subprocess operations
        violations.extend(self._check_subprocess_operations(code))
        
        return violations
    
    def _check_imports(self, code: str) -> List[str]:
        """Check for blocked imports"""
        violations = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check import statements
            if line.startswith('import ') or line.startswith('from '):
                # Extract module name
                if line.startswith('import '):
                    module = line.split('import ')[1].split()[0].split('.')[0]
                else:  # from ... import
                    module = line.split('from ')[1].split('import')[0].strip().split('.')[0]
                
                if module in self.policy.blocked_imports:
                    violations.append(f"Line {line_num}: Blocked import '{module}'")
                elif module not in self.policy.allowed_imports and not module.startswith('_'):
                    violations.append(f"Line {line_num}: Unknown/potentially unsafe import '{module}'")
        
        return violations
    
    def _check_dangerous_functions(self, code: str) -> List[str]:
        """Check for dangerous function calls"""
        violations = []
        dangerous_functions = [
            'exec', 'eval', 'compile', '__import__',
            'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr'
        ]
        
        for func in dangerous_functions:
            if func + '(' in code:
                violations.append(f"Dangerous function call: {func}")
        
        return violations
    
    def _check_file_operations(self, code: str) -> List[str]:
        """Check for file operations"""
        violations = []
        
        if not self.policy.max_file_operations:
            file_operations = ['open(', 'file(', 'with open', 'Path(']
            for op in file_operations:
                if op in code:
                    violations.append(f"File operation not allowed: {op}")
        
        return violations
    
    def _check_subprocess_operations(self, code: str) -> List[str]:
        """Check for subprocess operations"""
        violations = []
        
        if not self.policy.allow_subprocess:
            subprocess_patterns = [
                'subprocess.', 'os.system', 'os.popen',
                'commands.', 'popen'
            ]
            for pattern in subprocess_patterns:
                if pattern in code:
                    violations.append(f"Subprocess operation not allowed: {pattern}")
        
        return violations


class BaseCodeExecutor:
    """Base class for code execution environments"""
    
    def __init__(
        self,
        security_policy: Optional[SecurityPolicy] = None,
        working_dir: Optional[str] = None
    ):
        self.security_policy = security_policy or SecurityPolicy()
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="code_exec_")
        self.analyzer = CodeSecurityAnalyzer(self.security_policy)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for code executor"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def execute_code(
        self,
        code: str,
        timeout: int = 30,
        environment_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute Python code safely"""
        
        start_time = time.time()
        
        # Security analysis
        violations = self.analyzer.analyze_code(code)
        if violations:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                violations=violations,
                execution_time=time.time() - start_time
            )
        
        try:
            # Execute code in isolated environment
            result = self._execute_isolated(code, timeout, environment_vars)
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"Code execution completed: {result.status.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Code execution failed: {str(e)}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=str(e),
                execution_time=time.time() - start_time
            )
    
    def _execute_isolated(
        self,
        code: str,
        timeout: int,
        environment_vars: Optional[Dict[str, str]]
    ) -> ExecutionResult:
        """Execute code in isolated environment - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_isolated")
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup working directory: {e}")


class SubprocessCodeExecutor(BaseCodeExecutor):
    """Code executor using subprocess isolation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.python_executable = sys.executable
    
    def _execute_isolated(
        self,
        code: str,
        timeout: int,
        environment_vars: Optional[Dict[str, str]]
    ) -> ExecutionResult:
        """Execute code in subprocess with security restrictions"""
        
        # Create temporary Python file
        code_file = os.path.join(self.working_dir, "code_to_execute.py")
        
        # Wrap code with security measures
        wrapped_code = self._wrap_code_with_security(code)
        
        with open(code_file, 'w') as f:
            f.write(wrapped_code)
        
        # Prepare environment
        env = os.environ.copy()
        if environment_vars:
            env.update(environment_vars)
        
        # Restrict environment
        env['PYTHONDONTWRITEBYTECODE'] = '1'  # No .pyc files
        env['PYTHONPATH'] = ''  # Clear Python path
        
        # Execute subprocess with resource limits
        try:
            process = subprocess.Popen(
                [self.python_executable, code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
                text=True,
                preexec_fn=self._setup_subprocess_limits if os.name != 'nt' else None
            )
            
            # Monitor process with timeout and resource limits
            result = self._monitor_process(process, timeout)
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Subprocess execution failed: {str(e)}"
            )
    
    def _wrap_code_with_security(self, code: str) -> str:
        """Wrap user code with security monitoring"""
        
        security_wrapper = f'''
import sys
import signal
import resource
import time
from typing import Dict, Any

# Security monitoring
class SecurityMonitor:
    def __init__(self, max_memory_mb={self.security_policy.max_memory_mb}):
        self.max_memory_mb = max_memory_mb
        self.start_time = time.time()
    
    def check_memory(self):
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.max_memory_mb:
                print(f"Memory limit exceeded: {{memory_mb:.1f}}MB > {{self.max_memory_mb}}MB", file=sys.stderr)
                sys.exit(1)
        except ImportError:
            pass  # psutil not available
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Install security monitor
monitor = SecurityMonitor()

# Set up signal handler for timeout
def timeout_handler(signum, frame):
    print("Execution timed out", file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.security_policy.max_execution_time})

try:
    with monitor:
        monitor.check_memory()
        
        # User code starts here
{self._indent_code(code, 8)}
        
        # User code ends here
        monitor.check_memory()

except KeyboardInterrupt:
    print("Execution interrupted", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Execution error: {{str(e)}}", file=sys.stderr)
    sys.exit(1)
finally:
    signal.alarm(0)  # Cancel alarm
'''
        
        return security_wrapper
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces"""
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in code.split('\n'))
    
    def _setup_subprocess_limits(self):
        """Set up resource limits for subprocess (Unix only)"""
        try:
            import resource
            
            # Memory limit
            memory_limit = self.security_policy.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # CPU time limit
            cpu_limit = self.security_policy.max_execution_time
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # No core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            
            # Limit number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            
        except ImportError:
            pass  # resource module not available
    
    def _monitor_process(self, process: subprocess.Popen, timeout: int) -> ExecutionResult:
        """Monitor process execution with timeout and resource tracking"""
        
        start_time = time.time()
        max_memory = 0
        
        try:
            # Monitor process
            while process.poll() is None:
                elapsed = time.time() - start_time
                
                if elapsed > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        stderr=f"Execution timed out after {timeout} seconds",
                        execution_time=elapsed
                    )
                
                # Check memory usage
                try:
                    proc = psutil.Process(process.pid)
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, memory_mb)
                    
                    if memory_mb > self.security_policy.max_memory_mb:
                        process.terminate()
                        return ExecutionResult(
                            status=ExecutionStatus.RESOURCE_LIMIT_EXCEEDED,
                            stderr=f"Memory limit exceeded: {memory_mb:.1f}MB",
                            memory_usage=int(memory_mb)
                        )
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                time.sleep(0.1)  # Small delay
            
            # Get output
            stdout, stderr = process.communicate()
            
            # Determine status
            if process.returncode == 0:
                status = ExecutionStatus.SUCCESS
            else:
                status = ExecutionStatus.ERROR
            
            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                memory_usage=int(max_memory),
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Process monitoring failed: {str(e)}",
                execution_time=time.time() - start_time
            )


class MockServerIntegratedExecutor(SubprocessCodeExecutor):
    """Code executor with mock server integration"""
    
    def __init__(self, mock_server_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self.mock_server_url = mock_server_url
        self.server_process = None
    
    def execute_code_with_server(
        self,
        code: str,
        server_endpoints: Optional[List[Dict[str, Any]]] = None,
        timeout: int = 30
    ) -> ExecutionResult:
        """Execute code with mock server available"""
        
        # Prepare environment variables for server
        env_vars = {}
        if self.mock_server_url:
            env_vars['API_BASE_URL'] = self.mock_server_url
            env_vars['MOCK_SERVER_URL'] = self.mock_server_url
        
        # Add server connection code to the user code
        enhanced_code = self._enhance_code_with_server_access(code)
        
        # Execute with server environment
        result = self.execute_code(
            code=enhanced_code,
            timeout=timeout,
            environment_vars=env_vars
        )
        
        # Add server interaction metadata
        result.metadata['server_url'] = self.mock_server_url
        result.metadata['server_available'] = self.mock_server_url is not None
        
        return result
    
    def _enhance_code_with_server_access(self, code: str) -> str:
        """Enhance user code with server connection utilities"""
        
        server_utils = f'''
# Mock server utilities (automatically added)
import os

# Server configuration
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000')
MOCK_SERVER_URL = os.environ.get('MOCK_SERVER_URL', 'http://localhost:8000')

def get_api_base_url():
    """Get the base URL for API calls"""
    return API_BASE_URL

def test_server_connection():
    """Test if mock server is available"""
    try:
        import requests
        response = requests.get(f"{{API_BASE_URL}}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# Test server connection
if test_server_connection():
    print(f"✅ Mock server available at {{API_BASE_URL}}")
else:
    print(f"⚠️  Mock server not available at {{API_BASE_URL}}")

# User code starts here
{code}
'''
        
        return server_utils
    
    def test_api_integration(
        self,
        api_client_code: str,
        test_cases: List[Dict[str, Any]],
        timeout: int = 60
    ) -> Dict[str, ExecutionResult]:
        """Test API integration code with multiple test cases"""
        
        results = {}
        
        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'test_{i}')
            test_code = test_case.get('code', '')
            expected_result = test_case.get('expected', None)
            
            # Combine API client code with test code
            full_code = f'''
{api_client_code}

# Test case: {test_name}
try:
    {self._indent_code(test_code, 4)}
    print("✅ Test passed: {test_name}")
except Exception as e:
    print(f"❌ Test failed: {test_name} - {{str(e)}}")
    import traceback
    traceback.print_exc()
'''
            
            result = self.execute_code_with_server(
                code=full_code,
                timeout=timeout
            )
            
            result.metadata['test_name'] = test_name
            result.metadata['expected_result'] = expected_result
            results[test_name] = result
        
        return results


class CodeExecutionManager:
    """Manages multiple code execution environments"""
    
    def __init__(self):
        self.executors: Dict[str, BaseCodeExecutor] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_executor(
        self,
        executor_id: str,
        executor_type: str = "subprocess",
        security_policy: Optional[SecurityPolicy] = None,
        **kwargs
    ) -> BaseCodeExecutor:
        """Create a new code executor"""
        
        if executor_type == "subprocess":
            executor = SubprocessCodeExecutor(
                security_policy=security_policy,
                **kwargs
            )
        elif executor_type == "mock_server":
            executor = MockServerIntegratedExecutor(
                security_policy=security_policy,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
        
        self.executors[executor_id] = executor
        self.logger.info(f"Created {executor_type} executor: {executor_id}")
        
        return executor
    
    def get_executor(self, executor_id: str) -> Optional[BaseCodeExecutor]:
        """Get executor by ID"""
        return self.executors.get(executor_id)
    
    def remove_executor(self, executor_id: str) -> bool:
        """Remove and cleanup executor"""
        if executor_id in self.executors:
            executor = self.executors[executor_id]
            executor.cleanup()
            del self.executors[executor_id]
            self.logger.info(f"Removed executor: {executor_id}")
            return True
        return False
    
    def cleanup_all(self):
        """Cleanup all executors"""
        for executor_id in list(self.executors.keys()):
            self.remove_executor(executor_id)
    
    def execute_batch(
        self,
        code_batches: List[Dict[str, Any]],
        executor_type: str = "subprocess",
        max_concurrent: int = 3
    ) -> List[ExecutionResult]:
        """Execute multiple code snippets in parallel"""
        
        results = []
        
        # For now, execute sequentially (can be enhanced with threading)
        for i, batch in enumerate(code_batches):
            executor_id = f"batch_{i}"
            
            try:
                executor = self.create_executor(
                    executor_id=executor_id,
                    executor_type=executor_type
                )
                
                result = executor.execute_code(
                    code=batch.get('code', ''),
                    timeout=batch.get('timeout', 30)
                )
                
                result.metadata['batch_id'] = i
                result.metadata['batch_name'] = batch.get('name', f'batch_{i}')
                results.append(result)
                
            finally:
                self.remove_executor(executor_id)
        
        return results