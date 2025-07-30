import docker
import os
import tempfile
import time
import json
import tarfile
import io
from typing import Dict, List, Any, Optional
from pathlib import Path

from .code_executor import BaseCodeExecutor, ExecutionResult, ExecutionStatus, SecurityPolicy


class DockerCodeExecutor(BaseCodeExecutor):
    """Docker-based code executor for maximum isolation and security"""
    
    def __init__(
        self,
        image_name: str = "python:3.11-slim",
        container_name_prefix: str = "code_exec",
        security_policy: Optional[SecurityPolicy] = None,
        **kwargs
    ):
        super().__init__(security_policy=security_policy, **kwargs)
        
        self.image_name = image_name
        self.container_name_prefix = container_name_prefix
        self.docker_client = None
        self.container = None
        
        # Initialize Docker client
        self._init_docker_client()
        
        # Prepare execution image
        self._prepare_execution_image()
    
    def _init_docker_client(self):
        """Initialize Docker client with error handling"""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connection
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker not available: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Docker: {e}")
            raise RuntimeError(f"Docker initialization failed: {e}")
    
    def _prepare_execution_image(self):
        """Prepare Docker image with necessary dependencies"""
        dockerfile_content = f'''
FROM {self.image_name}

# Install required packages
RUN pip install --no-cache-dir requests httpx psutil

# Create non-root user for security
RUN useradd -m -u 1000 coderunner

# Set up working directory
WORKDIR /code
RUN chown coderunner:coderunner /code

# Switch to non-root user
USER coderunner

# Set up Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/code

CMD ["python3"]
'''
        
        # Build or use existing image
        execution_image_name = f"{self.container_name_prefix}_runner"
        
        try:
            # Check if image exists
            self.docker_client.images.get(execution_image_name)
            self.logger.info(f"Using existing image: {execution_image_name}")
        except docker.errors.ImageNotFound:
            self.logger.info(f"Building execution image: {execution_image_name}")
            
            # Build image
            dockerfile_io = io.BytesIO(dockerfile_content.encode('utf-8'))
            self.docker_client.images.build(
                fileobj=dockerfile_io,
                tag=execution_image_name,
                rm=True
            )
            self.logger.info("Execution image built successfully")
        
        self.execution_image_name = execution_image_name
    
    def _execute_isolated(
        self,
        code: str,
        timeout: int,
        environment_vars: Optional[Dict[str, str]]
    ) -> ExecutionResult:
        """Execute code in Docker container"""
        
        container_name = f"{self.container_name_prefix}_{int(time.time())}"
        
        try:
            # Prepare code file
            code_content = self._prepare_code_for_execution(code)
            
            # Create container
            container = self._create_container(
                container_name=container_name,
                code_content=code_content,
                environment_vars=environment_vars,
                timeout=timeout
            )
            
            # Execute code
            result = self._run_container(container, timeout)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Docker execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Docker execution error: {str(e)}"
            )
        finally:
            # Cleanup container
            self._cleanup_container(container_name)
    
    def _prepare_code_for_execution(self, code: str) -> str:
        """Prepare code with Docker-specific security wrapper"""
        
        security_wrapper = f'''
import sys
import time
import signal
import json
from typing import Dict, Any

# Security and monitoring setup
class DockerSecurityMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.max_memory_mb = {self.security_policy.max_memory_mb}
        self.max_execution_time = {self.security_policy.max_execution_time}
    
    def setup_limits(self):
        """Setup execution limits"""
        def timeout_handler(signum, frame):
            print("EXECUTION_TIMEOUT", file=sys.stderr)
            sys.exit(124)  # Timeout exit code
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_execution_time)
    
    def check_resources(self):
        """Check resource usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                print(f"MEMORY_LIMIT_EXCEEDED: {{memory_mb:.1f}}MB", file=sys.stderr)
                sys.exit(125)  # Resource limit exit code
        except ImportError:
            pass
    
    def cleanup(self):
        """Cleanup resources"""
        signal.alarm(0)  # Cancel alarm

# Initialize monitor
monitor = DockerSecurityMonitor()
monitor.setup_limits()

# Execution metadata
execution_metadata = {{
    "start_time": time.time(),
    "container_id": "docker_execution"
}}

try:
    # Check initial resources
    monitor.check_resources()
    
    # User code execution starts here
    print("=== CODE_EXECUTION_START ===")
    
{self._indent_code(code, 4)}
    
    print("=== CODE_EXECUTION_END ===")
    
    # Final resource check
    monitor.check_resources()
    
    # Success metadata
    execution_metadata["end_time"] = time.time()
    execution_metadata["status"] = "success"
    print(f"EXECUTION_METADATA: {{json.dumps(execution_metadata)}}", file=sys.stderr)

except KeyboardInterrupt:
    print("EXECUTION_INTERRUPTED", file=sys.stderr)
    sys.exit(130)  # Interrupted exit code
except Exception as e:
    print(f"EXECUTION_ERROR: {{str(e)}}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    monitor.cleanup()
'''
        
        return security_wrapper
    
    def _create_container(
        self,
        container_name: str,
        code_content: str,
        environment_vars: Optional[Dict[str, str]],
        timeout: int
    ) -> docker.models.containers.Container:
        """Create Docker container for code execution"""
        
        # Prepare environment
        env = {
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1'
        }
        if environment_vars:
            env.update(environment_vars)
        
        # Container resource limits
        mem_limit = f"{self.security_policy.max_memory_mb}m"
        
        # Create container
        container = self.docker_client.containers.create(
            image=self.execution_image_name,
            name=container_name,
            command=["python3", "-c", code_content],
            environment=env,
            mem_limit=mem_limit,
            memswap_limit=mem_limit,  # Disable swap
            cpu_period=100000,  # 100ms
            cpu_quota=50000,    # 50% CPU
            network_mode="bridge",  # Allow network access for API calls
            remove=False,  # Keep container for inspection
            detach=True,
            stdout=True,
            stderr=True,
            # Security options
            cap_drop=["ALL"],  # Drop all capabilities
            cap_add=["NET_BIND_SERVICE"] if self.security_policy.allow_network else [],
            read_only=True,  # Read-only filesystem
            tmpfs={'/tmp': 'noexec,nosuid,size=10m'},  # Temporary filesystem
            security_opt=['no-new-privileges:true']  # Prevent privilege escalation
        )
        
        self.logger.info(f"Created container: {container_name}")
        return container
    
    def _run_container(self, container: docker.models.containers.Container, timeout: int) -> ExecutionResult:
        """Run container and collect results"""
        
        start_time = time.time()
        
        try:
            # Start container
            container.start()
            
            # Wait for completion with timeout
            try:
                exit_code = container.wait(timeout=timeout)['StatusCode']
            except Exception:
                # Timeout occurred
                container.kill()
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stderr=f"Container execution timed out after {timeout} seconds",
                    execution_time=time.time() - start_time
                )
            
            # Get output
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            
            # Parse execution metadata from stderr
            metadata = self._parse_execution_metadata(stderr)
            
            # Clean output (remove metadata markers)
            stdout = self._clean_output(stdout)
            stderr = self._clean_stderr(stderr)
            
            # Determine execution status
            status = self._determine_status(exit_code, stderr)
            
            # Get resource usage
            stats = self._get_container_stats(container)
            
            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                return_code=exit_code,
                execution_time=time.time() - start_time,
                memory_usage=stats.get('memory_mb', 0),
                metadata=metadata
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Container execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _cleanup_container(self, container_name: str):
        """Clean up container and associated resources"""
        try:
            container = self.docker_client.containers.get(container_name)
            
            # Stop container if running
            if container.status == 'running':
                container.stop(timeout=5)
            
            # Remove container
            container.remove(force=True)
            
            self.logger.info(f"Cleaned up container: {container_name}")
            
        except docker.errors.NotFound:
            pass  # Container already removed
        except Exception as e:
            self.logger.warning(f"Failed to cleanup container {container_name}: {e}")
    
    def _parse_execution_metadata(self, stderr: str) -> Dict[str, Any]:
        """Parse execution metadata from stderr"""
        metadata = {}
        
        for line in stderr.split('\n'):
            if line.startswith('EXECUTION_METADATA: '):
                try:
                    metadata_json = line.replace('EXECUTION_METADATA: ', '')
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    pass
        
        return metadata
    
    def _clean_output(self, output: str) -> str:
        """Clean output by removing metadata markers"""
        lines = output.split('\n')
        cleaned_lines = []
        in_user_code = False
        
        for line in lines:
            if line == "=== CODE_EXECUTION_START ===":
                in_user_code = True
                continue
            elif line == "=== CODE_EXECUTION_END ===":
                in_user_code = False
                continue
            elif in_user_code:
                cleaned_lines.append(line)
        
        # If no markers found, return original output
        if not cleaned_lines and output:
            return output
        
        return '\n'.join(cleaned_lines)
    
    def _clean_stderr(self, stderr: str) -> str:
        """Clean stderr by removing internal messages"""
        lines = stderr.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip internal messages
            if (line.startswith('EXECUTION_') or
                line.startswith('MEMORY_LIMIT_') or
                line.startswith('Container')):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _determine_status(self, exit_code: int, stderr: str) -> ExecutionStatus:
        """Determine execution status from exit code and stderr"""
        
        if "EXECUTION_TIMEOUT" in stderr:
            return ExecutionStatus.TIMEOUT
        elif "MEMORY_LIMIT_EXCEEDED" in stderr:
            return ExecutionStatus.RESOURCE_LIMIT_EXCEEDED
        elif "EXECUTION_INTERRUPTED" in stderr:
            return ExecutionStatus.ERROR
        elif exit_code == 0:
            return ExecutionStatus.SUCCESS
        else:
            return ExecutionStatus.ERROR
    
    def _get_container_stats(self, container: docker.models.containers.Container) -> Dict[str, Any]:
        """Get container resource usage statistics"""
        
        try:
            stats = container.stats(stream=False)
            
            # Calculate memory usage
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_mb = memory_usage / 1024 / 1024
            
            # Calculate CPU usage (simplified)
            cpu_stats = stats.get('cpu_stats', {})
            cpu_usage = cpu_stats.get('cpu_usage', {}).get('total_usage', 0)
            
            return {
                'memory_mb': round(memory_mb, 2),
                'cpu_usage': cpu_usage,
                'network_rx': stats.get('networks', {}).get('eth0', {}).get('rx_bytes', 0),
                'network_tx': stats.get('networks', {}).get('eth0', {}).get('tx_bytes', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get container stats: {e}")
            return {}
    
    def cleanup(self):
        """Clean up Docker resources"""
        super().cleanup()
        
        # Clean up any remaining containers
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={'name': self.container_name_prefix}
            )
            
            for container in containers:
                try:
                    container.remove(force=True)
                    self.logger.info(f"Cleaned up container: {container.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup container {container.name}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup Docker containers: {e}")
    
    def create_custom_image(
        self,
        base_image: str,
        additional_packages: List[str],
        custom_setup: str = ""
    ) -> str:
        """Create custom Docker image with additional packages"""
        
        packages_str = " ".join(additional_packages)
        
        dockerfile_content = f'''
FROM {base_image}

# Install additional packages
RUN pip install --no-cache-dir {packages_str}

# Custom setup
{custom_setup}

# Create non-root user
RUN useradd -m -u 1000 coderunner
WORKDIR /code
RUN chown coderunner:coderunner /code
USER coderunner

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/code
'''
        
        # Build custom image
        custom_image_name = f"{self.container_name_prefix}_custom_{int(time.time())}"
        
        dockerfile_io = io.BytesIO(dockerfile_content.encode('utf-8'))
        self.docker_client.images.build(
            fileobj=dockerfile_io,
            tag=custom_image_name,
            rm=True
        )
        
        self.logger.info(f"Built custom image: {custom_image_name}")
        return custom_image_name


class DockerServerIntegratedExecutor(DockerCodeExecutor):
    """Docker executor with mock server integration"""
    
    def __init__(self, mock_server_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self.mock_server_url = mock_server_url
    
    def execute_code_with_server(
        self,
        code: str,
        timeout: int = 30,
        additional_env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute code with mock server access"""
        
        # Prepare environment variables
        env_vars = {
            'API_BASE_URL': self.mock_server_url or 'http://host.docker.internal:8000',
            'MOCK_SERVER_URL': self.mock_server_url or 'http://host.docker.internal:8000'
        }
        
        if additional_env:
            env_vars.update(additional_env)
        
        # Add server utilities to code
        enhanced_code = self._add_server_utilities(code)
        
        result = self.execute_code(
            code=enhanced_code,
            timeout=timeout,
            environment_vars=env_vars
        )
        
        # Add server metadata
        result.metadata['server_url'] = self.mock_server_url
        result.metadata['docker_execution'] = True
        
        return result
    
    def _add_server_utilities(self, code: str) -> str:
        """Add server connection utilities to code"""
        
        utilities = '''
# Docker server utilities
import os
import requests
import time

def get_api_base_url():
    """Get API base URL from environment"""
    return os.environ.get('API_BASE_URL', 'http://host.docker.internal:8000')

def wait_for_server(max_attempts=10):
    """Wait for server to be available"""
    base_url = get_api_base_url()
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server available at {base_url}")
                return True
        except requests.RequestException:
            if attempt < max_attempts - 1:
                time.sleep(1)
            continue
    
    print(f"⚠️  Server not available at {base_url}")
    return False

# Check server availability
server_available = wait_for_server()

'''
        
        return utilities + code
    
    def _create_container(self, container_name: str, code_content: str, environment_vars: Optional[Dict[str, str]], timeout: int):
        """Override to add Docker host networking for server access"""
        
        # Use host networking on Linux, or add host.docker.internal on macOS/Windows
        network_mode = "host" if os.name == 'posix' else "bridge"
        extra_hosts = None
        
        if network_mode == "bridge":
            extra_hosts = {'host.docker.internal': 'host-gateway'}
        
        # Prepare environment
        env = {
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1'
        }
        if environment_vars:
            env.update(environment_vars)
        
        # Container resource limits
        mem_limit = f"{self.security_policy.max_memory_mb}m"
        
        # Create container with network access
        container = self.docker_client.containers.create(
            image=self.execution_image_name,
            name=container_name,
            command=["python3", "-c", code_content],
            environment=env,
            mem_limit=mem_limit,
            memswap_limit=mem_limit,
            cpu_period=100000,
            cpu_quota=50000,
            network_mode=network_mode,
            extra_hosts=extra_hosts,
            remove=False,
            detach=True,
            stdout=True,
            stderr=True,
            # Relaxed security for server communication
            cap_drop=["ALL"],
            cap_add=["NET_BIND_SERVICE"],
            read_only=True,
            tmpfs={'/tmp': 'noexec,nosuid,size=10m'},
            security_opt=['no-new-privileges:true']
        )
        
        return container