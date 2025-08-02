import threading
import time
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import logging

from mock_servers.schema_server import SchemaBasedMockServer
from data_generation.api_schema_generator import APISchemaGenerator


class ServerManager:
    """
    Manages multiple mock server instances, providing utilities for
    server lifecycle management, monitoring, and coordination.
    """
    
    def __init__(self):
        self.servers: Dict[str, SchemaBasedMockServer] = {}
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = self._setup_logging()
        self._shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for server manager"""
        logger = logging.getLogger("server_manager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
        self.shutdown_all_servers()
        sys.exit(0)
    
    def create_server(
        self,
        server_name: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        title: str = "Mock API Server",
        description: str = "Dynamically generated mock API server",
        version: str = "1.0.0",
        log_level: str = "info",
        schema_generator: Optional[APISchemaGenerator] = None
    ) -> SchemaBasedMockServer:
        """Create a new mock server instance"""
        
        if server_name in self.servers:
            raise ValueError(f"Server '{server_name}' already exists")
        
        # Check if port is already in use
        if self._is_port_in_use(port):
            raise ValueError(f"Port {port} is already in use")
        
        try:
            server = SchemaBasedMockServer(
                title=title,
                description=description,
                version=version,
                host=host,
                port=port,
                log_level=log_level,
                schema_generator=schema_generator
            )
            
            self.servers[server_name] = server
            self.server_configs[server_name] = {
                "host": host,
                "port": port,
                "title": title,
                "description": description,
                "version": version,
                "log_level": log_level,
                "created_at": time.time()
            }
            
            self.logger.info(f"Created server '{server_name}' on {host}:{port}")
            return server
            
        except Exception as e:
            self.logger.error(f"Failed to create server '{server_name}': {str(e)}")
            raise
    
    def start_server(self, server_name: str, background: bool = True) -> None:
        """Start a specific server"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        server = self.servers[server_name]
        if server.is_server_running():
            self.logger.warning(f"Server '{server_name}' is already running")
            return
        
        try:
            server.start_server(background=background)
            self.logger.info(f"Started server '{server_name}' at {server.get_server_url()}")
        except Exception as e:
            self.logger.error(f"Failed to start server '{server_name}': {str(e)}")
            raise
    
    def stop_server(self, server_name: str) -> None:
        """Stop a specific server"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        server = self.servers[server_name]
        if not server.is_server_running():
            self.logger.warning(f"Server '{server_name}' is not running")
            return
        
        try:
            server.stop_server()
            self.logger.info(f"Stopped server '{server_name}'")
        except Exception as e:
            self.logger.error(f"Failed to stop server '{server_name}': {str(e)}")
            raise
    
    def restart_server(self, server_name: str) -> None:
        """Restart a specific server"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        self.logger.info(f"Restarting server '{server_name}'...")
        self.stop_server(server_name)
        time.sleep(1)  # Brief pause
        self.start_server(server_name)
    
    def remove_server(self, server_name: str) -> None:
        """Remove a server instance"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        # Stop server if running
        if self.servers[server_name].is_server_running():
            self.stop_server(server_name)
        
        # Remove from management
        del self.servers[server_name]
        del self.server_configs[server_name]
        
        self.logger.info(f"Removed server '{server_name}'")
    
    def start_all_servers(self) -> None:
        """Start all registered servers"""
        self.logger.info("Starting all servers...")
        
        for server_name in self.servers:
            try:
                self.start_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to start server '{server_name}': {str(e)}")
    
    def stop_all_servers(self) -> None:
        """Stop all running servers"""
        self.logger.info("Stopping all servers...")
        
        for server_name in self.servers:
            try:
                self.stop_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to stop server '{server_name}': {str(e)}")
    
    def shutdown_all_servers(self) -> None:
        """Gracefully shutdown all servers and clean up"""
        self.logger.info("Shutting down all servers...")
        self.stop_all_servers()
        self.servers.clear()
        self.server_configs.clear()
    
    def get_server(self, server_name: str) -> SchemaBasedMockServer:
        """Get a server instance by name"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        return self.servers[server_name]
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered servers with their status"""
        servers_info = []
        
        for name, server in self.servers.items():
            config = self.server_configs[name]
            
            info = {
                "name": name,
                "title": server.title,
                "url": server.get_server_url(),
                "status": "running" if server.is_server_running() else "stopped",
                "endpoints_count": len(server.registered_endpoints),
                "created_at": config["created_at"],
                "host": config["host"],
                "port": config["port"]
            }
            servers_info.append(info)
        
        return servers_info
    
    def get_server_status(self, server_name: str) -> Dict[str, Any]:
        """Get detailed status of a specific server"""
        if server_name not in self.servers:
            raise ValueError(f"Server '{server_name}' not found")
        
        server = self.servers[server_name]
        config = self.server_configs[server_name]
        
        return {
            "name": server_name,
            "title": server.title,
            "description": server.description,
            "version": server.version,
            "url": server.get_server_url(),
            "status": "running" if server.is_server_running() else "stopped",
            "host": config["host"],
            "port": config["port"],
            "log_level": config["log_level"],
            "endpoints_registered": len(server.registered_endpoints),
            "endpoints": [
                {
                    "path": ep.path,
                    "method": ep.method.value,
                    "summary": ep.summary,
                    "tags": ep.tags
                }
                for ep in server.registered_endpoints
            ],
            "created_at": config["created_at"],
            "uptime_seconds": time.time() - config["created_at"] if server.is_server_running() else 0
        }
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use by existing servers"""
        for server in self.servers.values():
            if server.port == port:
                return True
        return False
    
    def find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port"""
        import socket
        
        for port in range(start_port, start_port + max_attempts):
            if not self._is_port_in_use(port):
                # Double-check with socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('127.0.0.1', port))
                        return port
                    except OSError:
                        continue
        
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")
    
    @contextmanager
    def temporary_server(
        self,
        server_name: str,
        auto_start: bool = True,
        **server_kwargs
    ):
        """Context manager for temporary server instances"""
        try:
            # Find available port if not specified
            if 'port' not in server_kwargs:
                server_kwargs['port'] = self.find_available_port()
            
            server = self.create_server(server_name, **server_kwargs)
            
            if auto_start:
                self.start_server(server_name)
                time.sleep(0.5)  # Give server time to start
            
            yield server
            
        finally:
            if server_name in self.servers:
                self.remove_server(server_name)
    
    def create_server_cluster(
        self,
        cluster_name: str,
        server_count: int,
        base_port: int = 8000,
        load_balance: bool = False
    ) -> List[str]:
        """Create a cluster of servers for load testing"""
        server_names = []
        
        for i in range(server_count):
            server_name = f"{cluster_name}_{i}"
            port = base_port + i
            
            try:
                self.create_server(
                    server_name=server_name,
                    port=port,
                    title=f"{cluster_name} Server {i}",
                    description=f"Server {i} in {cluster_name} cluster"
                )
                server_names.append(server_name)
                
            except Exception as e:
                self.logger.error(f"Failed to create server {server_name}: {str(e)}")
        
        self.logger.info(f"Created cluster '{cluster_name}' with {len(server_names)} servers")
        return server_names
    
    def start_cluster(self, cluster_name: str) -> None:
        """Start all servers in a cluster"""
        cluster_servers = [name for name in self.servers.keys() if name.startswith(f"{cluster_name}_")]
        
        self.logger.info(f"Starting cluster '{cluster_name}' with {len(cluster_servers)} servers")
        
        for server_name in cluster_servers:
            try:
                self.start_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to start server {server_name}: {str(e)}")
    
    def stop_cluster(self, cluster_name: str) -> None:
        """Stop all servers in a cluster"""
        cluster_servers = [name for name in self.servers.keys() if name.startswith(f"{cluster_name}_")]
        
        self.logger.info(f"Stopping cluster '{cluster_name}' with {len(cluster_servers)} servers")
        
        for server_name in cluster_servers:
            try:
                self.stop_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to stop server {server_name}: {str(e)}")
    
    def get_cluster_status(self, cluster_name: str) -> Dict[str, Any]:
        """Get status of all servers in a cluster"""
        cluster_servers = [name for name in self.servers.keys() if name.startswith(f"{cluster_name}_")]
        
        running_count = sum(1 for name in cluster_servers if self.servers[name].is_server_running())
        
        return {
            "cluster_name": cluster_name,
            "total_servers": len(cluster_servers),
            "running_servers": running_count,
            "stopped_servers": len(cluster_servers) - running_count,
            "servers": [self.get_server_status(name) for name in cluster_servers]
        }
    
    def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all servers"""
        results = {}
        
        for server_name, server in self.servers.items():
            try:
                if server.is_server_running():
                    # In a real implementation, you'd make an HTTP request to /health
                    results[server_name] = {
                        "status": "healthy",
                        "url": server.get_server_url(),
                        "endpoints": len(server.registered_endpoints)
                    }
                else:
                    results[server_name] = {
                        "status": "stopped",
                        "url": server.get_server_url(),
                        "endpoints": len(server.registered_endpoints)
                    }
            except Exception as e:
                results[server_name] = {
                    "status": "error",
                    "error": str(e),
                    "url": server.get_server_url()
                }
        
        return {
            "timestamp": time.time(),
            "total_servers": len(self.servers),
            "healthy_servers": sum(1 for r in results.values() if r["status"] == "healthy"),
            "results": results
        }