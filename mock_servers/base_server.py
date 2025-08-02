import logging
import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from data_generation.endpoint_generator import EndpointSpec, HTTPMethod
from data_generation.data_generator import RandomDataGenerator


class MockServerError(Exception):
    """Custom exception for mock server errors"""
    pass


class MockServer:
    """
    FastAPI-based mock server that can dynamically create endpoints
    from API schemas and return appropriate mock responses.
    """
    
    def __init__(
        self,
        title: str = "Mock API Server",
        description: str = "Dynamically generated mock API server",
        version: str = "1.0.0",
        host: str = "127.0.0.1",
        port: int = 8000,
        log_level: str = "info"
    ):
        self.title = title
        self.description = description
        self.version = version
        self.host = host
        self.port = port
        self.log_level = log_level
        
        # Server state
        self.app: Optional[FastAPI] = None
        self.server_thread: Optional[threading.Thread] = None
        self.server_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.registered_endpoints: List[EndpointSpec] = []
        
        # Data generator for mock responses
        self.data_generator = RandomDataGenerator()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Custom response handlers
        self.custom_handlers: Dict[str, Callable] = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"mock_server_{self.port}")
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.logger.info(f"Mock server starting on {self.host}:{self.port}")
            yield
            self.logger.info("Mock server shutting down")
        
        app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            lifespan=lifespan
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request logging middleware
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            self.logger.info(
                f"Incoming request: {request.method} {request.url.path}"
            )
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                self.logger.info(
                    f"Request completed: {request.method} {request.url.path} "
                    f"- Status: {response.status_code} - Time: {process_time:.4f}s"
                )
                
                return response
            except Exception as e:
                process_time = time.time() - start_time
                self.logger.error(
                    f"Request failed: {request.method} {request.url.path} "
                    f"- Error: {str(e)} - Time: {process_time:.4f}s"
                )
                raise
        
        # Add health check endpoint
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "server": self.title,
                "version": self.version,
                "endpoints_registered": len(self.registered_endpoints)
            }
        
        # Add server info endpoint
        @app.get("/server-info")
        async def server_info():
            return {
                "title": self.title,
                "description": self.description,
                "version": self.version,
                "host": self.host,
                "port": self.port,
                "registered_endpoints": len(self.registered_endpoints),
                "endpoints": [
                    {
                        "path": ep.path,
                        "method": ep.method.value,
                        "summary": ep.summary,
                        "tags": ep.tags
                    }
                    for ep in self.registered_endpoints
                ]
            }
        
        return app
    
    def register_endpoint(self, endpoint_spec: EndpointSpec) -> None:
        """Register a single endpoint from an EndpointSpec"""
        if not self.app:
            raise MockServerError("Server app not initialized. Call start_server() first.")
        
        path = endpoint_spec.path
        method = endpoint_spec.method
        
        # Convert path parameters from {param} to {param:path} for FastAPI
        fastapi_path = self._convert_path_parameters(path)
        
        # Create the handler function
        handler = self._create_endpoint_handler(endpoint_spec)
        
        # Register the endpoint with FastAPI
        try:
            if method == HTTPMethod.GET:
                self.app.get(fastapi_path, summary=endpoint_spec.summary, tags=endpoint_spec.tags)(handler)
            elif method == HTTPMethod.POST:
                self.app.post(fastapi_path, summary=endpoint_spec.summary, tags=endpoint_spec.tags)(handler)
            elif method == HTTPMethod.PUT:
                self.app.put(fastapi_path, summary=endpoint_spec.summary, tags=endpoint_spec.tags)(handler)
            elif method == HTTPMethod.DELETE:
                self.app.delete(fastapi_path, summary=endpoint_spec.summary, tags=endpoint_spec.tags)(handler)
            
            self.registered_endpoints.append(endpoint_spec)
            self.logger.info(f"Registered endpoint: {method.value} {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to register endpoint {method.value} {path}: {str(e)}")
            raise MockServerError(f"Failed to register endpoint: {str(e)}")
    
    def register_endpoints(self, endpoint_specs: List[EndpointSpec]) -> None:
        """Register multiple endpoints from a list of EndpointSpecs"""
        for endpoint_spec in endpoint_specs:
            try:
                self.register_endpoint(endpoint_spec)
            except MockServerError as e:
                self.logger.warning(f"Skipping endpoint registration: {str(e)}")
                continue
    
    def _convert_path_parameters(self, path: str) -> str:
        """Convert OpenAPI path parameters to FastAPI format"""
        import re
        # Convert {param} to {param:path} for path parameters
        return re.sub(r'\{([^}]+)\}', r'{\1:path}', path)
    
    def _create_endpoint_handler(self, endpoint_spec: EndpointSpec) -> Callable:
        """Create a handler function for an endpoint"""
        async def handler(request: Request, **path_params):
            try:
                # Check for custom handler
                handler_key = f"{endpoint_spec.method.value}:{endpoint_spec.path}"
                if handler_key in self.custom_handlers:
                    return await self.custom_handlers[handler_key](request, **path_params)
                
                # Get query parameters
                query_params = dict(request.query_params)
                
                # Get request body for POST/PUT requests
                request_body = None
                if endpoint_spec.method in [HTTPMethod.POST, HTTPMethod.PUT]:
                    try:
                        request_body = await request.json()
                    except Exception:
                        request_body = {}
                
                # Log the request details
                self.logger.debug(f"Handler called for {endpoint_spec.method.value} {endpoint_spec.path}")
                self.logger.debug(f"Path params: {path_params}")
                self.logger.debug(f"Query params: {query_params}")
                if request_body:
                    self.logger.debug(f"Request body: {request_body}")
                
                # Generate mock response based on endpoint specification
                response_data = self._generate_mock_response(
                    endpoint_spec, path_params, query_params, request_body
                )
                
                # Determine status code
                status_code = self._determine_status_code(endpoint_spec, path_params, query_params)
                
                return JSONResponse(content=response_data, status_code=status_code)
                
            except Exception as e:
                self.logger.error(f"Error in handler for {endpoint_spec.method.value} {endpoint_spec.path}: {str(e)}")
                return JSONResponse(
                    content={
                        "error": "Internal Server Error",
                        "message": f"Mock server error: {str(e)}",
                        "status_code": 500
                    },
                    status_code=500
                )
        
        return handler
    
    def _generate_mock_response(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any],
        query_params: Dict[str, Any],
        request_body: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate mock response data based on endpoint specification"""
        
        # For DELETE requests, return empty response for 204
        if endpoint_spec.method == HTTPMethod.DELETE:
            return {}
        
        # Use data generator to create appropriate response
        return self.data_generator.generate_api_response_data(
            endpoint_spec.path,
            endpoint_spec.method.value,
            status_code=200
        )
    
    def _determine_status_code(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any],
        query_params: Dict[str, Any]
    ) -> int:
        """Determine appropriate status code based on request"""
        
        # For DELETE requests, return 204 (No Content)
        if endpoint_spec.method == HTTPMethod.DELETE:
            return 204
        
        # For POST requests, return 201 (Created)
        if endpoint_spec.method == HTTPMethod.POST:
            return 201
        
        # Default to 200 (OK)
        return 200
    
    def add_custom_handler(self, method: str, path: str, handler: Callable) -> None:
        """Add a custom handler for a specific endpoint"""
        handler_key = f"{method.upper()}:{path}"
        self.custom_handlers[handler_key] = handler
        self.logger.info(f"Added custom handler for {method.upper()} {path}")
    
    def start_server(self, background: bool = True) -> None:
        """Start the mock server"""
        if self.is_running:
            self.logger.warning("Server is already running")
            return
        
        try:
            self.app = self._create_app()
            
            if background:
                self._start_background_server()
            else:
                self._start_server_sync()
                
        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}")
            raise MockServerError(f"Failed to start server: {str(e)}")
    
    def _start_background_server(self) -> None:
        """Start server in background thread"""
        def run_server():
            try:
                uvicorn.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    log_level=self.log_level.lower(),
                    access_log=True
                )
            except Exception as e:
                self.logger.error(f"Server error in background thread: {str(e)}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        # Wait a moment for server to start
        time.sleep(1)
        self.logger.info(f"Mock server started in background on {self.host}:{self.port}")
    
    def _start_server_sync(self) -> None:
        """Start server synchronously"""
        self.is_running = True
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level.lower(),
            access_log=True
        )
    
    def stop_server(self) -> None:
        """Stop the mock server"""
        if not self.is_running:
            self.logger.warning("Server is not running")
            return
        
        self.is_running = False
        
        if self.server_thread and self.server_thread.is_alive():
            self.logger.info("Stopping background server...")
            # Note: uvicorn doesn't have a clean way to stop from thread
            # In production, you'd want to use a more sophisticated approach
        
        self.logger.info("Mock server stopped")
    
    def is_server_running(self) -> bool:
        """Check if server is running"""
        return self.is_running
    
    def get_server_url(self) -> str:
        """Get the server URL"""
        return f"http://{self.host}:{self.port}"
    
    def clear_endpoints(self) -> None:
        """Clear all registered endpoints"""
        if self.is_running:
            raise MockServerError("Cannot clear endpoints while server is running")
        
        self.registered_endpoints.clear()
        self.custom_handlers.clear()
        self.logger.info("Cleared all registered endpoints")