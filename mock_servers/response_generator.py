import random
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from data_generation.data_generator import RandomDataGenerator
from data_generation.endpoint_generator import EndpointSpec, HTTPMethod


class MockResponseGenerator:
    """
    Advanced mock response generator that creates realistic API responses
    based on endpoint specifications and request context.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.data_generator = RandomDataGenerator(seed=seed)
        self.response_templates: Dict[str, Dict[str, Any]] = {}
        self.error_scenarios: Dict[str, List[Dict[str, Any]]] = self._initialize_error_scenarios()
    
    def _initialize_error_scenarios(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize common error scenarios for different endpoints"""
        return {
            "users": [
                {"status_code": 400, "error": "Bad Request", "message": "Invalid user data provided"},
                {"status_code": 401, "error": "Unauthorized", "message": "Authentication required"},
                {"status_code": 403, "error": "Forbidden", "message": "Insufficient permissions"},
                {"status_code": 404, "error": "Not Found", "message": "User not found"},
                {"status_code": 409, "error": "Conflict", "message": "Username already exists"},
                {"status_code": 422, "error": "Unprocessable Entity", "message": "Validation failed"}
            ],
            "products": [
                {"status_code": 400, "error": "Bad Request", "message": "Invalid product data provided"},
                {"status_code": 401, "error": "Unauthorized", "message": "Authentication required"},
                {"status_code": 403, "error": "Forbidden", "message": "Insufficient permissions"},
                {"status_code": 404, "error": "Not Found", "message": "Product not found"},
                {"status_code": 409, "error": "Conflict", "message": "Product SKU already exists"},
                {"status_code": 422, "error": "Unprocessable Entity", "message": "Validation failed"}
            ],
            "generic": [
                {"status_code": 500, "error": "Internal Server Error", "message": "An unexpected error occurred"},
                {"status_code": 503, "error": "Service Unavailable", "message": "Service temporarily unavailable"},
                {"status_code": 429, "error": "Too Many Requests", "message": "Rate limit exceeded"}
            ]
        }
    
    def generate_response(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any],
        query_params: Dict[str, Any],
        request_body: Optional[Dict[str, Any]] = None,
        force_error: Optional[int] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a mock response for a given endpoint specification
        
        Args:
            endpoint_spec: The endpoint specification
            path_params: Path parameters from the request
            query_params: Query parameters from the request
            request_body: Request body data
            force_error: Force a specific error status code
            include_metadata: Include response metadata
        """
        
        # Handle forced errors
        if force_error:
            return self._generate_error_response(endpoint_spec, force_error)
        
        # Generate success response based on method and path
        if endpoint_spec.method == HTTPMethod.DELETE:
            return self._generate_delete_response(endpoint_spec, path_params)
        elif endpoint_spec.method == HTTPMethod.GET:
            return self._generate_get_response(endpoint_spec, path_params, query_params)
        elif endpoint_spec.method == HTTPMethod.POST:
            return self._generate_post_response(endpoint_spec, request_body, include_metadata)
        elif endpoint_spec.method == HTTPMethod.PUT:
            return self._generate_put_response(endpoint_spec, path_params, request_body, include_metadata)
        
        return {}
    
    def _generate_get_response(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any],
        query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate GET response"""
        
        # Check if this is a list endpoint (no ID in path)
        if not any(param in endpoint_spec.path for param in ["{id}", "{user_id}", "{product_id}"]):
            return self._generate_list_response(endpoint_spec, query_params)
        else:
            return self._generate_single_item_response(endpoint_spec, path_params)
    
    def _generate_list_response(
        self,
        endpoint_spec: EndpointSpec,
        query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate paginated list response"""
        
        # Extract pagination parameters
        page = int(query_params.get("page", 1))
        per_page = min(int(query_params.get("per_page", 10)), 100)
        
        # Generate appropriate number of items
        total_items = random.randint(50, 500)
        items_count = min(per_page, max(0, total_items - (page - 1) * per_page))
        
        items = []
        for _ in range(items_count):
            if "users" in endpoint_spec.path:
                item = self.data_generator.generate_user_data()
                # Apply query filters
                if "role" in query_params:
                    item["role"] = query_params["role"]
            elif "products" in endpoint_spec.path:
                item = self.data_generator.generate_product_data()
                # Apply query filters
                if "category" in query_params:
                    item["category"] = query_params["category"]
                if "min_price" in query_params:
                    item["price"] = max(float(query_params["min_price"]), item["price"])
                if "max_price" in query_params:
                    item["price"] = min(float(query_params["max_price"]), item["price"])
            else:
                item = {"id": random.randint(1, 1000), "name": f"Item {random.randint(1, 100)}"}
            
            items.append(item)
        
        return {
            "items": items,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_items,
                "total_pages": (total_items + per_page - 1) // per_page,
                "has_next": page * per_page < total_items,
                "has_prev": page > 1
            },
            "filters_applied": {k: v for k, v in query_params.items() if k not in ["page", "per_page"]},
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_single_item_response(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate single item response"""
        
        # Extract ID from path parameters
        item_id = None
        for key, value in path_params.items():
            if "id" in key.lower():
                try:
                    item_id = int(value)
                except (ValueError, TypeError):
                    item_id = random.randint(1, 1000)
                break
        
        if item_id is None:
            item_id = random.randint(1, 1000)
        
        if "users" in endpoint_spec.path:
            data = self.data_generator.generate_user_data()
            data["id"] = item_id
        elif "products" in endpoint_spec.path:
            data = self.data_generator.generate_product_data()
            data["id"] = item_id
        else:
            data = {
                "id": item_id,
                "name": f"Item {item_id}",
                "created_at": datetime.now().isoformat()
            }
        
        return data
    
    def _generate_post_response(
        self,
        endpoint_spec: EndpointSpec,
        request_body: Optional[Dict[str, Any]],
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Generate POST (create) response"""
        
        # Start with request body data and add generated fields
        if "users" in endpoint_spec.path:
            data = self.data_generator.generate_user_data()
            if request_body:
                data.update({k: v for k, v in request_body.items() if k != "password"})
        elif "products" in endpoint_spec.path:
            data = self.data_generator.generate_product_data()
            if request_body:
                data.update(request_body)
        else:
            data = request_body.copy() if request_body else {}
            data["id"] = random.randint(1, 10000)
            data["created_at"] = datetime.now().isoformat()
        
        # Ensure we have an ID and timestamps
        if "id" not in data:
            data["id"] = random.randint(1, 10000)
        if "created_at" not in data:
            data["created_at"] = datetime.now().isoformat()
        
        if include_metadata:
            data["_metadata"] = {
                "created_by": "mock_server",
                "created_at": datetime.now().isoformat(),
                "version": 1
            }
        
        return data
    
    def _generate_put_response(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any],
        request_body: Optional[Dict[str, Any]],
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Generate PUT (update) response"""
        
        # Start with existing item data
        existing_data = self._generate_single_item_response(endpoint_spec, path_params)
        
        # Apply updates from request body
        if request_body:
            existing_data.update({k: v for k, v in request_body.items() if k not in ["id", "created_at"]})
        
        # Update timestamp
        existing_data["updated_at"] = datetime.now().isoformat()
        
        if include_metadata:
            existing_data["_metadata"] = {
                "updated_by": "mock_server",
                "updated_at": datetime.now().isoformat(),
                "version": random.randint(2, 10)
            }
        
        return existing_data
    
    def _generate_delete_response(
        self,
        endpoint_spec: EndpointSpec,
        path_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate DELETE response (usually empty)"""
        
        # Most DELETE endpoints return empty response with 204 status
        # But some might return confirmation data
        if random.random() < 0.2:  # 20% chance to return confirmation
            item_id = None
            for key, value in path_params.items():
                if "id" in key.lower():
                    item_id = value
                    break
            
            return {
                "message": "Resource deleted successfully",
                "deleted_id": item_id,
                "deleted_at": datetime.now().isoformat()
            }
        
        return {}
    
    def _generate_error_response(
        self,
        endpoint_spec: EndpointSpec,
        status_code: int
    ) -> Dict[str, Any]:
        """Generate error response"""
        
        # Determine error category based on endpoint
        if "users" in endpoint_spec.path:
            error_category = "users"
        elif "products" in endpoint_spec.path:
            error_category = "products"
        else:
            error_category = "generic"
        
        # Find matching error scenario
        matching_errors = [
            error for error in self.error_scenarios[error_category]
            if error["status_code"] == status_code
        ]
        
        if matching_errors:
            error_data = random.choice(matching_errors).copy()
        else:
            # Fallback generic error
            error_data = {
                "status_code": status_code,
                "error": "Error",
                "message": f"HTTP {status_code} error occurred"
            }
        
        # Add additional error details
        error_data.update({
            "path": endpoint_spec.path,
            "method": endpoint_spec.method.value,
            "timestamp": datetime.now().isoformat(),
            "request_id": f"req_{random.randint(100000, 999999)}"
        })
        
        return error_data
    
    def add_response_template(
        self,
        endpoint_key: str,
        template: Dict[str, Any]
    ) -> None:
        """Add a custom response template for specific endpoints"""
        self.response_templates[endpoint_key] = template
    
    def simulate_server_load(self, load_factor: float = 1.0) -> Dict[str, Any]:
        """Generate response times and server metrics based on load"""
        base_response_time = 50  # ms
        response_time = base_response_time * load_factor * random.uniform(0.5, 2.0)
        
        return {
            "server_metrics": {
                "response_time_ms": round(response_time, 2),
                "load_factor": load_factor,
                "cpu_usage_percent": min(100, 20 + (load_factor * 30) + random.uniform(-10, 10)),
                "memory_usage_percent": min(100, 15 + (load_factor * 25) + random.uniform(-5, 15)),
                "active_connections": random.randint(1, int(50 * load_factor)),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def generate_realistic_delays(self) -> float:
        """Generate realistic API response delays in seconds"""
        # Simulate various delay scenarios
        delay_scenarios = [
            (0.05, 0.2, 0.7),   # Fast responses (70% chance)
            (0.2, 0.8, 0.25),   # Medium responses (25% chance)
            (0.8, 3.0, 0.05)    # Slow responses (5% chance)
        ]
        
        scenario = random.choices(delay_scenarios, weights=[0.7, 0.25, 0.05])[0]
        return random.uniform(scenario[0], scenario[1])
    
    def get_response_headers(self, endpoint_spec: EndpointSpec) -> Dict[str, str]:
        """Generate realistic response headers"""
        headers = {
            "X-Request-ID": f"req_{random.randint(100000, 999999)}",
            "X-Response-Time": f"{random.randint(10, 500)}ms",
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": str(random.randint(800, 999)),
            "X-RateLimit-Reset": str(int((datetime.now() + timedelta(hours=1)).timestamp())),
            "Cache-Control": "no-cache" if endpoint_spec.method != HTTPMethod.GET else "public, max-age=300"
        }
        
        if endpoint_spec.method == HTTPMethod.POST:
            headers["Location"] = f"{endpoint_spec.path}/{random.randint(1, 10000)}"
        
        return headers