import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
from data_generation.endpoint_generator import EndpointGenerator, EndpointSpec
from data_generation.data_generator import RandomDataGenerator


class APISchemaGenerator:
    def __init__(self, seed: int = None):
        self.endpoint_generator = EndpointGenerator()
        self.data_generator = RandomDataGenerator(seed=seed)
        self.seed = seed
    
    def generate_openapi_spec(
        self,
        title: str = "Generated API",
        version: str = "1.0.0",
        description: str = "Randomly generated API specification for RL training",
        include_endpoints: Optional[List[str]] = None,
        endpoint_count: Optional[int] = None
    ) -> Dict[str, Any]:
        if include_endpoints:
            endpoints = self._filter_endpoints_by_tags(include_endpoints)
        elif endpoint_count:
            endpoints = self.endpoint_generator.get_random_endpoints(endpoint_count)
        else:
            endpoints = self.endpoint_generator.get_all_endpoints()
        
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": title,
                "description": description,
                "version": version,
                "contact": {
                    "name": "API Generator",
                    "email": "api@example.com"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": self._generate_component_schemas(),
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                }
            },
            "security": [
                {"bearerAuth": []}
            ]
        }
        
        # Group endpoints by path
        paths = {}
        for endpoint in endpoints:
            path = endpoint.path
            method = endpoint.method.value.lower()
            
            if path not in paths:
                paths[path] = {}
            
            paths[path][method] = self._endpoint_to_openapi(endpoint)
        
        openapi_spec["paths"] = paths
        return openapi_spec
    
    def generate_api_scenario(
        self,
        scenario_name: str = "Random API Scenario",
        num_endpoints: int = 5,
        include_sample_data: bool = True
    ) -> Dict[str, Any]:
        endpoints = self.endpoint_generator.get_random_endpoints(num_endpoints)
        
        scenario = {
            "name": scenario_name,
            "description": f"API scenario with {num_endpoints} random endpoints",
            "created_at": datetime.now().isoformat(),
            "seed": self.seed,
            "endpoints": []
        }
        
        for endpoint in endpoints:
            endpoint_data = {
                "specification": endpoint.to_dict(),
                "sample_requests": [],
                "sample_responses": []
            }
            
            if include_sample_data:
                # Generate sample requests for POST/PUT
                if endpoint.method.value in ["POST", "PUT"]:
                    for _ in range(random.randint(1, 3)):
                        sample_request = self.data_generator.generate_api_request_data(
                            endpoint.path, endpoint.method.value
                        )
                        if sample_request:
                            endpoint_data["sample_requests"].append(sample_request)
                
                # Generate sample responses
                for status_code in [200, 201, 400, 404]:
                    if status_code == 201 and endpoint.method.value != "POST":
                        continue
                    if status_code in [400, 404] and random.random() > 0.3:
                        continue
                    
                    sample_response = self.data_generator.generate_api_response_data(
                        endpoint.path, endpoint.method.value, status_code
                    )
                    if sample_response:
                        endpoint_data["sample_responses"].append({
                            "status_code": status_code,
                            "data": sample_response
                        })
            
            scenario["endpoints"].append(endpoint_data)
        
        return scenario
    
    def generate_training_dataset(
        self,
        num_scenarios: int = 10,
        endpoints_per_scenario: int = 5
    ) -> List[Dict[str, Any]]:
        dataset = []
        
        for i in range(num_scenarios):
            scenario = self.generate_api_scenario(
                scenario_name=f"Training Scenario {i + 1}",
                num_endpoints=endpoints_per_scenario,
                include_sample_data=True
            )
            dataset.append(scenario)
        
        return dataset
    
    def _filter_endpoints_by_tags(self, tags: List[str]) -> List[EndpointSpec]:
        all_endpoints = self.endpoint_generator.get_all_endpoints()
        filtered = []
        
        for endpoint in all_endpoints:
            if any(tag in endpoint.tags for tag in tags):
                filtered.append(endpoint)
        
        return filtered
    
    def _generate_component_schemas(self) -> Dict[str, Any]:
        from data_generation.schemas import (
            User, UserCreate, UserUpdate, UserResponse,
            Product, ProductCreate, ProductUpdate, ProductResponse,
            PaginatedResponse, ErrorResponse
        )
        
        schemas = {}
        for model_class in [
            User, UserCreate, UserUpdate, UserResponse,
            Product, ProductCreate, ProductUpdate, ProductResponse,
            PaginatedResponse, ErrorResponse
        ]:
            schema = model_class.model_json_schema()
            schemas[model_class.__name__] = schema
        
        return schemas
    
    def _endpoint_to_openapi(self, endpoint: EndpointSpec) -> Dict[str, Any]:
        openapi_endpoint = {
            "summary": endpoint.summary,
            "description": endpoint.description,
            "tags": endpoint.tags,
            "parameters": endpoint.parameters,
            "responses": endpoint.responses
        }
        
        if endpoint.request_body:
            openapi_endpoint["requestBody"] = endpoint.request_body
        
        # Add security for non-GET endpoints
        if endpoint.method.value != "GET":
            openapi_endpoint["security"] = [{"bearerAuth": []}]
        
        return openapi_endpoint
    
    def save_spec_to_file(self, spec: Dict[str, Any], filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(spec, f, indent=2, default=str)
    
    def save_scenario_to_file(self, scenario: Dict[str, Any], filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(scenario, f, indent=2, default=str)
    
    def get_available_endpoint_tags(self) -> List[str]:
        all_endpoints = self.endpoint_generator.get_all_endpoints()
        tags = set()
        for endpoint in all_endpoints:
            tags.update(endpoint.tags)
        return list(tags)
    
    def generate_minimal_spec(self) -> Dict[str, Any]:
        return self.generate_openapi_spec(
            title="Minimal API",
            endpoint_count=3,
            description="Minimal API specification for quick testing"
        )
    
    def generate_user_management_spec(self) -> Dict[str, Any]:
        return self.generate_openapi_spec(
            title="User Management API",
            include_endpoints=["users"],
            description="Complete user management API specification"
        )
    
    def generate_product_catalog_spec(self) -> Dict[str, Any]:
        return self.generate_openapi_spec(
            title="Product Catalog API",
            include_endpoints=["products"],
            description="Complete product catalog API specification"
        )