from typing import Dict, List, Any, Optional
from enum import Enum
import random
from data_generation.schemas import (
    User, UserCreate, UserUpdate, UserResponse,
    Product, ProductCreate, ProductUpdate, ProductResponse,
    PaginatedResponse, ErrorResponse
)


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class EndpointSpec:
    def __init__(
        self,
        path: str,
        method: HTTPMethod,
        summary: str,
        description: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        request_body: Optional[Dict[str, Any]] = None,
        responses: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        self.path = path
        self.method = method
        self.summary = summary
        self.description = description
        self.parameters = parameters or []
        self.request_body = request_body
        self.responses = responses or {}
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        spec = {
            "path": self.path,
            "method": self.method.value,
            "summary": self.summary,
            "description": self.description,
            "tags": self.tags
        }
        
        if self.parameters:
            spec["parameters"] = self.parameters
        if self.request_body:
            spec["requestBody"] = self.request_body
        if self.responses:
            spec["responses"] = self.responses
            
        return spec


class EndpointGenerator:
    def __init__(self):
        self.user_endpoints = self._create_user_endpoints()
        self.product_endpoints = self._create_product_endpoints()
    
    def _create_user_endpoints(self) -> List[EndpointSpec]:
        endpoints = []
        
        # GET /users - List users with pagination
        endpoints.append(EndpointSpec(
            path="/users",
            method=HTTPMethod.GET,
            summary="List users",
            description="Retrieve a paginated list of users",
            parameters=[
                {
                    "name": "page",
                    "in": "query",
                    "description": "Page number",
                    "required": False,
                    "schema": {"type": "integer", "default": 1, "minimum": 1}
                },
                {
                    "name": "per_page",
                    "in": "query", 
                    "description": "Items per page",
                    "required": False,
                    "schema": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
                },
                {
                    "name": "role",
                    "in": "query",
                    "description": "Filter by user role",
                    "required": False,
                    "schema": {"type": "string", "enum": ["admin", "user", "moderator"]}
                }
            ],
            responses={
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": PaginatedResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["users"]
        ))
        
        # POST /users - Create user
        endpoints.append(EndpointSpec(
            path="/users",
            method=HTTPMethod.POST,
            summary="Create user",
            description="Create a new user account",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": UserCreate.model_json_schema()
                    }
                }
            },
            responses={
                "201": {
                    "description": "User created successfully",
                    "content": {
                        "application/json": {
                            "schema": UserResponse.model_json_schema()
                        }
                    }
                },
                "400": {
                    "description": "Bad request",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["users"]
        ))
        
        # GET /users/{user_id} - Get user by ID
        endpoints.append(EndpointSpec(
            path="/users/{user_id}",
            method=HTTPMethod.GET,
            summary="Get user by ID",
            description="Retrieve a specific user by their ID",
            parameters=[
                {
                    "name": "user_id",
                    "in": "path",
                    "description": "User ID",
                    "required": True,
                    "schema": {"type": "integer"}
                }
            ],
            responses={
                "200": {
                    "description": "User found",
                    "content": {
                        "application/json": {
                            "schema": UserResponse.model_json_schema()
                        }
                    }
                },
                "404": {
                    "description": "User not found",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["users"]
        ))
        
        # PUT /users/{user_id} - Update user
        endpoints.append(EndpointSpec(
            path="/users/{user_id}",
            method=HTTPMethod.PUT,
            summary="Update user",
            description="Update an existing user",
            parameters=[
                {
                    "name": "user_id",
                    "in": "path",
                    "description": "User ID",
                    "required": True,
                    "schema": {"type": "integer"}
                }
            ],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": UserUpdate.model_json_schema()
                    }
                }
            },
            responses={
                "200": {
                    "description": "User updated successfully",
                    "content": {
                        "application/json": {
                            "schema": UserResponse.model_json_schema()
                        }
                    }
                },
                "404": {
                    "description": "User not found",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["users"]
        ))
        
        # DELETE /users/{user_id} - Delete user
        endpoints.append(EndpointSpec(
            path="/users/{user_id}",
            method=HTTPMethod.DELETE,
            summary="Delete user",
            description="Delete a user account",
            parameters=[
                {
                    "name": "user_id",
                    "in": "path",
                    "description": "User ID",
                    "required": True,
                    "schema": {"type": "integer"}
                }
            ],
            responses={
                "204": {
                    "description": "User deleted successfully"
                },
                "404": {
                    "description": "User not found",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["users"]
        ))
        
        return endpoints
    
    def _create_product_endpoints(self) -> List[EndpointSpec]:
        endpoints = []
        
        # GET /products - List products
        endpoints.append(EndpointSpec(
            path="/products",
            method=HTTPMethod.GET,
            summary="List products",
            description="Retrieve a paginated list of products",
            parameters=[
                {
                    "name": "page",
                    "in": "query",
                    "description": "Page number",
                    "required": False,
                    "schema": {"type": "integer", "default": 1, "minimum": 1}
                },
                {
                    "name": "per_page",
                    "in": "query",
                    "description": "Items per page", 
                    "required": False,
                    "schema": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
                },
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by product category",
                    "required": False,
                    "schema": {"type": "string", "enum": ["electronics", "clothing", "books", "home", "sports"]}
                },
                {
                    "name": "min_price",
                    "in": "query",
                    "description": "Minimum price filter",
                    "required": False,
                    "schema": {"type": "number", "minimum": 0}
                },
                {
                    "name": "max_price",
                    "in": "query",
                    "description": "Maximum price filter",
                    "required": False,
                    "schema": {"type": "number", "minimum": 0}
                }
            ],
            responses={
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": PaginatedResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["products"]
        ))
        
        # POST /products - Create product
        endpoints.append(EndpointSpec(
            path="/products",
            method=HTTPMethod.POST,
            summary="Create product",
            description="Create a new product",
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": ProductCreate.model_json_schema()
                    }
                }
            },
            responses={
                "201": {
                    "description": "Product created successfully",
                    "content": {
                        "application/json": {
                            "schema": ProductResponse.model_json_schema()
                        }
                    }
                },
                "400": {
                    "description": "Bad request",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["products"]
        ))
        
        # GET /products/{product_id} - Get product by ID
        endpoints.append(EndpointSpec(
            path="/products/{product_id}",
            method=HTTPMethod.GET,
            summary="Get product by ID",
            description="Retrieve a specific product by its ID",
            parameters=[
                {
                    "name": "product_id",
                    "in": "path",
                    "description": "Product ID",
                    "required": True,
                    "schema": {"type": "integer"}
                }
            ],
            responses={
                "200": {
                    "description": "Product found",
                    "content": {
                        "application/json": {
                            "schema": ProductResponse.model_json_schema()
                        }
                    }
                },
                "404": {
                    "description": "Product not found",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["products"]
        ))
        
        # PUT /products/{product_id} - Update product
        endpoints.append(EndpointSpec(
            path="/products/{product_id}",
            method=HTTPMethod.PUT,
            summary="Update product",
            description="Update an existing product",
            parameters=[
                {
                    "name": "product_id",
                    "in": "path",
                    "description": "Product ID",
                    "required": True,
                    "schema": {"type": "integer"}
                }
            ],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": ProductUpdate.model_json_schema()
                    }
                }
            },
            responses={
                "200": {
                    "description": "Product updated successfully",
                    "content": {
                        "application/json": {
                            "schema": ProductResponse.model_json_schema()
                        }
                    }
                },
                "404": {
                    "description": "Product not found",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["products"]
        ))
        
        # DELETE /products/{product_id} - Delete product
        endpoints.append(EndpointSpec(
            path="/products/{product_id}",
            method=HTTPMethod.DELETE,
            summary="Delete product",
            description="Delete a product",
            parameters=[
                {
                    "name": "product_id",
                    "in": "path",
                    "description": "Product ID",
                    "required": True,
                    "schema": {"type": "integer"}
                }
            ],
            responses={
                "204": {
                    "description": "Product deleted successfully"
                },
                "404": {
                    "description": "Product not found",
                    "content": {
                        "application/json": {
                            "schema": ErrorResponse.model_json_schema()
                        }
                    }
                }
            },
            tags=["products"]
        ))
        
        return endpoints
    
    def get_random_endpoints(self, count: int = 5) -> List[EndpointSpec]:
        all_endpoints = self.user_endpoints + self.product_endpoints
        return random.sample(all_endpoints, min(count, len(all_endpoints)))
    
    def get_user_endpoints(self) -> List[EndpointSpec]:
        return self.user_endpoints
    
    def get_product_endpoints(self) -> List[EndpointSpec]:
        return self.product_endpoints
    
    def get_all_endpoints(self) -> List[EndpointSpec]:
        return self.user_endpoints + self.product_endpoints