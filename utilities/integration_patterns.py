from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import random

from ..data_generation.endpoint_generator import EndpointSpec, HTTPMethod
from .code_template_generator import (
    BaseCodeTemplateGenerator, MissingComponent, DifficultyLevel, 
    CodeTemplate, CodeGap
)
from .schema_analyzer import APISchemaAnalyzer, SchemaAnalysis


class IntegrationPattern(ABC):
    """Base class for different API integration patterns"""
    
    @abstractmethod
    def generate_template(
        self, 
        endpoints: List[EndpointSpec], 
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel
    ) -> CodeTemplate:
        """Generate template for this integration pattern"""
        pass
    
    @abstractmethod
    def get_pattern_name(self) -> str:
        """Get the name of this pattern"""
        pass


class CRUDIntegrationPattern(IntegrationPattern):
    """Template generator for CRUD-based API integrations"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
    
    def generate_template(
        self,
        endpoints: List[EndpointSpec],
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel
    ) -> CodeTemplate:
        """Generate CRUD integration template"""
        
        # Group endpoints by resource
        resources = self._group_by_resource(endpoints)
        primary_resource = list(resources.keys())[0] if resources else "resource"
        primary_endpoints = resources.get(primary_resource, endpoints[:5])
        
        code_parts = []
        gaps = []
        line_number = 1
        
        # Imports
        if MissingComponent.IMPORTS in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.IMPORTS, line_number,
                "# TODO: Import necessary libraries (requests, json, typing, etc.)",
                "Import HTTP client library and supporting modules"
            ))
            code_parts.append("# TODO: Import necessary libraries (requests, json, typing, etc.)")
        else:
            code_parts.extend([
                "import requests",
                "import json",
                "from typing import Dict, List, Optional, Any",
                "import logging"
            ])
        
        code_parts.append("")
        line_number += len([p for p in code_parts if p])
        
        # Main CRUD client class
        class_name = f"{primary_resource.title()}CRUDClient"
        code_parts.extend([
            f"class {class_name}:",
            f'    """CRUD client for {primary_resource} management"""',
            "",
            "    def __init__(self, base_url: str, api_key: str = None):",
            "        self.base_url = base_url.rstrip('/')",
            "        self.api_key = api_key"
        ])
        line_number += 6
        
        # Authentication setup
        if MissingComponent.AUTHENTICATION in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.AUTHENTICATION, line_number,
                "        # TODO: Set up authentication headers",
                "Configure API key or bearer token authentication"
            ))
            code_parts.append("        # TODO: Set up authentication headers")
        else:
            code_parts.extend([
                "        self.headers = {'Content-Type': 'application/json'}",
                "        if api_key:",
                "            self.headers['Authorization'] = f'Bearer {api_key}'"
            ])
        
        code_parts.append("")
        line_number += len([p for p in code_parts[-10:] if p])
        
        # Generate CRUD methods
        crud_methods = self._identify_crud_methods(primary_endpoints)
        
        for operation, endpoint in crud_methods.items():
            if not endpoint:
                continue
                
            method_code, method_gaps = self._generate_crud_method(
                operation, endpoint, missing_components, difficulty, line_number
            )
            code_parts.extend(method_code)
            gaps.extend(method_gaps)
            line_number += len(method_code)
        
        # Add helper methods
        if difficulty in [DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]:
            helper_code, helper_gaps = self._generate_helper_methods(
                missing_components, difficulty, line_number
            )
            code_parts.extend(helper_code)
            gaps.extend(helper_gaps)
        
        # Usage example
        code_parts.extend([
            "",
            "",
            "# Usage Example",
            "if __name__ == '__main__':",
            f"    client = {class_name}(",
            "        base_url='https://api.example.com',",
            "        api_key='your-api-key'",
            "    )",
            ""
        ])
        
        # Add example usage for each CRUD operation
        for operation in crud_methods.keys():
            example = self._generate_usage_example(operation, primary_resource)
            if example:
                code_parts.append(f"    # {example}")
        
        final_code = "\n".join(code_parts)
        
        metadata = {
            "pattern": "CRUD Integration",
            "primary_resource": primary_resource,
            "operations": list(crud_methods.keys()),
            "total_gaps": len(gaps),
            "difficulty": difficulty.value
        }
        
        return CodeTemplate(
            code=final_code,
            gaps=gaps,
            metadata=metadata,
            description=f"CRUD integration template for {primary_resource} resource"
        )
    
    def get_pattern_name(self) -> str:
        return "CRUD Integration"
    
    def _group_by_resource(self, endpoints: List[EndpointSpec]) -> Dict[str, List[EndpointSpec]]:
        """Group endpoints by resource type"""
        resources = {}
        
        for endpoint in endpoints:
            path_parts = endpoint.path.strip('/').split('/')
            resource = path_parts[0] if path_parts else "resource"
            
            if resource not in resources:
                resources[resource] = []
            resources[resource].append(endpoint)
        
        return resources
    
    def _identify_crud_methods(self, endpoints: List[EndpointSpec]) -> Dict[str, Optional[EndpointSpec]]:
        """Identify CRUD operations from endpoints"""
        crud_ops = {
            "list": None,
            "create": None,
            "read": None,
            "update": None,
            "delete": None
        }
        
        for endpoint in endpoints:
            has_id = "{" in endpoint.path and "}" in endpoint.path
            
            if endpoint.method == HTTPMethod.GET and not has_id:
                crud_ops["list"] = endpoint
            elif endpoint.method == HTTPMethod.POST and not has_id:
                crud_ops["create"] = endpoint
            elif endpoint.method == HTTPMethod.GET and has_id:
                crud_ops["read"] = endpoint
            elif endpoint.method == HTTPMethod.PUT and has_id:
                crud_ops["update"] = endpoint
            elif endpoint.method == HTTPMethod.DELETE and has_id:
                crud_ops["delete"] = endpoint
        
        return crud_ops
    
    def _generate_crud_method(
        self,
        operation: str,
        endpoint: EndpointSpec,
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel,
        start_line: int
    ) -> tuple[List[str], List[CodeGap]]:
        """Generate code for a specific CRUD method"""
        
        method_code = []
        gaps = []
        line_number = start_line
        
        # Method signature
        if operation == "list":
            method_code.extend([
                "    def list_items(self, page: int = 1, limit: int = 10) -> Dict[str, Any]:",
                '        """Retrieve a list of items with pagination"""'
            ])
        elif operation == "create":
            method_code.extend([
                "    def create_item(self, data: Dict[str, Any]) -> Dict[str, Any]:",
                '        """Create a new item"""'
            ])
        elif operation == "read":
            method_code.extend([
                "    def get_item(self, item_id: int) -> Dict[str, Any]:",
                '        """Retrieve a specific item by ID"""'
            ])
        elif operation == "update":
            method_code.extend([
                "    def update_item(self, item_id: int, data: Dict[str, Any]) -> Dict[str, Any]:",
                '        """Update an existing item"""'
            ])
        elif operation == "delete":
            method_code.extend([
                "    def delete_item(self, item_id: int) -> bool:",
                '        """Delete an item"""'
            ])
        
        line_number += 2
        
        # URL construction
        if MissingComponent.URL_CONSTRUCTION in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.URL_CONSTRUCTION, line_number,
                "        # TODO: Construct the API endpoint URL",
                f"Build URL for {endpoint.path} endpoint"
            ))
            method_code.append("        # TODO: Construct the API endpoint URL")
        else:
            if "{" in endpoint.path:
                url_template = endpoint.path.replace("{", "{").replace("}", "}")
                method_code.append(f"        url = self.base_url + f'{url_template}'")
            else:
                method_code.append(f"        url = self.base_url + '{endpoint.path}'")
        
        line_number += 1
        
        # Parameters for GET requests
        if operation == "list" and MissingComponent.PARAMETERS in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.PARAMETERS, line_number,
                "        # TODO: Set up query parameters",
                "Add pagination and filtering parameters"
            ))
            method_code.append("        # TODO: Set up query parameters")
        elif operation == "list":
            method_code.append("        params = {'page': page, 'limit': limit}")
        
        line_number += 1
        
        # Request body for POST/PUT
        if operation in ["create", "update"] and MissingComponent.REQUEST_BODY in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.REQUEST_BODY, line_number,
                "        # TODO: Prepare request body",
                "Serialize data to JSON format"
            ))
            method_code.append("        # TODO: Prepare request body")
        elif operation in ["create", "update"]:
            method_code.append("        json_data = json.dumps(data)")
        
        line_number += 1
        
        # HTTP request
        if MissingComponent.HTTP_METHOD in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.HTTP_METHOD, line_number,
                "        # TODO: Make HTTP request",
                f"Use {endpoint.method.value} method for this operation"
            ))
            method_code.append("        # TODO: Make HTTP request")
        else:
            method = endpoint.method.value.lower()
            if operation == "list":
                method_code.append(f"        response = requests.{method}(url, headers=self.headers, params=params)")
            elif operation in ["create", "update"]:
                method_code.append(f"        response = requests.{method}(url, headers=self.headers, data=json_data)")
            else:
                method_code.append(f"        response = requests.{method}(url, headers=self.headers)")
        
        line_number += 1
        
        # Error handling
        if MissingComponent.ERROR_HANDLING in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.ERROR_HANDLING, line_number,
                "        # TODO: Handle HTTP errors",
                "Check response status and handle errors appropriately"
            ))
            method_code.append("        # TODO: Handle HTTP errors")
        else:
            method_code.extend([
                "        try:",
                "            response.raise_for_status()",
                "        except requests.RequestException as e:",
                "            raise Exception(f'API request failed: {e}')"
            ])
            line_number += 4
        
        # Response parsing
        if MissingComponent.RESPONSE_PARSING in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.RESPONSE_PARSING, line_number,
                "        # TODO: Parse and return response",
                "Extract data from JSON response"
            ))
            method_code.append("        # TODO: Parse and return response")
        else:
            if operation == "delete":
                method_code.append("        return response.status_code == 204")
            else:
                method_code.append("        return response.json()")
        
        method_code.append("")
        
        return method_code, gaps
    
    def _generate_helper_methods(
        self,
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel,
        start_line: int
    ) -> tuple[List[str], List[CodeGap]]:
        """Generate helper methods for advanced functionality"""
        
        helper_code = []
        gaps = []
        line_number = start_line
        
        # Input validation helper
        helper_code.extend([
            "    def _validate_data(self, data: Dict[str, Any], required_fields: List[str]) -> None:",
            '        """Validate input data before API calls"""'
        ])
        line_number += 2
        
        if MissingComponent.VALIDATION in missing_components:
            gaps.append(self._create_gap(
                MissingComponent.VALIDATION, line_number,
                "        # TODO: Implement data validation logic",
                "Check for required fields and data types"
            ))
            helper_code.append("        # TODO: Implement data validation logic")
        else:
            helper_code.extend([
                "        for field in required_fields:",
                "            if field not in data:",
                "                raise ValueError(f'Required field missing: {field}')"
            ])
        
        helper_code.append("")
        line_number += 4
        
        # Retry logic helper
        if difficulty == DifficultyLevel.ADVANCED:
            helper_code.extend([
                "    def _make_request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:",
                '        """Make HTTP request with retry logic"""'
            ])
            line_number += 2
            
            if MissingComponent.RETRY_LOGIC in missing_components:
                gaps.append(self._create_gap(
                    MissingComponent.RETRY_LOGIC, line_number,
                    "        # TODO: Implement retry logic with exponential backoff",
                    "Retry failed requests with increasing delays"
                ))
                helper_code.append("        # TODO: Implement retry logic with exponential backoff")
            else:
                helper_code.extend([
                    "        import time",
                    "        max_retries = 3",
                    "        for attempt in range(max_retries):",
                    "            try:",
                    "                response = getattr(requests, method)(url, **kwargs)",
                    "                return response",
                    "            except requests.RequestException as e:",
                    "                if attempt == max_retries - 1:",
                    "                    raise e",
                    "                time.sleep(2 ** attempt)  # Exponential backoff"
                ])
            
            helper_code.append("")
        
        return helper_code, gaps
    
    def _generate_usage_example(self, operation: str, resource: str) -> Optional[str]:
        """Generate usage example for an operation"""
        examples = {
            "list": f"items = client.list_items(page=1, limit=10)",
            "create": f"new_item = client.create_item({{'name': 'New {resource}'}})",
            "read": f"item = client.get_item(item_id=123)",
            "update": f"updated = client.update_item(123, {{'name': 'Updated {resource}'}})",
            "delete": f"success = client.delete_item(item_id=123)"
        }
        return examples.get(operation)
    
    def _create_gap(
        self, 
        component: MissingComponent, 
        line_number: int, 
        placeholder: str, 
        hint: str
    ) -> CodeGap:
        """Create a code gap with the given parameters"""
        return CodeGap(
            component=component,
            line_number=line_number,
            placeholder=placeholder,
            hint=hint,
            expected_solution="# Implementation needed",
            difficulty=DifficultyLevel.INTERMEDIATE
        )


class AsyncIntegrationPattern(IntegrationPattern):
    """Template generator for async API integrations using httpx"""
    
    def generate_template(
        self,
        endpoints: List[EndpointSpec],
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel
    ) -> CodeTemplate:
        """Generate async integration template"""
        
        code_parts = []
        gaps = []
        line_number = 1
        
        # Async imports
        if MissingComponent.IMPORTS in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.IMPORTS,
                line_number=line_number,
                placeholder="# TODO: Import async HTTP client and asyncio",
                hint="Import httpx for async HTTP requests and asyncio for async functionality"
            ))
            code_parts.append("# TODO: Import async HTTP client and asyncio")
        else:
            code_parts.extend([
                "import asyncio",
                "import httpx",
                "import json",
                "from typing import Dict, List, Optional, Any"
            ])
        
        code_parts.append("")
        line_number += len([p for p in code_parts if p])
        
        # Async client class
        primary_endpoint = endpoints[0] if endpoints else None
        resource_name = self._extract_resource_name(primary_endpoint.path if primary_endpoint else "/items")
        
        code_parts.extend([
            f"class Async{resource_name.title()}Client:",
            f'    """Async API client for {resource_name} operations"""',
            "",
            "    def __init__(self, base_url: str, api_key: str = None):",
            "        self.base_url = base_url.rstrip('/')",
            "        self.api_key = api_key"
        ])
        line_number += 6
        
        # Authentication
        if MissingComponent.AUTHENTICATION in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.AUTHENTICATION,
                line_number=line_number,
                placeholder="        # TODO: Set up async client with authentication",
                hint="Configure httpx.AsyncClient with authentication headers"
            ))
            code_parts.append("        # TODO: Set up async client with authentication")
        else:
            code_parts.extend([
                "        self.headers = {'Content-Type': 'application/json'}",
                "        if api_key:",
                "            self.headers['Authorization'] = f'Bearer {api_key}'"
            ])
        
        code_parts.append("")
        line_number += len([p for p in code_parts[-10:] if p])
        
        # Generate async method for primary endpoint
        if primary_endpoint:
            method_code, method_gaps = self._generate_async_method(
                primary_endpoint, missing_components, line_number
            )
            code_parts.extend(method_code)
            gaps.extend(method_gaps)
        
        # Context manager methods
        code_parts.extend([
            "",
            "    async def __aenter__(self):",
            "        self.client = httpx.AsyncClient(headers=self.headers)",
            "        return self",
            "",
            "    async def __aexit__(self, exc_type, exc_val, exc_tb):",
            "        await self.client.aclose()",
            ""
        ])
        
        # Usage example
        code_parts.extend([
            "",
            "# Async Usage Example",
            "async def main():",
            f"    async with Async{resource_name.title()}Client(",
            "        base_url='https://api.example.com',",
            "        api_key='your-api-key'",
            "    ) as client:",
            f"        result = await client.fetch_{resource_name}()",
            "        print(result)",
            "",
            "if __name__ == '__main__':",
            "    asyncio.run(main())"
        ])
        
        final_code = "\n".join(code_parts)
        
        metadata = {
            "pattern": "Async Integration",
            "resource": resource_name,
            "uses_httpx": True,
            "total_gaps": len(gaps),
            "difficulty": difficulty.value
        }
        
        return CodeTemplate(
            code=final_code,
            gaps=gaps,
            metadata=metadata,
            description=f"Async integration template for {resource_name}"
        )
    
    def get_pattern_name(self) -> str:
        return "Async Integration"
    
    def _extract_resource_name(self, path: str) -> str:
        """Extract resource name from path"""
        parts = path.strip('/').split('/')
        return parts[0] if parts else "resource"
    
    def _generate_async_method(
        self,
        endpoint: EndpointSpec,
        missing_components: List[MissingComponent],
        start_line: int
    ) -> tuple[List[str], List[CodeGap]]:
        """Generate an async method for the endpoint"""
        
        method_code = []
        gaps = []
        line_number = start_line
        
        resource_name = self._extract_resource_name(endpoint.path)
        method_name = f"fetch_{resource_name}"
        
        method_code.extend([
            f"    async def {method_name}(self) -> Dict[str, Any]:",
            f'        """Async fetch {resource_name} data"""'
        ])
        line_number += 2
        
        # URL construction
        if MissingComponent.URL_CONSTRUCTION in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.URL_CONSTRUCTION,
                line_number=line_number,
                placeholder="        # TODO: Construct API URL",
                hint="Build the complete URL for the API endpoint"
            ))
            method_code.append("        # TODO: Construct API URL")
        else:
            method_code.append(f"        url = self.base_url + '{endpoint.path}'")
        
        line_number += 1
        
        # Async HTTP request
        if MissingComponent.HTTP_METHOD in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.HTTP_METHOD,
                line_number=line_number,
                placeholder="        # TODO: Make async HTTP request",
                hint="Use httpx client to make async GET request"
            ))
            method_code.append("        # TODO: Make async HTTP request")
        else:
            method = endpoint.method.value.lower()
            method_code.append(f"        response = await self.client.{method}(url)")
        
        line_number += 1
        
        # Error handling
        if MissingComponent.ERROR_HANDLING in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.ERROR_HANDLING,
                line_number=line_number,
                placeholder="        # TODO: Handle async errors",
                hint="Handle httpx exceptions and HTTP errors"
            ))
            method_code.append("        # TODO: Handle async errors")
        else:
            method_code.extend([
                "        try:",
                "            response.raise_for_status()",
                "        except httpx.HTTPError as e:",
                "            raise Exception(f'HTTP error occurred: {e}')"
            ])
            line_number += 4
        
        # Response parsing
        if MissingComponent.RESPONSE_PARSING in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.RESPONSE_PARSING,
                line_number=line_number,
                placeholder="        # TODO: Parse JSON response",
                hint="Extract and return JSON data from response"
            ))
            method_code.append("        # TODO: Parse JSON response")
        else:
            method_code.append("        return response.json()")
        
        method_code.append("")
        
        return method_code, gaps


class PatternBasedTemplateGenerator(BaseCodeTemplateGenerator):
    """Main generator that uses different patterns based on API analysis"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.analyzer = APISchemaAnalyzer()
        self.patterns = {
            "crud": CRUDIntegrationPattern(seed),
            "async": AsyncIntegrationPattern()
        }
    
    def generate_template(
        self,
        endpoints: List[EndpointSpec],
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    ) -> CodeTemplate:
        """Generate template using the most appropriate pattern"""
        
        # Analyze the API schema
        analysis = self.analyzer.analyze_schema(endpoints)
        
        # Choose pattern based on analysis
        pattern_name = self._choose_pattern(analysis)
        pattern = self.patterns[pattern_name]
        
        # Generate template
        template = pattern.generate_template(endpoints, missing_components, difficulty)
        
        # Add analysis metadata
        template.metadata.update({
            "schema_analysis": {
                "complexity": analysis.complexity.value,
                "total_endpoints": analysis.total_endpoints,
                "resources": analysis.resources,
                "auth_type": analysis.auth_requirements.value
            },
            "chosen_pattern": pattern_name
        })
        
        return template
    
    def _choose_pattern(self, analysis: SchemaAnalysis) -> str:
        """Choose the most appropriate pattern based on analysis"""
        
        # Check for CRUD patterns
        pattern_types = [p.pattern_type for p in analysis.patterns]
        if "crud_complete" in pattern_types:
            return "crud"
        
        # For complex APIs, suggest async pattern
        if analysis.complexity.value == "complex":
            return "async"
        
        # Default to CRUD for most cases
        return "crud"
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available patterns"""
        return list(self.patterns.keys())
    
    def generate_with_specific_pattern(
        self,
        pattern_name: str,
        endpoints: List[EndpointSpec],
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    ) -> CodeTemplate:
        """Generate template using a specific pattern"""
        
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.patterns[pattern_name]
        return pattern.generate_template(endpoints, missing_components, difficulty)