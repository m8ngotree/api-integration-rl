import re
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass

from ..data_generation.endpoint_generator import EndpointSpec, HTTPMethod


class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class MissingComponent(Enum):
    IMPORTS = "imports"
    AUTHENTICATION = "authentication"
    ERROR_HANDLING = "error_handling"
    REQUEST_BODY = "request_body"
    RESPONSE_PARSING = "response_parsing"
    URL_CONSTRUCTION = "url_construction"
    HTTP_METHOD = "http_method"
    HEADERS = "headers"
    PARAMETERS = "parameters"
    VALIDATION = "validation"
    LOGGING = "logging"
    RETRY_LOGIC = "retry_logic"


@dataclass
class CodeGap:
    """Represents a gap in the code that needs to be filled"""
    component: MissingComponent
    line_number: int
    placeholder: str
    hint: Optional[str] = None
    expected_solution: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE


@dataclass
class CodeTemplate:
    """Represents a generated code template with gaps"""
    code: str
    gaps: List[CodeGap]
    metadata: Dict[str, Any]
    language: str = "python"
    description: str = ""


class BaseCodeTemplateGenerator(ABC):
    """Base class for generating partial code templates"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
        
        # Common patterns for different components
        self.import_patterns = [
            "import requests",
            "import httpx",
            "from requests.auth import HTTPBasicAuth",
            "import json",
            "import os",
            "from typing import Dict, List, Optional, Any",
            "import logging",
            "from dataclasses import dataclass",
            "from pydantic import BaseModel",
            "import asyncio"
        ]
        
        self.auth_patterns = {
            "api_key": "headers['Authorization'] = f'Bearer {api_key}'",
            "basic_auth": "auth = HTTPBasicAuth(username, password)",
            "custom_header": "headers['X-API-Key'] = api_key"
        }
        
        self.error_handling_patterns = [
            "response.raise_for_status()",
            "if response.status_code != 200:",
            "try:\n    # API call\nexcept requests.RequestException as e:",
            "except requests.Timeout:",
            "except requests.ConnectionError:"
        ]
    
    @abstractmethod
    def generate_template(
        self,
        endpoints: List[EndpointSpec],
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    ) -> CodeTemplate:
        """Generate a code template with specified missing components"""
        pass
    
    def _create_placeholder(self, component: MissingComponent, context: str = "") -> str:
        """Create a placeholder for a missing component"""
        placeholders = {
            MissingComponent.IMPORTS: "# TODO: Add necessary imports",
            MissingComponent.AUTHENTICATION: "# TODO: Add authentication",
            MissingComponent.ERROR_HANDLING: "# TODO: Add error handling",
            MissingComponent.REQUEST_BODY: "# TODO: Construct request body",
            MissingComponent.RESPONSE_PARSING: "# TODO: Parse response data",
            MissingComponent.URL_CONSTRUCTION: "# TODO: Construct API URL",
            MissingComponent.HTTP_METHOD: "# TODO: Choose appropriate HTTP method",
            MissingComponent.HEADERS: "# TODO: Set request headers",
            MissingComponent.PARAMETERS: "# TODO: Add query parameters",
            MissingComponent.VALIDATION: "# TODO: Validate input data",
            MissingComponent.LOGGING: "# TODO: Add logging",
            MissingComponent.RETRY_LOGIC: "# TODO: Implement retry logic"
        }
        
        base_placeholder = placeholders.get(component, "# TODO: Implement missing component")
        
        if context:
            return f"{base_placeholder} ({context})"
        return base_placeholder
    
    def _create_hint(self, component: MissingComponent, endpoint: EndpointSpec) -> str:
        """Create a hint for solving the missing component"""
        hints = {
            MissingComponent.IMPORTS: "Consider what HTTP library to use (requests, httpx, urllib)",
            MissingComponent.AUTHENTICATION: f"This API might need API key, OAuth, or basic auth",
            MissingComponent.ERROR_HANDLING: "Handle HTTP errors, timeouts, and connection issues",
            MissingComponent.REQUEST_BODY: f"Need to serialize data for {endpoint.method.value} request",
            MissingComponent.RESPONSE_PARSING: "Parse JSON response and extract relevant data",
            MissingComponent.URL_CONSTRUCTION: f"Build URL for {endpoint.path} with parameters",
            MissingComponent.HTTP_METHOD: f"Use appropriate method for {endpoint.summary}",
            MissingComponent.HEADERS: "Set Content-Type, Accept, and authentication headers",
            MissingComponent.PARAMETERS: "Add query parameters or path parameters",
            MissingComponent.VALIDATION: "Validate input before making API call",
            MissingComponent.LOGGING: "Log requests, responses, and errors for debugging",
            MissingComponent.RETRY_LOGIC: "Implement exponential backoff for failed requests"
        }
        
        return hints.get(component, "Check the API documentation for requirements")
    
    def _get_expected_solution(self, component: MissingComponent, endpoint: EndpointSpec) -> str:
        """Get the expected solution for a missing component"""
        solutions = {
            MissingComponent.IMPORTS: "import requests\nimport json",
            MissingComponent.AUTHENTICATION: "headers['Authorization'] = f'Bearer {api_key}'",
            MissingComponent.ERROR_HANDLING: "response.raise_for_status()",
            MissingComponent.REQUEST_BODY: "data = json.dumps(payload)",
            MissingComponent.RESPONSE_PARSING: "return response.json()",
            MissingComponent.URL_CONSTRUCTION: f"url = base_url + '{endpoint.path}'",
            MissingComponent.HTTP_METHOD: f"response = requests.{endpoint.method.value.lower()}(url)",
            MissingComponent.HEADERS: "headers = {'Content-Type': 'application/json'}",
            MissingComponent.PARAMETERS: "params = {'page': 1, 'limit': 10}",
            MissingComponent.VALIDATION: "assert isinstance(data, dict)",
            MissingComponent.LOGGING: "logging.info(f'Making {method} request to {url}')",
            MissingComponent.RETRY_LOGIC: "for attempt in range(max_retries):"
        }
        
        return solutions.get(component, "# Implementation needed")
    
    def _shuffle_missing_components(self, components: List[MissingComponent]) -> List[MissingComponent]:
        """Shuffle components to create varied templates"""
        shuffled = components.copy()
        random.shuffle(shuffled)
        return shuffled


class APIIntegrationTemplateGenerator(BaseCodeTemplateGenerator):
    """Generates Python code templates for API integrations"""
    
    def generate_template(
        self,
        endpoints: List[EndpointSpec],
        missing_components: List[MissingComponent],
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    ) -> CodeTemplate:
        """Generate an API integration template with gaps"""
        
        if not endpoints:
            raise ValueError("At least one endpoint is required")
        
        # Select a primary endpoint for the template
        primary_endpoint = endpoints[0]
        
        # Generate base code structure
        code_parts = []
        gaps = []
        line_number = 1
        
        # Generate imports section
        if MissingComponent.IMPORTS in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.IMPORTS,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.IMPORTS),
                hint=self._create_hint(MissingComponent.IMPORTS, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.IMPORTS, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(self._create_placeholder(MissingComponent.IMPORTS))
        else:
            code_parts.append("import requests")
            code_parts.append("import json")
        
        code_parts.append("")
        line_number += len([p for p in code_parts if p])
        
        # Generate class structure
        class_name = self._generate_class_name(primary_endpoint)
        code_parts.append(f"class {class_name}:")
        code_parts.append('    """API client for interacting with the service"""')
        code_parts.append("")
        line_number += 3
        
        # Constructor
        code_parts.append("    def __init__(self, base_url: str, api_key: str = None):")
        code_parts.append("        self.base_url = base_url.rstrip('/')")
        code_parts.append("        self.api_key = api_key")
        code_parts.append("        self.session = requests.Session()")
        
        # Authentication setup
        if MissingComponent.AUTHENTICATION in missing_components:
            line_number += 4
            gaps.append(CodeGap(
                component=MissingComponent.AUTHENTICATION,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.AUTHENTICATION),
                hint=self._create_hint(MissingComponent.AUTHENTICATION, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.AUTHENTICATION, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.AUTHENTICATION)}")
        else:
            code_parts.append("        if api_key:")
            code_parts.append("            self.session.headers.update({'Authorization': f'Bearer {api_key}'})")
        
        code_parts.append("")
        line_number += len([p for p in code_parts[-10:] if p])
        
        # Generate method for primary endpoint
        method_name = self._generate_method_name(primary_endpoint)
        method_signature = self._generate_method_signature(primary_endpoint, difficulty)
        
        code_parts.append(f"    def {method_name}({method_signature}):")
        code_parts.append(f'        """')
        code_parts.append(f'        {primary_endpoint.summary}')
        code_parts.append(f'        """')
        line_number += 4
        
        # URL construction
        if MissingComponent.URL_CONSTRUCTION in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.URL_CONSTRUCTION,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.URL_CONSTRUCTION),
                hint=self._create_hint(MissingComponent.URL_CONSTRUCTION, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.URL_CONSTRUCTION, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.URL_CONSTRUCTION)}")
        else:
            url_path = primary_endpoint.path
            if "{" in url_path:
                # Handle path parameters
                url_path = re.sub(r'\{([^}]+)\}', r'{\\1}', url_path)
                code_parts.append(f"        url = self.base_url + f'{url_path}'")
            else:
                code_parts.append(f"        url = self.base_url + '{url_path}'")
        
        line_number += 1
        
        # Headers
        if MissingComponent.HEADERS in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.HEADERS,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.HEADERS),
                hint=self._create_hint(MissingComponent.HEADERS, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.HEADERS, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.HEADERS)}")
        else:
            code_parts.append("        headers = {'Content-Type': 'application/json'}")
        
        line_number += 1
        
        # Parameters handling
        if primary_endpoint.method == HTTPMethod.GET and MissingComponent.PARAMETERS in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.PARAMETERS,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.PARAMETERS),
                hint=self._create_hint(MissingComponent.PARAMETERS, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.PARAMETERS, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.PARAMETERS)}")
            line_number += 1
        
        # Request body for POST/PUT
        if primary_endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT]:
            if MissingComponent.REQUEST_BODY in missing_components:
                gaps.append(CodeGap(
                    component=MissingComponent.REQUEST_BODY,
                    line_number=line_number,
                    placeholder=self._create_placeholder(MissingComponent.REQUEST_BODY),
                    hint=self._create_hint(MissingComponent.REQUEST_BODY, primary_endpoint),
                    expected_solution=self._get_expected_solution(MissingComponent.REQUEST_BODY, primary_endpoint),
                    difficulty=difficulty
                ))
                code_parts.append(f"        {self._create_placeholder(MissingComponent.REQUEST_BODY)}")
            else:
                code_parts.append("        data = json.dumps(payload) if payload else None")
            line_number += 1
        
        # HTTP method call
        if MissingComponent.HTTP_METHOD in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.HTTP_METHOD,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.HTTP_METHOD),
                hint=self._create_hint(MissingComponent.HTTP_METHOD, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.HTTP_METHOD, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.HTTP_METHOD)}")
        else:
            method = primary_endpoint.method.value.lower()
            if primary_endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT]:
                code_parts.append(f"        response = self.session.{method}(url, headers=headers, data=data)")
            elif primary_endpoint.method == HTTPMethod.GET:
                code_parts.append(f"        response = self.session.{method}(url, headers=headers, params=params)")
            else:
                code_parts.append(f"        response = self.session.{method}(url, headers=headers)")
        
        line_number += 1
        
        # Error handling
        if MissingComponent.ERROR_HANDLING in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.ERROR_HANDLING,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.ERROR_HANDLING),
                hint=self._create_hint(MissingComponent.ERROR_HANDLING, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.ERROR_HANDLING, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.ERROR_HANDLING)}")
        else:
            code_parts.append("        response.raise_for_status()")
        
        line_number += 1
        
        # Response parsing
        if MissingComponent.RESPONSE_PARSING in missing_components:
            gaps.append(CodeGap(
                component=MissingComponent.RESPONSE_PARSING,
                line_number=line_number,
                placeholder=self._create_placeholder(MissingComponent.RESPONSE_PARSING),
                hint=self._create_hint(MissingComponent.RESPONSE_PARSING, primary_endpoint),
                expected_solution=self._get_expected_solution(MissingComponent.RESPONSE_PARSING, primary_endpoint),
                difficulty=difficulty
            ))
            code_parts.append(f"        {self._create_placeholder(MissingComponent.RESPONSE_PARSING)}")
        else:
            if primary_endpoint.method == HTTPMethod.DELETE:
                code_parts.append("        return response.status_code == 204")
            else:
                code_parts.append("        return response.json()")
        
        # Add usage example
        code_parts.append("")
        code_parts.append("")
        code_parts.append("# Usage example:")
        code_parts.append("if __name__ == '__main__':")
        code_parts.append("    client = " + class_name + "(")
        code_parts.append("        base_url='https://api.example.com',")
        code_parts.append("        api_key='your-api-key'")
        code_parts.append("    )")
        code_parts.append("")
        
        # Add example method call
        example_call = self._generate_example_call(primary_endpoint, method_name)
        code_parts.append(f"    result = client.{example_call}")
        code_parts.append("    print(result)")
        
        # Join all code parts
        final_code = "\n".join(code_parts)
        
        # Create metadata
        metadata = {
            "primary_endpoint": {
                "path": primary_endpoint.path,
                "method": primary_endpoint.method.value,
                "summary": primary_endpoint.summary
            },
            "total_gaps": len(gaps),
            "difficulty": difficulty.value,
            "estimated_completion_time": self._estimate_completion_time(gaps, difficulty),
            "learning_objectives": self._get_learning_objectives(missing_components)
        }
        
        return CodeTemplate(
            code=final_code,
            gaps=gaps,
            metadata=metadata,
            language="python",
            description=f"API integration template for {primary_endpoint.summary}"
        )
    
    def _generate_class_name(self, endpoint: EndpointSpec) -> str:
        """Generate a class name based on the endpoint"""
        if "users" in endpoint.path:
            return "UserAPIClient"
        elif "products" in endpoint.path:
            return "ProductAPIClient"
        else:
            return "APIClient"
    
    def _generate_method_name(self, endpoint: EndpointSpec) -> str:
        """Generate a method name based on the endpoint"""
        method_map = {
            HTTPMethod.GET: "get",
            HTTPMethod.POST: "create",
            HTTPMethod.PUT: "update",
            HTTPMethod.DELETE: "delete"
        }
        
        base_method = method_map[endpoint.method]
        
        if "users" in endpoint.path:
            if "{user_id}" in endpoint.path:
                return f"{base_method}_user"
            else:
                return f"{base_method}_users" if base_method == "get" else f"{base_method}_user"
        elif "products" in endpoint.path:
            if "{product_id}" in endpoint.path:
                return f"{base_method}_product"
            else:
                return f"{base_method}_products" if base_method == "get" else f"{base_method}_product"
        else:
            return f"{base_method}_resource"
    
    def _generate_method_signature(self, endpoint: EndpointSpec, difficulty: DifficultyLevel) -> str:
        """Generate method signature based on endpoint and difficulty"""
        base_sig = "self"
        
        # Add parameters based on endpoint
        if "{user_id}" in endpoint.path or "{product_id}" in endpoint.path:
            if "user_id" in endpoint.path:
                base_sig += ", user_id: int"
            else:
                base_sig += ", item_id: int"
        
        if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT]:
            base_sig += ", payload: dict"
        
        if endpoint.method == HTTPMethod.GET and "{" not in endpoint.path:
            if difficulty != DifficultyLevel.BEGINNER:
                base_sig += ", page: int = 1, limit: int = 10"
        
        return base_sig
    
    def _generate_example_call(self, endpoint: EndpointSpec, method_name: str) -> str:
        """Generate an example method call"""
        if "{user_id}" in endpoint.path or "{product_id}" in endpoint.path:
            if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT]:
                return f"{method_name}(1, {{'name': 'Example'}})"
            else:
                return f"{method_name}(1)"
        elif endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT]:
            return f"{method_name}({{'name': 'Example'}})"
        else:
            return f"{method_name}()"
    
    def _estimate_completion_time(self, gaps: List[CodeGap], difficulty: DifficultyLevel) -> int:
        """Estimate completion time in minutes"""
        base_time_per_gap = {
            DifficultyLevel.BEGINNER: 3,
            DifficultyLevel.INTERMEDIATE: 5,
            DifficultyLevel.ADVANCED: 8
        }
        
        return len(gaps) * base_time_per_gap[difficulty]
    
    def _get_learning_objectives(self, missing_components: List[MissingComponent]) -> List[str]:
        """Get learning objectives based on missing components"""
        objectives_map = {
            MissingComponent.IMPORTS: "Understanding Python HTTP libraries and their imports",
            MissingComponent.AUTHENTICATION: "Implementing API authentication mechanisms",
            MissingComponent.ERROR_HANDLING: "Handling HTTP errors and exceptions properly",
            MissingComponent.REQUEST_BODY: "Constructing and serializing request payloads",
            MissingComponent.RESPONSE_PARSING: "Parsing and extracting data from API responses",
            MissingComponent.URL_CONSTRUCTION: "Building URLs with parameters and paths",
            MissingComponent.HTTP_METHOD: "Choosing appropriate HTTP methods for operations",
            MissingComponent.HEADERS: "Setting proper HTTP headers for API requests",
            MissingComponent.PARAMETERS: "Handling query parameters and path variables",
            MissingComponent.VALIDATION: "Validating input data before API calls",
            MissingComponent.LOGGING: "Adding proper logging for debugging and monitoring",
            MissingComponent.RETRY_LOGIC: "Implementing resilient retry mechanisms"
        }
        
        return [objectives_map[component] for component in missing_components if component in objectives_map]