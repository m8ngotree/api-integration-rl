from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ..data_generation.endpoint_generator import EndpointSpec, HTTPMethod
from .code_template_generator import MissingComponent, DifficultyLevel


class APIComplexity(Enum):
    SIMPLE = "simple"          # Basic CRUD operations
    MODERATE = "moderate"      # Multiple resources, some relationships
    COMPLEX = "complex"        # Many resources, complex relationships, advanced features


class AuthType(Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


@dataclass
class APIPattern:
    """Represents a detected pattern in the API"""
    pattern_type: str
    description: str
    endpoints: List[EndpointSpec]
    complexity: APIComplexity
    suggested_components: List[MissingComponent]


@dataclass
class SchemaAnalysis:
    """Complete analysis of an API schema"""
    total_endpoints: int
    endpoints_by_method: Dict[str, int]
    resources: List[str]
    auth_requirements: AuthType
    complexity: APIComplexity
    patterns: List[APIPattern]
    recommended_difficulty: DifficultyLevel
    suggested_missing_components: List[MissingComponent]
    integration_challenges: List[str]


class APISchemaAnalyzer:
    """Analyzes API schemas to determine appropriate code template generation strategies"""
    
    def __init__(self):
        self.resource_keywords = [
            "users", "user", "customers", "customer",
            "products", "product", "items", "item",
            "orders", "order", "bookings", "booking",
            "files", "file", "documents", "document",
            "categories", "category", "tags", "tag"
        ]
        
        self.crud_patterns = {
            "list": (HTTPMethod.GET, lambda path: not self._has_path_params(path)),
            "create": (HTTPMethod.POST, lambda path: not self._has_path_params(path)),
            "read": (HTTPMethod.GET, lambda path: self._has_path_params(path)),
            "update": (HTTPMethod.PUT, lambda path: self._has_path_params(path)),
            "delete": (HTTPMethod.DELETE, lambda path: self._has_path_params(path))
        }
    
    def analyze_schema(self, endpoints: List[EndpointSpec]) -> SchemaAnalysis:
        """Perform comprehensive analysis of API schema"""
        if not endpoints:
            raise ValueError("No endpoints provided for analysis")
        
        # Basic statistics
        total_endpoints = len(endpoints)
        endpoints_by_method = self._count_methods(endpoints)
        resources = self._extract_resources(endpoints)
        
        # Detect authentication requirements
        auth_requirements = self._detect_auth_requirements(endpoints)
        
        # Determine API complexity
        complexity = self._determine_complexity(endpoints, resources)
        
        # Detect patterns
        patterns = self._detect_patterns(endpoints, resources)
        
        # Recommend difficulty level
        recommended_difficulty = self._recommend_difficulty(complexity, patterns, total_endpoints)
        
        # Suggest missing components based on analysis
        suggested_components = self._suggest_missing_components(
            endpoints, patterns, complexity, auth_requirements
        )
        
        # Identify integration challenges
        challenges = self._identify_challenges(endpoints, patterns, complexity)
        
        return SchemaAnalysis(
            total_endpoints=total_endpoints,
            endpoints_by_method=endpoints_by_method,
            resources=resources,
            auth_requirements=auth_requirements,
            complexity=complexity,
            patterns=patterns,
            recommended_difficulty=recommended_difficulty,
            suggested_missing_components=suggested_components,
            integration_challenges=challenges
        )
    
    def suggest_template_variations(self, analysis: SchemaAnalysis) -> List[Dict[str, Any]]:
        """Suggest different template variations based on analysis"""
        variations = []
        
        # Beginner variation
        beginner_components = self._filter_components_by_difficulty(
            analysis.suggested_missing_components, DifficultyLevel.BEGINNER
        )
        variations.append({
            "name": "Beginner Template",
            "difficulty": DifficultyLevel.BEGINNER,
            "missing_components": beginner_components[:3],  # Limit to 3 gaps
            "description": "Basic template focusing on core API interaction",
            "estimated_time": 15
        })
        
        # Intermediate variation
        intermediate_components = self._filter_components_by_difficulty(
            analysis.suggested_missing_components, DifficultyLevel.INTERMEDIATE
        )
        variations.append({
            "name": "Intermediate Template",
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "missing_components": intermediate_components[:5],  # Up to 5 gaps
            "description": "Template including error handling and authentication",
            "estimated_time": 30
        })
        
        # Advanced variation
        advanced_components = analysis.suggested_missing_components
        variations.append({
            "name": "Advanced Template",
            "difficulty": DifficultyLevel.ADVANCED,
            "missing_components": advanced_components,  # All components
            "description": "Complete template with all advanced features",
            "estimated_time": 60
        })
        
        # Pattern-specific variations
        for pattern in analysis.patterns:
            if pattern.pattern_type in ["crud_complete", "pagination_heavy"]:
                variations.append({
                    "name": f"{pattern.pattern_type.title()} Template",
                    "difficulty": analysis.recommended_difficulty,
                    "missing_components": pattern.suggested_components,
                    "description": f"Template optimized for {pattern.description}",
                    "estimated_time": self._estimate_pattern_time(pattern)
                })
        
        return variations
    
    def _count_methods(self, endpoints: List[EndpointSpec]) -> Dict[str, int]:
        """Count endpoints by HTTP method"""
        counts = {}
        for endpoint in endpoints:
            method = endpoint.method.value
            counts[method] = counts.get(method, 0) + 1
        return counts
    
    def _extract_resources(self, endpoints: List[EndpointSpec]) -> List[str]:
        """Extract resource names from endpoint paths"""
        resources = set()
        
        for endpoint in endpoints:
            path_parts = endpoint.path.strip('/').split('/')
            for part in path_parts:
                # Skip path parameters
                if not part.startswith('{') and not part.endswith('}'):
                    # Check if it's a known resource keyword
                    clean_part = part.lower().rstrip('s')  # Remove trailing 's'
                    if clean_part in [kw.rstrip('s') for kw in self.resource_keywords]:
                        resources.add(part.lower())
        
        return sorted(list(resources))
    
    def _detect_auth_requirements(self, endpoints: List[EndpointSpec]) -> AuthType:
        """Detect authentication requirements from endpoints"""
        # Check for auth-related parameters or responses
        auth_indicators = {
            "api_key": ["api_key", "apikey", "key"],
            "bearer_token": ["bearer", "token", "jwt"],
            "basic_auth": ["basic", "username", "password"],
            "oauth2": ["oauth", "authorization_code", "client_id"]
        }
        
        for endpoint in endpoints:
            # Check parameters
            for param in endpoint.parameters:
                param_name = param.get("name", "").lower()
                for auth_type, indicators in auth_indicators.items():
                    if any(indicator in param_name for indicator in indicators):
                        return AuthType(auth_type)
            
            # Check for security requirements in responses
            if endpoint.responses:
                for status_code, response in endpoint.responses.items():
                    if status_code in ["401", "403"]:
                        return AuthType.BEARER_TOKEN  # Most common
        
        # Default assumption for APIs
        return AuthType.API_KEY
    
    def _determine_complexity(self, endpoints: List[EndpointSpec], resources: List[str]) -> APIComplexity:
        """Determine API complexity based on various factors"""
        total_endpoints = len(endpoints)
        resource_count = len(resources)
        
        # Count different types of operations
        has_pagination = any("page" in str(endpoint.parameters) for endpoint in endpoints)
        has_filtering = any(len(endpoint.parameters) > 2 for endpoint in endpoints)
        has_nested_resources = any("/" in endpoint.path.strip("/").replace("{", "").replace("}", "") 
                                  and len(endpoint.path.strip("/").split("/")) > 2 
                                  for endpoint in endpoints)
        
        complexity_score = 0
        
        # Base score from endpoint count
        if total_endpoints > 20:
            complexity_score += 3
        elif total_endpoints > 10:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Resource diversity
        if resource_count > 5:
            complexity_score += 2
        elif resource_count > 2:
            complexity_score += 1
        
        # Advanced features
        if has_pagination:
            complexity_score += 1
        if has_filtering:
            complexity_score += 1
        if has_nested_resources:
            complexity_score += 2
        
        if complexity_score >= 7:
            return APIComplexity.COMPLEX
        elif complexity_score >= 4:
            return APIComplexity.MODERATE
        else:
            return APIComplexity.SIMPLE
    
    def _detect_patterns(self, endpoints: List[EndpointSpec], resources: List[str]) -> List[APIPattern]:
        """Detect common API patterns"""
        patterns = []
        
        # CRUD pattern detection
        crud_pattern = self._detect_crud_pattern(endpoints, resources)
        if crud_pattern:
            patterns.append(crud_pattern)
        
        # Pagination pattern
        pagination_pattern = self._detect_pagination_pattern(endpoints)
        if pagination_pattern:
            patterns.append(pagination_pattern)
        
        # Bulk operations pattern
        bulk_pattern = self._detect_bulk_operations(endpoints)
        if bulk_pattern:
            patterns.append(bulk_pattern)
        
        # Nested resources pattern
        nested_pattern = self._detect_nested_resources(endpoints)
        if nested_pattern:
            patterns.append(nested_pattern)
        
        return patterns
    
    def _detect_crud_pattern(self, endpoints: List[EndpointSpec], resources: List[str]) -> Optional[APIPattern]:
        """Detect CRUD (Create, Read, Update, Delete) patterns"""
        crud_endpoints = []
        resource_operations = {}
        
        for resource in resources:
            operations = []
            resource_endpoints = [ep for ep in endpoints if resource in ep.path.lower()]
            
            for endpoint in resource_endpoints:
                if endpoint.method == HTTPMethod.GET and not self._has_path_params(endpoint.path):
                    operations.append("list")
                elif endpoint.method == HTTPMethod.POST and not self._has_path_params(endpoint.path):
                    operations.append("create")
                elif endpoint.method == HTTPMethod.GET and self._has_path_params(endpoint.path):
                    operations.append("read")
                elif endpoint.method == HTTPMethod.PUT and self._has_path_params(endpoint.path):
                    operations.append("update")
                elif endpoint.method == HTTPMethod.DELETE and self._has_path_params(endpoint.path):
                    operations.append("delete")
            
            resource_operations[resource] = operations
            crud_endpoints.extend(resource_endpoints)
        
        # Check if we have complete CRUD for any resource
        complete_crud_resources = [
            resource for resource, ops in resource_operations.items()
            if len(set(ops) & {"list", "create", "read", "update", "delete"}) >= 4
        ]
        
        if complete_crud_resources:
            return APIPattern(
                pattern_type="crud_complete",
                description=f"Complete CRUD operations for {', '.join(complete_crud_resources)}",
                endpoints=crud_endpoints,
                complexity=APIComplexity.MODERATE,
                suggested_components=[
                    MissingComponent.HTTP_METHOD,
                    MissingComponent.REQUEST_BODY,
                    MissingComponent.ERROR_HANDLING,
                    MissingComponent.RESPONSE_PARSING
                ]
            )
        
        return None
    
    def _detect_pagination_pattern(self, endpoints: List[EndpointSpec]) -> Optional[APIPattern]:
        """Detect pagination patterns"""
        pagination_endpoints = []
        
        for endpoint in endpoints:
            if endpoint.method == HTTPMethod.GET:
                # Check for pagination parameters
                param_names = [p.get("name", "").lower() for p in endpoint.parameters]
                if any(param in param_names for param in ["page", "limit", "offset", "per_page"]):
                    pagination_endpoints.append(endpoint)
        
        if len(pagination_endpoints) >= 2:
            return APIPattern(
                pattern_type="pagination_heavy",
                description="Multiple endpoints with pagination support",
                endpoints=pagination_endpoints,
                complexity=APIComplexity.MODERATE,
                suggested_components=[
                    MissingComponent.PARAMETERS,
                    MissingComponent.RESPONSE_PARSING,
                    MissingComponent.VALIDATION
                ]
            )
        
        return None
    
    def _detect_bulk_operations(self, endpoints: List[EndpointSpec]) -> Optional[APIPattern]:
        """Detect bulk operation patterns"""
        bulk_endpoints = []
        
        for endpoint in endpoints:
            path_lower = endpoint.path.lower()
            summary_lower = endpoint.summary.lower()
            
            if any(keyword in path_lower or keyword in summary_lower 
                   for keyword in ["bulk", "batch", "multiple", "mass"]):
                bulk_endpoints.append(endpoint)
        
        if bulk_endpoints:
            return APIPattern(
                pattern_type="bulk_operations",
                description="Bulk/batch operations available",
                endpoints=bulk_endpoints,
                complexity=APIComplexity.COMPLEX,
                suggested_components=[
                    MissingComponent.REQUEST_BODY,
                    MissingComponent.ERROR_HANDLING,
                    MissingComponent.VALIDATION,
                    MissingComponent.RETRY_LOGIC
                ]
            )
        
        return None
    
    def _detect_nested_resources(self, endpoints: List[EndpointSpec]) -> Optional[APIPattern]:
        """Detect nested resource patterns"""
        nested_endpoints = []
        
        for endpoint in endpoints:
            path_parts = [part for part in endpoint.path.split("/") if part and not part.startswith("{")]
            if len(path_parts) > 1:
                nested_endpoints.append(endpoint)
        
        if len(nested_endpoints) >= 3:
            return APIPattern(
                pattern_type="nested_resources",
                description="Nested resource relationships",
                endpoints=nested_endpoints,
                complexity=APIComplexity.COMPLEX,
                suggested_components=[
                    MissingComponent.URL_CONSTRUCTION,
                    MissingComponent.PARAMETERS,
                    MissingComponent.VALIDATION
                ]
            )
        
        return None
    
    def _recommend_difficulty(
        self, 
        complexity: APIComplexity, 
        patterns: List[APIPattern], 
        endpoint_count: int
    ) -> DifficultyLevel:
        """Recommend difficulty level based on analysis"""
        if complexity == APIComplexity.SIMPLE and endpoint_count <= 5:
            return DifficultyLevel.BEGINNER
        elif complexity == APIComplexity.COMPLEX or endpoint_count > 15:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.INTERMEDIATE
    
    def _suggest_missing_components(
        self,
        endpoints: List[EndpointSpec],
        patterns: List[APIPattern],
        complexity: APIComplexity,
        auth_type: AuthType
    ) -> List[MissingComponent]:
        """Suggest which components to make missing based on analysis"""
        components = []
        
        # Always include basic components
        components.extend([
            MissingComponent.IMPORTS,
            MissingComponent.ERROR_HANDLING,
            MissingComponent.RESPONSE_PARSING
        ])
        
        # Add authentication if required
        if auth_type != AuthType.NONE:
            components.append(MissingComponent.AUTHENTICATION)
        
        # Add method-specific components
        methods = [ep.method for ep in endpoints]
        if any(method in [HTTPMethod.POST, HTTPMethod.PUT] for method in methods):
            components.append(MissingComponent.REQUEST_BODY)
        
        if any(method == HTTPMethod.GET for method in methods):
            components.append(MissingComponent.PARAMETERS)
        
        # Add complexity-based components
        if complexity in [APIComplexity.MODERATE, APIComplexity.COMPLEX]:
            components.extend([
                MissingComponent.URL_CONSTRUCTION,
                MissingComponent.HEADERS,
                MissingComponent.VALIDATION
            ])
        
        if complexity == APIComplexity.COMPLEX:
            components.extend([
                MissingComponent.LOGGING,
                MissingComponent.RETRY_LOGIC
            ])
        
        # Add pattern-specific components
        for pattern in patterns:
            components.extend(pattern.suggested_components)
        
        # Remove duplicates while preserving order
        unique_components = []
        seen = set()
        for component in components:
            if component not in seen:
                unique_components.append(component)
                seen.add(component)
        
        return unique_components
    
    def _identify_challenges(
        self,
        endpoints: List[EndpointSpec],
        patterns: List[APIPattern],
        complexity: APIComplexity
    ) -> List[str]:
        """Identify potential integration challenges"""
        challenges = []
        
        if complexity == APIComplexity.COMPLEX:
            challenges.append("Complex API with many endpoints requires careful error handling")
        
        # Check for authentication challenges
        challenges.append("API authentication setup required")
        
        # Pattern-specific challenges
        pattern_types = [p.pattern_type for p in patterns]
        if "pagination_heavy" in pattern_types:
            challenges.append("Pagination handling across multiple endpoints")
        
        if "nested_resources" in pattern_types:
            challenges.append("Complex URL construction for nested resources")
        
        if "bulk_operations" in pattern_types:
            challenges.append("Bulk operation error handling and validation")
        
        # Method-specific challenges
        methods = [ep.method.value for ep in endpoints]
        if "POST" in methods or "PUT" in methods:
            challenges.append("Request body serialization and validation")
        
        return challenges
    
    def _has_path_params(self, path: str) -> bool:
        """Check if path has parameters like {id}"""
        return "{" in path and "}" in path
    
    def _filter_components_by_difficulty(
        self, 
        components: List[MissingComponent], 
        difficulty: DifficultyLevel
    ) -> List[MissingComponent]:
        """Filter components appropriate for difficulty level"""
        difficulty_mapping = {
            DifficultyLevel.BEGINNER: [
                MissingComponent.IMPORTS,
                MissingComponent.HTTP_METHOD,
                MissingComponent.URL_CONSTRUCTION
            ],
            DifficultyLevel.INTERMEDIATE: [
                MissingComponent.IMPORTS,
                MissingComponent.AUTHENTICATION,
                MissingComponent.ERROR_HANDLING,
                MissingComponent.REQUEST_BODY,
                MissingComponent.RESPONSE_PARSING
            ],
            DifficultyLevel.ADVANCED: components  # All components
        }
        
        allowed_components = difficulty_mapping[difficulty]
        return [c for c in components if c in allowed_components]
    
    def _estimate_pattern_time(self, pattern: APIPattern) -> int:
        """Estimate completion time for pattern-specific templates"""
        base_times = {
            "crud_complete": 45,
            "pagination_heavy": 30,
            "bulk_operations": 60,
            "nested_resources": 50
        }
        return base_times.get(pattern.pattern_type, 30)