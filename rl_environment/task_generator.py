import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from data_generation.endpoint_generator import EndpointSpec, HTTPMethod
from data_generation.api_schema_generator import APISchemaGenerator
# Removed unused imports - MissingComponent, DifficultyLevel, RewardComponent


class TaskType(Enum):
    BASIC_GET_REQUEST = "basic_get_request"
    POST_CREATE_RESOURCE = "post_create_resource"
    PUT_UPDATE_RESOURCE = "put_update_resource"
    DELETE_RESOURCE = "delete_resource"
    AUTHENTICATION_SETUP = "authentication_setup"
    ERROR_HANDLING = "error_handling"
    PAGINATION_HANDLING = "pagination_handling"
    BULK_OPERATIONS = "bulk_operations"
    RESPONSE_VALIDATION = "response_validation"
    ASYNC_API_CALLS = "async_api_calls"


class TaskDifficulty(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class APIDocumentation:
    """API documentation for a task"""
    title: str
    base_url: str
    authentication: Dict[str, Any]
    endpoints: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    error_codes: Dict[str, str]
    rate_limits: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class SuccessCriteria:
    """Success criteria for task completion"""
    required_api_calls: List[Dict[str, Any]]
    expected_outputs: List[str]
    forbidden_patterns: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    code_quality_requirements: Dict[str, Any] = field(default_factory=dict)
    minimum_score: float = 5.0


@dataclass
class LearningTask:
    """Complete learning task with documentation, code, and criteria"""
    task_id: str
    task_type: TaskType
    difficulty: TaskDifficulty
    title: str
    description: str
    api_documentation: APIDocumentation
    starter_code: str
    solution_template: str
    success_criteria: SuccessCriteria
    hints: List[str] = field(default_factory=list)
    estimated_time: int = 30  # minutes
    learning_objectives: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTaskGenerator(ABC):
    """Base class for task generators"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
        
        self.schema_generator = APISchemaGenerator(seed=seed)
    
    @abstractmethod
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        """Generate a learning task"""
        pass
    
    def _create_task_id(self, task_type: TaskType, difficulty: TaskDifficulty) -> str:
        """Create unique task ID"""
        timestamp = random.randint(1000, 9999)
        return f"{task_type.value}_{difficulty.value}_{timestamp}"
    
    def _generate_api_documentation(
        self,
        endpoints: List[EndpointSpec],
        title: str,
        base_url: str = "https://api.example.com"
    ) -> APIDocumentation:
        """Generate API documentation from endpoints"""
        
        # Convert endpoints to documentation format
        endpoint_docs = []
        for endpoint in endpoints:
            doc = {
                "path": endpoint.path,
                "method": endpoint.method.value,
                "summary": endpoint.summary,
                "description": endpoint.description,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses,
                "tags": endpoint.tags
            }
            endpoint_docs.append(doc)
        
        # Generate examples
        examples = []
        for endpoint in endpoints[:2]:  # First 2 endpoints
            if endpoint.method == HTTPMethod.GET:
                examples.append({
                    "title": f"Get {endpoint.path.split('/')[-1]}",
                    "request": f"GET {endpoint.path}",
                    "response": {"status": 200, "data": "Sample response data"}
                })
            elif endpoint.method == HTTPMethod.POST:
                examples.append({
                    "title": f"Create {endpoint.path.split('/')[-1]}",
                    "request": f"POST {endpoint.path}",
                    "body": {"name": "Example", "value": "test"},
                    "response": {"status": 201, "data": {"id": 123, "name": "Example"}}
                })
        
        return APIDocumentation(
            title=title,
            base_url=base_url,
            authentication={
                "type": "Bearer Token",
                "header": "Authorization",
                "format": "Bearer {token}"
            },
            endpoints=endpoint_docs,
            examples=examples,
            error_codes={
                "400": "Bad Request - Invalid parameters",
                "401": "Unauthorized - Invalid or missing authentication",
                "403": "Forbidden - Insufficient permissions",
                "404": "Not Found - Resource not found",
                "422": "Unprocessable Entity - Validation failed",
                "500": "Internal Server Error - Server error"
            },
            rate_limits={
                "requests_per_minute": 100,
                "requests_per_hour": 1000
            }
        )


class BasicGetRequestGenerator(BaseTaskGenerator):
    """Generates basic GET request tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.BEGINNER,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        
        if random.choice([True, False]):
            endpoints = endpoint_gen.get_user_endpoints()[:1]  # GET /users
            resource_name = "users"
        else:
            endpoints = endpoint_gen.get_product_endpoints()[:1]  # GET /products
            resource_name = "products"
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            f"{resource_name.title()} API",
            "https://api.example.com"
        )
        
        # Generate starter code based on difficulty
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = f'''
# TODO: Import the necessary libraries for HTTP requests

# TODO: Set up the base URL and any required headers
base_url = "https://api.example.com"

# TODO: Make a GET request to retrieve {resource_name}
# The endpoint is: GET /{resource_name}

# TODO: Handle the response and print the results
'''
        else:
            starter_code = f'''
import requests

class {resource_name.title()}Client:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        # TODO: Set up any required headers
    
    def get_{resource_name}(self):
        """Retrieve {resource_name} from the API"""
        # TODO: Construct the URL
        
        # TODO: Make the GET request
        
        # TODO: Handle the response
        pass

# Usage
client = {resource_name.title()}Client("https://api.example.com")
result = client.get_{resource_name}()
print(result)
'''
        
        # Solution template
        solution_template = f'''
import requests

def get_{resource_name}(base_url: str) -> dict:
    """Retrieve {resource_name} from the API"""
    url = f"{{base_url}}/{resource_name}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching {resource_name}: {{e}}")
        return {{"error": str(e)}}

# Usage
result = get_{resource_name}("https://api.example.com")
print(f"Retrieved {{len(result.get('items', []))}} {resource_name}")
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "GET",
                    "path": f"/{resource_name}",
                    "expected_status": [200]
                }
            ],
            expected_outputs=[
                f"GET /{resource_name}",
                "200"
            ],
            performance_requirements={
                "max_execution_time": 10.0,
                "max_memory_mb": 50
            },
            minimum_score=3.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.BASIC_GET_REQUEST, difficulty),
            task_type=TaskType.BASIC_GET_REQUEST,
            difficulty=difficulty,
            title=f"Retrieve {resource_name.title()} via GET Request",
            description=f"Learn how to make a basic GET request to retrieve {resource_name} from an API endpoint.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Use the requests library for making HTTP calls",
                f"The endpoint URL should be base_url + '/{resource_name}'",
                "Don't forget to handle potential errors",
                "Parse the JSON response to access the data"
            ],
            estimated_time=15 if difficulty == TaskDifficulty.BEGINNER else 25,
            learning_objectives=[
                "Understand basic HTTP GET requests",
                "Learn to use the requests library",
                "Handle API responses and errors",
                "Parse JSON data from APIs"
            ],
            tags=["http", "get", "api", "requests", resource_name]
        )


class PostCreateResourceGenerator(BaseTaskGenerator):
    """Generates POST request tasks for creating resources"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        
        resource_choice = random.choice(["user", "product"])
        if resource_choice == "user":
            endpoints = [ep for ep in endpoint_gen.get_user_endpoints() if ep.method == HTTPMethod.POST]
            resource_name = "user"
            sample_data = {
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe"
            }
        else:
            endpoints = [ep for ep in endpoint_gen.get_product_endpoints() if ep.method == HTTPMethod.POST]
            resource_name = "product"
            sample_data = {
                "name": "Awesome Widget",
                "description": "A really great widget",
                "price": 29.99,
                "category": "electronics"
            }
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            f"{resource_name.title()} Creation API"
        )
        
        # Add specific example data
        api_docs.examples.append({
            "title": f"Create {resource_name}",
            "request": f"POST /{resource_name}s",
            "headers": {"Content-Type": "application/json"},
            "body": sample_data,
            "response": {"status": 201, "data": {**sample_data, "id": 123}}
        })
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = f'''
import requests
import json

# Sample {resource_name} data to create
{resource_name}_data = {json.dumps(sample_data, indent=4)}

base_url = "https://api.example.com"

# TODO: Create the URL for creating a {resource_name}
# Endpoint: POST /{resource_name}s

# TODO: Set up the headers (Content-Type: application/json)

# TODO: Make a POST request with the {resource_name} data

# TODO: Check if the request was successful (status code 201)

# TODO: Print the created {resource_name} information
'''
        else:
            starter_code = f'''
import requests
import json
from typing import Dict, Any, Optional

class {resource_name.title()}Creator:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {{"Content-Type": "application/json"}}
    
    def create_{resource_name}(self, {resource_name}_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new {resource_name}"""
        # TODO: Validate the input data
        
        # TODO: Construct the API endpoint URL
        
        # TODO: Make the POST request with proper error handling
        
        # TODO: Return the created {resource_name} data or None if failed
        pass
    
    def _validate_{resource_name}_data(self, data: Dict[str, Any]) -> bool:
        """Validate {resource_name} data before sending"""
        # TODO: Add validation logic
        return True

# Usage
creator = {resource_name.title()}Creator("https://api.example.com")
{resource_name}_data = {json.dumps(sample_data, indent=4)}

result = creator.create_{resource_name}({resource_name}_data)
if result:
    print(f"Created {resource_name}: {{result.get('id', 'unknown')}}")
else:
    print("Failed to create {resource_name}")
'''
        
        # Solution template
        solution_template = f'''
import requests
import json
from typing import Dict, Any, Optional

def create_{resource_name}(base_url: str, {resource_name}_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a new {resource_name} via POST request"""
    url = f"{{base_url}}/{resource_name}s"
    headers = {{"Content-Type": "application/json"}}
    
    try:
        response = requests.post(url, json={resource_name}_data, headers=headers)
        response.raise_for_status()
        
        if response.status_code == 201:
            created_{resource_name} = response.json()
            print(f"✅ Created {resource_name}: {{created_{resource_name}.get('id', 'unknown')}}")
            return created_{resource_name}
        else:
            print(f"Unexpected status code: {{response.status_code}}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Error creating {resource_name}: {{e}}")
        return None

# Usage
{resource_name}_data = {json.dumps(sample_data, indent=4)}
result = create_{resource_name}("https://api.example.com", {resource_name}_data)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "POST",
                    "path": f"/{resource_name}s",
                    "expected_status": [200, 201],
                    "requires_body": True
                }
            ],
            expected_outputs=[
                f"POST /{resource_name}s",
                "201",
                "Created"
            ],
            code_quality_requirements={
                "has_error_handling": True,
                "has_json_content_type": True
            },
            minimum_score=5.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.POST_CREATE_RESOURCE, difficulty),
            task_type=TaskType.POST_CREATE_RESOURCE,
            difficulty=difficulty,
            title=f"Create {resource_name.title()} via POST Request",
            description=f"Learn how to create a new {resource_name} by sending a POST request with JSON data.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Set Content-Type header to 'application/json'",
                "Use requests.post() with json parameter",
                "Check for 201 Created status code",
                "Handle potential validation errors (400, 422)"
            ],
            estimated_time=20 if difficulty == TaskDifficulty.BEGINNER else 35,
            learning_objectives=[
                "Understand HTTP POST requests",
                "Learn to send JSON data in requests",
                "Handle creation responses and errors",
                "Validate input data before sending"
            ],
            tags=["http", "post", "create", "json", resource_name]
        )


class AuthenticationSetupGenerator(BaseTaskGenerator):
    """Generates authentication setup tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate endpoints that require auth
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        endpoints = endpoint_gen.get_user_endpoints()[:2]
        
        # Choose auth type
        auth_types = ["bearer_token", "api_key"]
        auth_type = random.choice(auth_types)
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            "Authenticated User API"
        )
        
        if auth_type == "bearer_token":
            api_docs.authentication = {
                "type": "Bearer Token",
                "header": "Authorization",
                "format": "Bearer {token}",
                "example": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        else:
            api_docs.authentication = {
                "type": "API Key",
                "header": "X-API-Key",
                "format": "{api_key}",
                "example": "sk_test_1234567890abcdef"
            }
        
        # Add auth example
        api_docs.examples.append({
            "title": "Authenticated Request",
            "request": "GET /users",
            "headers": {
                api_docs.authentication["header"]: api_docs.authentication["example"]
            },
            "response": {"status": 200, "data": {"users": []}}
        })
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            if auth_type == "bearer_token":
                starter_code = '''
import requests

# Your Bearer token (normally from environment or config)
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example.token"

base_url = "https://api.example.com"

# TODO: Set up the Authorization header with Bearer token
# Format: "Bearer {token}"

# TODO: Make a GET request to /users with authentication

# TODO: Handle authentication errors (401, 403)
'''
            else:
                starter_code = '''
import requests

# Your API key (normally from environment or config)
api_key = "sk_test_1234567890abcdef"

base_url = "https://api.example.com"

# TODO: Set up the X-API-Key header

# TODO: Make a GET request to /users with authentication

# TODO: Handle authentication errors (401, 403)
'''
        else:
            starter_code = f'''
import requests
import os
from typing import Dict, Any, Optional

class AuthenticatedAPIClient:
    def __init__(self, base_url: str, {"bearer_token" if auth_type == "bearer_token" else "api_key"}: str):
        self.base_url = base_url.rstrip('/')
        self.{"bearer_token" if auth_type == "bearer_token" else "api_key"} = {"bearer_token" if auth_type == "bearer_token" else "api_key"}
        # TODO: Set up authentication headers
        self.headers = {{}}
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        # TODO: Return proper authentication headers
        pass
    
    def make_authenticated_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make an authenticated API request"""
        # TODO: Add authentication headers to the request
        
        # TODO: Make the request with proper error handling
        
        # TODO: Handle authentication errors specifically
        pass
    
    def get_users(self) -> Optional[Dict[str, Any]]:
        """Get users with authentication"""
        return self.make_authenticated_request("GET", "/users")

# Usage
# TODO: Get credentials from environment variables for security
client = AuthenticatedAPIClient(
    base_url="https://api.example.com",
    {"bearer_token" if auth_type == "bearer_token" else "api_key"}="your_{"token" if auth_type == "bearer_token" else "key"}_here"
)

users = client.get_users()
if users:
    print(f"Retrieved {{len(users.get('items', []))}} users")
else:
    print("Failed to retrieve users")
'''
        
        # Solution template
        if auth_type == "bearer_token":
            solution_template = '''
import requests
import os

def get_users_with_auth(base_url: str, bearer_token: str):
    """Get users with Bearer token authentication"""
    url = f"{base_url}/users"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:
            print("❌ Authentication failed: Invalid or expired token")
            return None
        elif response.status_code == 403:
            print("❌ Access denied: Insufficient permissions")
            return None
        
        response.raise_for_status()
        users = response.json()
        print(f"✅ Retrieved {len(users.get('items', []))} users")
        return users
        
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

# Usage (get token from environment for security)
token = os.getenv("BEARER_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.example")
result = get_users_with_auth("https://api.example.com", token)
'''
        else:
            solution_template = '''
import requests
import os

def get_users_with_api_key(base_url: str, api_key: str):
    """Get users with API key authentication"""
    url = f"{base_url}/users"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:
            print("❌ Authentication failed: Invalid API key")
            return None
        elif response.status_code == 403:
            print("❌ Access denied: API key lacks permissions")
            return None
        
        response.raise_for_status()
        users = response.json()
        print(f"✅ Retrieved {len(users.get('items', []))} users")
        return users
        
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

# Usage (get API key from environment for security)
api_key = os.getenv("API_KEY", "sk_test_1234567890abcdef")
result = get_users_with_api_key("https://api.example.com", api_key)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "GET",
                    "path": "/users",
                    "expected_status": [200],
                    "requires_auth": True
                }
            ],
            expected_outputs=[
                "Authorization" if auth_type == "bearer_token" else "X-API-Key",
                "200"
            ],
            code_quality_requirements={
                "has_authentication": True,
                "handles_auth_errors": True
            },
            minimum_score=6.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.AUTHENTICATION_SETUP, difficulty),
            task_type=TaskType.AUTHENTICATION_SETUP,
            difficulty=difficulty,
            title=f"API Authentication with {auth_type.replace('_', ' ').title()}",
            description=f"Learn how to authenticate API requests using {auth_type.replace('_', ' ')}.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                f"Set the {api_docs.authentication['header']} header",
                "Handle 401 (Unauthorized) and 403 (Forbidden) errors",
                "Store credentials securely (environment variables)",
                "Test authentication before making other requests"
            ],
            estimated_time=25 if difficulty == TaskDifficulty.BEGINNER else 40,
            learning_objectives=[
                "Understand API authentication methods",
                "Learn to set authentication headers",
                "Handle authentication errors properly",
                "Follow security best practices"
            ],
            tags=["authentication", auth_type, "security", "headers"]
        )


class ErrorHandlingGenerator(BaseTaskGenerator):
    """Generates error handling tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        endpoints = endpoint_gen.get_user_endpoints()[:3]
        
        # Create API documentation with error emphasis
        api_docs = self._generate_api_documentation(
            endpoints,
            "Error-Prone User API"
        )
        
        # Add detailed error information
        api_docs.error_codes.update({
            "429": "Too Many Requests - Rate limit exceeded",
            "503": "Service Unavailable - Server overloaded"
        })
        
        api_docs.notes.extend([
            "This API is known to have intermittent failures",
            "Network timeouts are common during peak hours",
            "Rate limiting is strictly enforced"
        ])
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = '''
import requests

base_url = "https://api.example.com"

# This API is unreliable and often returns errors
# Your task is to handle all possible error conditions

def get_user(user_id: int):
    """Get a user by ID with proper error handling"""
    url = f"{base_url}/users/{user_id}"
    
    # TODO: Make the request with error handling
    # Handle these specific cases:
    # - Connection errors (network issues)
    # - Timeout errors  
    # - 404 Not Found (user doesn't exist)
    # - 401 Unauthorized
    # - 429 Too Many Requests
    # - 500 Internal Server Error
    
    pass

# Test the function
result = get_user(123)
print(result)
'''
        else:
            starter_code = '''
import requests
import time
from typing import Dict, Any, Optional

class RobustAPIClient:
    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def make_request_with_retry(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make API request with retry logic and comprehensive error handling"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                # TODO: Make the request
                
                # TODO: Handle different HTTP status codes:
                # - 200-299: Success
                # - 400: Bad Request
                # - 401: Unauthorized  
                # - 403: Forbidden
                # - 404: Not Found
                # - 429: Rate Limited (implement backoff)
                # - 500-503: Server errors (retry)
                
                pass
                
            except requests.exceptions.Timeout:
                # TODO: Handle timeout
                pass
            except requests.exceptions.ConnectionError:
                # TODO: Handle connection errors
                pass
            except requests.exceptions.RequestException as e:
                # TODO: Handle other request exceptions
                pass
        
        return None
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user with robust error handling"""
        return self.make_request_with_retry("GET", f"/users/{user_id}")

# Usage
client = RobustAPIClient("https://api.example.com")
user = client.get_user(123)
'''
        
        # Solution template
        solution_template = '''
import requests
import time
import random
from typing import Dict, Any, Optional

def robust_api_call(base_url: str, endpoint: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Make API call with comprehensive error handling and retry logic"""
    url = f"{base_url}{endpoint}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            
            # Handle successful responses
            if 200 <= response.status_code < 300:
                return response.json()
            
            # Handle client errors (400-499)
            elif response.status_code == 400:
                print("❌ Bad Request: Check your request parameters")
                return None
            elif response.status_code == 401:
                print("❌ Unauthorized: Check your authentication")
                return None
            elif response.status_code == 403:
                print("❌ Forbidden: Insufficient permissions")
                return None
            elif response.status_code == 404:
                print("❌ Not Found: Resource doesn't exist")
                return None
            elif response.status_code == 429:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"⏳ Rate limited, waiting {wait_time:.1f}s before retry {attempt+1}")
                time.sleep(wait_time)
                continue
            
            # Handle server errors (500-599) - retry these
            elif response.status_code >= 500:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"🔄 Server error {response.status_code}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Server error {response.status_code} - max retries exceeded")
                    return None
            
            # Other status codes
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timed out (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
            
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error (attempt {attempt+1})")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    return None

# Usage with error handling
result = robust_api_call("https://api.example.com", "/users/123")
if result:
    print(f"✅ Success: {result}")
else:
    print("❌ Failed to retrieve data after all retries")
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "GET",
                    "path": "/users/123",
                    "expected_status": [200, 404, 429, 500]  # Various possible responses
                }
            ],
            expected_outputs=[
                "error",
                "retry",
                "failed"
            ],
            code_quality_requirements={
                "has_error_handling": True,
                "has_timeout_handling": True,
                "has_retry_logic": True if difficulty != TaskDifficulty.BEGINNER else False
            },
            minimum_score=7.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.ERROR_HANDLING, difficulty),
            task_type=TaskType.ERROR_HANDLING,
            difficulty=difficulty,
            title="Robust API Error Handling",
            description="Learn to handle various API errors, timeouts, and implement retry logic for robust integrations.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Use try-except blocks for different exception types",
                "Implement exponential backoff for retries",
                "Handle rate limiting with appropriate delays",
                "Log different types of errors for debugging"
            ],
            estimated_time=30 if difficulty == TaskDifficulty.BEGINNER else 45,
            learning_objectives=[
                "Understand different types of API errors",
                "Implement comprehensive error handling",
                "Learn retry strategies and backoff",
                "Handle network-related exceptions"
            ],
            tags=["error-handling", "retry", "robustness", "exceptions"]
        )


class PaginationHandlingGenerator(BaseTaskGenerator):
    """Generates pagination handling tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate endpoints with pagination
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        endpoints = [ep for ep in endpoint_gen.get_user_endpoints() if ep.method == HTTPMethod.GET and "{" not in ep.path]
        
        # Create API documentation with pagination details
        api_docs = self._generate_api_documentation(
            endpoints,
            "Paginated Users API"
        )
        
        # Add pagination documentation
        api_docs.examples.append({
            "title": "Paginated Request",
            "request": "GET /users?page=1&per_page=10",
            "response": {
                "status": 200,
                "data": {
                    "items": ["user1", "user2", "..."],
                    "pagination": {
                        "page": 1,
                        "per_page": 10,
                        "total": 150,
                        "total_pages": 15,
                        "has_next": True,
                        "has_prev": False
                    }
                }
            }
        })
        
        api_docs.notes.extend([
            "This API returns paginated results",
            "Default page size is 10, maximum is 100",
            "Use 'page' and 'per_page' query parameters"
        ])
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = '''
import requests

base_url = "https://api.example.com"

def get_all_users():
    """Retrieve ALL users by handling pagination"""
    all_users = []
    page = 1
    
    while True:
        # TODO: Make request with page parameter
        # URL should be: /users?page={page}&per_page=50
        
        # TODO: Parse the response
        
        # TODO: Add users from this page to all_users list
        
        # TODO: Check if there are more pages (has_next)
        # If not, break the loop
        
        pass
    
    return all_users

# Usage
users = get_all_users()
print(f"Retrieved {len(users)} users total")
'''
        else:
            starter_code = '''
import requests
from typing import Dict, List, Any, Optional, Iterator

class PaginatedAPIClient:
    def __init__(self, base_url: str, per_page: int = 50):
        self.base_url = base_url.rstrip('/')
        self.per_page = per_page
    
    def get_users_page(self, page: int) -> Optional[Dict[str, Any]]:
        """Get a single page of users"""
        # TODO: Make request to /users with page and per_page parameters
        pass
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users by iterating through all pages"""
        all_users = []
        
        # TODO: Implement pagination loop
        # - Start with page 1
        # - Keep fetching until no more pages
        # - Handle errors gracefully
        
        return all_users
    
    def iter_users(self) -> Iterator[Dict[str, Any]]:
        """Generator that yields users one by one across all pages"""
        # TODO: Implement a generator that yields individual users
        # This is more memory efficient for large datasets
        pass
    
    def get_users_with_filter(self, **filters) -> List[Dict[str, Any]]:
        """Get users with additional filter parameters"""
        # TODO: Add filter parameters to the pagination requests
        pass

# Usage
client = PaginatedAPIClient("https://api.example.com", per_page=25)

# Get all users
all_users = client.get_all_users()
print(f"Total users: {len(all_users)}")

# Or iterate efficiently
for user in client.iter_users():
    print(f"User: {user.get('username', 'unknown')}")
'''
        
        # Solution template
        solution_template = '''
import requests
from typing import Dict, List, Any, Optional

def get_all_users_paginated(base_url: str, per_page: int = 50) -> List[Dict[str, Any]]:
    """Retrieve all users by handling pagination automatically"""
    all_users = []
    page = 1
    
    while True:
        try:
            # Make request with pagination parameters
            response = requests.get(
                f"{base_url}/users",
                params={"page": page, "per_page": per_page},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract users from current page
            page_users = data.get("items", [])
            all_users.extend(page_users)
            
            # Check pagination info
            pagination = data.get("pagination", {})
            
            print(f"📄 Page {page}: {len(page_users)} users (Total so far: {len(all_users)})")
            
            # Check if there are more pages
            if not pagination.get("has_next", False):
                break
            
            page += 1
            
            # Safety check to prevent infinite loops
            if page > pagination.get("total_pages", 1000):
                print("⚠️  Safety limit reached, stopping pagination")
                break
                
        except requests.RequestException as e:
            print(f"❌ Error fetching page {page}: {e}")
            break
    
    print(f"✅ Retrieved {len(all_users)} users total across {page} pages")
    return all_users

def get_users_with_callback(base_url: str, callback=None) -> List[Dict[str, Any]]:
    """Get users with optional progress callback"""
    all_users = []
    page = 1
    
    while True:
        try:
            response = requests.get(
                f"{base_url}/users",
                params={"page": page, "per_page": 25}
            )
            response.raise_for_status()
            data = response.json()
            
            page_users = data.get("items", [])
            all_users.extend(page_users)
            
            # Call progress callback if provided
            if callback:
                callback(page, len(page_users), len(all_users))
            
            if not data.get("pagination", {}).get("has_next", False):
                break
                
            page += 1
            
        except requests.RequestException as e:
            print(f"Error on page {page}: {e}")
            break
    
    return all_users

# Usage examples
users = get_all_users_paginated("https://api.example.com")

# With progress callback
def progress_callback(page, page_count, total_count):
    print(f"Processing page {page}: {page_count} users (total: {total_count})")

users_with_progress = get_users_with_callback("https://api.example.com", progress_callback)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "GET",
                    "path": "/users",
                    "expected_status": [200],
                    "requires_pagination": True,
                    "min_calls": 2  # Should make multiple calls for pagination
                }
            ],
            expected_outputs=[
                "page",
                "per_page",
                "total"
            ],
            code_quality_requirements={
                "handles_pagination": True,
                "handles_empty_pages": True
            },
            minimum_score=6.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.PAGINATION_HANDLING, difficulty),
            task_type=TaskType.PAGINATION_HANDLING,
            difficulty=difficulty,
            title="Handle Paginated API Responses",
            description="Learn to efficiently retrieve all data from paginated API endpoints.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Use 'page' and 'per_page' query parameters",
                "Check 'has_next' field to know when to stop",
                "Handle the case where a page is empty",
                "Add safety limits to prevent infinite loops"
            ],
            estimated_time=25 if difficulty == TaskDifficulty.BEGINNER else 40,
            learning_objectives=[
                "Understand API pagination concepts",
                "Learn to iterate through paginated results",
                "Handle edge cases in pagination",
                "Implement memory-efficient data retrieval"
            ],
            tags=["pagination", "iteration", "data-retrieval", "query-parameters"]
        )


class PutUpdateResourceGenerator(BaseTaskGenerator):
    """Generates PUT request tasks for updating resources"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        
        resource_choice = random.choice(["user", "product"])
        if resource_choice == "user":
            endpoints = [ep for ep in endpoint_gen.get_user_endpoints() if ep.method == HTTPMethod.PUT]
            resource_name = "user"
            sample_data = {
                "username": "updated_user",
                "email": "updated@example.com",
                "full_name": "Updated User Name"
            }
        else:
            endpoints = [ep for ep in endpoint_gen.get_product_endpoints() if ep.method == HTTPMethod.PUT]
            resource_name = "product"
            sample_data = {
                "name": "Updated Product",
                "description": "Updated description",
                "price": 39.99,
                "category": "electronics"
            }
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            f"{resource_name.title()} Update API"
        )
        
        # Add update-specific example
        api_docs.examples.append({
            "title": f"Update {resource_name}",
            "request": f"PUT /{resource_name}s/123",
            "headers": {"Content-Type": "application/json"},
            "body": sample_data,
            "response": {"status": 200, "data": {**sample_data, "id": 123, "updated_at": "2023-01-01T00:00:00Z"}}
        })
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = f'''
import requests
import json

# Sample {resource_name} update data
update_data = {json.dumps(sample_data, indent=4)}

base_url = "https://api.example.com"
{resource_name}_id = 123

# TODO: Create the URL for updating a {resource_name}
# Endpoint: PUT /{resource_name}s/{{id}}

# TODO: Set up the headers (Content-Type: application/json)

# TODO: Make a PUT request with the updated {resource_name} data

# TODO: Check if the request was successful (status code 200)

# TODO: Print the updated {resource_name} information
'''
        else:
            starter_code = f'''
import requests
import json
from typing import Dict, Any, Optional

class {resource_name.title()}Updater:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {{"Content-Type": "application/json"}}
    
    def update_{resource_name}(self, {resource_name}_id: int, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing {resource_name}"""
        # TODO: Validate the update data
        
        # TODO: Construct the API endpoint URL with ID
        
        # TODO: Make the PUT request with error handling
        # Handle these cases:
        # - 404 Not Found (resource doesn't exist)
        # - 400 Bad Request (invalid data)
        # - 409 Conflict (version mismatch)
        
        # TODO: Return the updated {resource_name} data or None if failed
        pass
    
    def partial_update_{resource_name}(self, {resource_name}_id: int, partial_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform a partial update (PATCH-like behavior with PUT)"""
        # TODO: Get current {resource_name} data first
        
        # TODO: Merge with partial update data
        
        # TODO: Send complete updated data via PUT
        pass

# Usage
updater = {resource_name.title()}Updater("https://api.example.com")
update_data = {json.dumps(sample_data, indent=4)}

result = updater.update_{resource_name}(123, update_data)
if result:
    print(f"Updated {resource_name}: {{result.get('id', 'unknown')}}")
else:
    print("Failed to update {resource_name}")
'''
        
        # Solution template
        solution_template = f'''
import requests
import json
from typing import Dict, Any, Optional

def update_{resource_name}(base_url: str, {resource_name}_id: int, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a {resource_name} via PUT request"""
    url = f"{{base_url}}/{resource_name}s/{{" + f"{resource_name}_id" + "}}"
    headers = {{"Content-Type": "application/json"}}
    
    try:
        response = requests.put(url, json=update_data, headers=headers)
        
        if response.status_code == 404:
            print(f"❌ {resource_name.title()} not found: {{" + f"{resource_name}_id" + "}}")
            return None
        elif response.status_code == 400:
            print(f"❌ Invalid update data: {{response.text}}")
            return None
        elif response.status_code == 409:
            print(f"❌ Conflict: {resource_name} may have been modified by another process")
            return None
        
        response.raise_for_status()
        
        if response.status_code == 200:
            updated_{resource_name} = response.json()
            print(f"✅ Updated {resource_name}: {{updated_{resource_name}.get('id', 'unknown')}}")
            return updated_{resource_name}
        else:
            print(f"Unexpected status code: {{response.status_code}}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Error updating {resource_name}: {{e}}")
        return None

# Usage
update_data = {json.dumps(sample_data, indent=4)}
result = update_{resource_name}("https://api.example.com", 123, update_data)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "PUT",
                    "path": f"/{resource_name}s/123",
                    "expected_status": [200, 404],
                    "requires_body": True
                }
            ],
            expected_outputs=[
                f"PUT /{resource_name}s",
                "200",
                "Updated"
            ],
            code_quality_requirements={
                "has_error_handling": True,
                "handles_not_found": True
            },
            minimum_score=5.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.PUT_UPDATE_RESOURCE, difficulty),
            task_type=TaskType.PUT_UPDATE_RESOURCE,
            difficulty=difficulty,
            title=f"Update {resource_name.title()} via PUT Request",
            description=f"Learn how to update existing {resource_name}s using PUT requests with complete data replacement.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "PUT requests typically require complete resource data",
                "Handle 404 errors when resource doesn't exist",
                "Consider 409 Conflict for concurrent modifications",
                "Validate data before sending the request"
            ],
            estimated_time=25 if difficulty == TaskDifficulty.BEGINNER else 35,
            learning_objectives=[
                "Understand HTTP PUT semantics",
                "Learn difference between PUT and POST",
                "Handle resource not found scenarios",
                "Implement data validation before updates"
            ],
            tags=["http", "put", "update", "json", resource_name]
        )


class DeleteResourceGenerator(BaseTaskGenerator):
    """Generates DELETE request tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        
        resource_choice = random.choice(["user", "product"])
        if resource_choice == "user":
            endpoints = [ep for ep in endpoint_gen.get_user_endpoints() if ep.method == HTTPMethod.DELETE]
            resource_name = "user"
        else:
            endpoints = [ep for ep in endpoint_gen.get_product_endpoints() if ep.method == HTTPMethod.DELETE]
            resource_name = "product"
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            f"{resource_name.title()} Deletion API"
        )
        
        # Add delete-specific example
        api_docs.examples.append({
            "title": f"Delete {resource_name}",
            "request": f"DELETE /{resource_name}s/123",
            "response": {"status": 204, "data": None}
        })
        
        api_docs.notes.extend([
            f"Deleting a {resource_name} is irreversible",
            "Returns 204 No Content on successful deletion",
            "Returns 404 if resource doesn't exist"
        ])
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = f'''
import requests

base_url = "https://api.example.com"
{resource_name}_id = 123

# TODO: Create the URL for deleting a {resource_name}
# Endpoint: DELETE /{resource_name}s/{{id}}

# TODO: Make a DELETE request

# TODO: Check if the deletion was successful
# - 204 No Content: Successfully deleted
# - 404 Not Found: Resource doesn't exist
# - 409 Conflict: Cannot delete (has dependencies)

# TODO: Print appropriate success/failure message
'''
        else:
            starter_code = f'''
import requests
from typing import Dict, Any, Optional, List

class {resource_name.title()}Deleter:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def delete_{resource_name}(self, {resource_name}_id: int, force: bool = False) -> bool:
        """Delete a {resource_name} by ID"""
        # TODO: Construct the API endpoint URL
        
        # TODO: Add force parameter if needed (query param)
        
        # TODO: Make the DELETE request with error handling
        # Handle these cases:
        # - 204 No Content (success)
        # - 404 Not Found (doesn't exist)
        # - 409 Conflict (has dependencies)
        # - 403 Forbidden (no permission)
        
        pass
    
    def delete_multiple_{resource_name}s(self, {resource_name}_ids: List[int]) -> Dict[int, bool]:
        """Delete multiple {resource_name}s and return results"""
        results = {{}}
        
        # TODO: Delete each {resource_name} and track results
        # TODO: Consider implementing batch deletion if API supports it
        
        return results
    
    def safe_delete_{resource_name}(self, {resource_name}_id: int) -> bool:
        """Safely delete a {resource_name} with confirmation"""
        # TODO: First check if {resource_name} exists (GET request)
        
        # TODO: Check for dependencies or constraints
        
        # TODO: Perform the deletion
        
        pass

# Usage
deleter = {resource_name.title()}Deleter("https://api.example.com")

# Delete single {resource_name}
success = deleter.delete_{resource_name}(123)
if success:
    print("{resource_name.title()} deleted successfully")
else:
    print("Failed to delete {resource_name}")

# Delete multiple {resource_name}s
results = deleter.delete_multiple_{resource_name}s([123, 456, 789])
print(f"Deletion results: {{results}}")
'''
        
        # Solution template
        solution_template = f'''
import requests
from typing import List, Dict

def delete_{resource_name}(base_url: str, {resource_name}_id: int) -> bool:
    """Delete a {resource_name} via DELETE request"""
    url = f"{{base_url}}/{resource_name}s/{{" + f"{resource_name}_id" + "}}"
    
    try:
        response = requests.delete(url)
        
        if response.status_code == 204:
            print(f"✅ {resource_name.title()} {{" + f"{resource_name}_id" + "}} deleted successfully")
            return True
        elif response.status_code == 404:
            print(f"❌ {resource_name.title()} {{" + f"{resource_name}_id" + "}} not found")
            return False
        elif response.status_code == 409:
            print(f"❌ Cannot delete {resource_name} {{" + f"{resource_name}_id" + "}} - has dependencies")
            return False
        elif response.status_code == 403:
            print(f"❌ Forbidden: No permission to delete {resource_name} {{" + f"{resource_name}_id" + "}}")
            return False
        else:
            print(f"❌ Unexpected status code: {{response.status_code}}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Error deleting {resource_name}: {{e}}")
        return False

def delete_multiple_{resource_name}s(base_url: str, {resource_name}_ids: List[int]) -> Dict[int, bool]:
    """Delete multiple {resource_name}s and return results"""
    results = {{}}
    
    for {resource_name}_id in {resource_name}_ids:
        print(f"Deleting {resource_name} {{" + f"{resource_name}_id" + "}}...")
        results[{resource_name}_id] = delete_{resource_name}(base_url, {resource_name}_id)
    
    successful = sum(1 for success in results.values() if success)
    print(f"\\n📊 Deletion Summary: {{successful}}/{{len({resource_name}_ids)}} successful")
    
    return results

def safe_delete_with_check(base_url: str, {resource_name}_id: int) -> bool:
    """Safely delete after checking existence"""
    # First check if {resource_name} exists
    check_url = f"{{base_url}}/{resource_name}s/{{" + f"{resource_name}_id" + "}}"
    
    try:
        check_response = requests.get(check_url)
        if check_response.status_code == 404:
            print(f"❌ {resource_name.title()} {{" + f"{resource_name}_id" + "}} doesn't exist")
            return False
            
        # If exists, proceed with deletion
        return delete_{resource_name}(base_url, {resource_name}_id)
        
    except requests.RequestException as e:
        print(f"❌ Error checking {resource_name} existence: {{e}}")
        return False

# Usage examples
result = delete_{resource_name}("https://api.example.com", 123)

# Batch deletion
batch_results = delete_multiple_{resource_name}s("https://api.example.com", [123, 456, 789])

# Safe deletion with existence check
safe_result = safe_delete_with_check("https://api.example.com", 456)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "DELETE",
                    "path": f"/{resource_name}s/123",
                    "expected_status": [204, 404, 409]
                }
            ],
            expected_outputs=[
                f"DELETE /{resource_name}s",
                "204",
                "deleted"
            ],
            code_quality_requirements={
                "has_error_handling": True,
                "handles_not_found": True
            },
            minimum_score=5.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.DELETE_RESOURCE, difficulty),
            task_type=TaskType.DELETE_RESOURCE,
            difficulty=difficulty,
            title=f"Delete {resource_name.title()} via DELETE Request",
            description=f"Learn how to safely delete {resource_name}s using DELETE requests with proper error handling.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "DELETE requests typically return 204 No Content on success",
                "Always handle 404 (not found) and 409 (conflict) errors",
                "Consider checking resource existence before deletion",
                "Be careful with batch deletions - handle partial failures"
            ],
            estimated_time=20 if difficulty == TaskDifficulty.BEGINNER else 30,
            learning_objectives=[
                "Understand HTTP DELETE semantics",
                "Handle various deletion error scenarios",
                "Implement safe deletion patterns",
                "Learn batch deletion strategies"
            ],
            tags=["http", "delete", "removal", "error-handling", resource_name]
        )


class BulkOperationsGenerator(BaseTaskGenerator):
    """Generates bulk/batch operations tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.ADVANCED,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        
        resource_choice = random.choice(["user", "product"])
        if resource_choice == "user":
            endpoints = endpoint_gen.get_user_endpoints()
            resource_name = "user"
            sample_items = [
                {"username": f"user_{i}", "email": f"user{i}@example.com", "full_name": f"User {i}"}
                for i in range(1, 4)
            ]
        else:
            endpoints = endpoint_gen.get_product_endpoints()
            resource_name = "product"
            sample_items = [
                {"name": f"Product {i}", "description": f"Description {i}", "price": 10.0 + i, "category": "electronics"}
                for i in range(1, 4)
            ]
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            f"Bulk {resource_name.title()} Operations API"
        )
        
        # Add bulk operation examples
        api_docs.examples.extend([
            {
                "title": f"Bulk Create {resource_name}s",
                "request": f"POST /{resource_name}s/bulk",
                "body": {"items": sample_items},
                "response": {"status": 201, "data": {"created": 3, "failed": 0, "items": sample_items}}
            },
            {
                "title": f"Bulk Update {resource_name}s",
                "request": f"PUT /{resource_name}s/bulk",
                "body": {"updates": [{"id": 1, "data": sample_items[0]}]},
                "response": {"status": 200, "data": {"updated": 1, "failed": 0}}
            }
        ])
        
        api_docs.notes.extend([
            "Bulk operations are more efficient than individual requests",
            "API may have limits on batch sizes (typically 100-1000 items)",
            "Partial failures are possible - check individual item status"
        ])
        
        # Generate starter code
        if difficulty == TaskDifficulty.INTERMEDIATE:
            starter_code = f'''
import requests
import json
from typing import List, Dict, Any

# Sample {resource_name} data for bulk operations
{resource_name}_batch = {json.dumps(sample_items, indent=4)}

base_url = "https://api.example.com"

def bulk_create_{resource_name}s(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create multiple {resource_name}s in a single request"""
    # TODO: Create the bulk endpoint URL
    # Endpoint: POST /{resource_name}s/bulk
    
    # TODO: Prepare the request body with items
    
    # TODO: Make the POST request
    
    # TODO: Handle the response and check for partial failures
    
    pass

def process_in_batches(items: List[Dict[str, Any]], batch_size: int = 10):
    """Process large datasets in smaller batches"""
    total_processed = 0
    
    # TODO: Split items into batches of batch_size
    
    # TODO: Process each batch using bulk_create_{resource_name}s
    
    # TODO: Keep track of successes and failures
    
    pass

# Usage
result = bulk_create_{resource_name}s({resource_name}_batch)
print(f"Bulk operation result: {{result}}")

# Process large dataset in batches
large_dataset = [{resource_name}_batch[0]] * 25  # Simulate 25 items
process_in_batches(large_dataset, batch_size=5)
'''
        else:
            starter_code = f'''
import requests
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class Bulk{resource_name.title()}Operations:
    def __init__(self, base_url: str, max_batch_size: int = 50):
        self.base_url = base_url.rstrip('/')
        self.max_batch_size = max_batch_size
        self.headers = {{"Content-Type": "application/json"}}
    
    def bulk_create(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple {resource_name}s via bulk endpoint"""
        # TODO: Implement bulk creation
        # Handle batch size limits
        # Process partial failures
        pass
    
    def bulk_update(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update multiple {resource_name}s via bulk endpoint"""
        # TODO: Implement bulk updates
        # Each update should have 'id' and 'data' fields
        pass
    
    def bulk_delete(self, ids: List[int]) -> Dict[str, Any]:
        """Delete multiple {resource_name}s via bulk endpoint"""
        # TODO: Implement bulk deletion
        pass
    
    def parallel_individual_operations(self, items: List[Dict[str, Any]], operation: str = "create") -> Dict[str, Any]:
        """Perform operations in parallel using individual endpoints"""
        # TODO: Use ThreadPoolExecutor for parallel requests
        # TODO: Handle rate limiting and failures
        pass
    
    async def async_bulk_operations(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform bulk operations asynchronously"""
        # TODO: Implement async bulk operations using aiohttp
        pass
    
    def smart_batch_processor(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Intelligently choose between bulk and individual operations"""
        # TODO: Decide strategy based on:
        # - Number of items
        # - API capabilities  
        # - Error handling requirements
        pass

# Usage
bulk_ops = Bulk{resource_name.title()}Operations("https://api.example.com", max_batch_size=25)

{resource_name}_data = {json.dumps(sample_items, indent=4)}

# Synchronous bulk operations
sync_result = bulk_ops.bulk_create({resource_name}_data)

# Parallel individual operations
parallel_result = bulk_ops.parallel_individual_operations({resource_name}_data)

# Async bulk operations
async_result = asyncio.run(bulk_ops.async_bulk_operations({resource_name}_data))
'''
        
        # Solution template
        solution_template = f'''
import requests
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def bulk_create_{resource_name}s(base_url: str, items: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, Any]:
    """Create {resource_name}s in bulk with batch processing"""
    if not items:
        return {{"created": 0, "failed": 0, "errors": []}}
    
    total_created = 0
    total_failed = 0
    all_errors = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        try:
            response = requests.post(
                f"{{base_url}}/{resource_name}s/bulk",
                json={{"items": batch}},
                headers={{"Content-Type": "application/json"}},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            total_created += result.get("created", 0)
            total_failed += result.get("failed", 0)
            
            if result.get("errors"):
                all_errors.extend(result["errors"])
            
            print(f"📦 Batch {{i//batch_size + 1}}: {{result.get('created', 0)}} created, {{result.get('failed', 0)}} failed")
            
        except requests.RequestException as e:
            print(f"❌ Batch {{i//batch_size + 1}} failed: {{e}}")
            total_failed += len(batch)
            all_errors.append(f"Batch {{i//batch_size + 1}}: {{str(e)}}")
    
    print(f"\\n✅ Bulk operation complete: {{total_created}} created, {{total_failed}} failed")
    
    return {{
        "created": total_created,
        "failed": total_failed,
        "errors": all_errors
    }}

def parallel_create_{resource_name}s(base_url: str, items: List[Dict[str, Any]], max_workers: int = 5) -> Dict[str, Any]:
    """Create {resource_name}s in parallel using individual requests"""
    
    def create_single_{resource_name}(item_data):
        try:
            response = requests.post(
                f"{{base_url}}/{resource_name}s",
                json=item_data,
                timeout=10
            )
            response.raise_for_status()
            return {{"success": True, "data": response.json()}}
        except Exception as e:
            return {{"success": False, "error": str(e), "data": item_data}}
    
    results = {{"created": 0, "failed": 0, "errors": []}}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {{executor.submit(create_single_{resource_name}, item): item for item in items}}
        
        for future in as_completed(future_to_item):
            result = future.result()
            
            if result["success"]:
                results["created"] += 1
                print(f"✅ Created {resource_name}: {{result['data'].get('id', 'unknown')}}")
            else:
                results["failed"] += 1
                results["errors"].append(result["error"])
                print(f"❌ Failed to create {resource_name}: {{result['error']}}")
    
    print(f"\\n📊 Parallel operation complete: {{results['created']}} created, {{results['failed']}} failed")
    return results

def smart_bulk_processor(base_url: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose optimal strategy based on dataset size"""
    
    if len(items) <= 5:
        print("Using individual requests for small dataset")
        return parallel_create_{resource_name}s(base_url, items, max_workers=3)
    elif len(items) <= 100:
        print("Using bulk endpoint for medium dataset")
        return bulk_create_{resource_name}s(base_url, items, batch_size=25)
    else:
        print("Using batched bulk operations for large dataset")
        return bulk_create_{resource_name}s(base_url, items, batch_size=50)

# Usage examples
{resource_name}_data = {json.dumps(sample_items, indent=4)}

# Bulk creation
bulk_result = bulk_create_{resource_name}s("https://api.example.com", {resource_name}_data)

# Parallel creation
parallel_result = parallel_create_{resource_name}s("https://api.example.com", {resource_name}_data)

# Smart processing
smart_result = smart_bulk_processor("https://api.example.com", {resource_name}_data)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "POST",
                    "path": f"/{resource_name}s/bulk",
                    "expected_status": [200, 201],
                    "requires_body": True,
                    "min_calls": 1
                }
            ],
            expected_outputs=[
                "bulk",
                "batch",
                "created",
                "failed"
            ],
            code_quality_requirements={
                "handles_batch_processing": True,
                "handles_partial_failures": True,
                "has_error_tracking": True
            },
            minimum_score=8.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.BULK_OPERATIONS, difficulty),
            task_type=TaskType.BULK_OPERATIONS,
            difficulty=difficulty,
            title=f"Bulk {resource_name.title()} Operations",
            description=f"Learn to efficiently handle large datasets using bulk API operations and batch processing.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Use bulk endpoints when available for better performance",
                "Process large datasets in smaller batches",
                "Handle partial failures gracefully",
                "Consider parallel processing for individual operations",
                "Implement retry logic for failed batches"
            ],
            estimated_time=45 if difficulty == TaskDifficulty.INTERMEDIATE else 60,
            learning_objectives=[
                "Understand bulk API operation patterns",
                "Learn batch processing techniques",
                "Handle partial failure scenarios",
                "Optimize for large dataset processing",
                "Compare bulk vs parallel strategies"
            ],
            tags=["bulk", "batch", "performance", "scalability", resource_name]
        )


class ResponseValidationGenerator(BaseTaskGenerator):
    """Generates response validation tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        endpoints = endpoint_gen.get_user_endpoints()[:2]
        
        # Create API documentation with detailed response schemas
        api_docs = self._generate_api_documentation(
            endpoints,
            "User API with Response Validation"
        )
        
        # Add detailed schema information
        api_docs.examples.append({
            "title": "Valid User Response",
            "request": "GET /users/123",
            "response": {
                "status": 200,
                "data": {
                    "id": 123,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "full_name": "John Doe",
                    "created_at": "2023-01-01T00:00:00Z",
                    "is_active": True
                }
            }
        })
        
        api_docs.notes.extend([
            "All responses follow strict schema validation",
            "Date fields are in ISO 8601 format",
            "Email fields must be valid email addresses",
            "Numeric IDs must be positive integers"
        ])
        
        # Generate starter code
        if difficulty == TaskDifficulty.BEGINNER:
            starter_code = '''
import requests
import json
from typing import Dict, Any, Optional

def validate_user_response(response_data: Dict[str, Any]) -> bool:
    """Validate that user response contains required fields with correct types"""
    
    # TODO: Check that required fields exist:
    # - id (integer, positive)
    # - username (string, not empty)
    # - email (string, valid email format) 
    # - full_name (string, not empty)
    # - created_at (string, ISO date format)
    # - is_active (boolean)
    
    # TODO: Validate data types and formats
    
    # TODO: Return True if valid, False otherwise
    
    pass

def get_user_with_validation(base_url: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get user and validate the response"""
    
    # TODO: Make the API request
    
    # TODO: Validate the response structure
    
    # TODO: Return validated data or None if invalid
    
    pass

# Usage
user = get_user_with_validation("https://api.example.com", 123)
if user:
    print(f"Valid user data: {user['username']}")
else:
    print("Invalid or missing user data")
'''
        else:
            starter_code = '''
import requests
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

@dataclass
class ValidationError:
    field: str
    message: str
    value: Any

class ResponseValidator:
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')
        self.iso_date_pattern = re.compile(r'^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d{6})?Z?$')
    
    def validate_user_schema(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate user data against expected schema"""
        errors = []
        
        # TODO: Implement comprehensive validation:
        # - Required fields presence
        # - Data type validation
        # - Format validation (email, dates)
        # - Range validation (positive IDs)
        # - String length validation
        
        return errors
    
    def validate_users_list_schema(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate paginated users list response"""
        errors = []
        
        # TODO: Validate list response structure:
        # - items array exists and is list
        # - pagination object exists
        # - each user in items follows user schema
        
        return errors
    
    def is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        # TODO: Implement email validation
        pass
    
    def is_valid_iso_date(self, date_str: str) -> bool:
        """Validate ISO 8601 date format"""  
        # TODO: Implement date validation
        pass
    
    def validate_and_sanitize(self, data: Dict[str, Any], schema_type: str = "user") -> Optional[Dict[str, Any]]:
        """Validate data and return sanitized version"""
        # TODO: Validate data
        # TODO: Sanitize/normalize valid data
        # TODO: Return None if validation fails
        pass

class ValidatedAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.validator = ResponseValidator()
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user with response validation"""
        # TODO: Make API request
        # TODO: Validate response
        # TODO: Handle validation errors
        pass
    
    def get_users(self, page: int = 1, per_page: int = 10) -> Optional[Dict[str, Any]]:
        """Get users list with validation"""
        # TODO: Make API request with pagination
        # TODO: Validate list response
        # TODO: Validate each user in the list
        pass

# Usage
client = ValidatedAPIClient("https://api.example.com")
validator = ResponseValidator()

# Get and validate single user
user = client.get_user(123)

# Get and validate users list
users = client.get_users(page=1, per_page=25)
'''
        
        # Solution template
        solution_template = '''
import requests
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ValidationError:
    field: str
    message: str
    value: Any

class ResponseValidator:
    def __init__(self):
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')
    
    def validate_user_response(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Comprehensive user response validation"""
        errors = []
        
        # Required fields validation
        required_fields = ['id', 'username', 'email', 'full_name', 'created_at', 'is_active']
        
        for field in required_fields:
            if field not in data:
                errors.append(ValidationError(field, f"Missing required field", None))
        
        # Type and format validation
        if 'id' in data:
            if not isinstance(data['id'], int) or data['id'] <= 0:
                errors.append(ValidationError('id', "ID must be positive integer", data['id']))
        
        if 'username' in data:
            if not isinstance(data['username'], str) or len(data['username']) == 0:
                errors.append(ValidationError('username', "Username must be non-empty string", data['username']))
        
        if 'email' in data:
            if not isinstance(data['email'], str) or not self.email_pattern.match(data['email']):
                errors.append(ValidationError('email', "Invalid email format", data['email']))
        
        if 'full_name' in data:
            if not isinstance(data['full_name'], str) or len(data['full_name']) == 0:
                errors.append(ValidationError('full_name', "Full name must be non-empty string", data['full_name']))
        
        if 'is_active' in data:
            if not isinstance(data['is_active'], bool):
                errors.append(ValidationError('is_active', "is_active must be boolean", data['is_active']))
        
        if 'created_at' in data:
            if not self._is_valid_iso_date(data['created_at']):
                errors.append(ValidationError('created_at', "Invalid ISO date format", data['created_at']))
        
        return errors
    
    def _is_valid_iso_date(self, date_str: str) -> bool:
        """Validate ISO 8601 date format"""
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False

def get_user_with_validation(base_url: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get user with comprehensive response validation"""
    validator = ResponseValidator()
    
    try:
        response = requests.get(f"{base_url}/users/{user_id}")
        response.raise_for_status()
        
        if response.status_code == 200:
            user_data = response.json()
            
            # Validate response structure
            validation_errors = validator.validate_user_response(user_data)
            
            if validation_errors:
                print("❌ Response validation failed:")
                for error in validation_errors:
                    print(f"   • {error.field}: {error.message} (got: {error.value})")
                return None
            
            print(f"✅ Valid user response for ID {user_id}")
            return user_data
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def validate_users_list(base_url: str, page: int = 1) -> Optional[Dict[str, Any]]:
    """Get and validate paginated users list"""
    validator = ResponseValidator()
    
    try:
        response = requests.get(
            f"{base_url}/users",
            params={"page": page, "per_page": 10}
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Validate list structure
        if 'items' not in data or not isinstance(data['items'], list):
            print("❌ Response missing 'items' array")
            return None
        
        if 'pagination' not in data:
            print("❌ Response missing 'pagination' object")
            return None
        
        # Validate each user in the list
        valid_users = []
        invalid_count = 0
        
        for i, user in enumerate(data['items']):
            validation_errors = validator.validate_user_response(user)
            
            if validation_errors:
                print(f"❌ User {i+1} validation failed:")
                for error in validation_errors[:3]:  # Show first 3 errors
                    print(f"   • {error.field}: {error.message}")
                invalid_count += 1
            else:
                valid_users.append(user)
        
        print(f"✅ Validated users list: {len(valid_users)} valid, {invalid_count} invalid")
        
        # Return data with only valid users
        return {
            'items': valid_users,
            'pagination': data['pagination'],
            'validation_summary': {
                'total_items': len(data['items']),
                'valid_items': len(valid_users),
                'invalid_items': invalid_count
            }
        }
        
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

# Usage examples
user = get_user_with_validation("https://api.example.com", 123)
users_list = validate_users_list("https://api.example.com", page=1)
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "GET",
                    "path": "/users/123",
                    "expected_status": [200]
                }
            ],
            expected_outputs=[
                "validation",
                "valid",
                "invalid",
                "schema"
            ],
            code_quality_requirements={
                "has_validation_logic": True,
                "handles_invalid_data": True,
                "checks_data_types": True
            },
            minimum_score=7.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.RESPONSE_VALIDATION, difficulty),
            task_type=TaskType.RESPONSE_VALIDATION,
            difficulty=difficulty,
            title="API Response Validation and Schema Checking",
            description="Learn to validate API responses against expected schemas and handle malformed data gracefully.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Check both field presence and data types",
                "Validate email format using regex patterns",
                "Parse and validate ISO date formats",
                "Handle missing or null fields gracefully",
                "Provide detailed validation error messages"
            ],
            estimated_time=35 if difficulty == TaskDifficulty.BEGINNER else 50,
            learning_objectives=[
                "Understand API response schema validation",
                "Learn data type and format checking",
                "Handle invalid or malformed responses",
                "Implement comprehensive error reporting"
            ],
            tags=["validation", "schema", "data-types", "error-handling"]
        )


class AsyncApiCallsGenerator(BaseTaskGenerator):
    """Generates asynchronous API calls tasks"""
    
    def generate_task(
        self,
        difficulty: TaskDifficulty = TaskDifficulty.ADVANCED,
        **_kwargs
    ) -> LearningTask:
        
        # Generate API endpoints
        from data_generation.endpoint_generator import EndpointGenerator
        endpoint_gen = EndpointGenerator()
        endpoints = endpoint_gen.get_user_endpoints()[:3]
        
        # Create API documentation
        api_docs = self._generate_api_documentation(
            endpoints,
            "Asynchronous User API"
        )
        
        api_docs.notes.extend([
            "API supports concurrent requests for better performance",
            "Rate limiting applies: 100 requests per minute",
            "Use async/await for non-blocking operations"
        ])
        
        # Generate starter code
        if difficulty == TaskDifficulty.INTERMEDIATE:
            starter_code = '''
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional

async def fetch_user_async(session: aiohttp.ClientSession, base_url: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single user asynchronously"""
    
    # TODO: Make async HTTP request using aiohttp
    # TODO: Handle errors and timeouts
    # TODO: Return user data or None if failed
    
    pass

async def fetch_multiple_users_async(base_url: str, user_ids: List[int]) -> List[Optional[Dict[str, Any]]]:
    """Fetch multiple users concurrently"""
    
    # TODO: Create aiohttp session
    # TODO: Create tasks for all user requests
    # TODO: Wait for all tasks to complete
    # TODO: Return list of results
    
    pass

def compare_sync_vs_async():
    """Compare performance of sync vs async requests"""
    user_ids = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    base_url = "https://api.example.com"
    
    # TODO: Time synchronous requests
    sync_start = time.time()
    # ... sync implementation
    sync_time = time.time() - sync_start
    
    # TODO: Time asynchronous requests  
    async_start = time.time()
    # ... async implementation
    async_time = time.time() - async_start
    
    print(f"Sync time: {sync_time:.2f}s")
    print(f"Async time: {async_time:.2f}s") 
    print(f"Speedup: {sync_time/async_time:.1f}x")

# Usage
user_ids = [1, 2, 3, 4, 5]
results = asyncio.run(fetch_multiple_users_async("https://api.example.com", user_ids))
print(f"Fetched {len([r for r in results if r])} users")

compare_sync_vs_async()
'''
        else:
            starter_code = '''
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

@dataclass
class AsyncRequestResult:
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    response_time: float
    status_code: Optional[int]

class AsyncAPIClient:
    def __init__(self, base_url: str, max_concurrent: int = 10, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _make_request(self, session: aiohttp.ClientSession, method: str, endpoint: str, **kwargs) -> AsyncRequestResult:
        """Make a single async request with rate limiting"""
        async with self.semaphore:
            # TODO: Implement rate-limited async request
            # TODO: Handle different HTTP methods
            # TODO: Track response time and status
            # TODO: Return structured result
            pass
    
    async def fetch_users_batch(self, user_ids: List[int]) -> List[AsyncRequestResult]:
        """Fetch multiple users with controlled concurrency"""
        # TODO: Create aiohttp session
        # TODO: Create tasks with semaphore limiting
        # TODO: Gather all results
        # TODO: Handle partial failures
        pass
    
    async def fetch_with_retry(self, endpoint: str, max_retries: int = 3) -> AsyncRequestResult:
        """Fetch with exponential backoff retry"""
        # TODO: Implement retry logic with backoff
        # TODO: Handle different error types
        pass
    
    async def fetch_paginated_async(self, endpoint: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Fetch all pages of paginated data asynchronously"""
        # TODO: Fetch first page to get total pages
        # TODO: Create tasks for remaining pages
        # TODO: Combine results from all pages
        pass
    
    async def parallel_operations(self, operations: List[Dict[str, Any]]) -> List[AsyncRequestResult]:
        """Execute different operations in parallel"""
        # TODO: Support mixed operations (GET, POST, PUT, DELETE)
        # TODO: Handle different endpoints and payloads
        pass

class AsyncPerformanceMonitor:
    def __init__(self):
        self.results = []
    
    async def benchmark_async_performance(self, client: AsyncAPIClient, test_cases: List[Dict[str, Any]]):
        """Benchmark async performance with different concurrency levels"""
        # TODO: Test different concurrency levels
        # TODO: Measure throughput and response times
        # TODO: Generate performance report
        pass

# Usage
async def main():
    client = AsyncAPIClient("https://api.example.com", max_concurrent=5)
    
    # Fetch multiple users
    user_ids = list(range(1, 21))  # 20 users
    results = await client.fetch_users_batch(user_ids)
    
    # Fetch paginated data
    all_users = await client.fetch_paginated_async("/users")
    
    # Mixed operations
    operations = [
        {"method": "GET", "endpoint": "/users/1"},
        {"method": "POST", "endpoint": "/users", "data": {"username": "new_user"}},
        {"method": "PUT", "endpoint": "/users/2", "data": {"active": False}}
    ]
    op_results = await client.parallel_operations(operations)

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Solution template
        solution_template = '''
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional

async def fetch_user_async(session: aiohttp.ClientSession, base_url: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single user asynchronously"""
    try:
        async with session.get(f"{base_url}/users/{user_id}") as response:
            if response.status == 200:
                user_data = await response.json()
                print(f"✅ Fetched user {user_id}: {user_data.get('username', 'unknown')}")
                return user_data
            else:
                print(f"❌ User {user_id} returned status {response.status}")
                return None
    except asyncio.TimeoutError:
        print(f"⏰ User {user_id} request timed out")
        return None
    except Exception as e:
        print(f"❌ Error fetching user {user_id}: {e}")
        return None

async def fetch_multiple_users_concurrent(base_url: str, user_ids: List[int], max_concurrent: int = 5) -> List[Optional[Dict[str, Any]]]:
    """Fetch multiple users with controlled concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_semaphore(session, user_id):
        async with semaphore:
            return await fetch_user_async(session, base_url, user_id)
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [fetch_with_semaphore(session, user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to results
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Task failed with exception: {result}")
                clean_results.append(None)
            else:
                clean_results.append(result)
        
        return clean_results

async def fetch_all_users_paginated(base_url: str) -> List[Dict[str, Any]]:
    """Fetch all users using async pagination"""
    all_users = []
    
    async with aiohttp.ClientSession() as session:
        # Get first page to determine total pages
        async with session.get(f"{base_url}/users", params={"page": 1, "per_page": 25}) as response:
            if response.status != 200:
                print(f"❌ Failed to fetch first page: {response.status}")
                return []
            
            data = await response.json()
            all_users.extend(data.get('items', []))
            
            pagination = data.get('pagination', {})
            total_pages = pagination.get('total_pages', 1)
            
            print(f"📄 Page 1/{total_pages}: {len(data.get('items', []))} users")
            
            if total_pages > 1:
                # Create tasks for remaining pages
                async def fetch_page(page_num):
                    async with session.get(f"{base_url}/users", params={"page": page_num, "per_page": 25}) as resp:
                        if resp.status == 200:
                            page_data = await resp.json()
                            print(f"📄 Page {page_num}/{total_pages}: {len(page_data.get('items', []))} users")
                            return page_data.get('items', [])
                        return []
                
                remaining_pages = range(2, min(total_pages + 1, 11))  # Limit to 10 pages max
                page_tasks = [fetch_page(page) for page in remaining_pages]
                page_results = await asyncio.gather(*page_tasks)
                
                for page_users in page_results:
                    all_users.extend(page_users)
    
    print(f"✅ Fetched {len(all_users)} users total across {min(total_pages, 10)} pages")
    return all_users

async def compare_sync_vs_async_performance():
    """Compare sync vs async performance"""
    user_ids = list(range(1, 11))  # 10 users
    base_url = "https://api.example.com"
    
    # Simulate sync requests (for demonstration)
    print("🔄 Simulating synchronous requests...")
    sync_start = time.time()
    # In real sync code, this would be sequential HTTP requests
    await asyncio.sleep(len(user_ids) * 0.5)  # Simulate 0.5s per request
    sync_time = time.time() - sync_start
    
    # Async requests
    print("🚀 Performing asynchronous requests...")
    async_start = time.time()
    results = await fetch_multiple_users_concurrent(base_url, user_ids, max_concurrent=5)
    async_time = time.time() - async_start
    
    successful_requests = len([r for r in results if r is not None])
    
    print(f"\\n📊 Performance Comparison:")
    print(f"   • Synchronous time: {sync_time:.2f}s")
    print(f"   • Asynchronous time: {async_time:.2f}s")
    print(f"   • Speedup: {sync_time/async_time:.1f}x")
    print(f"   • Successful requests: {successful_requests}/{len(user_ids)}")
    
    return {
        "sync_time": sync_time,
        "async_time": async_time,
        "speedup": sync_time / async_time,
        "success_rate": successful_requests / len(user_ids)
    }

# Usage examples
async def main():
    base_url = "https://api.example.com"
    
    # Fetch multiple users concurrently
    user_ids = [1, 2, 3, 4, 5]
    users = await fetch_multiple_users_concurrent(base_url, user_ids)
    print(f"Fetched {len([u for u in users if u])} users")
    
    # Fetch all users with pagination
    all_users = await fetch_all_users_paginated(base_url)
    
    # Performance comparison
    perf_results = await compare_sync_vs_async_performance()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Success criteria
        success_criteria = SuccessCriteria(
            required_api_calls=[
                {
                    "method": "GET",
                    "path": "/users",
                    "expected_status": [200],
                    "min_calls": 3,  # Should make multiple concurrent calls
                    "requires_async": True
                }
            ],
            expected_outputs=[
                "async",
                "concurrent",
                "await",
                "asyncio"
            ],
            code_quality_requirements={
                "uses_async_await": True,
                "handles_concurrency": True,
                "has_error_handling": True
            },
            minimum_score=8.0
        )
        
        return LearningTask(
            task_id=self._create_task_id(TaskType.ASYNC_API_CALLS, difficulty),
            task_type=TaskType.ASYNC_API_CALLS,
            difficulty=difficulty,
            title="Asynchronous API Calls and Concurrency",
            description="Learn to make efficient concurrent API requests using async/await and aiohttp for improved performance.",
            api_documentation=api_docs,
            starter_code=starter_code,
            solution_template=solution_template,
            success_criteria=success_criteria,
            hints=[
                "Use aiohttp for async HTTP requests",
                "Control concurrency with asyncio.Semaphore",
                "Handle timeouts and exceptions in async context",
                "Use asyncio.gather() to wait for multiple tasks",
                "Compare performance with synchronous requests"
            ],
            estimated_time=50 if difficulty == TaskDifficulty.INTERMEDIATE else 70,
            learning_objectives=[
                "Understand asynchronous programming concepts",
                "Learn to use aiohttp for async HTTP requests",
                "Implement controlled concurrency patterns",
                "Handle errors in async contexts",
                "Compare sync vs async performance"
            ],
            tags=["async", "concurrency", "aiohttp", "performance", "asyncio"]
        )


class TaskGeneratorManager:
    """Main manager for orchestrating all task generators"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
        
        # Initialize all generators
        self.generators = {
            TaskType.BASIC_GET_REQUEST: BasicGetRequestGenerator(seed),
            TaskType.POST_CREATE_RESOURCE: PostCreateResourceGenerator(seed),
            TaskType.PUT_UPDATE_RESOURCE: PutUpdateResourceGenerator(seed),
            TaskType.DELETE_RESOURCE: DeleteResourceGenerator(seed),
            TaskType.AUTHENTICATION_SETUP: AuthenticationSetupGenerator(seed),
            TaskType.ERROR_HANDLING: ErrorHandlingGenerator(seed),
            TaskType.PAGINATION_HANDLING: PaginationHandlingGenerator(seed),
            TaskType.BULK_OPERATIONS: BulkOperationsGenerator(seed),
            TaskType.RESPONSE_VALIDATION: ResponseValidationGenerator(seed),
            TaskType.ASYNC_API_CALLS: AsyncApiCallsGenerator(seed)
        }
        
        # Task progression mappings
        self.difficulty_progression = {
            TaskDifficulty.BEGINNER: [
                TaskType.BASIC_GET_REQUEST,
                TaskType.POST_CREATE_RESOURCE,
                TaskType.ERROR_HANDLING
            ],
            TaskDifficulty.INTERMEDIATE: [
                TaskType.PUT_UPDATE_RESOURCE,
                TaskType.DELETE_RESOURCE,
                TaskType.AUTHENTICATION_SETUP,
                TaskType.PAGINATION_HANDLING,
                TaskType.RESPONSE_VALIDATION
            ],
            TaskDifficulty.ADVANCED: [
                TaskType.BULK_OPERATIONS,
                TaskType.ASYNC_API_CALLS
            ],
            TaskDifficulty.EXPERT: [
                TaskType.BULK_OPERATIONS,
                TaskType.ASYNC_API_CALLS,
                TaskType.RESPONSE_VALIDATION
            ]
        }
    
    def generate_task(
        self,
        task_type: Optional[TaskType] = None,
        difficulty: Optional[TaskDifficulty] = None,
        **_kwargs
    ) -> LearningTask:
        """Generate a single task"""
        
        # Auto-select task type if not provided
        if task_type is None:
            if difficulty:
                available_types = self.difficulty_progression.get(difficulty, list(TaskType))
            else:
                available_types = list(TaskType)
            task_type = random.choice(available_types)
        
        # Auto-select difficulty if not provided
        if difficulty is None:
            # Find appropriate difficulty for task type
            for diff, types in self.difficulty_progression.items():
                if task_type in types:
                    difficulty = diff
                    break
            else:
                difficulty = TaskDifficulty.INTERMEDIATE
        
        # Generate task using appropriate generator
        generator = self.generators[task_type]
        return generator.generate_task(difficulty=difficulty, **_kwargs)
    
    def generate_task_set(
        self,
        count: int = 5,
        difficulty: Optional[TaskDifficulty] = None,
        task_types: Optional[List[TaskType]] = None,
        ensure_variety: bool = True
    ) -> List[LearningTask]:
        """Generate a set of tasks"""
        
        tasks = []
        used_types = set()
        
        for _ in range(count):
            # Determine available task types
            if task_types:
                available_types = task_types.copy()
            elif difficulty:
                available_types = self.difficulty_progression.get(difficulty, list(TaskType))
            else:
                available_types = list(TaskType)
            
            # Ensure variety by avoiding recently used types
            if ensure_variety and used_types and len(available_types) > len(used_types):
                available_types = [t for t in available_types if t not in used_types]
            
            # Select task type
            task_type = random.choice(available_types)
            used_types.add(task_type)
            
            # Reset used types if we've used them all
            if len(used_types) >= len(self.difficulty_progression.get(difficulty, list(TaskType))):
                used_types.clear()
            
            # Generate task
            task = self.generate_task(task_type=task_type, difficulty=difficulty)
            tasks.append(task)
        
        return tasks
    
    def generate_progressive_curriculum(
        self,
        total_tasks: int = 20
    ) -> List[LearningTask]:
        """Generate a progressive curriculum from beginner to advanced"""
        
        curriculum = []
        
        # Define progression structure
        progression_plan = [
            (TaskDifficulty.BEGINNER, 6),      # 30% beginner
            (TaskDifficulty.INTERMEDIATE, 10), # 50% intermediate  
            (TaskDifficulty.ADVANCED, 4)       # 20% advanced
        ]
        
        # Adjust counts to match total_tasks
        total_planned = sum(count for _, count in progression_plan)
        scale_factor = total_tasks / total_planned
        
        for difficulty, planned_count in progression_plan:
            actual_count = max(1, int(planned_count * scale_factor))
            
            # Generate tasks for this difficulty level
            difficulty_tasks = self.generate_task_set(
                count=actual_count,
                difficulty=difficulty,
                ensure_variety=True
            )
            curriculum.extend(difficulty_tasks)
        
        # Shuffle slightly to avoid being too predictable
        # Keep general progression but allow some mixing
        for i in range(len(curriculum) - 1):
            if random.random() < 0.2:  # 20% chance to swap with next
                curriculum[i], curriculum[i + 1] = curriculum[i + 1], curriculum[i]
        
        return curriculum[:total_tasks]
    
    def generate_focused_session(
        self,
        focus_area: str,
        count: int = 5,
        difficulty: Optional[TaskDifficulty] = None
    ) -> List[LearningTask]:
        """Generate tasks focused on a specific area"""
        
        focus_mappings = {
            "crud": [
                TaskType.BASIC_GET_REQUEST,
                TaskType.POST_CREATE_RESOURCE,
                TaskType.PUT_UPDATE_RESOURCE,
                TaskType.DELETE_RESOURCE
            ],
            "error_handling": [
                TaskType.ERROR_HANDLING,
                TaskType.RESPONSE_VALIDATION,
                TaskType.AUTHENTICATION_SETUP
            ],
            "performance": [
                TaskType.BULK_OPERATIONS,
                TaskType.ASYNC_API_CALLS,
                TaskType.PAGINATION_HANDLING
            ],
            "security": [
                TaskType.AUTHENTICATION_SETUP,
                TaskType.RESPONSE_VALIDATION,
                TaskType.ERROR_HANDLING
            ],
            "advanced": [
                TaskType.BULK_OPERATIONS,
                TaskType.ASYNC_API_CALLS,
                TaskType.RESPONSE_VALIDATION
            ]
        }
        
        focus_types = focus_mappings.get(focus_area, list(TaskType))
        
        return self.generate_task_set(
            count=count,
            difficulty=difficulty,
            task_types=focus_types,
            ensure_variety=True
        )
    
    def generate_adaptive_next_task(
        self,
        previous_tasks: List[LearningTask],
        performance_scores: List[float],
        current_difficulty: TaskDifficulty = TaskDifficulty.INTERMEDIATE
    ) -> LearningTask:
        """Generate next task based on previous performance"""
        
        if not previous_tasks or not performance_scores:
            return self.generate_task(difficulty=current_difficulty)
        
        # Calculate recent performance
        recent_scores = performance_scores[-3:]  # Last 3 tasks
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Adjust difficulty based on performance
        if avg_score >= 8.0:  # High performance - increase difficulty
            if current_difficulty == TaskDifficulty.BEGINNER:
                new_difficulty = TaskDifficulty.INTERMEDIATE
            elif current_difficulty == TaskDifficulty.INTERMEDIATE:
                new_difficulty = TaskDifficulty.ADVANCED
            else:
                new_difficulty = TaskDifficulty.EXPERT
        elif avg_score <= 4.0:  # Low performance - decrease difficulty
            if current_difficulty == TaskDifficulty.EXPERT:
                new_difficulty = TaskDifficulty.ADVANCED
            elif current_difficulty == TaskDifficulty.ADVANCED:
                new_difficulty = TaskDifficulty.INTERMEDIATE
            else:
                new_difficulty = TaskDifficulty.BEGINNER
        else:
            new_difficulty = current_difficulty
        
        # Identify weak areas from previous tasks
        task_type_scores = {}
        
        for i, task in enumerate(previous_tasks[-5:]):
            if i < len(performance_scores):
                task_type_scores[task.task_type] = performance_scores[-(5-i)]
        
        # Find task types that need improvement
        weak_areas = [
            task_type for task_type, score in task_type_scores.items()
            if score < 6.0
        ]
        
        # Generate task focusing on weak areas if any
        if weak_areas:
            task_type = random.choice(weak_areas)
            return self.generate_task(task_type=task_type, difficulty=new_difficulty)
        
        # Otherwise generate variety task
        return self.generate_task(difficulty=new_difficulty)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about available tasks"""
        
        stats = {
            "total_generators": len(self.generators),
            "task_types": [t.value for t in TaskType],
            "difficulty_levels": [d.value for d in TaskDifficulty],
            "progression_mapping": {
                diff.value: [t.value for t in types]
                for diff, types in self.difficulty_progression.items()
            }
        }
        
        return stats
    
    def validate_task(self, task: LearningTask) -> List[str]:
        """Validate a generated task for completeness"""
        
        issues = []
        
        # Check required fields
        if not task.title:
            issues.append("Task missing title")
        
        if not task.description:
            issues.append("Task missing description")
        
        if not task.starter_code:
            issues.append("Task missing starter code")
        
        if not task.solution_template:
            issues.append("Task missing solution template")
        
        if not task.api_documentation.endpoints:
            issues.append("Task missing API endpoints")
        
        if not task.success_criteria.required_api_calls:
            issues.append("Task missing success criteria")
        
        if not task.hints:
            issues.append("Task missing hints")
        
        if not task.learning_objectives:
            issues.append("Task missing learning objectives")
        
        # Check code quality
        if "TODO" not in task.starter_code:
            issues.append("Starter code missing TODO items")
        
        if task.estimated_time <= 0:
            issues.append("Invalid estimated time")
        
        return issues
    
    def export_tasks_to_json(self, tasks: List[LearningTask], filepath: str):
        """Export tasks to JSON file"""
        
        def task_to_dict(task: LearningTask) -> Dict[str, Any]:
            return {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "difficulty": task.difficulty.value,
                "title": task.title,
                "description": task.description,
                "api_documentation": {
                    "title": task.api_documentation.title,
                    "base_url": task.api_documentation.base_url,
                    "authentication": task.api_documentation.authentication,
                    "endpoints": task.api_documentation.endpoints,
                    "examples": task.api_documentation.examples,
                    "error_codes": task.api_documentation.error_codes,
                    "rate_limits": task.api_documentation.rate_limits,
                    "notes": task.api_documentation.notes
                },
                "starter_code": task.starter_code,
                "solution_template": task.solution_template,
                "success_criteria": {
                    "required_api_calls": task.success_criteria.required_api_calls,
                    "expected_outputs": task.success_criteria.expected_outputs,
                    "forbidden_patterns": task.success_criteria.forbidden_patterns,
                    "performance_requirements": task.success_criteria.performance_requirements,
                    "code_quality_requirements": task.success_criteria.code_quality_requirements,
                    "minimum_score": task.success_criteria.minimum_score
                },
                "hints": task.hints,
                "estimated_time": task.estimated_time,
                "learning_objectives": task.learning_objectives,
                "tags": task.tags,
                "metadata": task.metadata
            }
        
        tasks_data = [task_to_dict(task) for task in tasks]
        
        with open(filepath, 'w') as f:
            json.dump(tasks_data, f, indent=2)
    
    def get_recommended_task_sequence(
        self,
        user_level: str = "beginner",
        learning_goals: Optional[List[str]] = None
    ) -> List[LearningTask]:
        """Get recommended sequence of tasks for a user"""
        
        if user_level == "beginner":
            return self.generate_task_set(
                count=8,
                difficulty=TaskDifficulty.BEGINNER,
                ensure_variety=True
            )
        elif user_level == "intermediate":
            return self.generate_progressive_curriculum(total_tasks=15)
        elif user_level == "advanced":
            return self.generate_task_set(
                count=10,
                difficulty=TaskDifficulty.ADVANCED,
                ensure_variety=True
            )
        else:
            # Custom based on learning goals
            if learning_goals:
                all_tasks = []
                for goal in learning_goals:
                    goal_tasks = self.generate_focused_session(
                        focus_area=goal,
                        count=3,
                        difficulty=TaskDifficulty.INTERMEDIATE
                    )
                    all_tasks.extend(goal_tasks)
                return all_tasks
            else:
                return self.generate_progressive_curriculum()