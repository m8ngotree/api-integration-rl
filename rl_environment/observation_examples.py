"""
Examples and tests for the observation processing system.

This script demonstrates the comprehensive observation space processing
including tokenization, AST parsing, embeddings, and semantic analysis.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any
import json

from rl_environment.observation_processor import (
    ObservationEncoder, CodeTokenizer, CodeASTAnalyzer, APISchemaProcessor
)
from rl_environment.code_embeddings import (
    CodeEmbeddingGenerator, CodeSemanticAnalyzer
)
from rl_environment.gym_environment import APIIntegrationEnv, EnvironmentConfig
from rl_environment.task_generator import TaskDifficulty


def demonstrate_code_tokenization():
    """Demonstrate code tokenization capabilities"""
    print("=== Code Tokenization Demo ===\n")
    
    tokenizer = CodeTokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample vocabulary: {tokenizer.vocabulary[:10]}")
    print()
    
    # Test different code samples
    code_samples = [
        # Basic API integration
        """
import requests
import json

def get_user_data(user_id):
    url = f"https://api.example.com/users/{user_id}"
    headers = {"Authorization": "Bearer token123"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None

data = get_user_data(123)
print(data)
        """,
        
        # Advanced async example
        """
import asyncio
import httpx

async def fetch_multiple_users(user_ids):
    async with httpx.AsyncClient() as client:
        tasks = []
        for user_id in user_ids:
            url = f"https://api.example.com/users/{user_id}"
            task = client.get(url)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses if r.status_code == 200]

# Usage
user_data = asyncio.run(fetch_multiple_users([1, 2, 3, 4, 5]))
        """,
        
        # Error handling and data processing
        """
import requests
from typing import Dict, Optional

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def get_paginated_data(self, endpoint: str, limit: int = 100) -> List[Dict]:
        all_data = []
        offset = 0
        
        while True:
            params = {"limit": limit, "offset": offset}
            response = self._make_request("GET", endpoint, params=params)
            
            if not response or "data" not in response:
                break
            
            data = response["data"]
            all_data.extend(data)
            
            if len(data) < limit:
                break
            
            offset += limit
        
        return all_data
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        """
    ]
    
    for i, code in enumerate(code_samples, 1):
        print(f"--- Code Sample {i} ---")
        print("Code preview:")
        print(code.strip()[:200] + "..." if len(code.strip()) > 200 else code.strip())
        print()
        
        # Tokenize
        tokens = tokenizer.tokenize_code(code)
        print(f"Total tokens: {len(tokens)}")
        
        # Show token type distribution
        token_types = {}
        for token in tokens:
            token_type = token.token_type.value
            token_types[token_type] = token_types.get(token_type, 0) + 1
        
        print("Token type distribution:")
        for token_type, count in sorted(token_types.items()):
            print(f"  {token_type}: {count}")
        
        # Show some sample tokens
        print("Sample tokens:")
        for j, token in enumerate(tokens[:10]):
            print(f"  {j+1}. '{token.text}' ({token.token_type.value})")
        
        # Encode to numerical representation
        encoded = tokenizer.encode_tokens(tokens, max_length=100)
        print(f"Encoded shape: {encoded.shape}")
        print(f"Non-zero positions: {np.count_nonzero(encoded)}")
        
        print("\n" + "="*50 + "\n")


def demonstrate_ast_analysis():
    """Demonstrate AST analysis capabilities"""
    print("=== AST Analysis Demo ===\n")
    
    analyzer = CodeASTAnalyzer()
    
    # Test different code complexities
    code_samples = [
        ("Simple script", """
import requests

url = "https://api.example.com/data"
response = requests.get(url)
print(response.json())
        """),
        
        ("Function-based code", """
import requests
import json
from typing import Dict, Optional

def fetch_user(user_id: int) -> Optional[Dict]:
    url = f"https://api.example.com/users/{user_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching user {user_id}: {e}")
        return None

def process_users(user_ids: List[int]) -> List[Dict]:
    users = []
    for user_id in user_ids:
        user = fetch_user(user_id)
        if user:
            users.append(user)
    return users
        """),
        
        ("Class-based code", """
import requests
import asyncio
from abc import ABC, abstractmethod

class BaseAPIClient(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    @abstractmethod
    async def authenticate(self) -> bool:
        pass

class UserAPIClient(BaseAPIClient):
    def __init__(self, base_url: str, api_key: str):
        super().__init__(base_url)
        self.api_key = api_key
        self.session = requests.Session()
    
    async def authenticate(self) -> bool:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.session.headers.update(headers)
        return True
    
    def get_user(self, user_id: int):
        response = self.session.get(f"{self.base_url}/users/{user_id}")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch user: {response.status_code}")
        """),
        
        ("Malformed code", """
import requests

def incomplete_function(
    # Missing closing parenthesis and body
    
url = "https://api.example.com/data"
response = requests.get(url
# Missing closing parenthesis

if response.status_code == 200:
    data = response.json()
    print(data
# Missing closing parenthesis
        """)
    ]
    
    for name, code in code_samples:
        print(f"--- {name} ---")
        print("Code:")
        print(code.strip())
        print()
        
        # Parse structure
        structure = analyzer.parse_code_structure(code)
        
        print("Structure analysis:")
        print(f"  Imports: {len(structure.imports)}")
        for imp in structure.imports:
            print(f"    - {imp}")
        
        print(f"  Functions: {len(structure.functions)}")
        for func in structure.functions:
            print(f"    - {func['name']}() [args: {len(func.get('args', []))}]")
        
        print(f"  Classes: {len(structure.classes)}")
        for cls in structure.classes:
            print(f"    - {cls['name']} [methods: {len(cls.get('methods', []))}]")
        
        print(f"  Variables: {len(structure.variables)}")
        for var in structure.variables[:5]:  # Show first 5
            print(f"    - {var['name']}")
        
        print(f"  API calls: {len(structure.api_calls)}")
        for call in structure.api_calls:
            print(f"    - {call['function']}()")
        
        print(f"  Control flow: {len(structure.control_flow)}")
        print(f"  Error handling: {len(structure.error_handling)}")
        
        if structure.syntax_errors:
            print(f"  Syntax errors: {structure.syntax_errors}")
        
        print("Complexity metrics:")
        for metric, value in structure.complexity_metrics.items():
            print(f"    {metric}: {value}")
        
        print("\n" + "="*50 + "\n")


def demonstrate_api_schema_processing():
    """Demonstrate API schema processing"""
    print("=== API Schema Processing Demo ===\n")
    
    processor = APISchemaProcessor()
    
    # Test different API documentation formats
    api_docs = [
        {
            "title": "User Management API",
            "base_url": "https://api.example.com",
            "version": "v1",
            "authentication": {
                "type": "bearer",
                "scheme": "JWT",
                "description": "Use JWT token in Authorization header"
            },
            "endpoints": [
                {
                    "path": "/users",
                    "method": "GET",
                    "description": "List all users",
                    "parameters": [
                        {"name": "limit", "type": "integer", "required": False, "default": 10},
                        {"name": "offset", "type": "integer", "required": False, "default": 0}
                    ],
                    "auth_required": True,
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "users": {"type": "array"},
                            "total": {"type": "integer"}
                        }
                    }
                },
                {
                    "path": "/users/{id}",
                    "method": "GET",
                    "description": "Get user by ID",
                    "parameters": [
                        {"name": "id", "type": "integer", "required": True, "in": "path"}
                    ],
                    "auth_required": True
                },
                {
                    "path": "/users",
                    "method": "POST",
                    "description": "Create new user",
                    "request_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"}
                        },
                        "required": ["name", "email"]
                    },
                    "auth_required": True
                }
            ],
            "error_codes": {
                "404": "Not Found",
                "401": "Unauthorized",
                "400": "Bad Request"
            },
            "rate_limits": {
                "requests_per_minute": 100,
                "burst_limit": 20
            }
        }
    ]
    
    for i, docs in enumerate(api_docs, 1):
        print(f"--- API Documentation {i} ---")
        print("Raw documentation:")
        print(json.dumps(docs, indent=2)[:500] + "..." if len(json.dumps(docs, indent=2)) > 500 else json.dumps(docs, indent=2))
        print()
        
        # Process schema
        schema = processor.process_api_documentation(docs)
        
        print("Processed schema:")
        print(f"  Title: {schema.title}")
        print(f"  Base URL: {schema.base_url}")
        print(f"  Version: {schema.version}")
        print(f"  Endpoints: {len(schema.endpoints)}")
        
        for endpoint in schema.endpoints:
            print(f"    - {endpoint['method']} {endpoint['path']}")
            print(f"      Parameters: {len(endpoint['parameters'])}")
            print(f"      Auth required: {endpoint['auth_required']}")
        
        print(f"  Authentication: {schema.authentication.get('type', 'none')}")
        print(f"  Error codes: {len(schema.error_codes)}")
        print(f"  Rate limits: {bool(schema.rate_limits)}")
        
        # Encode to numerical representation
        encoded = processor.encode_api_schema(schema, max_length=100)
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Non-zero features: {np.count_nonzero(encoded)}")
        print(f"  Sample features: {encoded[:10]}")
        
        print("\n" + "="*50 + "\n")


def demonstrate_semantic_analysis():
    """Demonstrate semantic pattern analysis"""
    print("=== Semantic Analysis Demo ===\n")
    
    analyzer = CodeSemanticAnalyzer()
    ast_analyzer = CodeASTAnalyzer()
    
    # Test code with various patterns
    code_samples = [
        ("Basic API integration", """
import requests

api_key = "your-api-key-here"
base_url = "https://api.example.com"

headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get(f"{base_url}/users", headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"Found {len(data)} users")
else:
    print(f"Error: {response.status_code}")
        """),
        
        ("Advanced with error handling", """
import requests
import logging

logger = logging.getLogger(__name__)

def make_api_request(url, method='GET', **kwargs):
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        raise
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
    except ValueError as e:
        logger.error(f"JSON decode error: {e}")
        raise

def get_paginated_data(endpoint, limit=100):
    all_data = []
    offset = 0
    
    while True:
        params = {'limit': limit, 'offset': offset}
        data = make_api_request(endpoint, params=params)
        
        if not data or 'results' not in data:
            break
        
        results = data['results']
        all_data.extend(results)
        
        if len(results) < limit:
            break
        
        offset += limit
    
    return all_data
        """),
        
        ("Async with authentication", """
import asyncio
import aiohttp
import os

class AsyncAPIClient:
    def __init__(self):
        self.base_url = "https://api.example.com"
        self.api_key = os.getenv("API_KEY")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def authenticate(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.get(f"{self.base_url}/auth/verify", headers=headers) as response:
            if response.status == 200:
                return True
            else:
                raise Exception(f"Authentication failed: {response.status}")
    
    async def fetch_user_data(self, user_ids):
        tasks = []
        for user_id in user_ids:
            task = self.session.get(f"{self.base_url}/users/{user_id}")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        
        for response in responses:
            if isinstance(response, Exception):
                continue
            if response.status == 200:
                data = await response.json()
                results.append(data)
        
        return results
        """)
    ]
    
    for name, code in code_samples:
        print(f"--- {name} ---")
        print("Code preview:")
        print(code.strip()[:300] + "..." if len(code.strip()) > 300 else code.strip())
        print()
        
        # Parse structure first
        structure = ast_analyzer.parse_code_structure(code)
        
        # Analyze semantic patterns
        patterns = analyzer.analyze_semantic_patterns(code, structure)
        
        print(f"Detected {len(patterns)} semantic patterns:")
        for pattern in patterns:
            print(f"  - {pattern.pattern_type}: {pattern.confidence:.2f}")
            if pattern.context:
                context_str = ", ".join(f"{k}={v}" for k, v in pattern.context.items() if k != 'examples')
                if context_str:
                    print(f"    Context: {context_str}")
        
        print("\n" + "="*50 + "\n")


def demonstrate_embeddings():
    """Demonstrate embedding generation"""
    print("=== Embedding Generation Demo ===\n")
    
    # Initialize components
    tokenizer = CodeTokenizer()
    ast_analyzer = CodeASTAnalyzer()
    api_processor = APISchemaProcessor()
    embedding_generator = CodeEmbeddingGenerator(embedding_dim=64)  # Smaller for demo
    semantic_analyzer = CodeSemanticAnalyzer()
    
    # Sample code and API schema
    code = """
import requests
import json

def get_user_profile(user_id, api_key):
    url = f"https://api.example.com/users/{user_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None
    """
    
    api_schema = {
        "title": "User API",
        "base_url": "https://api.example.com",
        "endpoints": [
            {"path": "/users/{id}", "method": "GET", "auth_required": True}
        ],
        "authentication": {"type": "bearer"}
    }
    
    print("Input code:")
    print(code.strip())
    print()
    
    print("API schema:")
    print(json.dumps(api_schema, indent=2))
    print()
    
    # Process components
    tokens = tokenizer.tokenize_code(code)
    code_structure = ast_analyzer.parse_code_structure(code)
    processed_api = api_processor.process_api_documentation(api_schema)
    semantic_patterns = semantic_analyzer.analyze_semantic_patterns(code, code_structure)
    
    print("Processing results:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Functions: {len(code_structure.functions)}")
    print(f"  API calls: {len(code_structure.api_calls)}")
    print(f"  Semantic patterns: {len(semantic_patterns)}")
    print()
    
    # Generate embeddings
    embeddings = embedding_generator.create_combined_embedding(
        api_schema=processed_api,
        code_structure=code_structure,
        tokens=tokens,
        semantic_patterns=semantic_patterns
    )
    
    print("Generated embeddings:")
    for name, embedding in embeddings.items():
        if name != 'token_sequence':  # Skip large token sequence
            print(f"  {name}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
            print(f"    sample values: {embedding[:5]}")
    
    # Test similarity computation
    print("\nEmbedding similarities:")
    emb_names = [name for name in embeddings.keys() if name != 'token_sequence']
    for i, name1 in enumerate(emb_names):
        for name2 in emb_names[i+1:]:
            similarity = embedding_generator.compute_similarity(
                embeddings[name1], embeddings[name2]
            )
            print(f"  {name1} <-> {name2}: {similarity:.3f}")
    
    print("\n" + "="*50 + "\n")


async def demonstrate_complete_observation_processing():
    """Demonstrate complete observation processing in environment"""
    print("=== Complete Observation Processing Demo ===\n")
    
    # Create environment with advanced observation processing
    config = EnvironmentConfig(
        max_episode_steps=5,
        task_difficulty=TaskDifficulty.INTERMEDIATE,
        max_code_tokens=256,
        max_api_tokens=512,
        embedding_dim=64,
        enable_embeddings=True,
        enable_semantic_analysis=True
    )
    
    env = APIIntegrationEnv(config)
    
    try:
        # Reset environment
        observation, info = await env.reset(seed=42)
        
        print(f"Task: {info['task_title']}")
        print()
        
        print("Observation space structure:")
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.size <= 10:
                    print(f"    values: {value}")
                else:
                    print(f"    sample: {value.flat[:5]}...")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        print(f"\nObservation space info:")
        obs_info = env.observation_encoder.get_observation_space_info()
        for key, value in obs_info.items():
            print(f"  {key}: {value}")
        
        if env.embedding_generator:
            print(f"\nEmbedding system info:")
            emb_info = env.embedding_generator.get_embedding_info()
            for key, value in emb_info.items():
                print(f"  {key}: {value}")
        
        # Take a step to see how observations change
        print("\n--- Taking a step ---")
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = await env.step(action)
        
        print(f"Reward: {reward:.3f}")
        print(f"Actions applied: {info.get('actions_applied', 0)}/{info.get('actions_total', 0)}")
        
        # Show key observation changes
        print("Key observation features:")
        print(f"  Code tokens (non-zero): {np.count_nonzero(observation['code_tokens'])}")
        print(f"  Code features (first 5): {observation['code_features'][:5]}")
        print(f"  Completion progress: {observation['completion_progress'][0]:.3f}")
        
        if 'semantic_patterns' in observation:
            print(f"  Semantic patterns (active): {np.count_nonzero(observation['semantic_patterns'])}")
        
        if 'code_embedding' in observation:
            print(f"  Code embedding norm: {np.linalg.norm(observation['code_embedding']):.3f}")
        
    finally:
        await env.close()


async def main():
    """Run all observation processing demonstrations"""
    print("Comprehensive Observation Processing Examples\n")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_code_tokenization()
    
    demonstrate_ast_analysis()
    
    demonstrate_api_schema_processing()
    
    demonstrate_semantic_analysis()
    
    demonstrate_embeddings()
    
    await demonstrate_complete_observation_processing()
    
    print("\nAll demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(main())