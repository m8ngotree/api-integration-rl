"""
Advanced embedding system for semantic code understanding.

This module provides embeddings for code tokens, API patterns, and semantic
relationships to enhance ML model understanding of code structure and intent.
"""

import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib

from rl_environment.observation_processor import (
    Token, TokenType, CodeStructure, APISchemaStructure
)


@dataclass
class CodeEmbedding:
    """Represents an embedding for a code element"""
    element: str
    embedding: np.ndarray
    element_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticPattern:
    """Represents a semantic pattern in code"""
    pattern_type: str
    pattern: str
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


class CodeSemanticAnalyzer:
    """Analyzes semantic patterns in code for better understanding"""
    
    def __init__(self):
        # API integration patterns
        self.api_patterns = {
            'authentication': [
                r'auth.*=.*token',
                r'headers.*=.*["\']authorization["\']',
                r'api[_-]?key',
                r'bearer.*token',
                r'basic.*auth'
            ],
            'http_request': [
                r'requests\.(get|post|put|delete|patch)',
                r'httpx\.(get|post|put|delete|patch)',
                r'urllib\.request',
                r'fetch\(',
                r'axios\.'
            ],
            'data_processing': [
                r'\.json\(\)',
                r'json\.loads',
                r'json\.dumps',
                r'response\.text',
                r'response\.content'
            ],
            'error_handling': [
                r'try:.*except.*requests\.',
                r'if.*status_code.*!=.*200',
                r'raise.*Exception',
                r'assert.*response',
                r'except.*HTTPError'
            ],
            'pagination': [
                r'limit.*=.*\d+',
                r'offset.*=.*\d+',
                r'page.*=.*\d+',
                r'next.*url',
                r'while.*has.*more'
            ],
            'url_construction': [
                r'f["\'].*\{.*\}.*["\']',
                r'url.*\+.*["\']',
                r'\.format\(',
                r'urljoin\(',
                r'base.*url'
            ]
        }
        
        # Code quality patterns
        self.quality_patterns = {
            'good_practices': [
                r'def.*\(.*\).*->.*:',  # Type hints
                r'""".*"""',            # Docstrings
                r'# .*',                # Comments
                r'if __name__ == ["\']__main__["\']:',  # Main guard
                r'with.*open\(',        # Context managers
                r'logging\.',           # Logging
                r'config\.',            # Configuration
                r'env\.',               # Environment variables
            ],
            'anti_patterns': [
                r'eval\(',
                r'exec\(',
                r'global\s+\w+',
                r'import \*',
                r'except:',  # Bare except
                r'pass.*#.*TODO',
                r'print\(.*\)',  # Debug prints (context dependent)
            ]
        }
        
        # Build pattern cache
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.compiled_patterns = {}
        
        for category, patterns in {**self.api_patterns, **self.quality_patterns}.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                for pattern in patterns
            ]
    
    def analyze_semantic_patterns(self, code: str, code_structure: CodeStructure) -> List[SemanticPattern]:
        """Analyze semantic patterns in code"""
        patterns = []
        
        # Analyze API patterns
        for category, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                matches = pattern.findall(code)
                if matches:
                    confidence = min(len(matches) / 5.0, 1.0)  # Normalize confidence
                    patterns.append(SemanticPattern(
                        pattern_type=category,
                        pattern=pattern.pattern,
                        confidence=confidence,
                        context={'matches': len(matches), 'examples': matches[:3]}
                    ))
        
        # Analyze structural patterns
        structural_patterns = self._analyze_structural_patterns(code_structure)
        patterns.extend(structural_patterns)
        
        # Analyze flow patterns
        flow_patterns = self._analyze_flow_patterns(code, code_structure)
        patterns.extend(flow_patterns)
        
        return patterns
    
    def _analyze_structural_patterns(self, structure: CodeStructure) -> List[SemanticPattern]:
        """Analyze structural patterns in code"""
        patterns = []
        
        # Import patterns
        if structure.imports:
            api_imports = sum(1 for imp in structure.imports 
                            if any(lib in str(imp) for lib in ['requests', 'httpx', 'urllib', 'aiohttp']))
            if api_imports > 0:
                patterns.append(SemanticPattern(
                    pattern_type='api_imports',
                    pattern='API library imports',
                    confidence=min(api_imports / 3.0, 1.0),
                    context={'count': api_imports, 'total_imports': len(structure.imports)}
                ))
        
        # Function patterns
        if structure.functions:
            async_functions = sum(1 for func in structure.functions if func.get('is_async', False))
            if async_functions > 0:
                patterns.append(SemanticPattern(
                    pattern_type='async_programming',
                    pattern='Async function definitions',
                    confidence=async_functions / len(structure.functions),
                    context={'async_count': async_functions, 'total_functions': len(structure.functions)}
                ))
        
        # API call patterns
        if structure.api_calls:
            http_methods = set()
            for call in structure.api_calls:
                func_name = call.get('function', '').lower()
                for method in ['get', 'post', 'put', 'delete', 'patch']:
                    if method in func_name:
                        http_methods.add(method)
            
            patterns.append(SemanticPattern(
                pattern_type='http_methods',
                pattern='HTTP method usage',
                confidence=min(len(http_methods) / 4.0, 1.0),
                context={'methods': list(http_methods), 'call_count': len(structure.api_calls)}
            ))
        
        return patterns
    
    def _analyze_flow_patterns(self, code: str, structure: CodeStructure) -> List[SemanticPattern]:
        """Analyze control flow and data flow patterns"""
        patterns = []
        
        lines = code.split('\n')
        
        # Request-response pattern
        request_response_pattern = self._detect_request_response_pattern(lines)
        if request_response_pattern:
            patterns.append(request_response_pattern)
        
        # Error handling flow
        error_handling_pattern = self._detect_error_handling_pattern(lines, structure)
        if error_handling_pattern:
            patterns.append(error_handling_pattern)
        
        # Data transformation pattern
        data_transform_pattern = self._detect_data_transformation_pattern(lines)
        if data_transform_pattern:
            patterns.append(data_transform_pattern)
        
        return patterns
    
    def _detect_request_response_pattern(self, lines: List[str]) -> Optional[SemanticPattern]:
        """Detect request-response patterns"""
        request_line = -1
        response_usage = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Find request line
            if any(pattern in line_clean for pattern in ['requests.', 'httpx.', '.get(', '.post(']):
                if 'response' in line_clean or '=' in line_clean:
                    request_line = i
            
            # Find response usage
            if request_line >= 0 and i > request_line:
                if 'response' in line_clean and any(usage in line_clean 
                    for usage in ['.json()', '.text', '.status_code', '.content']):
                    response_usage.append(i)
        
        if request_line >= 0 and response_usage:
            return SemanticPattern(
                pattern_type='request_response_flow',
                pattern='Request-response pattern',
                confidence=min(len(response_usage) / 3.0, 1.0),
                context={
                    'request_line': request_line,
                    'response_usage_lines': response_usage,
                    'usage_count': len(response_usage)
                }
            )
        
        return None
    
    def _detect_error_handling_pattern(self, lines: List[str], structure: CodeStructure) -> Optional[SemanticPattern]:
        """Detect error handling patterns"""
        if not structure.error_handling:
            return None
        
        error_types = []
        for line in lines:
            line_clean = line.strip().lower()
            if 'except' in line_clean:
                # Extract exception type
                except_match = re.search(r'except\s+(\w+)', line_clean)
                if except_match:
                    error_types.append(except_match.group(1))
        
        # Check for API-specific error handling
        api_error_handling = any(error_type in ['httperror', 'requestexception', 'connectionerror', 'timeout']
                               for error_type in error_types)
        
        confidence = 0.5
        if api_error_handling:
            confidence = 0.9
        elif error_types:
            confidence = 0.7
        
        return SemanticPattern(
            pattern_type='error_handling_flow',
            pattern='Error handling pattern',
            confidence=confidence,
            context={
                'error_types': error_types,
                'api_specific': api_error_handling,
                'handler_count': len(structure.error_handling)
            }
        )
    
    def _detect_data_transformation_pattern(self, lines: List[str]) -> Optional[SemanticPattern]:
        """Detect data transformation patterns"""
        transformations = []
        
        for line in lines:
            line_clean = line.strip().lower()
            
            # JSON operations
            if '.json()' in line_clean:
                transformations.append('json_parse')
            elif 'json.loads' in line_clean:
                transformations.append('json_loads')
            elif 'json.dumps' in line_clean:
                transformations.append('json_dumps')
            
            # Data manipulation
            elif any(op in line_clean for op in ['.get(', '.pop(', '.update(', '.extend(']):
                transformations.append('dict_manipulation')
            elif any(op in line_clean for op in ['.append(', '.extend(', '.sort(', '.filter(']):
                transformations.append('list_manipulation')
            
            # String operations
            elif any(op in line_clean for op in ['.strip()', '.split(', '.join(', '.format(']):
                transformations.append('string_manipulation')
        
        if transformations:
            transformation_counts = Counter(transformations)
            return SemanticPattern(
                pattern_type='data_transformation',
                pattern='Data transformation pattern',
                confidence=min(len(set(transformations)) / 5.0, 1.0),
                context={
                    'transformations': dict(transformation_counts),
                    'total_ops': len(transformations)
                }
            )
        
        return None


class CodeEmbeddingGenerator:
    """Generates embeddings for code elements and patterns"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.semantic_analyzer = CodeSemanticAnalyzer()
        
        # Initialize embedding matrices
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding matrices for different code elements"""
        # Token type embeddings
        self.token_type_embeddings = self._create_embedding_matrix(len(TokenType), self.embedding_dim)
        
        # API pattern embeddings
        api_pattern_count = sum(len(patterns) for patterns in self.semantic_analyzer.api_patterns.values())
        self.api_pattern_embeddings = self._create_embedding_matrix(api_pattern_count, self.embedding_dim)
        
        # Structural element embeddings
        structural_elements = [
            'import', 'function', 'class', 'variable', 'api_call',
            'control_flow', 'error_handling', 'comment'
        ]
        self.structural_embeddings = {
            element: self._create_random_embedding(self.embedding_dim)
            for element in structural_elements
        }
        
        # HTTP method embeddings
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        self.http_method_embeddings = {
            method: self._create_random_embedding(self.embedding_dim)
            for method in http_methods
        }
    
    def _create_embedding_matrix(self, vocab_size: int, embedding_dim: int) -> np.ndarray:
        """Create a random embedding matrix"""
        # Use Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        return np.random.normal(0, scale, (vocab_size, embedding_dim)).astype(np.float32)
    
    def _create_random_embedding(self, embedding_dim: int) -> np.ndarray:
        """Create a single random embedding vector"""
        return np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
    
    def generate_token_embeddings(self, tokens: List[Token]) -> np.ndarray:
        """Generate embeddings for a sequence of tokens"""
        if not tokens:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
        
        embeddings = []
        
        for token in tokens:
            # Base embedding from token type
            token_type_idx = list(TokenType).index(token.token_type)
            base_embedding = self.token_type_embeddings[token_type_idx].copy()
            
            # Modify embedding based on token content
            content_hash = self._hash_string(token.text) % 1000
            content_modifier = np.sin(np.arange(self.embedding_dim) * content_hash / 1000.0) * 0.1
            
            final_embedding = base_embedding + content_modifier
            embeddings.append(final_embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def generate_code_structure_embedding(self, structure: CodeStructure) -> np.ndarray:
        """Generate embedding for code structure"""
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Combine embeddings from different structural elements
        element_weights = {
            'imports': 0.15,
            'functions': 0.25,
            'classes': 0.15,
            'variables': 0.10,
            'api_calls': 0.20,
            'control_flow': 0.10,
            'error_handling': 0.05
        }
        
        for element, weight in element_weights.items():
            element_count = len(getattr(structure, element, []))
            if element_count > 0:
                element_embedding = self.structural_embeddings[element]
                # Scale by count (with saturation)
                count_factor = min(element_count / 5.0, 1.0)
                embedding += element_embedding * weight * count_factor
        
        return embedding
    
    def generate_api_schema_embedding(self, schema: APISchemaStructure) -> np.ndarray:
        """Generate embedding for API schema"""
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Endpoint method distribution
        method_counts = Counter()
        for endpoint in schema.endpoints:
            method = endpoint.get('method', 'GET')
            method_counts[method] += 1
        
        # Combine method embeddings
        total_endpoints = sum(method_counts.values())
        if total_endpoints > 0:
            for method, count in method_counts.items():
                if method in self.http_method_embeddings:
                    method_weight = count / total_endpoints
                    embedding += self.http_method_embeddings[method] * method_weight
        
        # Authentication embedding
        auth_type = schema.authentication.get('type', 'none')
        if auth_type != 'none':
            auth_hash = self._hash_string(auth_type)
            auth_modifier = np.sin(np.arange(self.embedding_dim) * auth_hash / 1000.0) * 0.2
            embedding += auth_modifier
        
        # Complexity factors
        complexity_factor = (
            len(schema.endpoints) / 20.0 +  # Endpoint count
            len(schema.error_codes) / 10.0 +  # Error diversity
            (1.0 if schema.rate_limits else 0.0)  # Rate limiting
        ) / 3.0
        
        embedding *= min(complexity_factor, 1.0)
        
        return embedding
    
    def generate_semantic_pattern_embedding(self, patterns: List[SemanticPattern]) -> np.ndarray:
        """Generate embedding for semantic patterns"""
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        if not patterns:
            return embedding
        
        # Group patterns by type
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)
        
        # Generate embedding for each pattern type
        for pattern_type, type_patterns in pattern_groups.items():
            # Create pattern type embedding
            type_hash = self._hash_string(pattern_type)
            type_embedding = np.sin(np.arange(self.embedding_dim) * type_hash / 1000.0)
            
            # Weight by pattern confidence
            avg_confidence = np.mean([p.confidence for p in type_patterns])
            pattern_count = len(type_patterns)
            
            # Combine with count and confidence
            weight = min(pattern_count / 5.0, 1.0) * avg_confidence
            embedding += type_embedding * weight * 0.1
        
        return embedding
    
    def generate_alignment_embedding(self, api_schema: APISchemaStructure, 
                                   code_structure: CodeStructure,
                                   semantic_patterns: List[SemanticPattern]) -> np.ndarray:
        """Generate embedding representing API-code alignment"""
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Method alignment
        api_methods = set()
        for endpoint in api_schema.endpoints:
            api_methods.add(endpoint.get('method', 'GET').lower())
        
        code_methods = set()
        for api_call in code_structure.api_calls:
            func_name = api_call.get('function', '').lower()
            for method in ['get', 'post', 'put', 'delete', 'patch']:
                if method in func_name:
                    code_methods.add(method)
        
        # Calculate alignment score
        if api_methods:
            method_alignment = len(api_methods.intersection(code_methods)) / len(api_methods)
            alignment_embedding = np.ones(self.embedding_dim) * method_alignment * 0.3
            embedding += alignment_embedding
        
        # Pattern-based alignment
        api_patterns = [p for p in semantic_patterns if p.pattern_type in self.semantic_analyzer.api_patterns]
        if api_patterns:
            pattern_strength = np.mean([p.confidence for p in api_patterns])
            pattern_embedding = np.ones(self.embedding_dim) * pattern_strength * 0.2
            embedding += pattern_embedding
        
        # Structural alignment
        has_error_handling = bool(code_structure.error_handling)
        has_error_codes = bool(api_schema.error_codes)
        if has_error_handling and has_error_codes:
            error_alignment = np.ones(self.embedding_dim) * 0.1
            embedding += error_alignment
        
        return embedding
    
    def _hash_string(self, s: str) -> int:
        """Generate a hash for a string"""
        return int(hashlib.md5(s.encode()).hexdigest(), 16)
    
    def create_combined_embedding(self, 
                                 api_schema: APISchemaStructure,
                                 code_structure: CodeStructure,
                                 tokens: List[Token],
                                 semantic_patterns: List[SemanticPattern]) -> Dict[str, np.ndarray]:
        """Create a comprehensive embedding combining all components"""
        
        # Generate individual embeddings
        code_structure_emb = self.generate_code_structure_embedding(code_structure)
        api_schema_emb = self.generate_api_schema_embedding(api_schema)
        semantic_patterns_emb = self.generate_semantic_pattern_embedding(semantic_patterns)
        alignment_emb = self.generate_alignment_embedding(api_schema, code_structure, semantic_patterns)
        
        # Token embeddings (summarized)
        token_embeddings = self.generate_token_embeddings(tokens)
        if token_embeddings.size > 0:
            token_summary = np.mean(token_embeddings, axis=0)
        else:
            token_summary = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Combine embeddings
        combined_embedding = (
            code_structure_emb * 0.3 +
            api_schema_emb * 0.25 +
            token_summary * 0.2 +
            semantic_patterns_emb * 0.15 +
            alignment_emb * 0.1
        )
        
        return {
            'combined': combined_embedding,
            'code_structure': code_structure_emb,
            'api_schema': api_schema_emb,
            'tokens': token_summary,
            'semantic_patterns': semantic_patterns_emb,
            'alignment': alignment_emb,
            'token_sequence': token_embeddings  # Full sequence
        }
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if emb1.size == 0 or emb2.size == 0:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding system"""
        return {
            'embedding_dim': self.embedding_dim,
            'token_types': len(TokenType),
            'structural_elements': len(self.structural_embeddings),
            'http_methods': len(self.http_method_embeddings),
            'api_patterns': sum(len(patterns) for patterns in self.semantic_analyzer.api_patterns.values()),
            'total_parameters': (
                self.token_type_embeddings.size +
                self.api_pattern_embeddings.size +
                sum(emb.size for emb in self.structural_embeddings.values()) +
                sum(emb.size for emb in self.http_method_embeddings.values())
            )
        }