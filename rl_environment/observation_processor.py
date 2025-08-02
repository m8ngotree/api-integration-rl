"""
Observation space processing for API integration tasks.

This module converts API documentation and code state into ML-ready formats
including text tokenization, AST parsing, and structured API schema representation.
"""

import ast
import re
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import warnings

# Suppress AST parsing warnings for malformed code
warnings.filterwarnings("ignore", category=SyntaxWarning)


class TokenType(Enum):
    """Types of tokens for code and API documentation"""
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    STRING = "string"
    NUMBER = "number"
    OPERATOR = "operator"
    API_METHOD = "api_method"
    API_ENDPOINT = "api_endpoint"
    HTTP_METHOD = "http_method"
    PARAMETER = "parameter"
    TYPE_ANNOTATION = "type_annotation"
    IMPORT = "import"
    FUNCTION_DEF = "function_def"
    CLASS_DEF = "class_def"
    VARIABLE = "variable"
    COMMENT = "comment"
    SPECIAL = "special"


@dataclass
class Token:
    """Represents a single token"""
    text: str
    token_type: TokenType
    position: int = 0
    line: int = 0
    column: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeStructure:
    """Structured representation of parsed code"""
    imports: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    api_calls: List[Dict[str, Any]] = field(default_factory=list)
    control_flow: List[Dict[str, Any]] = field(default_factory=list)
    error_handling: List[Dict[str, Any]] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    complexity_metrics: Dict[str, int] = field(default_factory=dict)
    syntax_errors: List[str] = field(default_factory=list)


@dataclass
class APISchemaStructure:
    """Structured representation of API documentation"""
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    authentication: Dict[str, Any] = field(default_factory=dict)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    response_schemas: List[Dict[str, Any]] = field(default_factory=list)
    error_codes: Dict[str, str] = field(default_factory=dict)
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    base_url: str = ""
    version: str = ""
    title: str = ""


class CodeTokenizer:
    """Tokenizes Python code with awareness of API integration patterns"""
    
    def __init__(self):
        # Python keywords
        self.python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'break',
            'continue', 'pass', 'raise', 'assert', 'lambda', 'and', 'or', 'not',
            'in', 'is', 'True', 'False', 'None', 'async', 'await'
        }
        
        # API-related patterns
        self.api_methods = {
            'get', 'post', 'put', 'delete', 'patch', 'head', 'options',
            'request', 'urlopen', 'fetch'
        }
        
        self.http_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'}
        
        # Common API libraries
        self.api_libraries = {
            'requests', 'httpx', 'urllib', 'aiohttp', 'fastapi', 'flask', 'django'
        }
        
        # Build vocabulary
        self.vocabulary = self._build_vocabulary()
        self.vocab_size = len(self.vocabulary)
        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
    
    def _build_vocabulary(self) -> List[str]:
        """Build tokenization vocabulary"""
        vocab = set()
        
        # Special tokens
        vocab.update(['<PAD>', '<UNK>', '<START>', '<END>', '<MASK>'])
        
        # Python keywords
        vocab.update(self.python_keywords)
        
        # API-related tokens
        vocab.update(self.api_methods)
        vocab.update(self.http_methods)
        vocab.update(self.api_libraries)
        
        # Common API integration tokens
        api_tokens = [
            'url', 'endpoint', 'headers', 'params', 'data', 'json', 'response',
            'status_code', 'content', 'text', 'auth', 'token', 'api_key',
            'timeout', 'verify', 'cookies', 'session', 'client', 'server',
            'request', 'method', 'query', 'body', 'payload', 'schema',
            'validation', 'error', 'exception', 'retry', 'limit', 'offset',
            'pagination', 'filter', 'sort', 'search', 'create', 'update',
            'retrieve', 'list', 'delete'
        ]
        vocab.update(api_tokens)
        
        # Common Python patterns
        python_patterns = [
            'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict',
            'tuple', 'set', 'type', 'isinstance', 'hasattr', 'getattr',
            'setattr', 'open', 'read', 'write', 'close', 'split', 'join',
            'strip', 'lower', 'upper', 'format', 'replace', 'find'
        ]
        vocab.update(python_patterns)
        
        # Operators and punctuation
        operators = [
            '+', '-', '*', '/', '//', '%', '**', '=', '==', '!=', '<', '>',
            '<=', '>=', '+=', '-=', '*=', '/=', '(', ')', '[', ']', '{', '}',
            ',', '.', ':', ';', '@', '->', '=>'
        ]
        vocab.update(operators)
        
        return sorted(list(vocab))
    
    def tokenize_code(self, code: str) -> List[Token]:
        """Tokenize Python code into structured tokens"""
        tokens = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines):
            line_tokens = self._tokenize_line(line, line_num)
            tokens.extend(line_tokens)
        
        return tokens
    
    def _tokenize_line(self, line: str, line_num: int) -> List[Token]:
        """Tokenize a single line of code"""
        tokens = []
        line = line.strip()
        
        if not line:
            return tokens
        
        # Handle comments
        if line.startswith('#'):
            tokens.append(Token(
                text=line,
                token_type=TokenType.COMMENT,
                line=line_num,
                metadata={'comment_type': 'line'}
            ))
            return tokens
        
        # Simple regex-based tokenization
        # This is simplified - in production, you'd use a proper parser
        token_pattern = r'''
            (?P<STRING>r?["'](?:[^"'\\]|\\.)*["']) |  # Strings
            (?P<NUMBER>\b\d+\.?\d*\b) |               # Numbers
            (?P<IDENTIFIER>\b[a-zA-Z_][a-zA-Z0-9_]*\b) |  # Identifiers
            (?P<OPERATOR>[+\-*/%=<>!&|^~]+) |        # Operators
            (?P<PUNCTUATION>[()[\]{},.:;@]) |         # Punctuation
            (?P<WHITESPACE>\s+)                       # Whitespace
        '''
        
        position = 0
        for match in re.finditer(token_pattern, line, re.VERBOSE):
            if match.lastgroup == 'WHITESPACE':
                continue
            
            text = match.group()
            token_type = self._classify_token(text, line)
            
            tokens.append(Token(
                text=text,
                token_type=token_type,
                position=position,
                line=line_num,
                column=match.start(),
                metadata={}
            ))
            position += 1
        
        return tokens
    
    def _classify_token(self, text: str, context_line: str) -> TokenType:
        """Classify a token based on its text and context"""
        # Python keywords
        if text in self.python_keywords:
            return TokenType.KEYWORD
        
        # String literals
        if text.startswith(("'", '"')):
            # Check if it looks like an API endpoint
            if any(pattern in text.lower() for pattern in ['http', 'api', 'endpoint', '.com']):
                return TokenType.API_ENDPOINT
            return TokenType.STRING
        
        # Numbers
        if re.match(r'^\d+\.?\d*$', text):
            return TokenType.NUMBER
        
        # HTTP methods
        if text.upper() in self.http_methods:
            return TokenType.HTTP_METHOD
        
        # API methods
        if text.lower() in self.api_methods:
            return TokenType.API_METHOD
        
        # Operators
        if re.match(r'^[+\-*/%=<>!&|^~]+$', text):
            return TokenType.OPERATOR
        
        # Special cases based on context
        if 'def ' in context_line and text not in self.python_keywords:
            return TokenType.FUNCTION_DEF
        
        if 'class ' in context_line and text not in self.python_keywords:
            return TokenType.CLASS_DEF
        
        if 'import ' in context_line or 'from ' in context_line:
            return TokenType.IMPORT
        
        # Default to identifier
        return TokenType.IDENTIFIER
    
    def encode_tokens(self, tokens: List[Token], max_length: int = 512) -> np.ndarray:
        """Encode tokens to numerical representation"""
        encoded = np.zeros(max_length, dtype=np.int32)
        
        for i, token in enumerate(tokens[:max_length]):
            token_id = self.token_to_id.get(token.text, self.token_to_id['<UNK>'])
            encoded[i] = token_id
        
        return encoded
    
    def decode_tokens(self, encoded: np.ndarray) -> List[str]:
        """Decode numerical representation back to tokens"""
        tokens = []
        for token_id in encoded:
            if token_id == 0:  # Padding
                break
            token = self.id_to_token.get(int(token_id), '<UNK>')
            tokens.append(token)
        return tokens


class CodeASTAnalyzer:
    """Analyzes Python code using AST parsing"""
    
    def __init__(self):
        self.api_call_patterns = [
            'requests', 'httpx', 'urllib', 'aiohttp', 'fetch'
        ]
    
    def parse_code_structure(self, code: str) -> CodeStructure:
        """Parse code into structured representation using AST"""
        structure = CodeStructure()
        
        try:
            tree = ast.parse(code)
            self._analyze_ast(tree, structure)
        except SyntaxError as e:
            structure.syntax_errors.append(str(e))
            # Try to parse partially valid code
            self._parse_partial_code(code, structure)
        except Exception as e:
            structure.syntax_errors.append(f"AST parsing error: {str(e)}")
        
        # Calculate complexity metrics
        structure.complexity_metrics = self._calculate_complexity(structure)
        
        return structure
    
    def _analyze_ast(self, node: ast.AST, structure: CodeStructure, depth: int = 0):
        """Recursively analyze AST nodes"""
        for child in ast.iter_child_nodes(node):
            self._process_node(child, structure, depth)
            self._analyze_ast(child, structure, depth + 1)
    
    def _process_node(self, node: ast.AST, structure: CodeStructure, depth: int):
        """Process individual AST nodes"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                structure.imports.append({
                    'name': alias.name,
                    'alias': alias.asname,
                    'type': 'import',
                    'line': getattr(node, 'lineno', 0)
                })
        
        elif isinstance(node, ast.ImportFrom):
            structure.imports.append({
                'module': node.module,
                'names': [alias.name for alias in node.names],
                'type': 'from_import',
                'line': getattr(node, 'lineno', 0)
            })
        
        elif isinstance(node, ast.FunctionDef):
            func_info = {
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [self._ast_to_string(dec) for dec in node.decorator_list],
                'line': getattr(node, 'lineno', 0),
                'is_async': False,
                'docstring': ast.get_docstring(node),
                'complexity': self._calculate_function_complexity(node)
            }
            structure.functions.append(func_info)
        
        elif isinstance(node, ast.AsyncFunctionDef):
            func_info = {
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'decorators': [self._ast_to_string(dec) for dec in node.decorator_list],
                'line': getattr(node, 'lineno', 0),
                'is_async': True,
                'docstring': ast.get_docstring(node),
                'complexity': self._calculate_function_complexity(node)
            }
            structure.functions.append(func_info)
        
        elif isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'bases': [self._ast_to_string(base) for base in node.bases],
                'decorators': [self._ast_to_string(dec) for dec in node.decorator_list],
                'line': getattr(node, 'lineno', 0),
                'methods': [],
                'docstring': ast.get_docstring(node)
            }
            
            # Extract methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_info['methods'].append(item.name)
            
            structure.classes.append(class_info)
        
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    structure.variables.append({
                        'name': target.id,
                        'type': 'assignment',
                        'line': getattr(node, 'lineno', 0),
                        'value': self._ast_to_string(node.value)[:100]  # Truncate long values
                    })
        
        elif isinstance(node, ast.Call):
            call_info = self._analyze_function_call(node)
            if call_info and self._is_api_call(call_info):
                structure.api_calls.append(call_info)
        
        elif isinstance(node, (ast.If, ast.While, ast.For)):
            structure.control_flow.append({
                'type': type(node).__name__.lower(),
                'line': getattr(node, 'lineno', 0),
                'depth': depth
            })
        
        elif isinstance(node, (ast.Try, ast.ExceptHandler, ast.Raise)):
            structure.error_handling.append({
                'type': type(node).__name__.lower(),
                'line': getattr(node, 'lineno', 0)
            })
    
    def _analyze_function_call(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """Analyze a function call node"""
        try:
            func_name = self._ast_to_string(node.func)
            args = [self._ast_to_string(arg) for arg in node.args]
            kwargs = {kw.arg: self._ast_to_string(kw.value) for kw in node.keywords if kw.arg}
            
            return {
                'function': func_name,
                'args': args,
                'kwargs': kwargs,
                'line': getattr(node, 'lineno', 0)
            }
        except Exception:
            return None
    
    def _is_api_call(self, call_info: Dict[str, Any]) -> bool:
        """Check if a function call is likely an API call"""
        func_name = call_info['function'].lower()
        
        # Check for common API libraries and methods
        for pattern in self.api_call_patterns:
            if pattern in func_name:
                return True
        
        # Check for HTTP methods
        http_methods = ['get', 'post', 'put', 'delete', 'patch']
        if any(method in func_name for method in http_methods):
            return True
        
        # Check for URL-like arguments
        all_args = call_info['args'] + list(call_info['kwargs'].values())
        for arg in all_args:
            if isinstance(arg, str) and ('http' in arg.lower() or 'api' in arg.lower()):
                return True
        
        return False
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation"""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._ast_to_string(node.value)}.{node.attr}"
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Str):  # Python < 3.8
                return repr(node.s)
            elif isinstance(node, ast.Num):  # Python < 3.8
                return str(node.n)
            else:
                # Fallback to basic representation
                return f"<{type(node).__name__}>"
        except Exception:
            return "<unknown>"
    
    def _parse_partial_code(self, code: str, structure: CodeStructure):
        """Parse partially valid code using regex patterns"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Import statements
            import_match = re.match(r'^(?:from\s+(\S+)\s+)?import\s+(.+)', line)
            if import_match:
                module, names = import_match.groups()
                structure.imports.append({
                    'module': module,
                    'names': names.split(',') if names else [],
                    'type': 'from_import' if module else 'import',
                    'line': i + 1
                })
            
            # Function definitions
            func_match = re.match(r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', line)
            if func_match:
                name, args = func_match.groups()
                structure.functions.append({
                    'name': name,
                    'args': [arg.strip() for arg in args.split(',') if arg.strip()],
                    'line': i + 1,
                    'is_async': 'async' in line
                })
            
            # Class definitions
            class_match = re.match(r'^class\s+(\w+)', line)
            if class_match:
                structure.classes.append({
                    'name': class_match.group(1),
                    'line': i + 1
                })
            
            # Variable assignments
            var_match = re.match(r'^(\w+)\s*=', line)
            if var_match:
                structure.variables.append({
                    'name': var_match.group(1),
                    'line': i + 1,
                    'type': 'assignment'
                })
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_complexity(self, structure: CodeStructure) -> Dict[str, int]:
        """Calculate overall code complexity metrics"""
        return {
            'total_functions': len(structure.functions),
            'total_classes': len(structure.classes),
            'total_imports': len(structure.imports),
            'total_variables': len(structure.variables),
            'total_api_calls': len(structure.api_calls),
            'control_flow_nodes': len(structure.control_flow),
            'error_handling_nodes': len(structure.error_handling),
            'avg_function_complexity': np.mean([f.get('complexity', 1) for f in structure.functions]) if structure.functions else 0,
            'max_function_complexity': max([f.get('complexity', 1) for f in structure.functions]) if structure.functions else 0
        }


class APISchemaProcessor:
    """Processes API documentation and schemas into structured format"""
    
    def __init__(self):
        self.http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        self.common_parameters = ['limit', 'offset', 'page', 'size', 'sort', 'filter', 'search', 'id']
        self.auth_types = ['bearer', 'basic', 'api_key', 'oauth', 'oauth2']
    
    def process_api_documentation(self, api_docs: Any) -> APISchemaStructure:
        """Process API documentation into structured format"""
        schema = APISchemaStructure()
        
        if hasattr(api_docs, '__dict__'):
            docs_dict = api_docs.__dict__
        elif isinstance(api_docs, dict):
            docs_dict = api_docs
        else:
            # Convert to string and try to parse as JSON
            try:
                docs_dict = json.loads(str(api_docs))
            except (json.JSONDecodeError, TypeError):
                docs_dict = {'raw': str(api_docs)}
        
        # Extract basic information
        schema.title = docs_dict.get('title', '')
        schema.base_url = docs_dict.get('base_url', '')
        schema.version = docs_dict.get('version', '')
        
        # Process endpoints
        endpoints = docs_dict.get('endpoints', [])
        if isinstance(endpoints, list):
            for endpoint in endpoints:
                schema.endpoints.append(self._process_endpoint(endpoint))
        
        # Process authentication
        auth = docs_dict.get('authentication', {})
        if isinstance(auth, dict):
            schema.authentication = self._process_authentication(auth)
        
        # Process error codes
        errors = docs_dict.get('error_codes', {})
        if isinstance(errors, dict):
            schema.error_codes = errors
        
        # Process rate limits
        rate_limits = docs_dict.get('rate_limits', {})
        if isinstance(rate_limits, dict):
            schema.rate_limits = rate_limits
        
        # Process examples
        examples = docs_dict.get('examples', [])
        if isinstance(examples, list):
            schema.examples = examples
        
        return schema
    
    def _process_endpoint(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single API endpoint"""
        processed = {
            'path': endpoint.get('path', ''),
            'method': endpoint.get('method', 'GET').upper(),
            'description': endpoint.get('description', ''),
            'parameters': [],
            'request_schema': {},
            'response_schema': {},
            'auth_required': endpoint.get('auth_required', False),
            'rate_limited': endpoint.get('rate_limited', False)
        }
        
        # Process parameters
        params = endpoint.get('parameters', [])
        if isinstance(params, list):
            for param in params:
                processed['parameters'].append(self._process_parameter(param))
        
        # Process schemas
        if 'request_schema' in endpoint:
            processed['request_schema'] = self._process_schema(endpoint['request_schema'])
        
        if 'response_schema' in endpoint:
            processed['response_schema'] = self._process_schema(endpoint['response_schema'])
        
        return processed
    
    def _process_parameter(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """Process an API parameter"""
        return {
            'name': param.get('name', ''),
            'type': param.get('type', 'string'),
            'required': param.get('required', False),
            'description': param.get('description', ''),
            'location': param.get('in', 'query'),  # query, path, header, body
            'default': param.get('default'),
            'enum': param.get('enum', []),
            'example': param.get('example')
        }
    
    def _process_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON schema"""
        processed = {
            'type': schema.get('type', 'object'),
            'properties': {},
            'required': schema.get('required', []),
            'description': schema.get('description', '')
        }
        
        # Process properties
        properties = schema.get('properties', {})
        if isinstance(properties, dict):
            for prop_name, prop_schema in properties.items():
                processed['properties'][prop_name] = {
                    'type': prop_schema.get('type', 'string'),
                    'description': prop_schema.get('description', ''),
                    'example': prop_schema.get('example'),
                    'enum': prop_schema.get('enum', [])
                }
        
        return processed
    
    def _process_authentication(self, auth: Dict[str, Any]) -> Dict[str, Any]:
        """Process authentication information"""
        return {
            'type': auth.get('type', 'none'),
            'scheme': auth.get('scheme', ''),
            'bearer_format': auth.get('bearerFormat', ''),
            'header_name': auth.get('name', 'Authorization'),
            'api_key_location': auth.get('in', 'header'),
            'description': auth.get('description', '')
        }
    
    def encode_api_schema(self, schema: APISchemaStructure, max_length: int = 1024) -> np.ndarray:
        """Encode API schema to numerical representation"""
        features = np.zeros(max_length, dtype=np.float32)
        
        # Basic information
        features[0] = len(schema.endpoints) / 100.0  # Normalized endpoint count
        features[1] = 1.0 if schema.authentication.get('type') != 'none' else 0.0
        features[2] = 1.0 if schema.rate_limits else 0.0
        features[3] = len(schema.error_codes) / 50.0  # Normalized error code count
        
        # Endpoint features
        endpoint_features = self._encode_endpoints(schema.endpoints)
        end_idx = min(len(endpoint_features), max_length - 20)
        features[20:20 + end_idx] = endpoint_features[:end_idx]
        
        # Authentication features
        auth_features = self._encode_authentication(schema.authentication)
        features[4:4 + len(auth_features)] = auth_features
        
        return features
    
    def _encode_endpoints(self, endpoints: List[Dict[str, Any]]) -> np.ndarray:
        """Encode endpoint information"""
        features = []
        
        method_counts = Counter()
        param_counts = []
        auth_required_count = 0
        
        for endpoint in endpoints:
            method_counts[endpoint.get('method', 'GET')] += 1
            param_counts.append(len(endpoint.get('parameters', [])))
            if endpoint.get('auth_required', False):
                auth_required_count += 1
        
        # Method distribution
        for method in self.http_methods:
            features.append(method_counts.get(method, 0) / max(len(endpoints), 1))
        
        # Parameter statistics
        if param_counts:
            features.extend([
                np.mean(param_counts) / 10.0,  # Average parameters per endpoint
                np.max(param_counts) / 20.0,   # Max parameters per endpoint
                auth_required_count / len(endpoints)  # Fraction requiring auth
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _encode_authentication(self, auth: Dict[str, Any]) -> np.ndarray:
        """Encode authentication information"""
        features = np.zeros(10, dtype=np.float32)
        
        auth_type = auth.get('type', 'none').lower()
        
        # One-hot encode auth type
        if auth_type == 'bearer':
            features[0] = 1.0
        elif auth_type == 'basic':
            features[1] = 1.0
        elif auth_type == 'api_key':
            features[2] = 1.0
        elif auth_type in ['oauth', 'oauth2']:
            features[3] = 1.0
        # else: no auth (all zeros)
        
        # Additional auth features
        features[4] = 1.0 if auth.get('scheme') else 0.0
        features[5] = 1.0 if auth.get('header_name') else 0.0
        
        return features


class ObservationEncoder:
    """Main encoder that combines all observation components"""
    
    def __init__(self, 
                 max_code_tokens: int = 512,
                 max_api_tokens: int = 1024,
                 embedding_dim: int = 256):
        self.max_code_tokens = max_code_tokens
        self.max_api_tokens = max_api_tokens
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.code_tokenizer = CodeTokenizer()
        self.ast_analyzer = CodeASTAnalyzer()
        self.api_processor = APISchemaProcessor()
        
        # Create combined vocabulary
        self.vocab_size = self.code_tokenizer.vocab_size
    
    def encode_observation(self, api_docs: Any, current_code: str, 
                          additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Encode complete observation into ML-ready format"""
        
        # Process API documentation
        api_schema = self.api_processor.process_api_documentation(api_docs)
        api_encoding = self.api_processor.encode_api_schema(api_schema, self.max_api_tokens)
        
        # Process code
        code_tokens = self.code_tokenizer.tokenize_code(current_code)
        code_token_encoding = self.code_tokenizer.encode_tokens(code_tokens, self.max_code_tokens)
        
        # AST analysis
        code_structure = self.ast_analyzer.parse_code_structure(current_code)
        structure_encoding = self._encode_code_structure(code_structure)
        
        # Code features
        code_features = self._extract_code_features(current_code, code_structure)
        
        # API-code alignment features
        alignment_features = self._compute_alignment_features(api_schema, code_structure)
        
        # Combine all encodings
        observation = {
            'api_schema': api_encoding,
            'code_tokens': code_token_encoding,
            'code_structure': structure_encoding,
            'code_features': code_features,
            'alignment_features': alignment_features,
            'vocab_size': np.array([self.vocab_size], dtype=np.int32)
        }
        
        # Add additional context if provided
        if additional_context:
            context_features = self._encode_additional_context(additional_context)
            observation['context'] = context_features
        
        return observation
    
    def _encode_code_structure(self, structure: CodeStructure) -> np.ndarray:
        """Encode code structure into numerical features"""
        features = np.zeros(128, dtype=np.float32)
        
        # Basic counts (normalized)
        features[0] = min(len(structure.imports) / 20.0, 1.0)
        features[1] = min(len(structure.functions) / 10.0, 1.0)
        features[2] = min(len(structure.classes) / 5.0, 1.0)
        features[3] = min(len(structure.variables) / 50.0, 1.0)
        features[4] = min(len(structure.api_calls) / 10.0, 1.0)
        features[5] = min(len(structure.control_flow) / 20.0, 1.0)
        features[6] = min(len(structure.error_handling) / 10.0, 1.0)
        features[7] = 1.0 if structure.syntax_errors else 0.0
        
        # Complexity metrics
        metrics = structure.complexity_metrics
        features[8] = min(metrics.get('avg_function_complexity', 0) / 10.0, 1.0)
        features[9] = min(metrics.get('max_function_complexity', 0) / 20.0, 1.0)
        
        # Import analysis
        import_types = defaultdict(int)
        for imp in structure.imports:
            if imp.get('type') == 'import':
                import_types['direct'] += 1
            else:
                import_types['from'] += 1
        
        features[10] = min(import_types['direct'] / 10.0, 1.0)
        features[11] = min(import_types['from'] / 10.0, 1.0)
        
        # API-related features
        api_libraries = ['requests', 'httpx', 'urllib', 'aiohttp']
        for i, lib in enumerate(api_libraries):
            has_lib = any(lib in imp.get('name', '') or lib in str(imp.get('names', [])) 
                         for imp in structure.imports)
            features[12 + i] = 1.0 if has_lib else 0.0
        
        # Function analysis
        if structure.functions:
            async_funcs = sum(1 for f in structure.functions if f.get('is_async', False))
            features[16] = async_funcs / len(structure.functions)
            
            avg_args = np.mean([len(f.get('args', [])) for f in structure.functions])
            features[17] = min(avg_args / 10.0, 1.0)
        
        return features
    
    def _extract_code_features(self, code: str, structure: CodeStructure) -> np.ndarray:
        """Extract high-level code features"""
        features = np.zeros(64, dtype=np.float32)
        
        # Basic text statistics
        features[0] = min(len(code) / 10000.0, 1.0)  # Code length
        features[1] = min(len(code.split('\n')) / 200.0, 1.0)  # Line count
        features[2] = min(len(code.split()) / 2000.0, 1.0)  # Word count
        
        # Code quality indicators
        features[3] = 1.0 if 'def ' in code else 0.0  # Has functions
        features[4] = 1.0 if 'class ' in code else 0.0  # Has classes
        features[5] = 1.0 if 'import ' in code or 'from ' in code else 0.0  # Has imports
        features[6] = 1.0 if any(keyword in code for keyword in ['try:', 'except', 'raise']) else 0.0  # Has error handling
        features[7] = 1.0 if '"""' in code or "'''" in code or code.count('#') > 2 else 0.0  # Has documentation
        
        # API integration patterns
        api_patterns = [
            'requests.', 'httpx.', '.get(', '.post(', '.put(', '.delete(',
            'json()', 'status_code', 'headers', 'params', 'data', 'response'
        ]
        
        for i, pattern in enumerate(api_patterns):
            features[8 + i] = 1.0 if pattern in code else 0.0
        
        # Code complexity indicators
        features[20] = min(code.count('if ') / 20.0, 1.0)  # Conditional complexity
        features[21] = min(code.count('for ') / 10.0, 1.0)  # Loop complexity
        features[22] = min(code.count('while ') / 5.0, 1.0)  # While loop count
        features[23] = min(code.count('lambda ') / 5.0, 1.0)  # Lambda usage
        
        return features
    
    def _compute_alignment_features(self, api_schema: APISchemaStructure, 
                                  code_structure: CodeStructure) -> np.ndarray:
        """Compute features that measure alignment between API and code"""
        features = np.zeros(32, dtype=np.float32)
        
        # API coverage - how many endpoints are potentially used
        endpoint_methods = set()
        for endpoint in api_schema.endpoints:
            endpoint_methods.add(endpoint.get('method', 'GET').lower())
        
        code_methods = set()
        for api_call in code_structure.api_calls:
            func_name = api_call.get('function', '').lower()
            for method in ['get', 'post', 'put', 'delete', 'patch']:
                if method in func_name:
                    code_methods.add(method)
                    break
        
        # Method alignment
        if endpoint_methods:
            method_overlap = len(endpoint_methods.intersection(code_methods))
            features[0] = method_overlap / len(endpoint_methods)
        
        # Authentication alignment
        auth_required = any(ep.get('auth_required', False) for ep in api_schema.endpoints)
        has_auth_code = any('auth' in var.get('name', '').lower() or 
                           'token' in var.get('name', '').lower() or
                           'key' in var.get('name', '').lower()
                           for var in code_structure.variables)
        features[1] = 1.0 if auth_required == has_auth_code else 0.0
        
        # Error handling alignment
        has_error_responses = bool(api_schema.error_codes)
        has_error_handling = bool(code_structure.error_handling)
        features[2] = 1.0 if has_error_responses == has_error_handling else 0.0
        
        # Parameter usage alignment
        total_params = sum(len(ep.get('parameters', [])) for ep in api_schema.endpoints)
        if total_params > 0:
            # Estimate parameter usage in code
            param_usage = 0
            for api_call in code_structure.api_calls:
                param_usage += len(api_call.get('kwargs', {}))
            
            features[3] = min(param_usage / total_params, 1.0)
        
        return features
    
    def _encode_additional_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode additional context information"""
        features = np.zeros(16, dtype=np.float32)
        
        # Task-related context
        if 'task_difficulty' in context:
            difficulty_map = {'beginner': 0.25, 'intermediate': 0.5, 'advanced': 0.75, 'expert': 1.0}
            features[0] = difficulty_map.get(context['task_difficulty'], 0.5)
        
        if 'step_count' in context:
            features[1] = min(context['step_count'] / 100.0, 1.0)
        
        if 'completion_progress' in context:
            features[2] = context['completion_progress']
        
        # Execution context
        if 'last_execution_success' in context:
            features[3] = 1.0 if context['last_execution_success'] else 0.0
        
        if 'syntax_errors' in context:
            features[4] = 1.0 if context['syntax_errors'] else 0.0
        
        return features
    
    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get information about the observation space dimensions"""
        return {
            'api_schema_dim': self.max_api_tokens,
            'code_tokens_dim': self.max_code_tokens,
            'code_structure_dim': 128,
            'code_features_dim': 64,
            'alignment_features_dim': 32,
            'context_dim': 16,
            'vocab_size': self.vocab_size,
            'total_dim': self.max_api_tokens + self.max_code_tokens + 128 + 64 + 32 + 16
        }