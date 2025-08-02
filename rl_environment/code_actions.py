"""
Comprehensive action space for code modifications in API integration tasks.

This module defines a vocabulary of code modification actions and provides
encoding/decoding functionality for structured code generation.
"""

import re
import ast
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ActionType(Enum):
    """High-level action categories"""
    ADD_IMPORT = "add_import"
    ADD_FUNCTION_CALL = "add_function_call"
    ADD_ERROR_HANDLING = "add_error_handling"
    MODIFY_PARAMETERS = "modify_parameters"
    ADD_VARIABLE = "add_variable"
    ADD_FUNCTION_DEF = "add_function_def"
    ADD_CLASS_DEF = "add_class_def"
    ADD_COMMENT = "add_comment"
    MODIFY_STRING = "modify_string"
    ADD_CONDITIONAL = "add_conditional"
    ADD_LOOP = "add_loop"
    DELETE_LINE = "delete_line"
    MODIFY_LINE = "modify_line"
    ADD_RETURN = "add_return"
    ADD_PRINT = "add_print"


class ImportType(Enum):
    """Types of import statements"""
    STANDARD_LIBRARY = "standard"
    THIRD_PARTY = "third_party"
    LOCAL = "local"


class HTTPMethod(Enum):
    """HTTP methods for API calls"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class CodePosition:
    """Represents a position in code"""
    line: int
    column: int = 0
    indent_level: int = 0


@dataclass
class ActionContext:
    """Context information for applying actions"""
    current_code: str
    position: CodePosition
    function_scope: Optional[str] = None
    class_scope: Optional[str] = None
    available_variables: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


class CodeActionVocabulary:
    """Vocabulary of code modification actions and templates"""
    
    def __init__(self):
        self.import_templates = self._create_import_templates()
        self.function_call_templates = self._create_function_call_templates()
        self.error_handling_templates = self._create_error_handling_templates()
        self.variable_templates = self._create_variable_templates()
        self.control_flow_templates = self._create_control_flow_templates()
        
        # Build action vocabulary
        self.action_vocab = self._build_action_vocabulary()
        self.vocab_size = len(self.action_vocab)
        
        # Create reverse mapping
        self.action_to_id = {action: i for i, action in enumerate(self.action_vocab)}
        self.id_to_action = {i: action for action, i in self.action_to_id.items()}
    
    def _create_import_templates(self) -> Dict[str, List[str]]:
        """Create templates for import statements"""
        return {
            ImportType.STANDARD_LIBRARY.value: [
                "import json",
                "import os",
                "import sys",
                "import time",
                "import datetime",
                "import urllib.parse",
                "import urllib.request",
                "from typing import Dict, List, Any, Optional",
                "from dataclasses import dataclass",
                "import asyncio",
                "import logging"
            ],
            ImportType.THIRD_PARTY.value: [
                "import requests",
                "import httpx",
                "import aiohttp",
                "import pandas as pd",
                "import numpy as np",
                "from fastapi import FastAPI",
                "from pydantic import BaseModel",
                "import pytest"
            ],
            ImportType.LOCAL.value: [
                "from .models import *",
                "from .utils import *",
                "from .config import settings"
            ]
        }
    
    def _create_function_call_templates(self) -> Dict[str, List[str]]:
        """Create templates for function calls"""
        return {
            "http_requests": [
                "requests.get(url)",
                "requests.post(url, json=data)",
                "requests.put(url, json=data)",
                "requests.delete(url)",
                "requests.patch(url, json=data)",
                "requests.get(url, headers=headers)",
                "requests.post(url, data=data, headers=headers)",
                "httpx.get(url)",
                "httpx.post(url, json=data)"
            ],
            "data_processing": [
                "json.loads(response.text)",
                "json.dumps(data)",
                "response.json()",
                "response.status_code",
                "response.headers",
                "str(data)",
                "int(value)",
                "float(value)",
                "len(data)",
                "list(data)",
                "dict(data)"
            ],
            "string_operations": [
                "f'{variable}'",
                "'{}'.format(variable)",
                "str.join(', ', items)",
                "text.strip()",
                "text.lower()",
                "text.upper()",
                "text.split(',')",
                "text.replace(old, new)"
            ],
            "validation": [
                "isinstance(obj, type)",
                "hasattr(obj, 'attr')",
                "callable(obj)",
                "obj is not None",
                "obj in collection",
                "len(obj) > 0"
            ]
        }
    
    def _create_error_handling_templates(self) -> Dict[str, List[str]]:
        """Create templates for error handling"""
        return {
            "try_except": [
                "try:\n    {code}\nexcept Exception as e:\n    print(f'Error: {e}')",
                "try:\n    {code}\nexcept requests.RequestException as e:\n    print(f'Request failed: {e}')",
                "try:\n    {code}\nexcept ValueError as e:\n    print(f'Invalid value: {e}')",
                "try:\n    {code}\nexcept KeyError as e:\n    print(f'Missing key: {e}')",
                "try:\n    {code}\nexcept TypeError as e:\n    print(f'Type error: {e}')"
            ],
            "validation": [
                "if response.status_code == 200:",
                "if response.status_code != 200:\n    raise Exception(f'Request failed: {response.status_code}')",
                "if not data:\n    raise ValueError('No data provided')",
                "if 'error' in response_data:\n    raise Exception(response_data['error'])",
                "assert response.status_code == 200, f'Expected 200, got {response.status_code}'"
            ],
            "logging": [
                "print(f'Request URL: {url}')",
                "print(f'Response status: {response.status_code}')",
                "print(f'Response data: {response.json()}')",
                "logging.info(f'Processing {len(items)} items')",
                "logging.error(f'Failed to process: {error}')"
            ]
        }
    
    def _create_variable_templates(self) -> Dict[str, List[str]]:
        """Create templates for variable definitions"""
        return {
            "api_related": [
                "url = '{endpoint}'",
                "headers = {{'Authorization': 'Bearer {token}'}}",
                "data = {{'key': 'value'}}",
                "params = {{'limit': 10, 'offset': 0}}",
                "response = None",
                "api_key = os.getenv('API_KEY')",
                "base_url = 'https://api.example.com'",
                "timeout = 30"
            ],
            "data_structures": [
                "items = []",
                "result = {}",
                "data = None",
                "config = {}",
                "options = {{'verbose': True}}",
                "cache = {{}}"
            ],
            "counters": [
                "count = 0",
                "index = 0",
                "total = 0",
                "page = 1",
                "limit = 100"
            ]
        }
    
    def _create_control_flow_templates(self) -> Dict[str, List[str]]:
        """Create templates for control flow"""
        return {
            "conditionals": [
                "if condition:",
                "if response.status_code == 200:",
                "if data is not None:",
                "if len(items) > 0:",
                "if hasattr(obj, 'attr'):",
                "if 'key' in data:",
                "elif condition:",
                "else:"
            ],
            "loops": [
                "for item in items:",
                "for i, item in enumerate(items):",
                "for key, value in data.items():",
                "while condition:",
                "for i in range(len(items)):",
                "for page in range(1, max_pages + 1):"
            ],
            "functions": [
                "def function_name():",
                "def process_data(data):",
                "def make_request(url, method='GET'):",
                "def validate_response(response):",
                "def handle_error(error):",
                "return result",
                "return None",
                "yield item"
            ]
        }
    
    def _build_action_vocabulary(self) -> List[str]:
        """Build complete action vocabulary"""
        vocab = []
        
        # Add all template categories
        for category_dict in [
            self.import_templates,
            self.function_call_templates,
            self.error_handling_templates,
            self.variable_templates,
            self.control_flow_templates
        ]:
            for subcategory, templates in category_dict.items():
                vocab.extend(templates)
        
        # Add common individual tokens/patterns
        common_tokens = [
            "pass", "break", "continue", "raise", "assert",
            "True", "False", "None", "[]", "{}", "()",
            "and", "or", "not", "in", "is", "==", "!=",
            "+", "-", "*", "/", "%", "=", "+=", "-=",
            "print()", "len()", "str()", "int()", "float()"
        ]
        vocab.extend(common_tokens)
        
        return sorted(list(set(vocab)))  # Remove duplicates and sort


@dataclass
class CodeAction:
    """Represents a single code modification action"""
    action_type: ActionType
    template: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: Optional[CodePosition] = None
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation"""
        return {
            'action_type': self.action_type.value,
            'template': self.template,
            'parameters': self.parameters,
            'position': {
                'line': self.position.line if self.position else 0,
                'column': self.position.column if self.position else 0,
                'indent_level': self.position.indent_level if self.position else 0
            },
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeAction':
        """Create action from dictionary representation"""
        position = None
        if 'position' in data and data['position']:
            pos_data = data['position']
            position = CodePosition(
                line=pos_data.get('line', 0),
                column=pos_data.get('column', 0),
                indent_level=pos_data.get('indent_level', 0)
            )
        
        return cls(
            action_type=ActionType(data['action_type']),
            template=data['template'],
            parameters=data.get('parameters', {}),
            position=position,
            priority=data.get('priority', 1.0)
        )


class CodeActionEncoder:
    """Encodes code actions into numerical representations"""
    
    def __init__(self, vocabulary: CodeActionVocabulary):
        self.vocab = vocabulary
        self.max_action_length = 512  # Maximum tokens per action
        
        # Parameter encoding
        self.param_types = [
            'string', 'integer', 'float', 'boolean', 'list', 'dict', 'none'
        ]
        self.param_type_to_id = {t: i for i, t in enumerate(self.param_types)}
    
    def encode_action(self, action: CodeAction) -> Dict[str, np.ndarray]:
        """Encode a single action into numerical representation"""
        # Encode action type
        action_type_id = list(ActionType).index(action.action_type)
        
        # Encode template
        template_id = self.vocab.action_to_id.get(action.template, 0)
        
        # Encode position
        position_encoding = np.zeros(3, dtype=np.float32)
        if action.position:
            position_encoding[0] = min(action.position.line / 1000.0, 1.0)
            position_encoding[1] = min(action.position.column / 100.0, 1.0)
            position_encoding[2] = min(action.position.indent_level / 10.0, 1.0)
        
        # Encode parameters
        param_encoding = self._encode_parameters(action.parameters)
        
        return {
            'action_type': np.array([action_type_id], dtype=np.int32),
            'template_id': np.array([template_id], dtype=np.int32),
            'position': position_encoding,
            'parameters': param_encoding,
            'priority': np.array([action.priority], dtype=np.float32)
        }
    
    def _encode_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Encode action parameters"""
        # Simplified parameter encoding - convert to feature vector
        param_features = np.zeros(64, dtype=np.float32)
        
        for i, (key, value) in enumerate(parameters.items()):
            if i >= 8:  # Limit number of parameters
                break
            
            base_idx = i * 8
            
            # Encode parameter name (hash-based)
            key_hash = hash(key) % 100
            param_features[base_idx] = key_hash / 100.0
            
            # Encode parameter type and value
            if isinstance(value, str):
                param_features[base_idx + 1] = 0.1
                param_features[base_idx + 2] = len(value) / 100.0
            elif isinstance(value, (int, float)):
                param_features[base_idx + 1] = 0.2
                param_features[base_idx + 2] = min(float(value) / 1000.0, 1.0)
            elif isinstance(value, bool):
                param_features[base_idx + 1] = 0.3
                param_features[base_idx + 2] = 1.0 if value else 0.0
            elif isinstance(value, (list, tuple)):
                param_features[base_idx + 1] = 0.4
                param_features[base_idx + 2] = len(value) / 100.0
            elif isinstance(value, dict):
                param_features[base_idx + 1] = 0.5
                param_features[base_idx + 2] = len(value) / 100.0
            else:
                param_features[base_idx + 1] = 0.6
        
        return param_features
    
    def decode_action_encoding(self, encoding: Dict[str, np.ndarray]) -> CodeAction:
        """Decode numerical representation back to CodeAction"""
        # Decode action type
        action_type_id = int(encoding['action_type'][0])
        action_type = list(ActionType)[action_type_id]
        
        # Decode template
        template_id = int(encoding['template_id'][0])
        template = self.vocab.id_to_action.get(template_id, "pass")
        
        # Decode position
        position_encoding = encoding['position']
        position = CodePosition(
            line=int(position_encoding[0] * 1000),
            column=int(position_encoding[1] * 100),
            indent_level=int(position_encoding[2] * 10)
        )
        
        # Decode parameters (simplified)
        parameters = self._decode_parameters(encoding['parameters'])
        
        priority = float(encoding['priority'][0])
        
        return CodeAction(
            action_type=action_type,
            template=template,
            parameters=parameters,
            position=position,
            priority=priority
        )
    
    def _decode_parameters(self, param_encoding: np.ndarray) -> Dict[str, Any]:
        """Decode parameter encoding back to dictionary"""
        parameters = {}
        
        for i in range(8):  # Match encoding limit
            base_idx = i * 8
            
            if base_idx + 2 >= len(param_encoding):
                break
            
            # Simple parameter reconstruction
            param_type_encoding = param_encoding[base_idx + 1]
            param_value_encoding = param_encoding[base_idx + 2]
            
            if param_type_encoding > 0:  # Parameter exists
                param_name = f"param_{i}"
                
                if 0.05 <= param_type_encoding < 0.15:  # String
                    parameters[param_name] = f"value_{int(param_value_encoding * 100)}"
                elif 0.15 <= param_type_encoding < 0.25:  # Number
                    parameters[param_name] = int(param_value_encoding * 1000)
                elif 0.25 <= param_type_encoding < 0.35:  # Boolean
                    parameters[param_name] = param_value_encoding > 0.5
                # Add more type decodings as needed
        
        return parameters


class CodeActionApplicator:
    """Applies code actions to source code"""
    
    def __init__(self, vocabulary: CodeActionVocabulary):
        self.vocab = vocabulary
    
    def apply_action(self, code: str, action: CodeAction, context: Optional[ActionContext] = None) -> str:
        """Apply a single action to code"""
        if context is None:
            context = self._create_context(code, action.position or CodePosition(0))
        
        # Select appropriate application method based on action type
        applicator_map = {
            ActionType.ADD_IMPORT: self._apply_import_action,
            ActionType.ADD_FUNCTION_CALL: self._apply_function_call_action,
            ActionType.ADD_ERROR_HANDLING: self._apply_error_handling_action,
            ActionType.MODIFY_PARAMETERS: self._apply_parameter_modification,
            ActionType.ADD_VARIABLE: self._apply_variable_action,
            ActionType.ADD_FUNCTION_DEF: self._apply_function_def_action,
            ActionType.ADD_COMMENT: self._apply_comment_action,
            ActionType.ADD_CONDITIONAL: self._apply_conditional_action,
            ActionType.ADD_LOOP: self._apply_loop_action,
            ActionType.DELETE_LINE: self._apply_delete_line_action,
            ActionType.MODIFY_LINE: self._apply_modify_line_action,
            ActionType.ADD_RETURN: self._apply_return_action,
            ActionType.ADD_PRINT: self._apply_print_action,
        }
        
        applicator = applicator_map.get(action.action_type, self._apply_generic_action)
        return applicator(code, action, context)
    
    def _create_context(self, code: str, position: CodePosition) -> ActionContext:
        """Create action context from code"""
        lines = code.split('\n')
        
        # Extract available variables (simplified)
        variables = []
        imports = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            elif '=' in line and not line.startswith('#'):
                # Simple variable extraction
                var_match = re.match(r'^(\w+)\s*=', line)
                if var_match:
                    variables.append(var_match.group(1))
        
        return ActionContext(
            current_code=code,
            position=position,
            available_variables=variables,
            imports=imports
        )
    
    def _apply_import_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply import statement addition"""
        lines = code.split('\n')
        
        # Find appropriate position for import (after existing imports or at top)
        import_position = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_position = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # Check if import already exists
        import_stmt = action.template.format(**action.parameters)
        if import_stmt not in context.imports:
            lines.insert(import_position, import_stmt)
        
        return '\n'.join(lines)
    
    def _apply_function_call_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply function call addition"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        # Format function call with parameters
        func_call = action.template.format(**action.parameters)
        
        # Add appropriate indentation
        indent = '    ' * position.indent_level
        formatted_call = f"{indent}{func_call}"
        
        # Insert at specified position
        insert_line = min(position.line, len(lines))
        lines.insert(insert_line, formatted_call)
        
        return '\n'.join(lines)
    
    def _apply_error_handling_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply error handling code"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        # Format error handling template
        error_handling = action.template.format(**action.parameters)
        
        # Handle multi-line templates
        if '\n' in error_handling:
            error_lines = error_handling.split('\n')
            indent = '    ' * position.indent_level
            formatted_lines = [f"{indent}{line}" if line.strip() else line for line in error_lines]
            
            # Insert all lines
            insert_pos = min(position.line, len(lines))
            for i, line in enumerate(formatted_lines):
                lines.insert(insert_pos + i, line)
        else:
            indent = '    ' * position.indent_level
            lines.insert(min(position.line, len(lines)), f"{indent}{error_handling}")
        
        return '\n'.join(lines)
    
    def _apply_parameter_modification(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Modify function parameters"""
        lines = code.split('\n')
        
        # Find function calls to modify (simplified)
        for i, line in enumerate(lines):
            if 'parameters' in action.parameters:
                old_params = action.parameters.get('old_parameters', '')
                new_params = action.parameters.get('new_parameters', '')
                if old_params in line:
                    lines[i] = line.replace(old_params, new_params)
                    break
        
        return '\n'.join(lines)
    
    def _apply_variable_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply variable definition"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        var_definition = action.template.format(**action.parameters)
        indent = '    ' * position.indent_level
        
        lines.insert(min(position.line, len(lines)), f"{indent}{var_definition}")
        return '\n'.join(lines)
    
    def _apply_generic_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply generic action (fallback)"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        formatted_code = action.template.format(**action.parameters)
        indent = '    ' * position.indent_level
        
        if '\n' in formatted_code:
            code_lines = formatted_code.split('\n')
            formatted_lines = [f"{indent}{line}" if line.strip() else line for line in code_lines]
            
            insert_pos = min(position.line, len(lines))
            for i, line in enumerate(formatted_lines):
                lines.insert(insert_pos + i, line)
        else:
            lines.insert(min(position.line, len(lines)), f"{indent}{formatted_code}")
        
        return '\n'.join(lines)
    
    # Additional specific applicators for other action types
    def _apply_function_def_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply function definition"""
        return self._apply_generic_action(code, action, context)
    
    def _apply_comment_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply comment addition"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        comment = f"# {action.parameters.get('text', 'Comment')}"
        indent = '    ' * position.indent_level
        
        lines.insert(min(position.line, len(lines)), f"{indent}{comment}")
        return '\n'.join(lines)
    
    def _apply_conditional_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply conditional statement"""
        return self._apply_generic_action(code, action, context)
    
    def _apply_loop_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply loop statement"""
        return self._apply_generic_action(code, action, context)
    
    def _apply_delete_line_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Delete a line of code"""
        lines = code.split('\n')
        position = action.position or CodePosition(0)
        
        if 0 <= position.line < len(lines):
            del lines[position.line]
        
        return '\n'.join(lines)
    
    def _apply_modify_line_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Modify an existing line"""
        lines = code.split('\n')
        position = action.position or CodePosition(0)
        
        if 0 <= position.line < len(lines):
            new_content = action.template.format(**action.parameters)
            indent = '    ' * position.indent_level
            lines[position.line] = f"{indent}{new_content}"
        
        return '\n'.join(lines)
    
    def _apply_return_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply return statement"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        return_stmt = f"return {action.parameters.get('value', 'None')}"
        indent = '    ' * position.indent_level
        
        lines.insert(min(position.line, len(lines)), f"{indent}{return_stmt}")
        return '\n'.join(lines)
    
    def _apply_print_action(self, code: str, action: CodeAction, context: ActionContext) -> str:
        """Apply print statement"""
        lines = code.split('\n')
        position = action.position or CodePosition(len(lines))
        
        print_stmt = f"print({action.parameters.get('value', '\"Debug\"')})"
        indent = '    ' * position.indent_level
        
        lines.insert(min(position.line, len(lines)), f"{indent}{print_stmt}")
        return '\n'.join(lines)


class ActionValidator:
    """Validates code actions for safety and correctness"""
    
    def __init__(self):
        self.forbidden_patterns = [
            r'__import__\(',
            r'eval\(',
            r'exec\(',
            r'open\(',
            r'file\(',
            r'input\(',
            r'raw_input\(',
            r'compile\(',
            r'globals\(',
            r'locals\(',
            r'vars\(',
            r'dir\(',
            r'getattr\(',
            r'setattr\(',
            r'delattr\(',
            r'hasattr\(',
            r'os\.',
            r'sys\.',
            r'subprocess\.',
        ]
        
        self.max_code_length = 10000
        self.max_line_length = 200
    
    def validate_action(self, action: CodeAction, context: ActionContext) -> Tuple[bool, List[str]]:
        """Validate if action is safe and appropriate"""
        errors = []
        
        # Check for forbidden patterns
        template = action.template
        for param_value in action.parameters.values():
            if isinstance(param_value, str):
                template = template.replace(f"{{{param_value}}}", str(param_value))
        
        for pattern in self.forbidden_patterns:
            if re.search(pattern, template, re.IGNORECASE):
                errors.append(f"Forbidden pattern detected: {pattern}")
        
        # Check code length constraints
        if len(template) > self.max_line_length:
            errors.append(f"Template too long: {len(template)} > {self.max_line_length}")
        
        # Check position validity
        if action.position:
            code_lines = context.current_code.split('\n')
            if action.position.line < 0 or action.position.line > len(code_lines):
                errors.append(f"Invalid line position: {action.position.line}")
        
        # Validate syntax (for complete statements)
        if action.action_type in [ActionType.ADD_FUNCTION_DEF, ActionType.ADD_CLASS_DEF]:
            try:
                ast.parse(template)
            except SyntaxError as e:
                errors.append(f"Syntax error in template: {e}")
        
        return len(errors) == 0, errors
    
    def validate_result_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate the resulting code after action application"""
        errors = []
        
        # Check total length
        if len(code) > self.max_code_length:
            errors.append(f"Code too long: {len(code)} > {self.max_code_length}")
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error in resulting code: {e}")
        
        # Check for security issues
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"Forbidden pattern in result: {pattern}")
        
        return len(errors) == 0, errors