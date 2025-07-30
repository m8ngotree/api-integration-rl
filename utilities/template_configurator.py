import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..data_generation.endpoint_generator import EndpointSpec
from .code_template_generator import MissingComponent, DifficultyLevel, CodeTemplate
from .schema_analyzer import APISchemaAnalyzer, SchemaAnalysis
from .integration_patterns import PatternBasedTemplateGenerator


class TemplateType(Enum):
    BASIC_CLIENT = "basic_client"
    CRUD_CLIENT = "crud_client"
    ASYNC_CLIENT = "async_client"
    BATCH_CLIENT = "batch_client"
    STREAMING_CLIENT = "streaming_client"


class LearningObjective(Enum):
    HTTP_BASICS = "http_basics"
    AUTHENTICATION = "authentication"
    ERROR_HANDLING = "error_handling"
    ASYNC_PROGRAMMING = "async_programming"
    DATA_VALIDATION = "data_validation"
    API_DESIGN = "api_design"
    TESTING = "testing"
    LOGGING = "logging"


@dataclass
class TemplateConfiguration:
    """Configuration for generating code templates"""
    template_type: TemplateType
    difficulty: DifficultyLevel
    missing_components: List[MissingComponent]
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    include_tests: bool = False
    include_documentation: bool = True
    max_gaps: int = 10
    min_gaps: int = 2
    randomize_gaps: bool = True
    add_hints: bool = True
    include_examples: bool = True
    custom_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DifficultyProfile:
    """Profile defining what components are appropriate for each difficulty level"""
    beginner_components: Set[MissingComponent]
    intermediate_components: Set[MissingComponent]
    advanced_components: Set[MissingComponent]
    complexity_weights: Dict[str, float]


class TemplateConfigurator:
    """Configures and customizes code template generation based on learning goals"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
        
        self.generator = PatternBasedTemplateGenerator(seed)
        self.analyzer = APISchemaAnalyzer()
        
        # Define difficulty profiles
        self.difficulty_profiles = self._create_difficulty_profiles()
        
        # Component dependencies and prerequisites
        self.component_dependencies = self._define_component_dependencies()
        
        # Learning objective mappings
        self.objective_mappings = self._define_objective_mappings()
    
    def create_configuration(
        self,
        difficulty: DifficultyLevel,
        learning_objectives: List[LearningObjective],
        template_type: TemplateType = TemplateType.BASIC_CLIENT,
        **kwargs
    ) -> TemplateConfiguration:
        """Create a template configuration based on learning objectives"""
        
        # Determine missing components based on objectives and difficulty
        missing_components = self._determine_missing_components(
            difficulty, learning_objectives, template_type
        )
        
        # Apply difficulty constraints
        missing_components = self._apply_difficulty_constraints(
            missing_components, difficulty
        )
        
        # Randomize if requested
        if kwargs.get('randomize_gaps', True):
            missing_components = self._randomize_components(missing_components, kwargs.get('max_gaps', 10))
        
        return TemplateConfiguration(
            template_type=template_type,
            difficulty=difficulty,
            missing_components=missing_components,
            learning_objectives=learning_objectives,
            **kwargs
        )
    
    def generate_progressive_templates(
        self,
        endpoints: List[EndpointSpec],
        learning_objectives: List[LearningObjective],
        num_levels: int = 3
    ) -> List[CodeTemplate]:
        """Generate a series of progressively difficult templates"""
        
        templates = []
        difficulties = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]
        
        for i in range(min(num_levels, len(difficulties))):
            difficulty = difficulties[i]
            
            # Create configuration for this level
            config = self.create_configuration(
                difficulty=difficulty,
                learning_objectives=learning_objectives,
                max_gaps=3 + i * 2  # Gradually increase complexity
            )
            
            # Generate template
            template = self.generate_template(endpoints, config)
            template.metadata['progression_level'] = i + 1
            template.metadata['total_levels'] = num_levels
            
            templates.append(template)
        
        return templates
    
    def generate_template(
        self,
        endpoints: List[EndpointSpec],
        config: TemplateConfiguration
    ) -> CodeTemplate:
        """Generate a template based on configuration"""
        
        # Generate base template
        template = self.generator.generate_template(
            endpoints=endpoints,
            missing_components=config.missing_components,
            difficulty=config.difficulty
        )
        
        # Apply customizations
        template = self._apply_customizations(template, config)
        
        # Add learning metadata
        template.metadata.update({
            'template_type': config.template_type.value,
            'learning_objectives': [obj.value for obj in config.learning_objectives],
            'configuration': {
                'include_tests': config.include_tests,
                'include_documentation': config.include_documentation,
                'max_gaps': config.max_gaps,
                'add_hints': config.add_hints
            }
        })
        
        return template
    
    def suggest_next_challenges(
        self,
        current_template: CodeTemplate,
        completed_components: List[MissingComponent]
    ) -> List[Dict[str, Any]]:
        """Suggest next challenges based on completed work"""
        
        current_difficulty = DifficultyLevel(current_template.metadata.get('difficulty', 'intermediate'))
        remaining_components = [
            gap.component for gap in current_template.gaps 
            if gap.component not in completed_components
        ]
        
        suggestions = []
        
        # Suggest completing remaining components
        if remaining_components:
            suggestions.append({
                'type': 'complete_current',
                'description': f'Complete remaining {len(remaining_components)} components in current template',
                'components': remaining_components,
                'estimated_time': len(remaining_components) * 10
            })
        
        # Suggest advancing difficulty
        if current_difficulty != DifficultyLevel.ADVANCED:
            next_difficulty = self._get_next_difficulty(current_difficulty)
            suggestions.append({
                'type': 'advance_difficulty',
                'description': f'Try {next_difficulty.value} level template',
                'difficulty': next_difficulty,
                'estimated_time': 45
            })
        
        # Suggest new learning objectives
        current_objectives = set(current_template.metadata.get('learning_objectives', []))
        available_objectives = set(obj.value for obj in LearningObjective) - current_objectives
        
        if available_objectives:
            new_objective = random.choice(list(available_objectives))
            suggestions.append({
                'type': 'new_objective',
                'description': f'Explore {new_objective.replace("_", " ")} concepts',
                'objective': new_objective,
                'estimated_time': 30
            })
        
        return suggestions
    
    def create_custom_challenge(
        self,
        endpoints: List[EndpointSpec],
        focus_components: List[MissingComponent],
        challenge_name: str,
        description: str = ""
    ) -> CodeTemplate:
        """Create a custom challenge focusing on specific components"""
        
        # Determine appropriate difficulty based on components
        difficulty = self._infer_difficulty_from_components(focus_components)
        
        # Create configuration
        config = TemplateConfiguration(
            template_type=TemplateType.BASIC_CLIENT,
            difficulty=difficulty,
            missing_components=focus_components,
            learning_objectives=[],
            add_hints=True,
            include_examples=True
        )
        
        # Generate template
        template = self.generate_template(endpoints, config)
        
        # Customize for challenge
        template.metadata.update({
            'challenge_name': challenge_name,
            'challenge_description': description,
            'focus_components': [comp.value for comp in focus_components],
            'is_custom_challenge': True
        })
        
        return template
    
    def analyze_template_difficulty(self, template: CodeTemplate) -> Dict[str, Any]:
        """Analyze the difficulty characteristics of a template"""
        
        gap_complexity = {}
        for gap in template.gaps:
            component = gap.component.value
            gap_complexity[component] = self._get_component_complexity(gap.component)
        
        total_complexity = sum(gap_complexity.values())
        avg_complexity = total_complexity / len(template.gaps) if template.gaps else 0
        
        return {
            'total_gaps': len(template.gaps),
            'complexity_score': total_complexity,
            'average_complexity': avg_complexity,
            'gap_breakdown': gap_complexity,
            'estimated_time': template.metadata.get('estimated_completion_time', 30),
            'difficulty_rating': self._calculate_difficulty_rating(total_complexity, len(template.gaps))
        }
    
    def _create_difficulty_profiles(self) -> Dict[DifficultyLevel, DifficultyProfile]:
        """Create difficulty profiles for different levels"""
        
        beginner_components = {
            MissingComponent.IMPORTS,
            MissingComponent.HTTP_METHOD,
            MissingComponent.URL_CONSTRUCTION
        }
        
        intermediate_components = beginner_components | {
            MissingComponent.AUTHENTICATION,
            MissingComponent.ERROR_HANDLING,
            MissingComponent.REQUEST_BODY,
            MissingComponent.RESPONSE_PARSING,
            MissingComponent.PARAMETERS
        }
        
        advanced_components = intermediate_components | {
            MissingComponent.HEADERS,
            MissingComponent.VALIDATION,
            MissingComponent.LOGGING,
            MissingComponent.RETRY_LOGIC
        }
        
        return {
            DifficultyLevel.BEGINNER: DifficultyProfile(
                beginner_components=beginner_components,
                intermediate_components=set(),
                advanced_components=set(),
                complexity_weights={'simple': 1.0, 'moderate': 0.3, 'complex': 0.0}
            ),
            DifficultyLevel.INTERMEDIATE: DifficultyProfile(
                beginner_components=beginner_components,
                intermediate_components=intermediate_components,
                advanced_components=set(),
                complexity_weights={'simple': 0.5, 'moderate': 1.0, 'complex': 0.3}
            ),
            DifficultyLevel.ADVANCED: DifficultyProfile(
                beginner_components=beginner_components,
                intermediate_components=intermediate_components,
                advanced_components=advanced_components,
                complexity_weights={'simple': 0.2, 'moderate': 0.7, 'complex': 1.0}
            )
        }
    
    def _define_component_dependencies(self) -> Dict[MissingComponent, List[MissingComponent]]:
        """Define prerequisites between components"""
        return {
            MissingComponent.AUTHENTICATION: [MissingComponent.HEADERS],
            MissingComponent.REQUEST_BODY: [MissingComponent.HEADERS],
            MissingComponent.RETRY_LOGIC: [MissingComponent.ERROR_HANDLING],
            MissingComponent.VALIDATION: [MissingComponent.IMPORTS],
            MissingComponent.LOGGING: [MissingComponent.IMPORTS]
        }
    
    def _define_objective_mappings(self) -> Dict[LearningObjective, List[MissingComponent]]:
        """Map learning objectives to relevant components"""
        return {
            LearningObjective.HTTP_BASICS: [
                MissingComponent.IMPORTS,
                MissingComponent.HTTP_METHOD,
                MissingComponent.URL_CONSTRUCTION
            ],
            LearningObjective.AUTHENTICATION: [
                MissingComponent.AUTHENTICATION,
                MissingComponent.HEADERS
            ],
            LearningObjective.ERROR_HANDLING: [
                MissingComponent.ERROR_HANDLING,
                MissingComponent.RETRY_LOGIC
            ],
            LearningObjective.DATA_VALIDATION: [
                MissingComponent.VALIDATION,
                MissingComponent.REQUEST_BODY,
                MissingComponent.RESPONSE_PARSING
            ],
            LearningObjective.LOGGING: [
                MissingComponent.LOGGING
            ]
        }
    
    def _determine_missing_components(
        self,
        difficulty: DifficultyLevel,
        objectives: List[LearningObjective],
        template_type: TemplateType
    ) -> List[MissingComponent]:
        """Determine which components should be missing"""
        
        components = set()
        
        # Add components based on learning objectives
        for objective in objectives:
            if objective in self.objective_mappings:
                components.update(self.objective_mappings[objective])
        
        # Add default components for template type
        default_components = self._get_default_components_for_type(template_type)
        components.update(default_components)
        
        # Ensure we have enough components for the difficulty level
        min_components = {
            DifficultyLevel.BEGINNER: 2,
            DifficultyLevel.INTERMEDIATE: 4,
            DifficultyLevel.ADVANCED: 6
        }
        
        profile = self.difficulty_profiles[difficulty]
        available_components = profile.advanced_components
        
        while len(components) < min_components[difficulty]:
            remaining = available_components - components
            if remaining:
                components.add(random.choice(list(remaining)))
            else:
                break
        
        return list(components)
    
    def _apply_difficulty_constraints(
        self,
        components: List[MissingComponent],
        difficulty: DifficultyLevel
    ) -> List[MissingComponent]:
        """Filter components based on difficulty level"""
        
        profile = self.difficulty_profiles[difficulty]
        allowed_components = profile.advanced_components
        
        return [comp for comp in components if comp in allowed_components]
    
    def _randomize_components(
        self,
        components: List[MissingComponent],
        max_gaps: int
    ) -> List[MissingComponent]:
        """Randomly select and shuffle components"""
        
        if len(components) > max_gaps:
            components = random.sample(components, max_gaps)
        
        random.shuffle(components)
        return components
    
    def _apply_customizations(
        self,
        template: CodeTemplate,
        config: TemplateConfiguration
    ) -> CodeTemplate:
        """Apply configuration customizations to template"""
        
        # Add/remove hints based on configuration
        if not config.add_hints:
            for gap in template.gaps:
                gap.hint = None
        
        # Add tests if requested
        if config.include_tests:
            template = self._add_test_code(template)
        
        # Enhance documentation if requested
        if config.include_documentation:
            template = self._enhance_documentation(template)
        
        return template
    
    def _add_test_code(self, template: CodeTemplate) -> CodeTemplate:
        """Add test code to the template"""
        
        test_code = """

# Unit Tests
import unittest
from unittest.mock import patch, Mock

class TestAPIClient(unittest.TestCase):
    def setUp(self):
        # TODO: Set up test client instance
        pass
    
    def test_successful_request(self):
        # TODO: Test successful API request
        pass
    
    def test_error_handling(self):
        # TODO: Test error handling
        pass

if __name__ == '__main__':
    unittest.main()
"""
        
        template.code += test_code
        template.metadata['includes_tests'] = True
        
        return template
    
    def _enhance_documentation(self, template: CodeTemplate) -> CodeTemplate:
        """Enhance template with better documentation"""
        
        # Add comprehensive docstrings and comments
        enhanced_code = template.code.replace(
            '"""', '"""\n    \n    Args:\n        # TODO: Document parameters\n    \n    Returns:\n        # TODO: Document return value\n    \n    Raises:\n        # TODO: Document exceptions\n    """'
        )
        
        template.code = enhanced_code
        template.metadata['enhanced_documentation'] = True
        
        return template
    
    def _get_default_components_for_type(self, template_type: TemplateType) -> Set[MissingComponent]:
        """Get default missing components for template type"""
        
        defaults = {
            TemplateType.BASIC_CLIENT: {
                MissingComponent.IMPORTS,
                MissingComponent.HTTP_METHOD,
                MissingComponent.ERROR_HANDLING
            },
            TemplateType.CRUD_CLIENT: {
                MissingComponent.REQUEST_BODY,
                MissingComponent.RESPONSE_PARSING,
                MissingComponent.URL_CONSTRUCTION
            },
            TemplateType.ASYNC_CLIENT: {
                MissingComponent.IMPORTS,
                MissingComponent.HTTP_METHOD,
                MissingComponent.ERROR_HANDLING
            },
            TemplateType.BATCH_CLIENT: {
                MissingComponent.REQUEST_BODY,
                MissingComponent.VALIDATION,
                MissingComponent.ERROR_HANDLING
            }
        }
        
        return defaults.get(template_type, set())
    
    def _get_next_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get the next difficulty level"""
        progression = {
            DifficultyLevel.BEGINNER: DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.INTERMEDIATE: DifficultyLevel.ADVANCED,
            DifficultyLevel.ADVANCED: DifficultyLevel.ADVANCED
        }
        return progression[current]
    
    def _infer_difficulty_from_components(self, components: List[MissingComponent]) -> DifficultyLevel:
        """Infer appropriate difficulty level from components"""
        
        advanced_components = {
            MissingComponent.RETRY_LOGIC,
            MissingComponent.LOGGING,
            MissingComponent.VALIDATION
        }
        
        intermediate_components = {
            MissingComponent.AUTHENTICATION,
            MissingComponent.ERROR_HANDLING,
            MissingComponent.REQUEST_BODY,
            MissingComponent.RESPONSE_PARSING
        }
        
        component_set = set(components)
        
        if component_set & advanced_components:
            return DifficultyLevel.ADVANCED
        elif component_set & intermediate_components:
            return DifficultyLevel.INTERMEDIATE
        else:
            return DifficultyLevel.BEGINNER
    
    def _get_component_complexity(self, component: MissingComponent) -> float:
        """Get complexity score for a component"""
        complexity_scores = {
            MissingComponent.IMPORTS: 1.0,
            MissingComponent.HTTP_METHOD: 1.5,
            MissingComponent.URL_CONSTRUCTION: 2.0,
            MissingComponent.AUTHENTICATION: 2.5,
            MissingComponent.ERROR_HANDLING: 3.0,
            MissingComponent.REQUEST_BODY: 2.5,
            MissingComponent.RESPONSE_PARSING: 2.0,
            MissingComponent.PARAMETERS: 2.0,
            MissingComponent.HEADERS: 1.5,
            MissingComponent.VALIDATION: 3.5,
            MissingComponent.LOGGING: 2.5,
            MissingComponent.RETRY_LOGIC: 4.0
        }
        return complexity_scores.get(component, 2.0)
    
    def _calculate_difficulty_rating(self, total_complexity: float, num_gaps: int) -> str:
        """Calculate overall difficulty rating"""
        
        avg_complexity = total_complexity / num_gaps if num_gaps > 0 else 0
        
        if avg_complexity < 2.0:
            return "Easy"
        elif avg_complexity < 3.0:
            return "Medium"
        elif avg_complexity < 4.0:
            return "Hard"
        else:
            return "Expert"