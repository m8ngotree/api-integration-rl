#!/usr/bin/env python3

import json
from typing import List

from .code_template_generator import MissingComponent, DifficultyLevel
from .schema_analyzer import APISchemaAnalyzer, LearningObjective
from .integration_patterns import PatternBasedTemplateGenerator
from .template_configurator import TemplateConfigurator, TemplateType
from ..data_generation.api_schema_generator import APISchemaGenerator
from ..data_generation.endpoint_generator import EndpointGenerator


def basic_template_generation_example():
    """Demonstrate basic template generation"""
    print("🔧 Basic Code Template Generation Example")
    print("=" * 60)
    
    # Create endpoint generator and get some endpoints
    endpoint_gen = EndpointGenerator()
    endpoints = endpoint_gen.get_user_endpoints()[:3]  # Get first 3 user endpoints
    
    print(f"\n📋 Using {len(endpoints)} endpoints:")
    for ep in endpoints:
        print(f"   • {ep.method.value} {ep.path} - {ep.summary}")
    
    # Create template generator
    generator = PatternBasedTemplateGenerator(seed=42)
    
    # Define what components to leave missing
    missing_components = [
        MissingComponent.IMPORTS,
        MissingComponent.AUTHENTICATION,
        MissingComponent.ERROR_HANDLING,
        MissingComponent.RESPONSE_PARSING
    ]
    
    print(f"\n🔨 Generating template with {len(missing_components)} missing components:")
    for comp in missing_components:
        print(f"   • {comp.value}")
    
    # Generate template
    template = generator.generate_template(
        endpoints=endpoints,
        missing_components=missing_components,
        difficulty=DifficultyLevel.INTERMEDIATE
    )
    
    print(f"\n✅ Generated template with {len(template.gaps)} gaps")
    print(f"📝 Template description: {template.description}")
    print(f"⏱️  Estimated completion time: {template.metadata.get('estimated_completion_time', 'N/A')} minutes")
    
    # Show template code (first 30 lines)
    print("\n📄 Template code preview (first 30 lines):")
    print("-" * 50)
    lines = template.code.split('\n')
    for i, line in enumerate(lines[:30], 1):
        print(f"{i:2d}: {line}")
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    
    # Show gaps details
    print(f"\n🔍 Gap details:")
    for i, gap in enumerate(template.gaps, 1):
        print(f"   {i}. Line {gap.line_number}: {gap.component.value}")
        print(f"      Hint: {gap.hint}")
        print()
    
    return template


def schema_analysis_example():
    """Demonstrate API schema analysis"""
    print("\n🔬 API Schema Analysis Example")
    print("=" * 60)
    
    # Generate a complex API schema
    schema_gen = APISchemaGenerator(seed=123)
    openapi_spec = schema_gen.generate_openapi_spec(
        title="E-commerce API",
        endpoint_count=8
    )
    
    # Extract endpoints from the schema
    endpoint_gen = EndpointGenerator()
    all_endpoints = endpoint_gen.get_all_endpoints()
    
    print(f"\n📊 Analyzing API with {len(all_endpoints)} endpoints")
    
    # Analyze the schema
    analyzer = APISchemaAnalyzer()
    analysis = analyzer.analyze_schema(all_endpoints)
    
    print(f"\n📈 Analysis Results:")
    print(f"   • Complexity: {analysis.complexity.value}")
    print(f"   • Resources: {', '.join(analysis.resources)}")
    print(f"   • Auth type: {analysis.auth_requirements.value}")
    print(f"   • Recommended difficulty: {analysis.recommended_difficulty.value}")
    print(f"   • Integration challenges: {len(analysis.integration_challenges)}")
    
    print(f"\n🎯 Detected patterns:")
    for pattern in analysis.patterns:
        print(f"   • {pattern.pattern_type}: {pattern.description}")
        print(f"     Suggested components: {[c.value for c in pattern.suggested_components]}")
    
    print(f"\n💡 Suggested missing components:")
    for comp in analysis.suggested_missing_components:
        print(f"   • {comp.value}")
    
    # Get template variations
    variations = analyzer.suggest_template_variations(analysis)
    print(f"\n🎨 Template variations ({len(variations)}):")
    for var in variations:
        print(f"   • {var['name']} ({var['difficulty'].value})")
        print(f"     Components: {len(var['missing_components'])}, Time: {var['estimated_time']}min")
        print(f"     Description: {var['description']}")
        print()
    
    return analysis


def progressive_template_example():
    """Demonstrate progressive template generation"""
    print("\n📈 Progressive Template Generation Example")
    print("=" * 60)
    
    # Setup
    endpoint_gen = EndpointGenerator()
    endpoints = endpoint_gen.get_product_endpoints()
    
    configurator = TemplateConfigurator(seed=456)
    
    # Define learning objectives
    learning_objectives = [
        LearningObjective.HTTP_BASICS,
        LearningObjective.AUTHENTICATION,
        LearningObjective.ERROR_HANDLING
    ]
    
    print(f"\n🎯 Learning objectives:")
    for obj in learning_objectives:
        print(f"   • {obj.value.replace('_', ' ').title()}")
    
    # Generate progressive templates
    templates = configurator.generate_progressive_templates(
        endpoints=endpoints,
        learning_objectives=learning_objectives,
        num_levels=3
    )
    
    print(f"\n📚 Generated {len(templates)} progressive templates:")
    
    for i, template in enumerate(templates, 1):
        level = template.metadata['progression_level']
        difficulty = template.metadata['difficulty']
        gaps = len(template.gaps)
        
        print(f"\n   Level {level} ({difficulty}):")
        print(f"   • Gaps: {gaps}")
        print(f"   • Components: {[gap.component.value for gap in template.gaps]}")
        print(f"   • Estimated time: {template.metadata.get('estimated_completion_time', 'N/A')} min")
        
        # Show difficulty analysis
        analysis = configurator.analyze_template_difficulty(template)
        print(f"   • Complexity score: {analysis['complexity_score']:.1f}")
        print(f"   • Difficulty rating: {analysis['difficulty_rating']}")
    
    return templates


def custom_challenge_example():
    """Demonstrate custom challenge creation"""
    print("\n🎮 Custom Challenge Creation Example")
    print("=" * 60)
    
    # Setup
    endpoint_gen = EndpointGenerator()
    endpoints = endpoint_gen.get_user_endpoints()[:2]
    
    configurator = TemplateConfigurator(seed=789)
    
    # Create a custom challenge focusing on specific components
    focus_components = [
        MissingComponent.AUTHENTICATION,
        MissingComponent.ERROR_HANDLING,
        MissingComponent.RETRY_LOGIC
    ]
    
    print(f"\n🎯 Creating custom challenge with focus on:")
    for comp in focus_components:
        print(f"   • {comp.value}")
    
    # Generate custom challenge
    challenge = configurator.create_custom_challenge(
        endpoints=endpoints,
        focus_components=focus_components,
        challenge_name="Authentication & Resilience Challenge",
        description="Master API authentication and error recovery patterns"
    )
    
    print(f"\n🏆 Challenge: {challenge.metadata['challenge_name']}")
    print(f"📝 Description: {challenge.metadata['challenge_description']}")
    print(f"🔧 Difficulty: {challenge.metadata['difficulty']}")
    print(f"📊 Total gaps: {len(challenge.gaps)}")
    
    # Analyze the challenge
    analysis = configurator.analyze_template_difficulty(challenge)
    print(f"\n📈 Challenge Analysis:")
    print(f"   • Complexity score: {analysis['complexity_score']:.1f}")
    print(f"   • Average complexity: {analysis['average_complexity']:.1f}")
    print(f"   • Difficulty rating: {analysis['difficulty_rating']}")
    print(f"   • Estimated time: {analysis['estimated_time']} minutes")
    
    return challenge


def pattern_comparison_example():
    """Compare different integration patterns"""
    print("\n🔄 Integration Pattern Comparison Example")
    print("=" * 60)
    
    # Setup
    endpoint_gen = EndpointGenerator()
    endpoints = endpoint_gen.get_all_endpoints()[:4]
    
    generator = PatternBasedTemplateGenerator(seed=999)
    patterns = generator.get_available_patterns()
    
    missing_components = [
        MissingComponent.IMPORTS,
        MissingComponent.HTTP_METHOD,
        MissingComponent.ERROR_HANDLING
    ]
    
    print(f"\n🎨 Comparing {len(patterns)} integration patterns:")
    
    results = {}
    for pattern_name in patterns:
        try:
            template = generator.generate_with_specific_pattern(
                pattern_name=pattern_name,
                endpoints=endpoints,
                missing_components=missing_components,
                difficulty=DifficultyLevel.INTERMEDIATE
            )
            
            results[pattern_name] = {
                'template': template,
                'lines': len(template.code.split('\n')),
                'gaps': len(template.gaps),
                'pattern_name': template.metadata.get('chosen_pattern', pattern_name)
            }
            
            print(f"\n   🔧 {pattern_name.upper()} Pattern:")
            print(f"      • Lines of code: {results[pattern_name]['lines']}")
            print(f"      • Number of gaps: {results[pattern_name]['gaps']}")
            print(f"      • Description: {template.description}")
            
        except Exception as e:
            print(f"   ❌ {pattern_name}: {str(e)}")
    
    # Show best pattern for learning
    if results:
        best_for_learning = min(results.items(), key=lambda x: x[1]['gaps'])
        most_comprehensive = max(results.items(), key=lambda x: x[1]['lines'])
        
        print(f"\n🏅 Pattern Recommendations:")
        print(f"   • Best for beginners: {best_for_learning[0]} ({best_for_learning[1]['gaps']} gaps)")
        print(f"   • Most comprehensive: {most_comprehensive[0]} ({most_comprehensive[1]['lines']} lines)")
    
    return results


def save_templates_example():
    """Demonstrate saving templates to files"""
    print("\n💾 Template Saving Example")
    print("=" * 60)
    
    # Generate a sample template
    endpoint_gen = EndpointGenerator()
    endpoints = endpoint_gen.get_user_endpoints()[:2]
    
    configurator = TemplateConfigurator()
    config = configurator.create_configuration(
        difficulty=DifficultyLevel.INTERMEDIATE,
        learning_objectives=[LearningObjective.HTTP_BASICS, LearningObjective.AUTHENTICATION],
        include_tests=True,
        include_documentation=True
    )
    
    template = configurator.generate_template(endpoints, config)
    
    # Save template code
    template_filename = "sample_api_client_template.py"
    with open(template_filename, 'w') as f:
        f.write(template.code)
    
    print(f"✅ Saved template code to: {template_filename}")
    
    # Save metadata
    metadata_filename = "template_metadata.json"
    with open(metadata_filename, 'w') as f:
        # Convert gaps to serializable format
        gaps_data = []
        for gap in template.gaps:
            gaps_data.append({
                'component': gap.component.value,
                'line_number': gap.line_number,
                'placeholder': gap.placeholder,
                'hint': gap.hint,
                'difficulty': gap.difficulty.value
            })
        
        metadata = template.metadata.copy()
        metadata['gaps'] = gaps_data
        metadata['total_lines'] = len(template.code.split('\n'))
        
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✅ Saved template metadata to: {metadata_filename}")
    
    # Create solution hints file
    hints_filename = "solution_hints.md"
    with open(hints_filename, 'w') as f:
        f.write(f"# Solution Hints for {template.description}\n\n")
        
        for i, gap in enumerate(template.gaps, 1):
            f.write(f"## Gap {i}: {gap.component.value}\n")
            f.write(f"**Line {gap.line_number}**\n\n")
            f.write(f"**Hint:** {gap.hint}\n\n")
            if gap.expected_solution:
                f.write(f"**Expected Solution:**\n```python\n{gap.expected_solution}\n```\n\n")
            f.write("---\n\n")
    
    print(f"✅ Saved solution hints to: {hints_filename}")
    
    return {
        'template_file': template_filename,
        'metadata_file': metadata_filename,
        'hints_file': hints_filename
    }


def main():
    """Run all code template generation examples"""
    print("🎯 Code Template Generation System Demo")
    print("=" * 70)
    
    try:
        # Run all examples
        print("\n" + "="*70)
        basic_template_generation_example()
        
        print("\n" + "="*70)
        schema_analysis_example()
        
        print("\n" + "="*70)
        progressive_template_example()
        
        print("\n" + "="*70)
        custom_challenge_example()
        
        print("\n" + "="*70)
        pattern_comparison_example()
        
        print("\n" + "="*70)
        save_templates_example()
        
        print("\n" + "="*70)
        print("🎉 All examples completed successfully!")
        
        print("\n💡 Key Features Demonstrated:")
        print("   • Automatic schema analysis and complexity detection")
        print("   • Multiple integration patterns (CRUD, Async, etc.)")
        print("   • Progressive difficulty levels for learning")
        print("   • Custom challenge creation with specific focus areas")
        print("   • Configurable missing components and hints")
        print("   • Template metadata and analysis")
        print("   • File export for use in RL training")
        
        print("\n🚀 Next Steps:")
        print("   • Use generated templates for RL agent training")
        print("   • Create custom learning curricula")
        print("   • Integrate with mock servers for complete testing")
        print("   • Build adaptive difficulty based on performance")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()