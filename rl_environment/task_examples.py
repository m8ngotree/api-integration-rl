#!/usr/bin/env python3

import asyncio
import json
from typing import Dict, List, Any

from .task_generator import (
    TaskGeneratorManager, TaskType, TaskDifficulty, LearningTask
)


async def basic_task_generation_example():
    """Demonstrate basic task generation"""
    print("🎯 Basic Task Generation Example")
    print("=" * 60)
    
    # Create task generator manager
    manager = TaskGeneratorManager(seed=42)
    
    # Generate a single random task
    task = manager.generate_task()
    
    print(f"📋 Generated Task:")
    print(f"   • ID: {task.task_id}")
    print(f"   • Type: {task.task_type.value}")
    print(f"   • Difficulty: {task.difficulty.value}")
    print(f"   • Title: {task.title}")
    print(f"   • Estimated Time: {task.estimated_time} minutes")
    print(f"   • Learning Objectives: {len(task.learning_objectives)} items")
    print(f"   • API Endpoints: {len(task.api_documentation.endpoints)} endpoints")
    
    # Show starter code preview
    print(f"\n📝 Starter Code Preview (first 300 chars):")
    print(f"```python")
    print(task.starter_code[:300] + "..." if len(task.starter_code) > 300 else task.starter_code)
    print(f"```")
    
    # Show hints
    print(f"\n💡 Hints:")
    for i, hint in enumerate(task.hints[:3], 1):
        print(f"   {i}. {hint}")


async def specific_task_type_example():
    """Demonstrate generating specific task types"""
    print("\n🔍 Specific Task Type Generation Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Generate specific task types
    task_types = [
        TaskType.BASIC_GET_REQUEST,
        TaskType.AUTHENTICATION_SETUP,
        TaskType.BULK_OPERATIONS
    ]
    
    for task_type in task_types:
        task = manager.generate_task(task_type=task_type)
        
        print(f"\n📋 {task_type.value.replace('_', ' ').title()}:")
        print(f"   • Title: {task.title}")
        print(f"   • Difficulty: {task.difficulty.value}")
        print(f"   • Tags: {', '.join(task.tags)}")
        print(f"   • Success Criteria: {len(task.success_criteria.required_api_calls)} API calls required")


async def difficulty_progression_example():
    """Demonstrate difficulty-based task generation"""
    print("\n📈 Difficulty Progression Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    difficulties = [
        TaskDifficulty.BEGINNER,
        TaskDifficulty.INTERMEDIATE,
        TaskDifficulty.ADVANCED
    ]
    
    for difficulty in difficulties:
        print(f"\n🎚️  {difficulty.value.upper()} Level Tasks:")
        
        tasks = manager.generate_task_set(count=3, difficulty=difficulty)
        
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task.title} ({task.task_type.value})")
            print(f"      Time: {task.estimated_time}min | Score: {task.success_criteria.minimum_score}")


async def progressive_curriculum_example():
    """Demonstrate progressive curriculum generation"""
    print("\n🎓 Progressive Curriculum Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Generate a complete curriculum
    curriculum = manager.generate_progressive_curriculum(total_tasks=12)
    
    print(f"📚 Generated {len(curriculum)} task curriculum:")
    
    difficulty_counts = {}
    for task in curriculum:
        diff = task.difficulty.value
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\n📊 Difficulty Distribution:")
    for difficulty, count in difficulty_counts.items():
        percentage = (count / len(curriculum)) * 100
        print(f"   • {difficulty.title()}: {count} tasks ({percentage:.1f}%)")
    
    print(f"\n📋 Curriculum Overview:")
    for i, task in enumerate(curriculum, 1):
        difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(task.difficulty.value, "⚪")
        print(f"   {i:2d}. {difficulty_icon} {task.title}")
        print(f"       {task.task_type.value} | {task.estimated_time}min")


async def focused_session_example():
    """Demonstrate focused learning sessions"""
    print("\n🎯 Focused Learning Sessions Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    focus_areas = ["crud", "error_handling", "performance", "security"]
    
    for focus_area in focus_areas:
        print(f"\n🔍 {focus_area.upper().replace('_', ' ')} Focus Session:")
        
        tasks = manager.generate_focused_session(
            focus_area=focus_area,
            count=3,
            difficulty=TaskDifficulty.INTERMEDIATE
        )
        
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task.title}")
            print(f"      Type: {task.task_type.value} | Time: {task.estimated_time}min")


async def adaptive_task_example():
    """Demonstrate adaptive task generation based on performance"""
    print("\n🤖 Adaptive Task Generation Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Simulate learning session with performance tracking
    previous_tasks = []
    performance_scores = []
    current_difficulty = TaskDifficulty.BEGINNER
    
    print("🏃 Simulating adaptive learning session...")
    
    for session in range(5):
        print(f"\n📅 Session {session + 1}:")
        
        # Generate next task based on previous performance
        if previous_tasks:
            task = manager.generate_adaptive_next_task(
                previous_tasks=previous_tasks,
                performance_scores=performance_scores,
                current_difficulty=current_difficulty
            )
        else:
            task = manager.generate_task(difficulty=current_difficulty)
        
        # Simulate performance (varying scores to show adaptation)
        if session < 2:
            # Start with good performance
            simulated_score = 7.5 + (session * 0.5)
        elif session == 2:
            # Drop performance to trigger difficulty adjustment
            simulated_score = 3.5
        else:
            # Recover performance
            simulated_score = 6.0 + session
        
        previous_tasks.append(task)
        performance_scores.append(simulated_score)
        current_difficulty = task.difficulty
        
        print(f"   📋 Task: {task.title}")
        print(f"   🎚️  Difficulty: {task.difficulty.value}")
        print(f"   📊 Simulated Score: {simulated_score:.1f}")
        
        # Show adaptation logic
        if len(performance_scores) >= 3:
            recent_avg = sum(performance_scores[-3:]) / 3
            print(f"   📈 Recent Average: {recent_avg:.1f}")
            
            if recent_avg >= 8.0:
                print("   ⬆️  High performance - difficulty may increase")
            elif recent_avg <= 4.0:
                print("   ⬇️  Low performance - difficulty may decrease")
            else:
                print("   ➡️  Stable performance - difficulty maintained")


async def task_validation_example():
    """Demonstrate task validation"""
    print("\n✅ Task Validation Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Generate some tasks and validate them
    tasks = manager.generate_task_set(count=3)
    
    print("🔍 Validating generated tasks...")
    
    all_valid = True
    for i, task in enumerate(tasks, 1):
        issues = manager.validate_task(task)
        
        if issues:
            all_valid = False
            print(f"\n❌ Task {i} ({task.title}) has {len(issues)} issues:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n✅ Task {i} ({task.title}) is valid")
    
    if all_valid:
        print(f"\n🎉 All {len(tasks)} tasks passed validation!")


async def task_export_example():
    """Demonstrate task export functionality"""
    print("\n💾 Task Export Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Generate a small set of tasks
    tasks = manager.generate_task_set(count=2, difficulty=TaskDifficulty.BEGINNER)
    
    # Export to JSON (would normally save to file)
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = f.name
    
    try:
        manager.export_tasks_to_json(tasks, export_path)
        
        # Read back and show preview
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        print(f"📄 Exported {len(exported_data)} tasks to JSON")
        print(f"🔍 JSON Structure Preview:")
        
        if exported_data:
            task_keys = list(exported_data[0].keys())
            print(f"   📋 Task fields: {', '.join(task_keys[:5])}{'...' if len(task_keys) > 5 else ''}")
            
            # Show size information
            json_size = len(json.dumps(exported_data))
            print(f"   📏 Export size: {json_size:,} characters")
    
    finally:
        # Clean up temp file
        if os.path.exists(export_path):
            os.unlink(export_path)


async def statistics_example():
    """Demonstrate task statistics"""
    print("\n📊 Task Statistics Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Get comprehensive statistics
    stats = manager.get_task_statistics()
    
    print("🔢 Task Generator Statistics:")
    print(f"   • Total Generators: {stats['total_generators']}")
    print(f"   • Task Types: {len(stats['task_types'])}")
    print(f"   • Difficulty Levels: {len(stats['difficulty_levels'])}")
    
    print(f"\n📋 Available Task Types:")
    for task_type in stats['task_types']:
        print(f"   • {task_type.replace('_', ' ').title()}")
    
    print(f"\n🎚️  Difficulty Progression:")
    for difficulty, task_types in stats['progression_mapping'].items():
        print(f"   • {difficulty.title()}: {len(task_types)} task types")
        for task_type in task_types:
            print(f"     - {task_type.replace('_', ' ').title()}")


async def recommended_sequences_example():
    """Demonstrate recommended task sequences"""
    print("\n🎯 Recommended Task Sequences Example")
    print("=" * 60)
    
    manager = TaskGeneratorManager(seed=42)
    
    user_levels = ["beginner", "intermediate", "advanced"]
    
    for user_level in user_levels:
        print(f"\n👤 {user_level.title()} User Sequence:")
        
        tasks = manager.get_recommended_task_sequence(user_level=user_level)
        
        print(f"   📚 {len(tasks)} recommended tasks:")
        
        # Show first 3 tasks as preview
        for i, task in enumerate(tasks[:3], 1):
            difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(task.difficulty.value, "⚪")
            print(f"   {i}. {difficulty_icon} {task.title}")
            print(f"      {task.task_type.value} | {task.estimated_time}min")
        
        if len(tasks) > 3:
            print(f"   ... and {len(tasks) - 3} more tasks")
    
    # Show custom learning goals
    print(f"\n🎯 Custom Learning Goals Example:")
    custom_tasks = manager.get_recommended_task_sequence(
        user_level="custom",
        learning_goals=["crud", "security"]
    )
    
    print(f"   📚 {len(custom_tasks)} tasks for CRUD + Security focus:")
    for i, task in enumerate(custom_tasks[:4], 1):
        print(f"   {i}. {task.title} ({task.task_type.value})")


async def comprehensive_example():
    """Run a comprehensive example showcasing all features"""
    print("\n🚀 Comprehensive Task Generator Example")
    print("=" * 80)
    
    manager = TaskGeneratorManager(seed=42)
    
    # Demonstrate complete workflow
    print("📋 Step 1: Generate a diverse curriculum")
    curriculum = manager.generate_progressive_curriculum(total_tasks=8)
    
    print("📋 Step 2: Validate all tasks")
    validation_results = []
    for task in curriculum:
        issues = manager.validate_task(task)
        validation_results.append(len(issues) == 0)
    
    valid_count = sum(validation_results)
    print(f"✅ {valid_count}/{len(curriculum)} tasks are valid")
    
    print("📋 Step 3: Show curriculum overview")
    total_time = sum(task.estimated_time for task in curriculum)
    print(f"   • Total estimated time: {total_time} minutes ({total_time/60:.1f} hours)")
    
    task_type_counts = {}
    for task in curriculum:
        task_type = task.task_type.value
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    print(f"   • Task type variety: {len(task_type_counts)} different types")
    
    print("📋 Step 4: Sample task breakdown")
    sample_task = curriculum[0]
    print(f"   📝 Sample Task: {sample_task.title}")
    print(f"   🎯 Success Criteria: {len(sample_task.success_criteria.required_api_calls)} API calls required")
    print(f"   💡 Hints provided: {len(sample_task.hints)}")
    print(f"   🎓 Learning objectives: {len(sample_task.learning_objectives)}")
    print(f"   🏷️  Tags: {', '.join(sample_task.tags[:3])}{'...' if len(sample_task.tags) > 3 else ''}")
    
    print("\n🎉 Task generator successfully demonstrated all capabilities!")


async def main():
    """Run all task generator examples"""
    print("🎯 API Integration RL Task Generator Examples")
    print("=" * 80)
    
    try:
        # Run all examples
        await basic_task_generation_example()
        await specific_task_type_example()
        await difficulty_progression_example()
        await progressive_curriculum_example()
        await focused_session_example()
        await adaptive_task_example()
        await task_validation_example()
        await task_export_example()
        await statistics_example()
        await recommended_sequences_example()
        await comprehensive_example()
        
        print("\n" + "=" * 80)
        print("🎉 All task generator examples completed successfully!")
        
        print("\n💡 Key Features Demonstrated:")
        print("   • 10 different task types covering all API integration patterns")
        print("   • 4 difficulty levels with intelligent progression")
        print("   • Adaptive task generation based on performance")
        print("   • Focused learning sessions for specific skills")
        print("   • Progressive curriculum generation")
        print("   • Task validation and quality assurance")
        print("   • JSON export for integration with RL systems")
        print("   • Comprehensive statistics and analytics")
        
        print("\n🎯 Task Types Available:")
        print("   • Basic GET Request - Learn fundamental HTTP requests")
        print("   • POST Create Resource - Master data creation")
        print("   • PUT Update Resource - Handle data updates")
        print("   • DELETE Resource - Manage resource deletion")
        print("   • Authentication Setup - Implement API security")
        print("   • Error Handling - Build robust error handling")
        print("   • Pagination Handling - Process large datasets")
        print("   • Bulk Operations - Optimize for performance")
        print("   • Response Validation - Ensure data integrity")
        print("   • Async API Calls - Master concurrent programming")
        
        print("\n🎚️  Difficulty Levels:")
        print("   • Beginner: Basic concepts and simple implementations")
        print("   • Intermediate: Real-world scenarios with complexity")
        print("   • Advanced: Performance optimization and edge cases")
        print("   • Expert: Complex integrations and advanced patterns")
        
        print("\n🤖 Perfect for RL Training:")
        print("   • Diverse task generation for comprehensive learning")
        print("   • Adaptive difficulty based on agent performance")
        print("   • Rich reward signals from detailed success criteria")
        print("   • Comprehensive API documentation for each task")
        print("   • Progressive curriculum for structured learning")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())