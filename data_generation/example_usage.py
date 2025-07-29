#!/usr/bin/env python3

import json
from api_schema_generator import APISchemaGenerator


def main():
    print("🚀 API Schema Generator Demo")
    print("=" * 50)
    
    # Initialize the generator with a seed for reproducible results
    generator = APISchemaGenerator(seed=42)
    
    print("\n1. Generating a complete OpenAPI specification...")
    complete_spec = generator.generate_openapi_spec(
        title="Demo E-commerce API",
        version="2.0.0",
        description="A complete e-commerce API with user management and product catalog"
    )
    print(f"✅ Generated spec with {len(complete_spec['paths'])} endpoints")
    
    # Save the complete spec
    generator.save_spec_to_file(complete_spec, "complete_api_spec.json")
    print("💾 Saved to: complete_api_spec.json")
    
    print("\n2. Generating a minimal API specification...")
    minimal_spec = generator.generate_minimal_spec()
    print(f"✅ Generated minimal spec with {len(minimal_spec['paths'])} endpoints")
    
    # Save the minimal spec
    generator.save_spec_to_file(minimal_spec, "minimal_api_spec.json")
    print("💾 Saved to: minimal_api_spec.json")
    
    print("\n3. Generating user management only API...")
    user_spec = generator.generate_user_management_spec()
    print(f"✅ Generated user management spec with {len(user_spec['paths'])} endpoints")
    
    print("\n4. Generating product catalog only API...")
    product_spec = generator.generate_product_catalog_spec()
    print(f"✅ Generated product catalog spec with {len(product_spec['paths'])} endpoints")
    
    print("\n5. Generating API scenario with sample data...")
    scenario = generator.generate_api_scenario(
        scenario_name="E-commerce Training Scenario",
        num_endpoints=6,
        include_sample_data=True
    )
    print(f"✅ Generated scenario with {len(scenario['endpoints'])} endpoints")
    
    # Save the scenario
    generator.save_scenario_to_file(scenario, "training_scenario.json")
    print("💾 Saved to: training_scenario.json")
    
    print("\n6. Generating training dataset...")
    dataset = generator.generate_training_dataset(
        num_scenarios=5,
        endpoints_per_scenario=4
    )
    print(f"✅ Generated dataset with {len(dataset)} scenarios")
    
    # Save the dataset
    with open("training_dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    print("💾 Saved to: training_dataset.json")
    
    print("\n7. Available endpoint tags:")
    tags = generator.get_available_endpoint_tags()
    print(f"📋 Tags: {', '.join(tags)}")
    
    print("\n8. Sample endpoint details:")
    all_endpoints = generator.endpoint_generator.get_all_endpoints()
    sample_endpoint = all_endpoints[0]
    print(f"📝 Sample endpoint: {sample_endpoint.method.value} {sample_endpoint.path}")
    print(f"   Summary: {sample_endpoint.summary}")
    print(f"   Tags: {', '.join(sample_endpoint.tags)}")
    
    print("\n🎉 Demo completed! Check the generated files:")
    print("   - complete_api_spec.json")
    print("   - minimal_api_spec.json")
    print("   - training_scenario.json")
    print("   - training_dataset.json")


if __name__ == "__main__":
    main()