"""
Example: Enhanced metadata workflow for gaiapy

This example demonstrates the full metadata-enhanced workflow for geographic
ancestry inference, including:

1. Creating tree sequences with location metadata
2. Using metadata-aware MPR functions
3. Augmenting tree sequences with inferred locations
4. Comparing discrete vs continuous reconstruction methods

This workflow showcases the enhanced functionality that supports both
traditional external location arrays and modern tree sequence metadata.
"""

import numpy as np
import msprime
import tskit
import gaiapy
import json


def create_tree_sequence_with_metadata():
    """
    Create a tree sequence with geographic location metadata.
    
    Demonstrates how to add location information to individual metadata
    in formats compatible with gaiapy's metadata-aware functions.
    """
    print("Creating tree sequence with location metadata...")
    
    # Simulate basic tree sequence
    ts = msprime.simulate(
        sample_size=8,
        Ne=10000,
        length=1000000,
        recombination_rate=1e-8,
        random_seed=42
    )
    
    # Create location data for samples
    sample_locations = [
        {"location": {"coordinates": [0.0, 0.0], "name": "Site_A"}},    # Samples 0,1
        {"location": {"coordinates": [0.0, 0.0], "name": "Site_A"}},
        {"location": {"coordinates": [50.0, 30.0], "name": "Site_B"}},  # Samples 2,3
        {"location": {"coordinates": [50.0, 30.0], "name": "Site_B"}}, 
        {"location": {"coordinates": [100.0, 60.0], "name": "Site_C"}}, # Samples 4,5
        {"location": {"coordinates": [100.0, 60.0], "name": "Site_C"}},
        {"location": {"coordinates": [150.0, 90.0], "name": "Site_D"}}, # Samples 6,7
        {"location": {"coordinates": [150.0, 90.0], "name": "Site_D"}},
    ]
    
    # Create discrete state version
    discrete_states = [
        {"geographic_state": 0, "region": "West"},     # Samples 0,1
        {"geographic_state": 0, "region": "West"},
        {"geographic_state": 1, "region": "Central"},  # Samples 2,3
        {"geographic_state": 1, "region": "Central"},
        {"geographic_state": 2, "region": "East"},     # Samples 4,5
        {"geographic_state": 2, "region": "East"},
        {"geographic_state": 3, "region": "Far_East"}, # Samples 6,7
        {"geographic_state": 3, "region": "Far_East"},
    ]
    
    # Add metadata to individuals
    tables = ts.dump_tables()
    
    # Create individuals table if it doesn't exist
    if ts.num_individuals == 0:
        for i in range(ts.num_samples):
            # Add individual with both continuous and discrete metadata
            metadata = {
                **sample_locations[i],
                **discrete_states[i]
            }
            tables.individuals.add_row(
                metadata=json.dumps(metadata).encode()
            )
        
        # Link nodes to individuals
        for i, node in enumerate(tables.nodes):
            if i < ts.num_samples:
                tables.nodes[i] = node.replace(individual=i)
    
    # Create new tree sequence with metadata
    ts_with_metadata = tables.tree_sequence()
    
    print(f"Created tree sequence with {ts_with_metadata.num_individuals} individuals")
    print(f"Sample locations stored in individual metadata")
    
    return ts_with_metadata


def demonstrate_discrete_workflow():
    """
    Demonstrate discrete ancestry inference with metadata support.
    """
    print("\n" + "="*60)
    print("DISCRETE ANCESTRY INFERENCE WORKFLOW")
    print("="*60)
    
    # Create tree sequence with metadata
    ts = create_tree_sequence_with_metadata()
    
    # Method 1: Traditional approach with external location array
    print("\nMethod 1: External location array")
    external_locations = np.array([
        [0, 0], [1, 0], [2, 1], [3, 1], [4, 2], [5, 2], [6, 3], [7, 3]
    ])
    
    cost_matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2], 
        [2, 1, 0, 1],
        [3, 2, 1, 0]
    ])
    
    try:
        # Traditional discrete MPR
        result_traditional = gaiapy.discrete_mpr(ts, external_locations, cost_matrix)
        states_traditional = gaiapy.discrete_mpr_minimize(result_traditional)
        
        print(f"✓ Traditional discrete MPR completed")
        print(f"  Sample states: {states_traditional[:ts.num_samples]}")
        print(f"  Mean tree length: {result_traditional.mean_tree_length:.3f}")
        
    except NotImplementedError as e:
        print(f"⚠ Traditional method: {e}")
    
    # Method 2: Metadata-aware approach
    print("\nMethod 2: Metadata-aware approach")
    
    try:
        # Read locations from metadata and return augmented tree sequence
        result_metadata, ts_augmented = gaiapy.discrete_mpr_with_metadata(
            ts, 
            cost_matrix=cost_matrix,
            location_key="geographic_state",
            return_augmented_ts=True
        )
        
        print(f"✓ Metadata-aware discrete MPR completed")
        print(f"✓ Returned augmented tree sequence with inferred states")
        
        # Analyze results from augmented tree sequence
        for node in ts_augmented.nodes():
            if hasattr(node, 'metadata') and node.metadata:
                metadata = json.loads(node.metadata.decode())
                if "inferred_state" in metadata:
                    print(f"  Node {node.id}: inferred state {metadata['inferred_state']}")
                    
    except NotImplementedError as e:
        print(f"⚠ Metadata method: {e}")


def demonstrate_continuous_workflow():
    """
    Demonstrate continuous space ancestry inference with metadata support.
    """
    print("\n" + "="*60)
    print("CONTINUOUS SPACE ANCESTRY INFERENCE WORKFLOW")
    print("="*60)
    
    # Create tree sequence with metadata
    ts = create_tree_sequence_with_metadata()
    
    # Method 1: Traditional approach with external coordinates
    print("\nMethod 1: External coordinate array")
    external_coordinates = np.array([
        [0, 0.0, 0.0],      # Sample 0 at (0,0)
        [1, 0.0, 0.0],      # Sample 1 at (0,0)
        [2, 50.0, 30.0],    # Sample 2 at (50,30)
        [3, 50.0, 30.0],    # Sample 3 at (50,30)
        [4, 100.0, 60.0],   # Sample 4 at (100,60)
        [5, 100.0, 60.0],   # Sample 5 at (100,60)
        [6, 150.0, 90.0],   # Sample 6 at (150,90)
        [7, 150.0, 90.0],   # Sample 7 at (150,90)
    ])
    
    # Test both quadratic and linear methods
    methods = [
        ("Quadratic (L2)", gaiapy.quadratic_mpr, gaiapy.quadratic_mpr_minimize),
        ("Linear (L1)", gaiapy.linear_mpr, gaiapy.linear_mpr_minimize),
    ]
    
    for method_name, mpr_func, minimize_func in methods:
        print(f"\n{method_name} reconstruction:")
        
        try:
            # Traditional approach
            result = mpr_func(ts, external_coordinates, use_branch_lengths=True)
            locations = minimize_func(result)
            
            print(f"✓ Traditional {method_name} MPR completed")
            print(f"  Sample coordinates:")
            for i in range(ts.num_samples):
                x, y = locations[i]
                print(f"    Sample {i}: ({x:.1f}, {y:.1f})")
            print(f"  Mean tree length: {result.mean_tree_length:.3f}")
            
        except NotImplementedError as e:
            print(f"⚠ Traditional {method_name}: {e}")
    
    # Method 2: Metadata-aware approach
    print("\nMethod 2: Metadata-aware approaches")
    
    metadata_methods = [
        ("Quadratic with metadata", gaiapy.quadratic_mpr_with_metadata),
        ("Linear with metadata", gaiapy.linear_mpr_with_metadata),
    ]
    
    for method_name, metadata_func in metadata_methods:
        print(f"\n{method_name}:")
        
        try:
            # Read coordinates from metadata and return augmented tree sequence
            result, ts_augmented = metadata_func(
                ts,
                location_key="location",
                coordinates_key="coordinates", 
                return_augmented_ts=True
            )
            
            print(f"✓ {method_name} completed")
            print(f"✓ Returned augmented tree sequence with inferred coordinates")
            
            # Analyze results from augmented tree sequence
            for node_id in range(min(5, ts_augmented.num_nodes)):  # Show first 5 nodes
                node = ts_augmented.node(node_id)
                if hasattr(node, 'metadata') and node.metadata:
                    metadata = json.loads(node.metadata.decode())
                    if "inferred_location" in metadata:
                        coords = metadata["inferred_location"]
                        print(f"  Node {node_id}: inferred location {coords}")
                        
        except NotImplementedError as e:
            print(f"⚠ {method_name}: {e}")


def demonstrate_ancestry_analysis():
    """
    Demonstrate ancestry coefficient and migration flux analysis.
    """
    print("\n" + "="*60)
    print("ANCESTRY AND MIGRATION ANALYSIS")
    print("="*60)
    
    # Create tree sequence with metadata
    ts = create_tree_sequence_with_metadata()
    
    # Mock MPR result for demonstration (since implementation not complete)
    print("\nCreating mock MPR result for ancestry analysis...")
    mpr_matrix = np.random.rand(ts.num_nodes, 4)  # 4 discrete states
    tree_lengths = np.random.rand(ts.num_trees)
    
    mock_result = gaiapy.MPRResult(
        mpr_matrix=mpr_matrix,
        tree_lengths=tree_lengths,
        mean_tree_length=np.mean(tree_lengths),
        node_weights=np.ones(ts.num_nodes)
    )
    
    # Ancestry coefficient analysis
    print("\nAncestry coefficient analysis:")
    try:
        ancestry_results = gaiapy.discrete_mpr_ancestry(
            ts, mock_result, time_bins=np.array([0, 1000, 2000, 3000])
        )
        
        print("✓ Ancestry coefficients computed")
        print("  Time bins:", ancestry_results['time_bins'])
        print("  Ancestry matrix shape:", ancestry_results['ancestry_matrix'].shape)
        
    except NotImplementedError as e:
        print(f"⚠ Ancestry analysis: {e}")
    
    # Migration flux analysis
    print("\nMigration flux analysis:")
    try:
        flux_results = gaiapy.discrete_mpr_ancestry_flux(
            ts, mock_result, time_bins=np.array([0, 1000, 2000])
        )
        
        print("✓ Migration flux computed")
        print("  Time intervals:", flux_results['time_intervals'])
        print("  Flux matrices shape:", flux_results['flux_matrices'].shape)
        
    except NotImplementedError as e:
        print(f"⚠ Migration flux analysis: {e}")


def demonstrate_metadata_utilities():
    """
    Demonstrate metadata utility functions.
    """
    print("\n" + "="*60)
    print("METADATA UTILITY FUNCTIONS")
    print("="*60)
    
    # Create tree sequence with metadata
    ts = create_tree_sequence_with_metadata()
    
    # Extract locations from metadata
    print("\nExtracting sample locations from metadata:")
    try:
        locations = gaiapy.extract_sample_locations_from_metadata(
            ts, location_key="location", coordinates_key="coordinates"
        )
        print(f"✓ Extracted locations with shape: {locations.shape}")
        
    except NotImplementedError as e:
        print(f"⚠ Location extraction: {e}")
    
    # Validate metadata
    print("\nValidating location metadata:")
    try:
        validation_results = gaiapy.validate_location_metadata(
            ts, location_key="location", expected_dims=2
        )
        
        print(f"✓ Metadata validation completed")
        print(f"  Valid: {validation_results['valid']}")
        print(f"  Format: {validation_results['metadata_format']}")
        print(f"  Coverage: {validation_results['sample_coverage']:.1%}")
        
    except NotImplementedError as e:
        print(f"⚠ Metadata validation: {e}")
    
    # Demonstrate augmenting tree sequence
    print("\nAugmenting tree sequence with inferred locations:")
    try:
        # Mock inferred locations
        inferred_coords = np.random.rand(ts.num_nodes, 2) * 100
        
        ts_augmented = gaiapy.augment_tree_sequence_with_locations(
            ts, inferred_coords, location_key="inferred_location"
        )
        
        print(f"✓ Augmented tree sequence created")
        print(f"  Original nodes: {ts.num_nodes}")
        print(f"  Augmented nodes: {ts_augmented.num_nodes}")
        
    except NotImplementedError as e:
        print(f"⚠ Tree sequence augmentation: {e}")


def main():
    """
    Run the complete metadata workflow demonstration.
    """
    print("GAIAPY ENHANCED METADATA WORKFLOW DEMONSTRATION")
    print("=" * 80)
    print("\nThis example demonstrates the enhanced functionality of gaiapy")
    print("that supports both traditional location arrays and tree sequence")
    print("metadata for geographic ancestry inference.")
    
    # Run all demonstrations
    demonstrate_discrete_workflow()
    demonstrate_continuous_workflow()
    demonstrate_ancestry_analysis()
    demonstrate_metadata_utilities()
    
    print("\n" + "="*80)
    print("WORKFLOW SUMMARY")
    print("="*80)
    print("\nThe enhanced gaiapy package provides:")
    print("✓ Discrete ancestry inference with metadata support")
    print("✓ Continuous space reconstruction (quadratic & linear)")
    print("✓ Metadata-aware functions for streamlined workflows")
    print("✓ Augmented tree sequences with inferred locations")
    print("✓ Ancestry coefficient and migration flux analysis")
    print("✓ Comprehensive metadata utilities")
    print("\nNote: Many functions show NotImplementedError as they are")
    print("placeholders for future implementation. The discrete MPR")
    print("functions are fully implemented and functional.")
    
    print(f"\nReferences to original C implementation:")
    print(f"• Discrete MPR: src/treeseq_sankoff_discrete.c")
    print(f"• Quadratic MPR: src/treeseq_sankoff_quadratic.c") 
    print(f"• Linear MPR: src/treeseq_sankoff_linear.c")
    print(f"• Ancestry analysis: src/treeseq_sankoff_discrete_ancestry.c")
    print(f"• Migration flux: src/treeseq_sankoff_discrete_flux.c")


if __name__ == "__main__":
    main() 