"""
Basic usage example for gaiapy.

This example demonstrates how to use gaiapy for discrete geographic
ancestry inference using a simple simulated tree sequence.
"""

import numpy as np
import tskit
import msprime
import gaiapy


def create_example_tree_sequence():
    """Create a simple tree sequence for testing."""
    
    # Simulate a simple tree sequence with 4 samples
    ts = msprime.simulate(
        sample_size=4,
        Ne=1000,
        length=1000,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        random_seed=42
    )
    
    return ts


def basic_discrete_example():
    """Demonstrate basic discrete ancestry inference."""
    
    print("=== Basic Discrete Ancestry Inference ===")
    
    # Create example tree sequence
    ts = create_example_tree_sequence()
    print(f"Tree sequence: {ts.num_samples} samples, {ts.num_nodes} nodes, {ts.num_trees} trees")
    
    # Define sample locations
    # Let's say samples 0,1 are from state 0 and samples 2,3 are from state 1
    sample_locations = np.array([
        [0, 0],  # Sample 0 in state 0
        [1, 0],  # Sample 1 in state 0
        [2, 1],  # Sample 2 in state 1
        [3, 1],  # Sample 3 in state 1
    ])
    
    print(f"Sample locations: {sample_locations}")
    
    # Define cost matrix - cost of 1 to migrate between states
    cost_matrix = np.array([
        [0, 1],  # From state 0: stay=0, to_1=1
        [1, 0]   # From state 1: to_0=1, stay=0
    ])
    
    print(f"Cost matrix:\n{cost_matrix}")
    
    # Compute discrete MPR
    try:
        mpr_result = gaiapy.discrete_mpr(ts, sample_locations, cost_matrix)
        print(f"\nMPR Result: {mpr_result}")
        print(f"Mean tree length: {mpr_result.mean_tree_length:.4f}")
        
        # Get optimal state assignments
        optimal_states = gaiapy.discrete_mpr_minimize(mpr_result)
        print(f"\nOptimal state assignments:")
        for node in range(ts.num_nodes):
            print(f"  Node {node}: state {optimal_states[node]}")
        
        # Get migration histories
        history = gaiapy.discrete_mpr_edge_history(ts, mpr_result, cost_matrix)
        print(f"\nMigration histories: {len(history['paths'])} edges processed")
        
        print("\n=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def compare_with_branch_lengths():
    """Compare results with and without branch length scaling."""
    
    print("\n=== Comparison: Branch Length Scaling ===")
    
    ts = create_example_tree_sequence()
    
    sample_locations = np.array([
        [0, 0], [1, 0], [2, 1], [3, 1]
    ])
    
    cost_matrix = np.array([[0, 1], [1, 0]])
    
    # Without branch lengths
    mpr_no_bl = gaiapy.discrete_mpr(ts, sample_locations, cost_matrix, 
                                   use_branch_lengths=False)
    
    # With branch lengths  
    mpr_with_bl = gaiapy.discrete_mpr(ts, sample_locations, cost_matrix,
                                     use_branch_lengths=True)
    
    print(f"Mean tree length without branch lengths: {mpr_no_bl.mean_tree_length:.4f}")
    print(f"Mean tree length with branch lengths: {mpr_with_bl.mean_tree_length:.4f}")


if __name__ == "__main__":
    basic_discrete_example()
    compare_with_branch_lengths()
