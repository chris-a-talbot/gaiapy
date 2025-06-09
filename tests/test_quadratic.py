#!/usr/bin/env python3
"""
Test script for quadratic MPR functionality.
"""

import sys
import os
# Add the parent directory to the path so we can import gaiapy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gaiapy
import numpy as np
import msprime

def test_quadratic_mpr():
    print('Testing quadratic MPR functionality...')

    # Create a simple tree sequence
    ts = msprime.simulate(sample_size=4, Ne=1000, length=100, random_seed=42)
    print(f'Tree sequence: {ts.num_samples} samples, {ts.num_nodes} nodes, {ts.num_trees} trees')

    # Create sample locations (node_id, x, y)
    sample_locations = np.array([
        [0, 0.0, 0.0],  # Sample 0 at origin
        [1, 1.0, 0.0],  # Sample 1 at (1,0)
        [2, 0.0, 1.0],  # Sample 2 at (0,1)
        [3, 1.0, 1.0],  # Sample 3 at (1,1)
    ])
    print(f'Sample locations shape: {sample_locations.shape}')

    # Test quadratic MPR
    try:
        result = gaiapy.quadratic_mpr(ts, sample_locations)
        print(f'✓ quadratic_mpr completed successfully')
        print(f'  Result type: {type(result)}')
        print(f'  Reconstruction type: {result.reconstruction_type}')
        print(f'  MPR matrix shape: {result.mpr_matrix.shape}')
        print(f'  Mean tree length: {result.mean_tree_length:.6f}')
        print(f'  Number of trees: {result.num_trees}')
        
        # Test quadratic_mpr_minimize
        optimal_locations = gaiapy.quadratic_mpr_minimize(result)
        print(f'✓ quadratic_mpr_minimize completed successfully')
        print(f'  Optimal locations shape: {optimal_locations.shape}')
        print(f'  Sample optimal locations:')
        for i in range(min(4, len(optimal_locations))):
            x, y = optimal_locations[i]
            print(f'    Node {i}: ({x:.3f}, {y:.3f})')
            
        # Test with branch lengths
        print('\nTesting with branch lengths...')
        result_bl = gaiapy.quadratic_mpr(ts, sample_locations, use_branch_lengths=True)
        print(f'✓ quadratic_mpr with branch lengths completed')
        print(f'  Mean tree length: {result_bl.mean_tree_length:.6f}')
        
        optimal_locations_bl = gaiapy.quadratic_mpr_minimize(result_bl)
        print(f'✓ quadratic_mpr_minimize with branch lengths completed')
        
        return True
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quadratic_mpr()
    if success:
        print('\n🎉 All quadratic MPR tests passed!')
    else:
        print('\n❌ Some tests failed!') 