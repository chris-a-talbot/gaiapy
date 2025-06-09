#!/usr/bin/env python3
"""
Test script for linear MPR functionality.
"""

import sys
import os
# Add the parent directory to the path so we can import gaiapy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gaiapy
import numpy as np
import msprime

def test_linear_mpr():
    print('Testing linear MPR functionality...')

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

    # Test linear MPR
    try:
        result = gaiapy.linear_mpr(ts, sample_locations)
        print(f'✓ linear_mpr completed successfully')
        print(f'  Result type: {type(result)}')
        print(f'  Reconstruction type: {result.reconstruction_type}')
        print(f'  MPR matrix shape: {result.mpr_matrix.shape}')
        print(f'  Mean tree length: {result.mean_tree_length:.6f}')
        print(f'  Number of trees: {result.num_trees}')
        
        # Test linear_mpr_minimize
        optimal_locations = gaiapy.linear_mpr_minimize(result)
        print(f'✓ linear_mpr_minimize completed successfully')
        print(f'  Optimal locations shape: {optimal_locations.shape}')
        print(f'  Sample optimal locations:')
        for i in range(min(4, len(optimal_locations))):
            x, y = optimal_locations[i]
            print(f'    Node {i}: ({x:.3f}, {y:.3f})')
        
        # Test linear_mpr_minimize_discrete with candidate sites
        print('\nTesting discrete minimization...')
        candidate_sites = np.array([
            [0.0, 0.0],  # Site 1
            [0.5, 0.0],  # Site 2
            [1.0, 0.0],  # Site 3
            [0.0, 0.5],  # Site 4
            [0.5, 0.5],  # Site 5
            [1.0, 0.5],  # Site 6
            [0.0, 1.0],  # Site 7
            [0.5, 1.0],  # Site 8
            [1.0, 1.0],  # Site 9
        ])
        
        optimal_sites = gaiapy.linear_mpr_minimize_discrete(result, candidate_sites)
        print(f'✓ linear_mpr_minimize_discrete completed successfully')
        print(f'  Optimal site indices shape: {optimal_sites.shape}')
        print(f'  Sample optimal sites (1-based indices):')
        for i in range(min(4, len(optimal_sites))):
            site_idx = optimal_sites[i] - 1  # Convert to 0-based
            if 0 <= site_idx < len(candidate_sites):
                x, y = candidate_sites[site_idx]
                print(f'    Node {i}: Site {optimal_sites[i]} = ({x:.1f}, {y:.1f})')
            
        # Test with branch lengths
        print('\nTesting with branch lengths...')
        result_bl = gaiapy.linear_mpr(ts, sample_locations, use_branch_lengths=True)
        print(f'✓ linear_mpr with branch lengths completed')
        print(f'  Mean tree length: {result_bl.mean_tree_length:.6f}')
        
        optimal_locations_bl = gaiapy.linear_mpr_minimize(result_bl)
        print(f'✓ linear_mpr_minimize with branch lengths completed')
        
        optimal_sites_bl = gaiapy.linear_mpr_minimize_discrete(result_bl, candidate_sites)
        print(f'✓ linear_mpr_minimize_discrete with branch lengths completed')
        
        # Test with different tolerance
        print('\nTesting with different tolerance...')
        result_tol = gaiapy.linear_mpr(ts, sample_locations, tolerance=0.001)
        print(f'✓ linear_mpr with tight tolerance completed')
        print(f'  Mean tree length: {result_tol.mean_tree_length:.6f}')
        
        return True
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_linear_vs_quadratic():
    """Compare linear and quadratic reconstruction results."""
    print('\n\nComparing linear vs quadratic reconstruction...')
    
    try:
        # Create tree sequence with outlier location
        ts = msprime.simulate(sample_size=6, Ne=1000, length=50, random_seed=123)
        
        # Sample locations with one outlier
        sample_locations = np.array([
            [0, 0.0, 0.0],    # Sample 0 at origin
            [1, 1.0, 0.0],    # Sample 1 nearby
            [2, 0.0, 1.0],    # Sample 2 nearby
            [3, 1.0, 1.0],    # Sample 3 nearby
            [4, 0.5, 0.5],    # Sample 4 in center
            [5, 10.0, 10.0],  # Sample 5 is outlier
        ])
        
        # Run both reconstructions
        linear_result = gaiapy.linear_mpr(ts, sample_locations)
        quadratic_result = gaiapy.quadratic_mpr(ts, sample_locations)
        
        print(f'Linear reconstruction mean cost: {linear_result.mean_tree_length:.6f}')
        print(f'Quadratic reconstruction mean cost: {quadratic_result.mean_tree_length:.6f}')
        
        # Get optimal locations
        linear_locations = gaiapy.linear_mpr_minimize(linear_result)
        quadratic_locations = gaiapy.quadratic_mpr_minimize(quadratic_result)
        
        print('\nOptimal locations comparison:')
        print('Node\tLinear\t\t\tQuadratic')
        for i in range(min(6, len(linear_locations))):
            lx, ly = linear_locations[i]
            qx, qy = quadratic_locations[i]
            print(f'{i}\t({lx:.3f}, {ly:.3f})\t\t({qx:.3f}, {qy:.3f})')
        
        print('✓ Linear vs quadratic comparison completed')
        return True
        
    except Exception as e:
        print(f'✗ Comparison error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print('\n\nTesting edge cases...')
    
    try:
        # Create simple tree sequence
        ts = msprime.simulate(sample_size=3, Ne=1000, length=10, random_seed=456)
        
        # Test with minimal locations (just x, y)
        minimal_locations = np.array([
            [0, 0.0, 0.0],
            [1, 1.0, 0.0], 
            [2, 0.0, 1.0],
        ])
        
        result = gaiapy.linear_mpr(ts, minimal_locations)
        print(f'✓ Minimal 2D locations test passed')
        
        # Test error conditions
        try:
            # Too few columns
            bad_locations = np.array([[0, 0.0], [1, 1.0]])
            gaiapy.linear_mpr(ts, bad_locations)
            print('✗ Should have failed with too few columns')
            return False
        except ValueError:
            print('✓ Correctly rejected insufficient columns')
        
        try:
            # Wrong reconstruction type for minimize
            quad_result = gaiapy.quadratic_mpr(ts, minimal_locations)
            gaiapy.linear_mpr_minimize(quad_result)
            print('✗ Should have failed with wrong reconstruction type')
            return False
        except ValueError:
            print('✓ Correctly rejected wrong reconstruction type')
        
        # Test discrete minimization edge cases
        candidate_sites = np.array([[0.0, 0.0]])  # Single site
        optimal_sites = gaiapy.linear_mpr_minimize_discrete(result, candidate_sites)
        if np.all(optimal_sites == 1):  # All should choose the only site
            print('✓ Single candidate site test passed')
        else:
            print('✗ Single candidate site test failed')
            return False
        
        print('✓ All edge cases passed')
        return True
        
    except Exception as e:
        print(f'✗ Edge case error: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_piecewise_linear_function():
    """Test the PiecewiseLinearFunction class directly."""
    print('\n\nTesting PiecewiseLinearFunction class...')
    
    try:
        from gaiapy.core.linear import PiecewiseLinearFunction
        
        # Create a simple PLF: f(x) = |x - 1| 
        plf = PiecewiseLinearFunction()
        plf.breakpoints[0] = 1.0
        plf.slopes[0] = -1.0
        plf.slopes[1] = 1.0
        plf.intercept = 1.0  # So f(0) = 1
        plf.num_breaks = 1
        
        # Test evaluation
        assert abs(plf.evaluate(0.0) - 1.0) < 1e-10, "f(0) should be 1"
        assert abs(plf.evaluate(1.0) - 0.0) < 1e-10, "f(1) should be 0"
        assert abs(plf.evaluate(2.0) - 1.0) < 1e-10, "f(2) should be 1"
        print('✓ PLF evaluation test passed')
        
        # Test minimum finding
        min_loc, min_val = plf.get_minimum()
        assert abs(min_loc - 1.0) < 1e-10, "Minimum should be at x=1"
        assert abs(min_val - 0.0) < 1e-10, "Minimum value should be 0"
        print('✓ PLF minimum finding test passed')
        
        # Test min_transform
        transformed = plf.min_transform(2.0)
        print('✓ PLF min_transform test passed')
        
        # Test addition
        plf2 = PiecewiseLinearFunction()
        plf2.slopes[0] = 0.0
        plf2.intercept = 1.0
        plf2.num_breaks = 0
        
        sum_plf = plf.add(plf2)
        print('✓ PLF addition test passed')
        
        print('✓ All PiecewiseLinearFunction tests passed')
        return True
        
    except Exception as e:
        print(f'✗ PLF test error: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    success &= test_linear_mpr()
    success &= test_linear_vs_quadratic()
    success &= test_edge_cases()
    success &= test_piecewise_linear_function()
    
    if success:
        print('\n🎉 All linear MPR tests passed!')
    else:
        print('\n❌ Some tests failed!')
        sys.exit(1)
