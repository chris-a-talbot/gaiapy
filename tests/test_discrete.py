"""
Tests for discrete parsimony functionality.
"""

import numpy as np
import pytest
import tskit
import msprime
import sys
import os

# Add parent directory to path to import gaiapy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import gaiapy


@pytest.fixture
def simple_tree_sequence():
    """Create a simple tree sequence for testing."""
    return msprime.simulate(
        sample_size=4,
        Ne=1000,
        length=100,
        recombination_rate=1e-8,
        random_seed=42
    )


@pytest.fixture  
def sample_locations():
    """Sample locations for testing."""
    return np.array([
        [0, 0],  # Sample 0 in state 0
        [1, 0],  # Sample 1 in state 0
        [2, 1],  # Sample 2 in state 1
        [3, 1],  # Sample 3 in state 1
    ])


@pytest.fixture
def cost_matrix():
    """Simple cost matrix for testing."""
    return np.array([
        [0, 1],
        [1, 0]
    ])


def test_discrete_mpr_basic(simple_tree_sequence, sample_locations, cost_matrix):
    """Test basic discrete MPR functionality."""
    
    result = gaiapy.discrete_mpr(simple_tree_sequence, sample_locations, cost_matrix)
    
    # Check result structure
    assert isinstance(result, gaiapy.MPRResult)
    assert result.reconstruction_type == "discrete"
    assert result.num_nodes == simple_tree_sequence.num_nodes
    assert result.num_trees == simple_tree_sequence.num_trees
    
    # Check MPR matrix shape
    assert result.mpr_matrix.shape == (simple_tree_sequence.num_nodes, 2)
    
    # Check that costs are non-negative
    assert np.all(result.mpr_matrix >= 0)
    
    # Check that mean tree length is reasonable
    assert result.mean_tree_length >= 0
    assert np.isfinite(result.mean_tree_length)


def test_discrete_mpr_minimize(simple_tree_sequence, sample_locations, cost_matrix):
    """Test state assignment minimization."""
    
    result = gaiapy.discrete_mpr(simple_tree_sequence, sample_locations, cost_matrix)
    states = gaiapy.discrete_mpr_minimize(result)
    
    # Check output format
    assert len(states) == simple_tree_sequence.num_nodes
    assert np.all(states >= 0)
    assert np.all(states < 2)  # Should be 0 or 1
    
    # Sample nodes should have their assigned states
    assert states[0] == 0  # Sample 0 assigned to state 0
    assert states[1] == 0  # Sample 1 assigned to state 0  
    assert states[2] == 1  # Sample 2 assigned to state 1
    assert states[3] == 1  # Sample 3 assigned to state 1


def test_discrete_mpr_with_branch_lengths(simple_tree_sequence, sample_locations, cost_matrix):
    """Test MPR with branch length scaling."""
    
    result_no_bl = gaiapy.discrete_mpr(simple_tree_sequence, sample_locations, 
                                      cost_matrix, use_branch_lengths=False)
    
    result_with_bl = gaiapy.discrete_mpr(simple_tree_sequence, sample_locations,
                                        cost_matrix, use_branch_lengths=True)
    
    # Results should be different (unless branch lengths are all equal)
    # At minimum, both should be valid
    assert isinstance(result_no_bl, gaiapy.MPRResult)
    assert isinstance(result_with_bl, gaiapy.MPRResult)
    assert np.all(np.isfinite(result_no_bl.mpr_matrix))
    assert np.all(np.isfinite(result_with_bl.mpr_matrix))


def test_discrete_mpr_edge_history(simple_tree_sequence, sample_locations, cost_matrix):
    """Test edge migration history reconstruction."""
    
    result = gaiapy.discrete_mpr(simple_tree_sequence, sample_locations, cost_matrix)
    history = gaiapy.discrete_mpr_edge_history(simple_tree_sequence, result, cost_matrix)
    
    # Check output structure
    assert 'paths' in history
    assert 'node_states' in history
    assert 'edge_costs' in history
    
    # Check node states
    assert len(history['node_states']) == simple_tree_sequence.num_nodes
    
    # Check that edge costs are non-negative
    assert np.all(history['edge_costs'] >= 0)


def test_input_validation():
    """Test input validation."""
    
    ts = msprime.simulate(sample_size=3, Ne=1000, length=100, random_seed=42)
    
    # Test invalid sample locations
    with pytest.raises(ValueError):
        # Wrong number of columns
        gaiapy.discrete_mpr(ts, np.array([[0], [1]]), np.array([[0, 1], [1, 0]]))
    
    with pytest.raises(ValueError):
        # Invalid node ID
        gaiapy.discrete_mpr(ts, np.array([[10, 0]]), np.array([[0, 1], [1, 0]]))
    
    # Test invalid cost matrix
    with pytest.raises(ValueError):
        # Non-square matrix
        gaiapy.discrete_mpr(ts, np.array([[0, 0]]), np.array([[0, 1, 2], [1, 0, 1]]))
    
    with pytest.raises(ValueError):
        # Negative costs
        gaiapy.discrete_mpr(ts, np.array([[0, 0]]), np.array([[0, -1], [-1, 0]]))


if __name__ == "__main__":
    # Run basic tests
    ts = msprime.simulate(sample_size=4, Ne=1000, length=100, random_seed=42)
    samples = np.array([[0, 0], [1, 0], [2, 1], [3, 1]])
    costs = np.array([[0, 1], [1, 0]])
    
    print("Running basic test...")
    result = gaiapy.discrete_mpr(ts, samples, costs)
    print(f"Result: {result}")
    
    states = gaiapy.discrete_mpr_minimize(result)
    print(f"States: {states}")
    
    print("Tests completed!")
