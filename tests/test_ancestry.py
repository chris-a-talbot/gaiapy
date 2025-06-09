"""
Tests for ancestry analysis functions.
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
from gaiapy.core.ancestry import AncestryTracker, MigrationFluxAnalyzer
from gaiapy.core.sankoff import MPRResult


@pytest.fixture
def simple_tree_sequence():
    """Create a simple tree sequence for testing."""
    return msprime.simulate(
        sample_size=6,
        Ne=1000,
        length=1000,
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
        [4, 2],  # Sample 4 in state 2
        [5, 2],  # Sample 5 in state 2
    ])


@pytest.fixture
def mock_mpr_result(simple_tree_sequence):
    """Create a mock MPR result for testing."""
    ts = simple_tree_sequence
    num_nodes = ts.num_nodes
    num_states = 3
    
    # Create mock MPR matrix
    mpr_matrix = np.random.rand(num_nodes, num_states)
    
    # Create mock tree lengths
    tree_lengths = np.random.rand(ts.num_trees)
    
    return MPRResult(
        mpr_matrix=mpr_matrix,
        tree_lengths=tree_lengths,
        mean_tree_length=np.mean(tree_lengths),
        node_weights=np.ones(num_nodes)
    )


class TestAncestryTracker:
    """Test the AncestryTracker class."""
    
    def test_init(self, simple_tree_sequence, mock_mpr_result):
        """Test AncestryTracker initialization."""
        tracker = AncestryTracker(simple_tree_sequence, mock_mpr_result)
        
        assert tracker.ts == simple_tree_sequence
        assert tracker.mpr_result == mock_mpr_result
        assert tracker.num_states == mock_mpr_result.mpr_matrix.shape[1]
        assert len(tracker.node_times) == simple_tree_sequence.num_nodes
    
    def test_compute_ancestry_coefficients_not_implemented(self, simple_tree_sequence, mock_mpr_result):
        """Test that ancestry coefficient computation raises NotImplementedError."""
        tracker = AncestryTracker(simple_tree_sequence, mock_mpr_result)
        
        with pytest.raises(NotImplementedError):
            tracker.compute_ancestry_coefficients()
    
    def test_compute_ancestry_trajectories_not_implemented(self, simple_tree_sequence, mock_mpr_result):
        """Test that ancestry trajectory computation raises NotImplementedError."""
        tracker = AncestryTracker(simple_tree_sequence, mock_mpr_result)
        
        with pytest.raises(NotImplementedError):
            tracker.compute_ancestry_trajectories()


class TestMigrationFluxAnalyzer:
    """Test the MigrationFluxAnalyzer class."""
    
    def test_init(self, simple_tree_sequence, mock_mpr_result):
        """Test MigrationFluxAnalyzer initialization."""
        analyzer = MigrationFluxAnalyzer(simple_tree_sequence, mock_mpr_result)
        
        assert analyzer.ts == simple_tree_sequence
        assert analyzer.mpr_result == mock_mpr_result
        assert analyzer.num_states == mock_mpr_result.mpr_matrix.shape[1]
        assert len(analyzer.node_times) == simple_tree_sequence.num_nodes
    
    def test_compute_migration_flux_not_implemented(self, simple_tree_sequence, mock_mpr_result):
        """Test that migration flux computation raises NotImplementedError."""
        analyzer = MigrationFluxAnalyzer(simple_tree_sequence, mock_mpr_result)
        
        with pytest.raises(NotImplementedError):
            analyzer.compute_migration_flux()
    
    def test_compute_effective_migration_rates_not_implemented(self, simple_tree_sequence, mock_mpr_result):
        """Test that effective migration rate computation raises NotImplementedError."""
        analyzer = MigrationFluxAnalyzer(simple_tree_sequence, mock_mpr_result)
        
        with pytest.raises(NotImplementedError):
            analyzer.compute_effective_migration_rates()


class TestAncestryFunctions:
    """Test the main ancestry analysis functions."""
    
    def test_discrete_mpr_ancestry_not_implemented(self, simple_tree_sequence, mock_mpr_result):
        """Test that discrete_mpr_ancestry raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            gaiapy.discrete_mpr_ancestry(simple_tree_sequence, mock_mpr_result)
    
    def test_discrete_mpr_ancestry_flux_not_implemented(self, simple_tree_sequence, mock_mpr_result):
        """Test that discrete_mpr_ancestry_flux raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            gaiapy.discrete_mpr_ancestry_flux(simple_tree_sequence, mock_mpr_result)
    
    def test_discrete_mpr_ancestry_with_time_bins(self, simple_tree_sequence, mock_mpr_result):
        """Test discrete_mpr_ancestry with custom time bins."""
        time_bins = np.array([0, 1000, 2000, 3000])
        
        with pytest.raises(NotImplementedError):
            gaiapy.discrete_mpr_ancestry(simple_tree_sequence, mock_mpr_result, time_bins)
    
    def test_discrete_mpr_ancestry_flux_with_adjacency(self, simple_tree_sequence, mock_mpr_result):
        """Test discrete_mpr_ancestry_flux with adjacency matrix."""
        # Create adjacency matrix allowing all migrations
        adjacency = np.ones((3, 3), dtype=int)
        np.fill_diagonal(adjacency, 0)
        
        time_bins = np.array([0, 1000, 2000])
        
        with pytest.raises(NotImplementedError):
            gaiapy.discrete_mpr_ancestry_flux(
                simple_tree_sequence, mock_mpr_result, time_bins
            )


class TestValidation:
    """Test input validation for ancestry functions."""
    
    def test_valid_inputs_ancestry(self, simple_tree_sequence, mock_mpr_result):
        """Test that valid inputs don't raise validation errors before hitting NotImplementedError."""
        # These should fail with NotImplementedError, not validation errors
        with pytest.raises(NotImplementedError):
            gaiapy.discrete_mpr_ancestry(simple_tree_sequence, mock_mpr_result)
    
    def test_valid_inputs_flux(self, simple_tree_sequence, mock_mpr_result):
        """Test that valid inputs don't raise validation errors before hitting NotImplementedError."""
        # These should fail with NotImplementedError, not validation errors
        with pytest.raises(NotImplementedError):
            gaiapy.discrete_mpr_ancestry_flux(simple_tree_sequence, mock_mpr_result)


if __name__ == "__main__":
    # Run basic tests to check that placeholders are properly set up
    ts = msprime.simulate(sample_size=4, Ne=1000, length=100, random_seed=42)
    
    # Create mock MPR result
    mpr_matrix = np.random.rand(ts.num_nodes, 3)
    tree_lengths = np.random.rand(ts.num_trees)
    mpr_result = MPRResult(
        mpr_matrix=mpr_matrix,
        tree_lengths=tree_lengths,
        mean_tree_length=np.mean(tree_lengths),
        node_weights=np.ones(ts.num_nodes)
    )
    
    print("Testing AncestryTracker...")
    tracker = AncestryTracker(ts, mpr_result)
    print(f"AncestryTracker created with {tracker.num_states} states")
    
    print("Testing MigrationFluxAnalyzer...")
    analyzer = MigrationFluxAnalyzer(ts, mpr_result)
    print(f"MigrationFluxAnalyzer created with {analyzer.num_states} states")
    
    print("Testing that functions properly raise NotImplementedError...")
    try:
        gaiapy.discrete_mpr_ancestry(ts, mpr_result)
    except NotImplementedError:
        print("discrete_mpr_ancestry properly raises NotImplementedError")
    
    try:
        gaiapy.discrete_mpr_ancestry_flux(ts, mpr_result)
    except NotImplementedError:
        print("discrete_mpr_ancestry_flux properly raises NotImplementedError")
    
    print("All ancestry placeholder tests completed!") 