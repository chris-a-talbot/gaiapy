"""
Main API for gaiapy - Geographic Ancestry Inference Algorithm.

This module provides the user-facing functions that mirror the functionality
of the original GAIA R package but adapted for Python and tskit.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
import tskit

from .core.discrete import discrete_mpr as _discrete_mpr
from .core.discrete import discrete_mpr_minimize as _discrete_mpr_minimize
from .core.discrete import discrete_mpr_edge_history as _discrete_mpr_edge_history
from .core.quadratic import quadratic_mpr as _quadratic_mpr
from .core.quadratic import quadratic_mpr_minimize as _quadratic_mpr_minimize
from .core.quadratic import quadratic_mpr_minimize_discrete as _quadratic_mpr_minimize_discrete
from .core.linear import linear_mpr as _linear_mpr
from .core.linear import linear_mpr_minimize as _linear_mpr_minimize
from .core.linear import linear_mpr_minimize_discrete as _linear_mpr_minimize_discrete
from .core.ancestry import discrete_mpr_ancestry as _discrete_mpr_ancestry
from .core.ancestry import discrete_mpr_ancestry_flux as _discrete_mpr_ancestry_flux
from .core.sankoff import MPRResult
from .utils.validation import validate_tree_sequence, validate_sample_locations
from .utils.metadata import (
    extract_sample_locations_from_metadata,
    augment_tree_sequence_with_locations,
    merge_location_sources
)


def discrete_mpr(ts: tskit.TreeSequence,
                sample_locations: Union[np.ndarray, list],
                cost_matrix: Union[np.ndarray, list],
                use_branch_lengths: bool = False) -> MPRResult:
    """
    Compute minimum migration costs for discrete geographic ancestry reconstruction.
    
    Uses generalized (Sankoff) parsimony to compute the minimum migration costs
    needed to explain sampled geographic locations under different possible
    ancestral state assignments.
    
    Args:
        ts: A tree sequence object loaded via tskit.load()
        sample_locations: Array-like with shape (n_samples, 2) containing:
                         - Column 0: Node IDs for sampled genomes (0-based)
                         - Column 1: Geographic state assignments (0-based)
        cost_matrix: Symmetric matrix where entry [i,j] gives the migration
                    cost between states i and j. Must have non-negative values.
                    Diagonal elements are ignored.
        use_branch_lengths: Whether to scale migration costs by inverse
                           branch lengths (True) or treat all branches equally (False)
    
    Returns:
        MPRResult object containing:
        - mpr_matrix: Matrix where entry [i,j] gives minimum migration cost
                     to explain sample locations when node i is in state j
        - tree_lengths: Array of migration costs for each local tree
        - mean_tree_length: Genome-wide average migration cost
        - node_weights: Genomic span weights for each node
    
    Examples:
        >>> import gaiapy
        >>> import tskit
        >>> import numpy as np
        >>> 
        >>> # Load tree sequence
        >>> ts = tskit.load("example.trees")
        >>> 
        >>> # Define sample locations (0-based indexing)
        >>> samples = np.array([
        ...     [0, 0],  # node 0 in state 0
        ...     [1, 0],  # node 1 in state 0  
        ...     [2, 1],  # node 2 in state 1
        ... ])
        >>> 
        >>> # Create cost matrix (cost 1 to migrate between states)
        >>> costs = np.array([[0, 1], [1, 0]])
        >>> 
        >>> # Compute MPR
        >>> result = gaiapy.discrete_mpr(ts, samples, costs)
    
    Notes:
        This implements the discrete Sankoff parsimony algorithm from the original
        GAIA C implementation in src/treeseq_sankoff_discrete.c. The algorithm
        performs a two-pass traversal (downpass and uppass) on each tree to
        compute minimum migration costs.
    """
    
    # Convert inputs to numpy arrays
    sample_locations = np.asarray(sample_locations)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    
    # Validate inputs
    validate_tree_sequence(ts)
    validate_sample_locations(sample_locations, ts.num_nodes)
    
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be a square matrix")
    
    if np.any(cost_matrix < 0):
        raise ValueError("cost_matrix must have non-negative values")
    
    if not np.allclose(cost_matrix, cost_matrix.T):
        raise ValueError("cost_matrix must be symmetric")
    
    # Ensure diagonal is zero
    np.fill_diagonal(cost_matrix, 0)
    
    # Call implementation
    return _discrete_mpr(ts, sample_locations, cost_matrix, use_branch_lengths)


def discrete_mpr_minimize(mpr_result: MPRResult,
                         return_zero_based: bool = True) -> np.ndarray:
    """
    Determine optimal geographic states from minimum migration costs.
    
    Uses the migration costs computed by discrete_mpr() to identify the
    optimal geographic state for each ancestral node. When multiple states
    achieve the minimum cost, one is chosen randomly.
    
    Args:
        mpr_result: Result object from discrete_mpr()
        return_zero_based: Whether to return 0-based state indices (True)
                          or 1-based indices (False, for R compatibility)
    
    Returns:
        Array of optimal geographic state assignments for each node
    
    Examples:
        >>> # Continuing from discrete_mpr example
        >>> states = gaiapy.discrete_mpr_minimize(result)
        >>> print(f"Node 3 optimal state: {states[3]}")
    """
    
    return _discrete_mpr_minimize(mpr_result, return_zero_based)


def discrete_mpr_edge_history(ts: tskit.TreeSequence,
                             mpr_result: MPRResult,
                             cost_matrix: Union[np.ndarray, list],
                             adjacency_matrix: Optional[Union[np.ndarray, list]] = None,
                             return_zero_based: bool = True) -> Dict[str, Any]:
    """
    Sample migration paths for each edge in a tree sequence.
    
    For each edge in the tree sequence, samples a minimum-cost migration path
    between the geographic states of parent and child nodes.
    
    Args:
        ts: Tree sequence object
        mpr_result: Result object from discrete_mpr()
        cost_matrix: Migration cost matrix used in reconstruction
        adjacency_matrix: Optional binary matrix specifying allowed transitions.
                         Entry [i,j] should be 1 if direct transitions are allowed
                         between states i and j, 0 otherwise. If None, all
                         transitions are allowed.
        return_zero_based: Whether to use 0-based state indexing
    
    Returns:
        Dictionary containing:
        - 'paths': List of migration paths for each edge
        - 'node_states': Optimal state assignments for all nodes  
        - 'edge_costs': Total migration cost for each edge
    
    Examples:
        >>> # Get detailed migration histories
        >>> history = gaiapy.discrete_mpr_edge_history(ts, result, costs)
        >>> print(f"Found {len(history['paths'])} migration paths")
    
    Notes:
        Based on the algorithm in src/treeseq_sankoff_discrete_history.c which
        uses Dijkstra's algorithm to reconstruct minimum-cost migration paths
        between ancestral states.
    """
    
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if adjacency_matrix is not None:
        adjacency_matrix = np.asarray(adjacency_matrix)
    
    return _discrete_mpr_edge_history(
        ts, mpr_result, cost_matrix, adjacency_matrix, return_zero_based
    )


def discrete_mpr_ancestry(ts: tskit.TreeSequence,
                         mpr_result: MPRResult,
                         time_bins: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate ancestry coefficients through time.
    
    Computes the proportion of genetic ancestry from each geographic state
    at different time points in the past.
    
    Args:
        ts: Tree sequence object
        mpr_result: Result from discrete_mpr()
        time_bins: Time points at which to calculate ancestry coefficients.
                  If None, uses natural time points from the tree sequence.
    
    Returns:
        Dictionary containing ancestry coefficients over time
    
    Notes:
        Implements the ancestry coefficient algorithm from 
        src/treeseq_sankoff_discrete_ancestry.c which tracks lineage
        proportions backward through time.
    """
    
    return _discrete_mpr_ancestry(ts, mpr_result, time_bins)


def discrete_mpr_ancestry_flux(ts: tskit.TreeSequence,
                              mpr_result: MPRResult,
                              time_bins: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate migration flux between regions over time.
    
    Computes the rate of migration between different geographic states
    across different time periods.
    
    Args:
        ts: Tree sequence object  
        mpr_result: Result from discrete_mpr()
        time_bins: Time points at which to calculate migration flux.
                  If None, uses natural time points from the tree sequence.
    
    Returns:
        Dictionary containing migration flux matrices over time
    
    Notes:
        Based on the flux computation algorithm in src/treeseq_sankoff_discrete_flux.c
        which analyzes state transitions along genealogical branches.
    """
    
    return _discrete_mpr_ancestry_flux(ts, mpr_result, time_bins)


def quadratic_mpr(ts: tskit.TreeSequence,
                 sample_locations: Union[np.ndarray, list],
                 use_branch_lengths: bool = False) -> MPRResult:
    """
    Compute continuous space reconstruction using squared distances.
    
    Uses generalized parsimony to infer ancestral locations in continuous
    space by minimizing the sum of squared Euclidean distances.
    
    Args:
        ts: Tree sequence object
        sample_locations: Array-like with shape (n_samples, 3) containing:
                         - Column 0: Node IDs (0-based)
                         - Column 1: X coordinates
                         - Column 2: Y coordinates  
        use_branch_lengths: Whether to weight distances by branch lengths
    
    Returns:
        MPRResult object for continuous space reconstruction
    
    Notes:
        Implements the squared-distance parsimony algorithm from
        src/treeseq_sankoff_quadratic.c based on Maddison (1991).
        The algorithm maintains quadratic cost functions for each node.
    """
    
    # Convert inputs to numpy arrays
    sample_locations = np.asarray(sample_locations)
    
    # Validate inputs
    validate_tree_sequence(ts)
    validate_sample_locations(sample_locations, ts.num_nodes)
    
    if sample_locations.shape[1] < 3:
        raise ValueError("sample_locations must have at least 3 columns for continuous reconstruction")
    
    return _quadratic_mpr(ts, sample_locations, use_branch_lengths)


def quadratic_mpr_minimize(mpr_result: MPRResult) -> np.ndarray:
    """
    Find optimal continuous locations from quadratic MPR results.
    
    Args:
        mpr_result: Result from quadratic_mpr()
    
    Returns:
        Array of optimal (x, y) coordinates for each node
    """
    
    return _quadratic_mpr_minimize(mpr_result)


def quadratic_mpr_minimize_discrete(mpr_result: MPRResult, 
                                   candidate_sites: Union[np.ndarray, list]) -> np.ndarray:
    """
    Assign ancestral nodes to discrete locations using quadratic parsimony costs.
    
    Uses the squared distance costs computed by quadratic_mpr() to assign 
    each ancestral node to the discrete location (from a provided set) that 
    minimizes the total squared distance cost needed to explain sample locations.
    
    Args:
        mpr_result: Result object from quadratic_mpr()
        candidate_sites: Array-like with shape (n_sites, n_dims) where each row 
                        represents a possible location with columns corresponding 
                        to spatial dimensions (x, y, etc.) in the same order as 
                        used in the original sample_locations
    
    Returns:
        Array of 1-based indices into candidate_sites for each node (R compatibility)
    
    Examples:
        >>> # Continuing from quadratic_mpr example
        >>> sites = np.array([[0, 0], [1, 1], [2, 2]])
        >>> indices = gaiapy.quadratic_mpr_minimize_discrete(result, sites)
        >>> print(f"Node 3 assigned to site: {indices[3]}")
    
    Notes:
        This function evaluates the quadratic cost function at each candidate
        location and selects the one with minimum cost. Indices are 1-based
        for compatibility with the R implementation.
    """
    candidate_sites = np.asarray(candidate_sites)
    return _quadratic_mpr_minimize_discrete(mpr_result, candidate_sites)


def linear_mpr(ts: tskit.TreeSequence,
              sample_locations: Union[np.ndarray, list],
              use_branch_lengths: bool = False,
              tolerance: float = 0.01) -> MPRResult:
    """
    Compute continuous space reconstruction using absolute distances.
    
    Uses generalized parsimony to infer ancestral locations in continuous
    space by minimizing the sum of Manhattan (L1) distances.
    
    Args:
        ts: Tree sequence object
        sample_locations: Array-like with shape (n_samples, 3) containing:
                         - Column 0: Node IDs (0-based)
                         - Column 1: X coordinates
                         - Column 2: Y coordinates
        use_branch_lengths: Whether to weight distances by branch lengths
        tolerance: Tolerance for merging breakpoints in piecewise linear functions
    
    Returns:
        MPRResult object for continuous space reconstruction
    
    Notes:
        Implements the linear distance parsimony algorithm from
        src/treeseq_sankoff_linear.c based on Csurös (2008). The algorithm
        maintains piecewise linear cost functions for efficient computation.
    """
    
    # Convert inputs to numpy arrays
    sample_locations = np.asarray(sample_locations)
    
    # Validate inputs
    validate_tree_sequence(ts)
    validate_sample_locations(sample_locations, ts.num_nodes)
    
    if sample_locations.shape[1] < 3:
        raise ValueError("sample_locations must have at least 3 columns for continuous reconstruction")
    
    return _linear_mpr(ts, sample_locations, use_branch_lengths, tolerance)


def linear_mpr_minimize(mpr_result: MPRResult) -> np.ndarray:
    """
    Find optimal continuous locations from linear MPR results.
    
    Args:
        mpr_result: Result from linear_mpr()
    
    Returns:
        Array of optimal (x, y) coordinates for each node
    """
    
    return _linear_mpr_minimize(mpr_result)


def linear_mpr_minimize_discrete(mpr_result: MPRResult, 
                                candidate_sites: Union[np.ndarray, list]) -> np.ndarray:
    """
    Find optimal discrete locations from candidate sites using linear MPR results.
    
    Evaluates the piecewise linear cost functions at each candidate site and
    returns the site that minimizes the total cost for each node.
    
    Args:
        mpr_result: Result from linear_mpr()
        candidate_sites: Array of candidate locations with shape (n_sites, n_dims)
                        where each row is a potential location (x, y, ...)
    
    Returns:
        Array of 1-based indices into candidate_sites for each node (R compatibility)
    
    Examples:
        >>> import gaiapy
        >>> import numpy as np
        >>> 
        >>> # After running linear_mpr...
        >>> result = gaiapy.linear_mpr(ts, sample_locations)
        >>> 
        >>> # Define candidate sites
        >>> sites = np.array([
        ...     [0.0, 0.0],  # Site 1
        ...     [1.0, 0.0],  # Site 2  
        ...     [0.0, 1.0],  # Site 3
        ...     [1.0, 1.0],  # Site 4
        ... ])
        >>> 
        >>> # Find optimal discrete locations
        >>> optimal_sites = gaiapy.linear_mpr_minimize_discrete(result, sites)
        >>> print(f"Node 3 optimal site: {optimal_sites[3]}")  # 1-based index
    
    Notes:
        Returns 1-based indices for compatibility with R. Subtract 1 to get
        0-based Python indices. This function implements the discrete minimization
        from src/treeseq_sankoff_linear.c which evaluates piecewise linear functions
        at discrete candidate locations.
    """
    
    candidate_sites = np.asarray(candidate_sites)
    return _linear_mpr_minimize_discrete(mpr_result, candidate_sites)


# Enhanced functions that support metadata input/output

def discrete_mpr_with_metadata(ts: tskit.TreeSequence,
                              sample_locations: Optional[Union[np.ndarray, list]] = None,
                              cost_matrix: Union[np.ndarray, list] = None,
                              use_branch_lengths: bool = False,
                              location_key: str = "location",
                              return_augmented_ts: bool = True) -> Union[MPRResult, Tuple[MPRResult, tskit.TreeSequence]]:
    """
    Discrete MPR with support for metadata input and augmented tree sequence output.
    
    Enhanced version of discrete_mpr() that can read sample locations from tree
    sequence metadata and optionally return an augmented tree sequence with
    inferred ancestral states.
    
    Args:
        ts: Tree sequence object
        sample_locations: Sample location array, or None to read from metadata
        cost_matrix: Migration cost matrix
        use_branch_lengths: Whether to weight costs by branch lengths
        location_key: Metadata key for location information (if reading from metadata)
        return_augmented_ts: Whether to return augmented tree sequence with inferred states
    
    Returns:
        If return_augmented_ts=False: MPRResult object
        If return_augmented_ts=True: Tuple of (MPRResult, augmented TreeSequence)
    
    Examples:
        >>> # Read locations from metadata
        >>> result, ts_with_states = gaiapy.discrete_mpr_with_metadata(
        ...     ts, cost_matrix=costs, location_key="geographic_state"
        ... )
        >>> 
        >>> # Use external locations but return augmented tree sequence
        >>> result, ts_augmented = gaiapy.discrete_mpr_with_metadata(
        ...     ts, sample_locations=samples, cost_matrix=costs
        ... )
    
    Notes:
        This function provides a streamlined workflow for discrete ancestry
        inference with tree sequence integration. When return_augmented_ts=True,
        the returned tree sequence contains inferred states for all nodes.
    """
    raise NotImplementedError("discrete_mpr_with_metadata not yet implemented")


def quadratic_mpr_with_metadata(ts: tskit.TreeSequence,
                               sample_locations: Optional[Union[np.ndarray, list]] = None,
                               use_branch_lengths: bool = False,
                               location_key: str = "location",
                               coordinates_key: str = "coordinates",
                               return_augmented_ts: bool = True) -> Union[MPRResult, Tuple[MPRResult, tskit.TreeSequence]]:
    """
    Quadratic MPR with metadata support for continuous space reconstruction.
    
    Enhanced version of quadratic_mpr() that can read sample coordinates from
    tree sequence metadata and return an augmented tree sequence with inferred
    ancestral locations.
    
    Args:
        ts: Tree sequence object
        sample_locations: Sample coordinate array, or None to read from metadata
        use_branch_lengths: Whether to weight distances by branch lengths
        location_key: Metadata key for location information
        coordinates_key: Key for coordinate arrays in nested metadata
        return_augmented_ts: Whether to return augmented tree sequence
    
    Returns:
        If return_augmented_ts=False: MPRResult object
        If return_augmented_ts=True: Tuple of (MPRResult, augmented TreeSequence)
    
    Examples:
        >>> # Read coordinates from individual metadata
        >>> result, ts_with_coords = gaiapy.quadratic_mpr_with_metadata(
        ...     ts, location_key="sampling_location", coordinates_key="coordinates"
        ... )
        >>> 
        >>> # Access inferred locations from augmented tree sequence
        >>> for node in ts_with_coords.nodes():
        ...     if node.metadata and "inferred_location" in node.metadata:
        ...         print(f"Node {node.id}: {node.metadata['inferred_location']}")
    
    Notes:
        Particularly useful for continuous space ancestry inference where
        sample coordinates are stored as part of individual metadata.
        The augmented tree sequence contains inferred (x,y) coordinates
        for all ancestral nodes.
        
        Based on the quadratic parsimony implementation in 
        src/treeseq_sankoff_quadratic.c with enhanced metadata integration.
    """
    raise NotImplementedError("quadratic_mpr_with_metadata not yet implemented")


def linear_mpr_with_metadata(ts: tskit.TreeSequence,
                            sample_locations: Optional[Union[np.ndarray, list]] = None,
                            use_branch_lengths: bool = False,
                            location_key: str = "location",
                            coordinates_key: str = "coordinates",
                            return_augmented_ts: bool = True) -> Union[MPRResult, Tuple[MPRResult, tskit.TreeSequence]]:
    """
    Linear MPR with metadata support for robust continuous space reconstruction.
    
    Enhanced version of linear_mpr() that provides metadata integration and
    augmented tree sequence output for Manhattan distance-based ancestry inference.
    
    Args:
        ts: Tree sequence object
        sample_locations: Sample coordinate array, or None to read from metadata
        use_branch_lengths: Whether to weight distances by branch lengths
        location_key: Metadata key for location information
        coordinates_key: Key for coordinate arrays in nested metadata
        return_augmented_ts: Whether to return augmented tree sequence
    
    Returns:
        If return_augmented_ts=False: MPRResult object
        If return_augmented_ts=True: Tuple of (MPRResult, augmented TreeSequence)
    
    Examples:
        >>> # Robust reconstruction from metadata
        >>> result, ts_robust = gaiapy.linear_mpr_with_metadata(
        ...     ts, location_key="collection_site"
        ... )
        >>> 
        >>> # Compare with quadratic reconstruction
        >>> quad_result, ts_quad = gaiapy.quadratic_mpr_with_metadata(ts)
        >>> # Linear reconstruction is more robust to outliers
    
    Notes:
        Linear (Manhattan) distance parsimony is more robust to outlier
        locations than quadratic distance, making it suitable for datasets
        with potential location errors or extreme geographic distributions.
        
        Implements the piecewise linear algorithm from src/treeseq_sankoff_linear.c
        based on Csurös (2008) with full metadata integration.
    """
    raise NotImplementedError("linear_mpr_with_metadata not yet implemented")


def extract_mpr_summary(ts: tskit.TreeSequence,
                       mpr_result: MPRResult,
                       include_node_details: bool = True) -> Dict[str, Any]:
    """
    Extract comprehensive summary of MPR reconstruction results.
    
    Provides detailed analysis of ancestry reconstruction including tree-by-tree
    breakdown, node-level statistics, and genome-wide summaries.
    
    Args:
        ts: Tree sequence object
        mpr_result: Result from any MPR function
        include_node_details: Whether to include per-node statistics
    
    Returns:
        Dictionary containing:
        - 'genome_wide_cost': Total reconstruction cost across genome
        - 'tree_costs': Reconstruction cost for each local tree
        - 'mean_cost_per_tree': Average cost per tree
        - 'cost_variance': Variance in costs across trees
        - 'node_statistics': Per-node reconstruction statistics (if requested)
        - 'convergence_metrics': Algorithm convergence information
    
    Notes:
        Useful for assessing reconstruction quality and identifying regions
        of high uncertainty or complex ancestry patterns.
    """
    raise NotImplementedError("extract_mpr_summary not yet implemented") 