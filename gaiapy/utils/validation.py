"""
Input validation utilities for gaiapy.

This module provides functions to validate inputs to the main API functions,
ensuring that tree sequences, sample locations, and other parameters are
properly formatted and consistent.
"""

import numpy as np
import tskit
from typing import Union


def validate_tree_sequence(ts: tskit.TreeSequence) -> None:
    """
    Validate that the input is a proper tree sequence.
    
    Args:
        ts: Object to validate
        
    Raises:
        TypeError: If ts is not a TreeSequence
        ValueError: If tree sequence is empty or invalid
    """
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("ts must be a tskit.TreeSequence object")
    
    if ts.num_nodes == 0:
        raise ValueError("Tree sequence must contain at least one node")
    
    if ts.num_samples == 0:
        raise ValueError("Tree sequence must contain at least one sample")
    
    if ts.sequence_length <= 0:
        raise ValueError("Tree sequence must have positive sequence length")


def validate_sample_locations(sample_locations: np.ndarray, 
                            max_node_id: int) -> None:
    """
    Validate sample location data.
    
    Args:
        sample_locations: Array with columns [node_id, state_id] or 
                         [node_id, x_coord, y_coord]
        max_node_id: Maximum valid node ID (exclusive)
        
    Raises:
        ValueError: If sample locations are invalid
    """
    if sample_locations.ndim != 2:
        raise ValueError("sample_locations must be a 2D array")
    
    if sample_locations.shape[0] == 0:
        raise ValueError("sample_locations must contain at least one sample")
    
    if sample_locations.shape[1] < 2:
        raise ValueError("sample_locations must have at least 2 columns")
    
    # Check node IDs are valid
    node_ids = sample_locations[:, 0]
    if not np.all(np.isfinite(node_ids)):
        raise ValueError("Node IDs must be finite")
    
    if not np.all(node_ids >= 0):
        raise ValueError("Node IDs must be non-negative")
    
    if not np.all(node_ids < max_node_id):
        raise ValueError(f"Node IDs must be less than {max_node_id}")
    
    if not np.all(node_ids == np.round(node_ids)):
        raise ValueError("Node IDs must be integers")
    
    # Check for duplicate node IDs
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Node IDs must be unique")
    
    # For discrete case (2 columns), check state IDs
    if sample_locations.shape[1] == 2:
        state_ids = sample_locations[:, 1]
        if not np.all(np.isfinite(state_ids)):
            raise ValueError("State IDs must be finite")
        
        if not np.all(state_ids >= 0):
            raise ValueError("State IDs must be non-negative")
        
        if not np.all(state_ids == np.round(state_ids)):
            raise ValueError("State IDs must be integers")
    
    # For continuous case (3+ columns), check coordinates
    elif sample_locations.shape[1] >= 3:
        coords = sample_locations[:, 1:]
        if not np.all(np.isfinite(coords)):
            raise ValueError("Coordinates must be finite")


def validate_cost_matrix(cost_matrix: np.ndarray) -> None:
    """
    Validate a cost matrix for discrete parsimony.
    
    Args:
        cost_matrix: Square symmetric matrix of migration costs
        
    Raises:
        ValueError: If cost matrix is invalid
    """
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be a 2D array")
    
    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be square")
    
    if cost_matrix.shape[0] == 0:
        raise ValueError("cost_matrix must have at least one state")
    
    if not np.all(np.isfinite(cost_matrix)):
        raise ValueError("cost_matrix must have finite values")
    
    if not np.all(cost_matrix >= 0):
        raise ValueError("cost_matrix must have non-negative values")
    
    if not np.allclose(cost_matrix, cost_matrix.T):
        raise ValueError("cost_matrix must be symmetric")


def validate_adjacency_matrix(adjacency_matrix: np.ndarray,
                             num_states: int) -> None:
    """
    Validate an adjacency matrix for migration constraints.
    
    Args:
        adjacency_matrix: Binary matrix specifying allowed transitions
        num_states: Expected number of states
        
    Raises:
        ValueError: If adjacency matrix is invalid
    """
    if adjacency_matrix.ndim != 2:
        raise ValueError("adjacency_matrix must be a 2D array")
    
    if adjacency_matrix.shape != (num_states, num_states):
        raise ValueError(f"adjacency_matrix must be {num_states}x{num_states}")
    
    if not np.all(np.isin(adjacency_matrix, [0, 1])):
        raise ValueError("adjacency_matrix must contain only 0s and 1s")
    
    if not np.allclose(adjacency_matrix, adjacency_matrix.T):
        raise ValueError("adjacency_matrix must be symmetric")
