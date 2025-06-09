"""
Core Sankoff parsimony algorithm implementation.

This module contains the foundational classes and functions for performing
generalized (Sankoff) parsimony on tree sequences, which forms the basis
for all geographic ancestry inference methods in gaiapy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import tskit


@dataclass
class MPRResult:
    """
    Results from Maximum Parsimony Reconstruction (MPR) analysis.
    
    This class stores the results of ancestry reconstruction using various
    parsimony algorithms (discrete, quadratic, linear).
    
    Attributes:
        mpr_matrix: Matrix where entry [i,j] gives the cost/function parameters
                   for node i in state/location j. For discrete parsimony,
                   this contains migration costs. For continuous parsimony,
                   this contains quadratic/linear function parameters.
        tree_lengths: Array of reconstruction costs for each local tree
        mean_tree_length: Genome-wide average reconstruction cost
        node_weights: Genomic span weights for each node
        reconstruction_type: Type of reconstruction ("discrete", "quadratic", "linear")
        num_nodes: Number of nodes in the tree sequence
        num_trees: Number of trees in the tree sequence
        sample_locations: Array of sample locations with columns [node_id, x, y, ...]
        sample_node_ids: Array of sample node IDs
    """
    mpr_matrix: np.ndarray
    tree_lengths: np.ndarray
    mean_tree_length: float
    node_weights: np.ndarray
    reconstruction_type: str = "unknown"
    num_nodes: Optional[int] = None
    num_trees: Optional[int] = None
    sample_locations: Optional[np.ndarray] = None
    sample_node_ids: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Set num_nodes and num_trees if not provided."""
        if self.num_nodes is None:
            self.num_nodes = self.mpr_matrix.shape[0]
        if self.num_trees is None:
            self.num_trees = len(self.tree_lengths)


@dataclass
class TreeState:
    """
    State information for a single tree during parsimony computation.
    
    This class maintains the cost functions and tree structure information
    needed during the dynamic programming algorithm.
    
    Attributes:
        node_costs: Current node cost functions
        stem_costs: Current stem cost functions  
        parent: Parent array for the tree
        children: Children arrays for the tree
        num_samples: Number of samples below each node
        time: Node times (if using branch lengths)
    """
    node_costs: Dict[int, np.ndarray]
    stem_costs: Dict[int, np.ndarray]
    parent: np.ndarray
    children: Dict[int, list]
    num_samples: np.ndarray
    time: Optional[np.ndarray] = None


class SankoffTree:
    """
    Base class for tree-based parsimony computations.
    
    This class provides the framework for performing parsimony algorithms
    on individual trees within a tree sequence.
    """
    
    def __init__(self, ts: tskit.TreeSequence, use_branch_lengths: bool = False):
        """
        Initialize the Sankoff tree framework.
        
        Args:
            ts: Tree sequence object
            use_branch_lengths: Whether to use branch lengths in computations
        """
        self.ts = ts
        self.use_branch_lengths = use_branch_lengths
        self.num_nodes = ts.num_nodes
        self.num_samples = ts.num_samples
        
    def postorder_traversal(self, tree: tskit.Tree):
        """Get nodes in postorder traversal."""
        nodes = []
        stack = list(tree.roots)
        
        while stack:
            node = stack.pop()
            nodes.append(node)
            # Add children to stack (they'll be processed first)
            children = list(tree.children(node))
            stack.extend(reversed(children))
        
        return reversed(nodes)
    
    def preorder_traversal(self, tree: tskit.Tree):
        """Get nodes in preorder traversal."""
        nodes = []
        stack = list(tree.roots)
        
        while stack:
            node = stack.pop()
            nodes.append(node)
            # Add children to stack
            children = list(tree.children(node))
            stack.extend(reversed(children))
        
        return nodes


def compute_tree_cost_summary(trees_costs: np.ndarray, 
                            tree_spans: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for tree costs.
    
    Args:
        trees_costs: Array of costs for each tree
        tree_spans: Array of genomic spans for each tree
        
    Returns:
        Dictionary with summary statistics
    """
    total_span = np.sum(tree_spans)
    weighted_mean = np.sum(trees_costs * tree_spans) / total_span
    
    return {
        'mean_cost': np.mean(trees_costs),
        'weighted_mean_cost': weighted_mean,
        'min_cost': np.min(trees_costs),
        'max_cost': np.max(trees_costs),
        'cost_variance': np.var(trees_costs),
        'total_span': total_span
    }
