"""
Quadratic (squared-distance) parsimony implementation.

This module implements Sankoff parsimony for continuous geographic locations
using squared Euclidean distances. The algorithm computes quadratic functions
that represent the minimum sum of squared distances between ancestor-descendant
pairs for different ancestral location assignments.

Based on the C implementation in src/treeseq_sankoff_quadratic.c from the original
GAIA package, which implements the algorithm described in:

Maddison, W.P. 1991. Squared-change parsimony reconstructions of ancestral states
for continuous-valued characters on a phylogenetic tree. Systematic Zoology 40(3): 304-314.
https://www.jstor.org/stable/2992324

The core algorithm maintains quadratic cost functions F(x,y) = ax² + by² + cx + dy + e
for each node, representing the minimum cost to explain descendant locations when
the node is at position (x,y).
"""

import numpy as np
from typing import Optional, Tuple, Union
import tskit

from .sankoff import MPRResult, SankoffTree


class QuadraticFunction:
    """
    Represents a multivariate quadratic function.
    
    For n dimensions, represents:
    F(x1,x2,...,xn) = p0*(x1² + x2² + ... + xn²) + p1*x1 + p2*x2 + ... + pn*xn + p_const
    
    Note: All squared terms have the same coefficient p0 (spherical quadratic).
    """
    
    def __init__(self, num_dims: int):
        """
        Initialize quadratic function.
        
        Args:
            num_dims: Number of spatial dimensions
        """
        self.num_dims = num_dims
        # Parameters: [p0, p1, p2, ..., pn, p_const]
        self.params = np.zeros(num_dims + 2)
    
    @property
    def quadratic_coeff(self) -> float:
        """Coefficient of all squared terms."""
        return self.params[0]
    
    @quadratic_coeff.setter
    def quadratic_coeff(self, value: float):
        self.params[0] = value
    
    @property
    def linear_coeffs(self) -> np.ndarray:
        """Coefficients of linear terms."""
        return self.params[1:self.num_dims+1]
    
    @linear_coeffs.setter
    def linear_coeffs(self, values: np.ndarray):
        self.params[1:self.num_dims+1] = values
    
    @property
    def constant(self) -> float:
        """Constant term."""
        return self.params[-1]
    
    @constant.setter
    def constant(self, value: float):
        self.params[-1] = value
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate function at point x."""
        if len(x) != self.num_dims:
            raise ValueError(f"Expected {self.num_dims} dimensions, got {len(x)}")
        
        quad_term = self.quadratic_coeff * np.sum(x * x)
        linear_term = np.sum(self.linear_coeffs * x)
        return quad_term + linear_term + self.constant
    
    def minimize(self) -> Tuple[np.ndarray, float]:
        """
        Find minimum of quadratic function.
        
        Returns:
            Tuple of (optimal_point, minimum_value)
        """
        if self.quadratic_coeff <= 0:
            raise ValueError("Cannot minimize quadratic with non-positive leading coefficient")
        
        # Optimal point: x_i = -p_i / (2 * p0)
        optimal_point = -self.linear_coeffs / (2 * self.quadratic_coeff)
        
        # Minimum value: substitute back into function
        min_value = self.evaluate(optimal_point)
        
        return optimal_point, min_value


class QuadraticMPR(SankoffTree):
    """
    Quadratic parsimony reconstruction for continuous locations.
    """
    
    def __init__(self, ts: tskit.TreeSequence, sample_locations: np.ndarray,
                 use_branch_lengths: bool = False):
        """
        Initialize quadratic MPR.
        
        Args:
            ts: Tree sequence
            sample_locations: Array with columns [node_id, x, y, ...]
            use_branch_lengths: Whether to use branch lengths
        """
        super().__init__(ts, use_branch_lengths)
        
        self.sample_locations = np.asarray(sample_locations)
        if self.sample_locations.shape[1] < 3:
            raise ValueError("sample_locations must have at least 3 columns (node_id, x, y)")

        # Determine dimensions and parameter count
        self.num_dims = self.sample_locations.shape[1] - 1  # Subtract node_id column
        self.num_pars = self.num_dims + 2  # p0, p1, ..., pn, p_const

        # Build dense per-node coordinate matrix x (num_nodes x num_dims)
        self.x = np.zeros((self.num_nodes, self.num_dims), dtype=float)

        # Validate that all samples have provided coordinates and populate x
        provided_nodes = self.sample_locations[:, 0].astype(int)
        if not np.all(provided_nodes == np.round(provided_nodes)):
            raise ValueError("sample_locations node_id must be integers")
        if np.any(provided_nodes < 0) or np.any(provided_nodes >= self.num_nodes):
            raise ValueError("sample_locations node_id out of range")
        if len(np.unique(provided_nodes)) != len(provided_nodes):
            raise ValueError("Duplicate node_id entries in sample_locations")

        # Check every sample in the tree sequence has coordinates
        ts_sample_ids = np.array(list(self.ts.samples()), dtype=int)
        missing = np.setdiff1d(ts_sample_ids, provided_nodes, assume_unique=False)
        if missing.size > 0:
            raise ValueError(f"Missing coordinates for sample node IDs: {missing.tolist()}")

        # Populate x for provided nodes
        self.x[provided_nodes] = self.sample_locations[:, 1:1+self.num_dims]
        
        # Initialize result storage
        self.node_weights = np.zeros(self.num_nodes)
        # Final averaged costs in R orientation: (num_pars, num_nodes)
        self.F = np.zeros((self.num_pars, self.num_nodes))
        self.tree_lengths = []
        self.tree_weights = []
    
    def compute_mpr(self) -> MPRResult:
        """
        Compute quadratic MPR for all trees.
        
        Returns:
            MPRResult object containing reconstruction results
        """
        total_weight = 0.0
        mean_tree_length = 0.0
        
        # Process each tree in the sequence
        for tree in self.ts.trees():
            tree_span = tree.interval.right - tree.interval.left
            
            # Initialize cost arrays for this tree
            g = np.zeros((self.num_nodes, self.num_pars))  # Node costs
            h = np.zeros((self.num_nodes, self.num_pars))  # Stem costs
            f = np.zeros((self.num_nodes, self.num_pars))  # Final costs
            
            # Perform sankoff algorithm for this tree
            self._sankoff_algorithm(tree, g, h, f)
            
            # Compute tree length
            tree_length = self._compute_tree_length(tree, g)
            self.tree_lengths.append(tree_length)
            self.tree_weights.append(tree_span)
            
            # Update weighted averages
            total_weight += tree_span
            mean_tree_length += tree_span * (tree_length - mean_tree_length) / total_weight
            
            # Update node weights and average final costs
            self._update_node_averages(tree, f, tree_span)
        
        return MPRResult(
            mpr_matrix=self.F.copy(),
            tree_lengths=np.array(self.tree_lengths),
            mean_tree_length=mean_tree_length,
            node_weights=self.node_weights.copy(),
            reconstruction_type="quadratic",
            num_nodes=self.num_nodes,
            num_trees=len(self.tree_lengths),
            sample_locations=self.sample_locations.copy(),
            sample_node_ids=np.array(list(self.ts.samples()), dtype=int)
        )
    
    def _sankoff_algorithm(self, tree: tskit.Tree, g: np.ndarray, h: np.ndarray, f: np.ndarray):
        """
        Run the Sankoff algorithm for a single tree.
        
        This uses a simplified approach with postorder then preorder traversal.
        """
        # Use simpler approach: postorder then preorder traversal
        self._postorder_pass(tree, g, h)
        self._preorder_pass(tree, g, h, f)
    
    def _postorder_pass(self, tree: tskit.Tree, g: np.ndarray, h: np.ndarray):
        """Postorder traversal to compute node and stem costs."""
        # Get nodes in postorder
        visited = set()
        
        def postorder_visit(node):
            if node in visited:
                return
            visited.add(node)
            
            # Visit children first
            for child in tree.children(node):
                postorder_visit(child)
                
                # Calculate stem cost for child
                self._calc_stem_cost(node, child, tree, g, h)
                
                # Add stem cost to node cost
                g[node] += h[child]
        
        # Start from roots
        for root in tree.roots:
            postorder_visit(root)
    
    def _preorder_pass(self, tree: tskit.Tree, g: np.ndarray, h: np.ndarray, f: np.ndarray):
        """Preorder traversal to compute final costs."""
        # Get nodes in preorder
        visited = set()
        
        def preorder_visit(node):
            if node in visited:
                return
            visited.add(node)
            
            # Calculate final cost for this node
            self._calc_final_cost(node, tree, g, h, f)
            
            # Visit children
            for child in tree.children(node):
                preorder_visit(child)
        
        # Start from roots
        for root in tree.roots:
            preorder_visit(root)
    
    def _calc_stem_cost(self, u: int, v: int, tree: tskit.Tree, 
                       g: np.ndarray, h: np.ndarray):
        """
        Calculate stem cost for edge u->v following the C implementation.
        """
        # Get branch length
        b = 1.0
        if self.use_branch_lengths:
            u_time = tree.time(u)
            v_time = tree.time(v)
            if u_time is not None and v_time is not None:
                b = u_time - v_time
        
        if tree.is_sample(v):
            # Sample node case: coordinates from dense x matrix
            x_v = self.x[v]
            # h_v(x) = (1/b) * ||x - x_v||²
            h[v, 0] = 1.0 / b  # Quadratic coefficient
            h[v, 1:self.num_dims+1] = -2.0 * x_v / b  # Linear coefficients
            h[v, -1] = np.sum(x_v * x_v) / b  # Constant term
        else:
            # Internal node case
            p0 = g[v, 0]
            
            denominator = b * p0 + 1
            h[v, 0] = p0 / denominator
            h[v, 1:self.num_dims+1] = g[v, 1:self.num_dims+1] / denominator
            
            # Constant term calculation
            linear_coeffs = g[v, 1:self.num_dims+1]
            linear_sq_sum = np.sum(linear_coeffs * linear_coeffs)
            h[v, -1] = g[v, -1] - linear_sq_sum / (4 * (p0 + 1/b))
    
    def _calc_final_cost(self, v: int, tree: tskit.Tree, 
                        g: np.ndarray, h: np.ndarray, f: np.ndarray):
        """
        Calculate final cost for node v following the C implementation.
        """
        u = tree.parent(v)
        
        if u == tskit.NULL:
            # Root node - final cost is same as node cost
            f[v] = g[v].copy()
            return
        
        # Get branch length
        b = 1.0
        if self.use_branch_lengths:
            u_time = tree.time(u)
            v_time = tree.time(v)
            if u_time is not None and v_time is not None:
                b = u_time - v_time
        
        # Combine parent's final cost with edge cost
        p0 = f[u, 0] - h[v, 0]
        
        denominator = b * p0 + 1
        f[v, 0] = p0 / denominator
        f[v, 1:self.num_dims+1] = (f[u, 1:self.num_dims+1] - h[v, 1:self.num_dims+1]) / denominator
        
        # Constant term
        linear_diff = f[u, 1:self.num_dims+1] - h[v, 1:self.num_dims+1]
        linear_sq_sum = np.sum(linear_diff * linear_diff)
        f[v, -1] = (f[u, -1] - h[v, -1]) - linear_sq_sum / (4 * (p0 + 1/b))
        
        # Add node cost for internal nodes
        if not tree.is_sample(v):
            f[v] += g[v]
    
    def _compute_tree_length(self, tree: tskit.Tree, g: np.ndarray) -> float:
        """
        Compute total parsimony length for a tree following C implementation.
        
        The C code only considers children of virtual_root (i.e., tree roots)
        and computes the minimum of their quadratic functions, then divides by edges.
        """
        total_length = 0.0
        
        # Only consider root nodes (children of virtual_root in C implementation)
        for root in tree.roots:
            if not tree.is_sample(root):
                p0 = g[root, 0]
                # Minimum of quadratic: g[num_dims1] - sum(g[i]²)/(4*g[0])
                # where num_dims1 = num_dims + 1 (constant term index)
                constant_term = g[root, -1]  # g[num_dims1]
                linear_coeffs = g[root, 1:self.num_dims+1]  # g[1] to g[num_dims]
                min_val = constant_term - np.sum(linear_coeffs * linear_coeffs) / (4 * p0)
                total_length += min_val
        
        # Divide by number of edges in the tree (matches C implementation)
        num_edges = tree.num_edges
        return total_length / max(num_edges, 1)
    
    def _update_node_averages(self, tree: tskit.Tree, f: np.ndarray, tree_span: float):
        """Update running averages for node costs."""
        for node in tree.nodes():
            self.node_weights[node] += tree_span

            if self.node_weights[node] > 0:
                # Update running average, writing column-wise per node
                weight = tree_span / self.node_weights[node]
                self.F[:, node] = (1 - weight) * self.F[:, node] + weight * f[node]


def quadratic_mpr(ts: tskit.TreeSequence,
                 sample_locations: Union[np.ndarray, list],
                 use_branch_lengths: bool = False) -> MPRResult:
    """
    Compute quadratic parsimony reconstruction.
    
    Args:
        ts: Tree sequence object
        sample_locations: Array with columns [node_id, x, y, ...]
        use_branch_lengths: Whether to use branch lengths
        
    Returns:
        MPRResult object containing reconstruction results
    """
    mpr = QuadraticMPR(ts, sample_locations, use_branch_lengths)
    return mpr.compute_mpr()


def quadratic_mpr_minimize(mpr_result: MPRResult) -> np.ndarray:
    """
    Find optimal locations by minimizing quadratic functions.
    
    Args:
        mpr_result: Result from quadratic_mpr()
        
    Returns:
        Array of optimal (x, y, ...) coordinates for each node
    """
    if mpr_result.reconstruction_type != "quadratic":
        raise ValueError("Expected quadratic MPR result")
    
    num_nodes = mpr_result.num_nodes
    num_params = mpr_result.mpr_matrix.shape[0]  # (num_pars, num_nodes)
    num_dims = num_params - 2
    
    optimal_locations = np.zeros((num_nodes, num_dims))
    
    for node in range(num_nodes):
        params = mpr_result.mpr_matrix[:, node]
        p0 = params[0]

        linear_coeffs = params[1:num_dims+1]
        optimal_locations[node] = -linear_coeffs / (2 * p0)
    
    return optimal_locations


def quadratic_mpr_minimize_discrete(mpr_result: MPRResult, 
                                   candidate_sites: np.ndarray) -> np.ndarray:
    """
    Find optimal discrete locations from candidate sites.
    
    Args:
        mpr_result: Result from quadratic_mpr()
        candidate_sites: Array of candidate locations with shape (n_sites, n_dims)
        
    Returns:
        Array of 1-based indices into candidate_sites for each node (R compatibility)
    """
    if mpr_result.reconstruction_type != "quadratic":
        raise ValueError("Expected quadratic MPR result")
    
    num_nodes = mpr_result.num_nodes
    num_params = mpr_result.mpr_matrix.shape[0]
    num_dims = num_params - 2
    num_sites = candidate_sites.shape[0]
    
    optimal_indices = np.zeros(num_nodes, dtype=int)
    
    for node in range(num_nodes):
        params = mpr_result.mpr_matrix[:, node]
        p0 = params[0]

        # Evaluate function at all candidate sites
        min_cost = np.inf
        best_site = 0

        for site_idx in range(num_sites):
            site_coords = candidate_sites[site_idx]

            quad_term = p0 * np.sum(site_coords * site_coords)
            linear_term = np.sum(params[1:num_dims+1] * site_coords)
            cost = quad_term + linear_term + params[-1]

            if cost < min_cost:
                min_cost = cost
                best_site = site_idx
        
        optimal_indices[node] = best_site + 1  # Convert to 1-based for R compatibility
    
    return optimal_indices