"""
Linear (Manhattan distance) parsimony implementation.

This module implements Sankoff parsimony for continuous geographic locations
using Manhattan (L1) distances. The algorithm computes piecewise linear functions
that represent the minimum sum of absolute distances between ancestor-descendant
pairs for different ancestral location assignments.

Based on the C implementation in src/treeseq_sankoff_linear.c from the original
GAIA package, which implements the algorithm described in:

Miklós Csurös. 2008. Ancestral Reconstruction by Asymmetric Wagner
Parsimony over Continuous Characters and Squared Parsimony over
Distributions. In: Nelson, C.E., Vialette, S. (eds) Comparative Genomics.

The core algorithm maintains piecewise linear cost functions for each node,
representing the minimum cost to explain descendant locations when
the node is at different positions in continuous space.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import tskit
import warnings

from .sankoff import MPRResult, SankoffTree


class PiecewiseLinearFunction:
    """
    Represents a convex piecewise linear function.
    
    The function has the form:
    
    F(x) = {
        intercept + slopes[0] * x                        if x <= breakpoints[0]
        intercept + slopes[0]*breakpoints[0] + 
            slopes[1]*(x - breakpoints[0])               if breakpoints[0] < x <= breakpoints[1]
        ...
        s[k] + slopes[k+1] * (x - breakpoints[k])       if x > breakpoints[k]
    }
    
    Where slopes and breakpoints are stored in non-decreasing order.
    """
    
    def __init__(self, max_breakpoints: int = 100):
        """
        Initialize piecewise linear function.
        
        Args:
            max_breakpoints: Maximum number of breakpoints to allow
        """
        self.slopes = np.zeros(max_breakpoints + 1)
        self.breakpoints = np.zeros(max_breakpoints)
        self.intercept = 0.0
        self.num_breaks = 0
        self.max_num_breaks = max_breakpoints
    
    def copy(self) -> 'PiecewiseLinearFunction':
        """Create a copy of this PLF."""
        new_plf = PiecewiseLinearFunction(self.max_num_breaks)
        new_plf.slopes[:len(self.slopes)] = self.slopes
        new_plf.breakpoints[:len(self.breakpoints)] = self.breakpoints
        new_plf.intercept = self.intercept
        new_plf.num_breaks = self.num_breaks
        return new_plf
    
    def evaluate(self, x: float) -> float:
        """Evaluate the function at point x."""
        if self.num_breaks == 0:
            return self.intercept + self.slopes[0] * x
        
        if x <= self.breakpoints[0]:
            return self.intercept + self.slopes[0] * x
        
        # Find the appropriate segment
        score = self.intercept + self.slopes[0] * self.breakpoints[0]
        i = 0
        while i < self.num_breaks - 1 and x > self.breakpoints[i + 1]:
            score += self.slopes[i + 1] * (self.breakpoints[i + 1] - self.breakpoints[i])
            i += 1
        
        # Add the final segment
        score += self.slopes[i + 1] * (x - self.breakpoints[i])
        return score
    
    def get_minimum(self) -> Tuple[float, float]:
        """
        Find the minimum value and location of the function.
        
        Returns:
            Tuple of (min_location, min_value)
        """
        if self.num_breaks == 0:
            if abs(self.slopes[0]) < 1e-10:
                return 0.0, self.intercept
            else:
                # Linear function - no minimum in finite domain
                return float('inf'), float('inf')
        
        # Find where slope changes from negative to non-negative
        i = 0
        while i < self.num_breaks and self.slopes[i + 1] < 0:
            i += 1
        
        if i >= self.num_breaks:
            # All slopes are negative - minimum is at +infinity
            return float('inf'), float('inf')
        
        min_location = self.breakpoints[i]
        min_value = self.evaluate(min_location)
        
        # If the slope is exactly zero, we can choose any point in the flat region
        if abs(self.slopes[i + 1]) < 1e-10:
            # Find the end of the flat region
            while i + 1 < self.num_breaks and abs(self.slopes[i + 2]) < 1e-10:
                i += 1
            # Return midpoint of flat region
            if i + 1 < self.num_breaks:
                min_location = (self.breakpoints[i] + self.breakpoints[i + 1]) / 2
            min_value = self.evaluate(min_location)
        
        return min_location, min_value
    
    def min_transform(self, b: float) -> 'PiecewiseLinearFunction':
        """
        Compute the Minkowski sum with absolute value function.
        
        ret(x) = min_z { b * |x - z| + f(z) }
        
        Args:
            b: Scaling factor (must be positive)
            
        Returns:
            New PLF representing the transformed function
        """
        if b <= 0:
            raise ValueError("b must be positive")
        
        result = PiecewiseLinearFunction(self.max_num_breaks)
        minus_b = -b
        
        # Handle case where -b <= first slope
        if minus_b <= self.slopes[0]:
            x_left = 0
            result.intercept = self.intercept
        else:
            # Find where -b intersects the function
            i = 0
            shift = self.intercept + self.slopes[0] * self.breakpoints[0]
            while i < self.num_breaks and minus_b >= self.slopes[i + 1]:
                i += 1
                if i < self.num_breaks:
                    shift += self.slopes[i] * (self.breakpoints[i] - self.breakpoints[i - 1])
            
            x_left = i + 1
            if i < self.num_breaks:
                result.intercept = shift + b * self.breakpoints[i]
                result.slopes[0] = minus_b
                result.breakpoints[0] = self.breakpoints[i]
                result.num_breaks = 1
            else:
                result.intercept = shift
                x_left = self.num_breaks
        
        # Copy remaining slopes and breakpoints
        num_breaks = result.num_breaks
        if b >= self.slopes[self.num_breaks]:
            # Copy all remaining breakpoints
            for i in range(x_left, self.num_breaks):
                if num_breaks < result.max_num_breaks:
                    result.slopes[num_breaks] = self.slopes[i]
                    result.breakpoints[num_breaks] = self.breakpoints[i]
                    num_breaks += 1
            result.slopes[num_breaks] = self.slopes[self.num_breaks]
        else:
            # Copy until we reach slope b
            i = x_left
            while i < self.num_breaks and b >= self.slopes[i]:
                if num_breaks < result.max_num_breaks:
                    result.slopes[num_breaks] = self.slopes[i]
                    result.breakpoints[num_breaks] = self.breakpoints[i]
                    num_breaks += 1
                i += 1
            
            # Check for colinearity to avoid superfluous breakpoints
            if num_breaks > 0 and abs(b - result.slopes[num_breaks - 1]) > 1e-10:
                result.slopes[num_breaks] = b
        
        result.num_breaks = max(0, num_breaks)
        return result
    
    def add(self, other: 'PiecewiseLinearFunction', sign: int = 1, 
            self_scale: float = 1.0, other_scale: float = 1.0) -> 'PiecewiseLinearFunction':
        """
        Add two piecewise linear functions.
        
        result = self_scale * self + sign * other_scale * other
        
        Args:
            other: Other PLF to add
            sign: Sign for the other function (+1 or -1)
            self_scale: Scaling factor for self
            other_scale: Scaling factor for other
            
        Returns:
            New PLF representing the sum
        """
        result = PiecewiseLinearFunction(max(self.max_num_breaks, other.max_num_breaks))
        
        i = j = 0
        num_breaks = 0
        prev_slope = float('-inf')
        
        while i < self.num_breaks and j < other.num_breaks:
            if self.breakpoints[i] < other.breakpoints[j]:
                next_break = self.breakpoints[i]
                next_slope = self_scale * self.slopes[i] + sign * other_scale * other.slopes[j]
                i += 1
            elif self.breakpoints[i] > other.breakpoints[j]:
                next_break = other.breakpoints[j]
                next_slope = self_scale * self.slopes[i] + sign * other_scale * other.slopes[j]
                j += 1
            else:
                next_break = self.breakpoints[i]
                next_slope = self_scale * self.slopes[i] + sign * other_scale * other.slopes[j]
                i += 1
                j += 1
            
            if abs(prev_slope - next_slope) > 1e-10:
                if num_breaks < result.max_num_breaks:
                    result.breakpoints[num_breaks] = next_break
                    result.slopes[num_breaks] = next_slope
                    prev_slope = next_slope
                    num_breaks += 1
            else:
                if num_breaks > 0:
                    result.breakpoints[num_breaks - 1] = next_break
        
        # Handle remaining breakpoints
        while i < self.num_breaks:
            next_break = self.breakpoints[i]
            next_slope = self_scale * self.slopes[i] + sign * other_scale * other.slopes[j]
            i += 1
            
            if abs(prev_slope - next_slope) > 1e-10:
                if num_breaks < result.max_num_breaks:
                    result.breakpoints[num_breaks] = next_break
                    result.slopes[num_breaks] = next_slope
                    prev_slope = next_slope
                    num_breaks += 1
            else:
                if num_breaks > 0:
                    result.breakpoints[num_breaks - 1] = next_break
        
        while j < other.num_breaks:
            next_break = other.breakpoints[j]
            next_slope = self_scale * self.slopes[i] + sign * other_scale * other.slopes[j]
            j += 1
            
            if abs(prev_slope - next_slope) > 1e-10:
                if num_breaks < result.max_num_breaks:
                    result.breakpoints[num_breaks] = next_break
                    result.slopes[num_breaks] = next_slope
                    prev_slope = next_slope
                    num_breaks += 1
            else:
                if num_breaks > 0:
                    result.breakpoints[num_breaks - 1] = next_break
        
        # Final slope
        final_slope = self_scale * self.slopes[i] + sign * other_scale * other.slopes[j]
        if num_breaks == 0 or abs(prev_slope - final_slope) > 1e-10:
            result.slopes[num_breaks] = final_slope
        else:
            num_breaks -= 1
            result.slopes[num_breaks] = final_slope
        
        result.num_breaks = max(0, num_breaks)
        result.intercept = self_scale * self.intercept + sign * other_scale * other.intercept
        
        return result
    
    def defragment(self, tolerance: float = 1e-10):
        """
        Remove redundant breakpoints with colinear slopes.
        
        Args:
            tolerance: Tolerance for considering slopes equal
        """
        if self.num_breaks <= 1:
            return
        
        # Create temporary arrays
        new_slopes = np.zeros_like(self.slopes)
        new_breaks = np.zeros_like(self.breakpoints)
        
        new_slopes[0] = self.slopes[0]
        new_breaks[0] = self.breakpoints[0]
        new_num_breaks = 1
        
        for i in range(1, self.num_breaks):
            if abs(self.slopes[i - 1] - self.slopes[i]) > tolerance:
                new_slopes[new_num_breaks] = self.slopes[i]
                new_breaks[new_num_breaks] = self.breakpoints[i]
                new_num_breaks += 1
            else:
                # Extend the previous segment
                new_breaks[new_num_breaks - 1] = self.breakpoints[i]
        
        # Handle final slope
        if abs(self.slopes[self.num_breaks] - self.slopes[self.num_breaks - 1]) > tolerance:
            new_slopes[new_num_breaks] = self.slopes[self.num_breaks]
        else:
            new_num_breaks -= 1
            new_slopes[new_num_breaks] = self.slopes[self.num_breaks]
        
        # Update arrays
        self.slopes[:len(new_slopes)] = new_slopes
        self.breakpoints[:len(new_breaks)] = new_breaks
        self.num_breaks = max(0, new_num_breaks)


class LinearMPR(SankoffTree):
    """
    Linear (Manhattan distance) parsimony reconstruction for continuous locations.
    """
    
    def __init__(self, ts: tskit.TreeSequence, sample_locations: np.ndarray,
                 use_branch_lengths: bool = False, tolerance: float = 0.01):
        """
        Initialize linear MPR.
        
        Args:
            ts: Tree sequence
            sample_locations: Array with columns [node_id, x, y, ...]
            use_branch_lengths: Whether to use branch lengths
            tolerance: Tolerance for merging breakpoints
        """
        super().__init__(ts, use_branch_lengths)
        
        self.sample_locations = np.asarray(sample_locations)
        if self.sample_locations.shape[1] < 3:
            raise ValueError("sample_locations must have at least 3 columns (node_id, x, y)")
        
        self.num_dims = self.sample_locations.shape[1] - 1  # Subtract node_id column
        self.tolerance = tolerance
        
        # Create lookup for sample coordinates
        self.sample_coords = {}
        for row in self.sample_locations:
            node_id = int(row[0])
            coords = row[1:1+self.num_dims]
            self.sample_coords[node_id] = coords
        
        # Estimate maximum breakpoints needed
        max_breakpoints = max(100, self.num_samples * 2)
        
        # Initialize result storage
        self.node_weights = np.zeros(self.num_nodes)
        self.F = [[PiecewiseLinearFunction(max_breakpoints) for _ in range(self.num_dims)] 
                  for _ in range(self.num_nodes)]
        self.tree_lengths = []
        self.tree_weights = []
    
    def compute_mpr(self) -> MPRResult:
        """
        Compute linear MPR for all trees.
        
        Returns:
            MPRResult object containing reconstruction results
        """
        total_weight = 0.0
        mean_tree_length = 0.0
        
        # Process each tree in the sequence
        for tree in self.ts.trees():
            tree_span = tree.interval.right - tree.interval.left
            
            # Initialize cost arrays for this tree
            max_breakpoints = max(100, self.num_samples * 2)
            g = [[PiecewiseLinearFunction(max_breakpoints) for _ in range(self.num_dims)] 
                 for _ in range(self.num_nodes)]
            h = [[PiecewiseLinearFunction(max_breakpoints) for _ in range(self.num_dims)] 
                 for _ in range(self.num_nodes)]
            f = [[PiecewiseLinearFunction(max_breakpoints) for _ in range(self.num_dims)] 
                 for _ in range(self.num_nodes)]
            
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
        
        # Convert PLF results to matrix format for compatibility
        mpr_matrix = self._plf_to_matrix()
        
        return MPRResult(
            mpr_matrix=mpr_matrix,
            tree_lengths=np.array(self.tree_lengths),
            mean_tree_length=mean_tree_length,
            node_weights=self.node_weights.copy(),
            reconstruction_type="linear",
            num_nodes=self.num_nodes,
            num_trees=len(self.tree_lengths),
            sample_locations=self.sample_locations.copy(),
            sample_node_ids=np.array(list(self.sample_coords.keys()))
        )
    
    def _sankoff_algorithm(self, tree: tskit.Tree, g: List[List[PiecewiseLinearFunction]], 
                          h: List[List[PiecewiseLinearFunction]], f: List[List[PiecewiseLinearFunction]]):
        """
        Run the Sankoff algorithm for a single tree using simplified traversal.
        """
        # Postorder pass
        for node in self.postorder_traversal(tree):
            if not tree.is_sample(node):
                # Initialize node cost to zero
                for dim in range(self.num_dims):
                    g[node][dim] = PiecewiseLinearFunction()
                    g[node][dim].slopes[0] = 0.0
                    g[node][dim].intercept = 0.0
                    g[node][dim].num_breaks = 0
                
                # Add contributions from children
                for child in tree.children(node):
                    self._calc_stem_cost(node, child, tree, g, h)
                    for dim in range(self.num_dims):
                        g[node][dim] = g[node][dim].add(h[child][dim])
        
        # Preorder pass
        for node in self.preorder_traversal(tree):
            self._calc_final_cost(node, tree, g, h, f)
    
    def _calc_stem_cost(self, u: int, v: int, tree: tskit.Tree,
                       g: List[List[PiecewiseLinearFunction]], h: List[List[PiecewiseLinearFunction]]):
        """
        Calculate stem cost for edge u->v.
        """
        # Get branch length scaling factor
        b = 1.0
        if self.use_branch_lengths:
            u_time = tree.time(u)
            v_time = tree.time(v)
            if u_time is not None and v_time is not None:
                time_diff = u_time - v_time
                if time_diff > 0:
                    b = 1.0 / time_diff  # Reciprocal like C code!
                else:
                    b = 1.0
        
        if tree.is_sample(v):
            # Sample node case - create PLF for |x - x_v|
            if v in self.sample_coords:
                x_v = self.sample_coords[v]
                for dim in range(self.num_dims):
                    coord = x_v[dim]
                    # h(x) = b * |x - coord|
                    h[v][dim] = PiecewiseLinearFunction()
                    h[v][dim].breakpoints[0] = coord
                    h[v][dim].slopes[0] = -b
                    h[v][dim].slopes[1] = b
                    h[v][dim].intercept = b * abs(coord)
                    h[v][dim].num_breaks = 1
        else:
            # Internal node case - apply min transform
            for dim in range(self.num_dims):
                h[v][dim] = g[v][dim].min_transform(b)
    
    def _calc_final_cost(self, v: int, tree: tskit.Tree,
                        g: List[List[PiecewiseLinearFunction]], h: List[List[PiecewiseLinearFunction]], 
                        f: List[List[PiecewiseLinearFunction]]):
        """
        Calculate final cost for node v.
        """
        u = tree.parent(v)
        
        if u == tskit.NULL:
            # Root node - final cost is same as node cost
            for dim in range(self.num_dims):
                if tree.is_sample(v):
                    f[v][dim] = PiecewiseLinearFunction()
                    f[v][dim].slopes[0] = 0.0
                    f[v][dim].intercept = 0.0
                    f[v][dim].num_breaks = 0
                else:
                    f[v][dim] = g[v][dim].copy()
            return
        
        # Get branch length
        b = 1.0
        if self.use_branch_lengths:
            u_time = tree.time(u)
            v_time = tree.time(v)
            if u_time is not None and v_time is not None:
                time_diff = u_time - v_time
                if time_diff > 0:
                    b = 1.0 / time_diff  # Reciprocal like C code!
                else:
                    b = 1.0
        
        if tree.is_sample(v):
            # For sample nodes: final cost is always zero (location is fixed)
            for dim in range(self.num_dims):
                f[v][dim] = PiecewiseLinearFunction()
                f[v][dim].slopes[0] = 0.0
                f[v][dim].intercept = 0.0
                f[v][dim].num_breaks = 0
        else:
            # For internal nodes: f_v = min_transform(f_u - h_v, b) + g_v
            for dim in range(self.num_dims):
                combined = f[u][dim].add(h[v][dim], sign=-1)
                temp = combined.min_transform(b)
                f[v][dim] = temp.add(g[v][dim])
    
    def _compute_tree_length(self, tree: tskit.Tree, g: List[List[PiecewiseLinearFunction]]) -> float:
        """
        Compute total parsimony length for a tree.
        """
        total_length = 0.0
        
        # Sum minimum values for all root nodes (non-sample)
        for root in tree.roots:
            if not tree.is_sample(root):
                for dim in range(self.num_dims):
                    _, min_val = g[root][dim].get_minimum()
                    if not np.isinf(min_val):
                        total_length += min_val
        
        # Divide by number of edges in the tree
        num_edges = tree.num_edges
        return total_length / max(num_edges, 1)
    
    def _update_node_averages(self, tree: tskit.Tree, f: List[List[PiecewiseLinearFunction]], tree_span: float):
        """Update running averages for node costs."""
        for node in tree.nodes():
            # Skip sample nodes (matches C implementation's early return)
            if tree.is_sample(node):
                continue
                
            old_weight = self.node_weights[node]
            self.node_weights[node] += tree_span
            new_weight = self.node_weights[node]
            
            if new_weight > 0:
                # Update running average using weighted combination
                t = tree_span / new_weight
                for dim in range(self.num_dims):
                    if old_weight == 0:
                        # First contribution
                        self.F[node][dim] = f[node][dim].copy()
                    else:
                        # Weighted average: (1-t) * old + t * new
                        self.F[node][dim] = self.F[node][dim].add(f[node][dim], 
                                                                 self_scale=(1-t), 
                                                                 other_scale=t)
                    
                    # Apply defragmentation
                    self.F[node][dim].defragment(self.tolerance)
    
    def _plf_to_matrix(self) -> np.ndarray:
        """
        Convert PLF results to matrix format.
        
        Returns a matrix where each row contains the serialized PLF parameters
        for one node across all dimensions.
        """
        # Calculate maximum size needed
        max_params = 0
        for node in range(self.num_nodes):
            for dim in range(self.num_dims):
                # Each PLF needs: intercept + slopes + breakpoints
                plf = self.F[node][dim]
                params_needed = 3 + plf.num_breaks * 2  # intercept, num_breaks, final_slope, then (slope,break) pairs
                max_params = max(max_params, params_needed)
        
        # Create matrix to hold all parameters
        # Format: [intercept, num_breaks, slope_0, break_0, slope_1, break_1, ..., final_slope]
        mpr_matrix = np.zeros((self.num_nodes, self.num_dims, max_params))
        
        for node in range(self.num_nodes):
            for dim in range(self.num_dims):
                plf = self.F[node][dim]
                row = mpr_matrix[node, dim]
                idx = 0
                
                # Store intercept and number of breakpoints
                row[idx] = plf.intercept
                idx += 1
                row[idx] = plf.num_breaks
                idx += 1
                
                # Store slope-breakpoint pairs
                for i in range(plf.num_breaks):
                    row[idx] = plf.slopes[i]
                    idx += 1
                    row[idx] = plf.breakpoints[i]
                    idx += 1
                
                # Store final slope
                row[idx] = plf.slopes[plf.num_breaks]
        
        return mpr_matrix


def linear_mpr(ts: tskit.TreeSequence,
              sample_locations: Union[np.ndarray, list],
              use_branch_lengths: bool = False,
              tolerance: float = 0.01) -> MPRResult:
    """
    Compute linear (Manhattan distance) parsimony reconstruction.
    
    Args:
        ts: Tree sequence object
        sample_locations: Array with columns [node_id, x, y, ...]
        use_branch_lengths: Whether to use branch lengths
        tolerance: Tolerance for merging breakpoints in piecewise linear functions
        
    Returns:
        MPRResult object containing reconstruction results
    """
    mpr = LinearMPR(ts, sample_locations, use_branch_lengths, tolerance)
    return mpr.compute_mpr()


def linear_mpr_minimize(mpr_result: MPRResult) -> np.ndarray:
    """
    Find optimal continuous locations from linear MPR results.
    
    For each node, finds the location that minimizes the piecewise linear cost function.
    When there are tied optima (flat regions), randomly selects within the optimal region.
    
    This implementation matches the R C code exactly, including its reservoir sampling.
    
    Args:
        mpr_result: Result from linear_mpr()
        
    Returns:
        Array of optimal (x, y, ...) coordinates for each node
    """
    if mpr_result.reconstruction_type != "linear":
        raise ValueError("Expected linear MPR result")
    
    num_nodes = mpr_result.num_nodes
    mpr_matrix = mpr_result.mpr_matrix
    num_dims = mpr_matrix.shape[1]
    
    optimal_locations = np.zeros((num_nodes, num_dims))
    
    # Use seed 1 to match R's behavior
    np.random.seed(1)
    
    # Match R's nested loop structure: for j (dims), for i (nodes)
    for j in range(num_dims):  # R's outer loop: dimensions
        for i in range(num_nodes):  # R's inner loop: nodes
            # Extract PLF data (what R calls 'a' and 'b')
            row = mpr_matrix[i, j]
            intercept = row[0]
            num_breaks = int(row[1])
            
            if num_breaks == 0:
                slopes = np.array([row[2]])  # a = [final_slope]
                breakpoints = np.array([])   # b = [] (empty)
            else:
                slopes = np.zeros(num_breaks + 1)
                breakpoints = np.zeros(num_breaks)
                
                idx = 2
                for k in range(num_breaks):
                    slopes[k] = row[idx]
                    idx += 1
                    breakpoints[k] = row[idx]
                    idx += 1
                slopes[num_breaks] = row[idx]
            
            # Exactly match R's algorithm
            k = 0
            # while (a[k+1] < 0) ++k;
            while k + 1 < len(slopes) and slopes[k + 1] < 0:
                k += 1
            
            # x[i+j*n] = b[k];
            if k < len(breakpoints):
                x_val = breakpoints[k]
            else:
                # R bug: accessing invalid memory, we use 0.0
                x_val = 0.0
            
            # Reservoir sampling for ties: l = 1; while (a[k+1] == 0) ...
            l = 1
            while k + 1 < len(slopes) and abs(slopes[k + 1]) < 1e-15:  # exactly 0
                l += 1
                k += 1
                prob = 1.0 / l
                
                if np.random.random() < prob:
                    if k < len(breakpoints):
                        x_val = breakpoints[k]
                    # else: keep previous value
            
            optimal_locations[i, j] = x_val
    
    return optimal_locations


def linear_mpr_minimize_discrete(mpr_result: MPRResult, 
                                candidate_sites: np.ndarray) -> np.ndarray:
    """
    Find optimal discrete locations from candidate sites.
    
    Args:
        mpr_result: Result from linear_mpr()
        candidate_sites: Array of candidate locations with shape (n_sites, n_dims)
        
    Returns:
        Array of 1-based indices into candidate_sites for each node (R compatibility)
    """
    if mpr_result.reconstruction_type != "linear":
        raise ValueError("Expected linear MPR result")
    
    num_nodes = mpr_result.num_nodes
    mpr_matrix = mpr_result.mpr_matrix
    num_dims = mpr_matrix.shape[1]
    num_sites = candidate_sites.shape[0]
    
    if candidate_sites.shape[1] != num_dims:
        raise ValueError("Candidate sites must have same number of dimensions as MPR result")
    
    optimal_indices = np.zeros(num_nodes, dtype=int)
    
    for node in range(num_nodes):
        min_total_cost = float('inf')
        best_site = 0
        
        for site_idx in range(num_sites):
            site_coords = candidate_sites[site_idx]
            total_cost = 0.0
            
            # Evaluate cost at this site for each dimension
            for dim in range(num_dims):
                x = site_coords[dim]
                
                # Reconstruct PLF from matrix
                row = mpr_matrix[node, dim]
                intercept = row[0]
                num_breaks = int(row[1])
                
                if num_breaks == 0:
                    # Simple linear function
                    final_slope = row[2]
                    cost = intercept + final_slope * x
                else:
                    # Evaluate piecewise linear function
                    idx = 2
                    slopes = np.zeros(num_breaks + 1)
                    breakpoints = np.zeros(num_breaks)
                    
                    for i in range(num_breaks):
                        slopes[i] = row[idx]
                        idx += 1
                        breakpoints[i] = row[idx]
                        idx += 1
                    slopes[num_breaks] = row[idx]
                    
                    # Evaluate at x
                    if x <= breakpoints[0]:
                        cost = intercept + slopes[0] * x
                    else:
                        cost = intercept + slopes[0] * breakpoints[0]
                        j = 0
                        while j < num_breaks - 1 and x > breakpoints[j + 1]:
                            cost += slopes[j + 1] * (breakpoints[j + 1] - breakpoints[j])
                            j += 1
                        cost += slopes[j + 1] * (x - breakpoints[j])
                
                total_cost += cost
            
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_site = site_idx
        
        optimal_indices[node] = best_site + 1  # Convert to 1-based for R compatibility
    
    return optimal_indices
