"""
Discrete state parsimony implementation.

This module implements Sankoff parsimony for discrete geographic states,
which is the core algorithm for inferring ancestral locations when
ancestors are restricted to a finite set of possible locations.
"""

import numpy as np
from typing import Union, Optional, Dict, Any
import tskit

from .sankoff import MPRResult


def discrete_mpr(ts: tskit.TreeSequence,
                sample_locations: Union[np.ndarray, list],
                cost_matrix: Union[np.ndarray, list],
                use_branch_lengths: bool = False) -> MPRResult:
    """
    Compute minimum migration costs for discrete geographic ancestry reconstruction.
    
    THIS IS A PLACEHOLDER IMPLEMENTATION - NOT YET IMPLEMENTED
    """
    raise NotImplementedError("discrete_mpr not yet implemented")


def discrete_mpr_minimize(mpr_result: MPRResult,
                         return_zero_based: bool = True) -> np.ndarray:
    """
    Determine optimal geographic states from minimum migration costs.
    
    THIS IS A PLACEHOLDER IMPLEMENTATION - NOT YET IMPLEMENTED
    """
    raise NotImplementedError("discrete_mpr_minimize not yet implemented")


def discrete_mpr_edge_history(ts: tskit.TreeSequence,
                             mpr_result: MPRResult,
                             cost_matrix: Union[np.ndarray, list],
                             adjacency_matrix: Optional[Union[np.ndarray, list]] = None,
                             return_zero_based: bool = True) -> Dict[str, Any]:
    """
    Sample migration paths for each edge in a tree sequence.
    
    THIS IS A PLACEHOLDER IMPLEMENTATION - NOT YET IMPLEMENTED
    """
    raise NotImplementedError("discrete_mpr_edge_history not yet implemented")
