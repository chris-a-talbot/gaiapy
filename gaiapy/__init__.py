"""
gaiapy: Geographic Ancestry Inference Algorithm for Python

A Python port of the GAIA R package for inferring geographic locations 
of genetic ancestors using tree sequences and generalized parsimony methods.

This package implements discrete and continuous space ancestry reconstruction
algorithms with full support for tree sequence metadata integration.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main API functions
from .api import (
    discrete_mpr,
    discrete_mpr_minimize,
    discrete_mpr_edge_history,
    discrete_mpr_ancestry,
    discrete_mpr_ancestry_flux,
    quadratic_mpr,
    quadratic_mpr_minimize,
    quadratic_mpr_minimize_discrete,
    linear_mpr,
    linear_mpr_minimize,
    linear_mpr_minimize_discrete,
    # Enhanced metadata-aware functions
    discrete_mpr_with_metadata,
    quadratic_mpr_with_metadata,
    linear_mpr_with_metadata,
    extract_mpr_summary,
)

# Import metadata utilities
from .utils.metadata import (
    extract_sample_locations_from_metadata,
    augment_tree_sequence_with_locations,
    validate_location_metadata,
    merge_location_sources,
    export_locations_to_file,
    import_locations_from_file,
)

# Import core data structures
from .core.sankoff import MPRResult

__all__ = [
    # Core functions
    "discrete_mpr",
    "discrete_mpr_minimize", 
    "discrete_mpr_edge_history",
    "discrete_mpr_ancestry",
    "discrete_mpr_ancestry_flux",
    "quadratic_mpr",
    "quadratic_mpr_minimize",
    "quadratic_mpr_minimize_discrete",
    "linear_mpr",
    "linear_mpr_minimize",
    "linear_mpr_minimize_discrete",
    
    # Enhanced metadata-aware functions
    "discrete_mpr_with_metadata",
    "quadratic_mpr_with_metadata", 
    "linear_mpr_with_metadata",
    "extract_mpr_summary",
    
    # Metadata utilities
    "extract_sample_locations_from_metadata",
    "augment_tree_sequence_with_locations",
    "validate_location_metadata",
    "merge_location_sources",
    "export_locations_to_file",
    "import_locations_from_file",
    
    # Data structures
    "MPRResult",
]
