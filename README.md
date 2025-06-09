# gaiapy: Geographic Ancestry Inference Algorithm (Python)

`gaiapy` is a Python port of the GAIA R package for inferring the geographic locations of genetic ancestors using tree sequences. This package implements generalized parsimony methods for ancestral location reconstruction in continuous geographic space.

**Note**: This package is distributed on PyPI as `geoancestry`.

## Current Implementation Status

✅ **Implemented and Ready to Use:**
- **Quadratic parsimony** - for ancestors in continuous space, minimizing sum of squared Euclidean distances
- **Linear parsimony** - for ancestors in continuous space, minimizing sum of absolute (Manhattan) distances
- Full metadata integration and tree sequence augmentation
- Comprehensive validation and utility functions

🚧 **Not Yet Implemented:**
- **Discrete parsimony** - for ancestors restricted to finite location sets (coming soon)
- **Ancestry coefficients** - temporal analysis of ancestry proportions (coming soon)  
- **Migration flux** - migration flow analysis between regions (coming soon)

This package leverages the Python `tskit` API directly, avoiding the need for C wrappers and making the implementation more accessible for Python users and web applications.

## Installation

Install from PyPI:
```bash
pip install geoancestry
```

Install from source for development:
```bash
git clone https://github.com/chris-a-talbot/gaiapy
cd gaiapy
pip install -e ".[dev]"
```

## Quick Start

### Basic Continuous Space Reconstruction

```python
import geoancestry as gp
import tskit
import numpy as np

# Load your tree sequence
ts = tskit.load("path/to/treesequence.trees")

# Define sample locations as [node_id, x_coord, y_coord]
# node_id: Tree sequence node IDs (0-based)
# x_coord, y_coord: Geographic coordinates (any coordinate system)
samples = np.array([
    [0, 1.5, 2.0],  # node 0 at coordinates (1.5, 2.0)
    [1, 4.2, 3.1],  # node 1 at coordinates (4.2, 3.1) 
    [2, 6.7, 5.5],  # node 2 at coordinates (6.7, 5.5)
    # ... more samples
])

# Quadratic reconstruction (minimizes sum of squared Euclidean distances)
mpr_quad = gp.quadratic_mpr(ts, samples)
locations_quad = gp.quadratic_mpr_minimize(mpr_quad)

# Linear reconstruction (minimizes sum of Manhattan distances)
mpr_lin = gp.linear_mpr(ts, samples)
locations_lin = gp.linear_mpr_minimize(mpr_lin)

print(f"Quadratic reconstruction shape: {locations_quad.shape}")
print(f"Linear reconstruction shape: {locations_lin.shape}")
```

### Working with Tree Sequence Metadata

```python
# If your tree sequence has location metadata, you can extract it automatically
sample_locs = gp.extract_sample_locations_from_metadata(ts)

# Or augment a tree sequence with location data
ts_with_locs = gp.augment_tree_sequence_with_locations(ts, samples)

# Use metadata-aware reconstruction
mpr_meta = gp.quadratic_mpr_with_metadata(ts_with_locs)
locations_meta = gp.quadratic_mpr_minimize(mpr_meta)
```

### Advanced Options

```python
# Discrete coordinate system reconstruction (useful for grid-based coordinates)
locations_discrete = gp.quadratic_mpr_minimize_discrete(mpr_quad)

# Alternative linear parsimony with discrete output
locations_lin_discrete = gp.linear_mpr_minimize_discrete(mpr_lin)

# Export results for further analysis
gp.export_locations_to_file(locations_quad, "ancestral_locations.tsv")
```

## Key Functions (Currently Implemented)

### Continuous Space Functions
- `quadratic_mpr()` - Continuous space reconstruction using squared distances
- `linear_mpr()` - Continuous space reconstruction using absolute distances
- `quadratic_mpr_minimize()` - Find optimal continuous locations (quadratic)
- `linear_mpr_minimize()` - Find optimal continuous locations (linear)
- `quadratic_mpr_minimize_discrete()` - Discrete coordinate optimization (quadratic)
- `linear_mpr_minimize_discrete()` - Discrete coordinate optimization (linear)

### Metadata Integration
- `quadratic_mpr_with_metadata()` - Metadata-aware quadratic reconstruction
- `linear_mpr_with_metadata()` - Metadata-aware linear reconstruction
- `extract_sample_locations_from_metadata()` - Extract locations from tree sequence metadata
- `augment_tree_sequence_with_locations()` - Add location data to tree sequences
- `validate_location_metadata()` - Validate location data format
- `export_locations_to_file()` / `import_locations_from_file()` - I/O utilities

## Input Data Format

Sample locations should be provided as a NumPy array with shape `(n_samples, 3)`:
```python
samples = np.array([
    [node_id, x_coordinate, y_coordinate],
    [node_id, x_coordinate, y_coordinate],
    # ...
])
```

- **node_id**: Tree sequence node ID (0-based, integer)
- **x_coordinate, y_coordinate**: Geographic coordinates (float, any coordinate system)

## Output Format

Reconstructed locations are returned as NumPy arrays with shape `(n_nodes, 2)` where:
- Row index corresponds to tree sequence node ID
- Column 0: x-coordinate of reconstructed location
- Column 1: y-coordinate of reconstructed location

## Coming Soon

The following features from the original GAIA R package are planned for future releases:

- **Discrete parsimony** - `discrete_mpr()`, `discrete_mpr_minimize()`, `discrete_mpr_edge_history()`
- **Ancestry analysis** - `discrete_mpr_ancestry()`, `discrete_mpr_ancestry_flux()`

## References

Grundler, M.C., Terhorst, J., and Bradburd, G.S. (2025) A geographic history of human genetic ancestry. *Science* 387(6741): 1391-1397. DOI: [10.1126/science.adp4642](https://doi.org/10.1126/science.adp4642)

## License

MIT License (adapted from original CC-BY 4.0 International)
