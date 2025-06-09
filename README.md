# gaiapy: Geographic Ancestry Inference Algorithm (Python)

`gaiapy` is a Python port of the GAIA R package for inferring the geographic locations of genetic ancestors using tree sequences. It implements three approaches to ancestral location reconstruction:

1. **Discrete parsimony** - for ancestors restricted to a finite set of locations
2. **Squared change parsimony** - for ancestors in continuous space, minimizing squared distances  
3. **Linear parsimony** - for ancestors in continuous space, minimizing absolute distances

This package leverages the Python `tskit` API directly, avoiding the need for C wrappers and making the implementation more accessible and maintainable.

## Installation

Install from source:
```bash
git clone <repository-url>
cd gaiapy
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Working with Discrete Locations

```python
import gaiapy
import tskit
import numpy as np

# Load your tree sequence
ts = tskit.load("path/to/treesequence.trees")

# Define sample locations - each sample must be assigned to a discrete state
# node_id: Tree sequence node IDs (0-based)
# state_id: Location state IDs (0-based in Python, unlike R version)
samples = np.array([
    [0, 0],  # node 0 -> state 0
    [1, 0],  # node 1 -> state 0  
    [2, 1],  # node 2 -> state 1
    # ... more samples
])

# Create cost matrix for migrations between states
# Must be symmetric with non-negative values
# Entry [i,j] = cost of migrating from state i to state j
num_states = 2
costs = np.ones((num_states, num_states))  # Default cost of 1 between states
np.fill_diagonal(costs, 0)  # No cost to stay in same state

# Compute migration costs
mpr = gaiapy.discrete_mpr(ts, samples, costs)

# Get optimal state assignments for ancestors
states = gaiapy.discrete_mpr_minimize(mpr)

# Get detailed migration histories (optional)
history = gaiapy.discrete_mpr_edge_history(ts, mpr, costs)
```

### Working with Continuous Space

```python
# For ancestors in continuous space, provide sample coordinates
samples = np.array([
    [0, 1.5, 2.0],  # node 0 at coordinates (1.5, 2.0)
    [1, 4.2, 3.1],  # node 1 at coordinates (4.2, 3.1) 
    [2, 6.7, 5.5],  # node 2 at coordinates (6.7, 5.5)
    # ... more samples
])

# Using squared distance (minimizes sum of squared Euclidean distances)
mpr_quad = gaiapy.quadratic_mpr(ts, samples)
locations_quad = gaiapy.quadratic_mpr_minimize(mpr_quad)

# Using absolute distance (minimizes sum of Manhattan distances)
mpr_lin = gaiapy.linear_mpr(ts, samples)
locations_lin = gaiapy.linear_mpr_minimize(mpr_lin)
```

## Key Functions

- `discrete_mpr()` - Discrete state reconstruction
- `quadratic_mpr()` - Continuous space reconstruction using squared distances
- `linear_mpr()` - Continuous space reconstruction using absolute distances
- `discrete_mpr_minimize()` - Find optimal discrete state assignments
- `discrete_mpr_edge_history()` - Detailed migration histories
- `discrete_mpr_ancestry()` - Ancestry coefficients through time
- `discrete_mpr_ancestry_flux()` - Migration flux between regions

## Differences from R Version

- Uses 0-based indexing throughout (consistent with Python/tskit conventions)
- Returns NumPy arrays instead of R matrices/data frames
- Leverages tskit Python API directly instead of C wrappers
- More Pythonic API design and error handling

## References

Grundler et al. (2024) A geographic history of human genetic ancestry. bioRxiv doi: 10.1101/2024.03.27.586858

## License

MIT License (adapted from original CC-BY 4.0 International)
