#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tskit

# Ensure project root is first on sys.path so local sources are preferred over installed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from helpers import common  # type: ignore

# gaiapy is the module name, package is geoancestry
try:
    import gaiapy as gp
    # Optional: show which gaiapy is being used when verbose later
    GP_PATH = getattr(gp, "__file__", None)
except Exception as e:  # pragma: no cover
    print("Failed to import 'gaiapy' (PyPI: geoancestry). Install it first.", file=sys.stderr)
    raise


def discover_trees(trees_dir: Path):
    return sorted(trees_dir.rglob("*.trees"))


def extract_samples_xy(ts: tskit.TreeSequence) -> np.ndarray:
    """Return Nx3 array [node_id, x, y].

    Tries gaiapy metadata extraction; falls back to tskit individual locations.
    """
    # Try gaiapy metadata-aware extraction if available
    try:
        if hasattr(gp, "extract_sample_locations_from_metadata"):
            arr = gp.extract_sample_locations_from_metadata(ts)  # type: ignore[attr-defined]
            if arr is not None and len(arr) > 0:
                out = np.zeros((arr.shape[0], 3), dtype=float)
                out[:, 0] = arr[:, 0].astype(int)
                out[:, 1:] = arr[:, 1:3].astype(float)
                return out
    except Exception:
        pass
    # Fallback: read from individuals' location for sample nodes
    rows = []
    for node_id, x, y, _z in common.iter_sample_node_locations(ts):
        rows.append([int(node_id), float(x), float(y)])
    if not rows:
        raise RuntimeError("No sample nodes found to extract locations from")
    return np.asarray(rows, dtype=float)


def write_samples_csv(ts: tskit.TreeSequence, path: Path):
    common.write_sample_locations_csv(ts, path)


def run_r_gaia(rscript: str, run_gaia_r: Path, tree_path: Path, samples_csv: Path, out_csv: Path, verbose: bool):
    cmd = [
        rscript,
        str(run_gaia_r),
        "--tree", str(tree_path),
        "--samples_csv", str(samples_csv),
        "--out_csv", str(out_csv),
    ]
    if verbose:
        cmd.append("--verbose")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"R GAIA failed for {tree_path.name} with exit code {result.returncode}")
    if verbose:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)


def run_python_gaiapy(ts: tskit.TreeSequence, samples_xy: np.ndarray) -> np.ndarray:
    # Match R call which sets use_branch_lengths=TRUE
    mpr = gp.quadratic_mpr(ts, samples_xy, use_branch_lengths=True)
    locs = gp.quadratic_mpr_minimize(mpr)  # shape (n_nodes, 2)
    return np.asarray(locs, dtype=float)


def compare_results(r_df: pd.DataFrame, py_df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    # Ensure both have node_id, x, y, z (py z=0)
    py_df = py_df.copy()
    if "z" not in py_df.columns:
        py_df["z"] = 0.0
    # Align on node_id intersection
    common_nodes = sorted(set(r_df.node_id.astype(int)).intersection(set(py_df.node_id.astype(int))))
    r_sub = r_df.set_index("node_id").loc[common_nodes].sort_index()
    p_sub = py_df.set_index("node_id").loc[common_nodes].sort_index()
    dx = (r_sub["x"] - p_sub["x"]).to_numpy()
    dy = (r_sub["y"] - p_sub["y"]).to_numpy()
    dist = np.sqrt(dx * dx + dy * dy)
    out = pd.DataFrame({
        "node_id": common_nodes,
        "x_r": r_sub["x"].to_numpy(),
        "y_r": r_sub["y"].to_numpy(),
        "x_py": p_sub["x"].to_numpy(),
        "y_py": p_sub["y"].to_numpy(),
        "diff_xy": dist,
    })
    out["ok"] = out["diff_xy"] <= tolerance
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare R gaia and Python gaiapy quadratic MPR outputs.")
    parser.add_argument("--trees-dir", type=str, default="test/trees", help="Directory containing .trees files")
    parser.add_argument("--out-dir", type=str, default="gaiapy_tests/out", help="Output directory for reports and intermediates")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for x,y differences")
    parser.add_argument("--rscript", type=str, default="Rscript", help="Path to Rscript executable")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    trees_dir = Path(args.trees_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_gaia_r = Path(__file__).resolve().parent / "run_gaia.R"

    tree_files = discover_trees(trees_dir)
    if len(tree_files) == 0:
        print(f"No .trees files found under {trees_dir}")
        sys.exit(0)

    overall = []

    for tree_path in tree_files:
        try:
            if args.verbose:
                print(f"[py] Processing {tree_path.name}")
                try:
                    if GP_PATH:
                        print(f"[py] Using gaiapy from: {GP_PATH}")
                except Exception:
                    pass
            ts = tskit.load(str(tree_path))

            # Extract samples (metadata if available; otherwise individual locations)
            samples_xy = extract_samples_xy(ts)
            # Prepare R CSV from individuals as required (z=0)
            samples_csv = out_dir / f"{tree_path.stem}.samples.csv"
            write_samples_csv(ts, samples_csv)

            # Run R GAIA
            r_out_csv = out_dir / f"{tree_path.stem}.r_ancestors.csv"
            run_r_gaia(args.rscript, run_gaia_r, tree_path, samples_csv, r_out_csv, args.verbose)

            # Python gaiapy
            locs = run_python_gaiapy(ts, samples_xy)
            py_df = pd.DataFrame({
                "node_id": np.arange(locs.shape[0], dtype=int),
                "x": locs[:, 0],
                "y": locs[:, 1],
                "z": 0.0,
            })
            sample_nodes = samples_xy[:, 0].astype(int)
            py_df = py_df[~py_df["node_id"].isin(sample_nodes)]

            # Load R output
            r_df = pd.read_csv(r_out_csv)

            # Compare
            comp = compare_results(r_df, py_df, args.tolerance)
            comp_csv = out_dir / f"{tree_path.stem}.comparison.csv"
            comp.to_csv(comp_csv, index=False)

            # Stats
            n_nodes = int(ts.num_nodes)
            n_samples = int(len(ts.samples()))
            n_comp = int(len(comp))
            diffs = comp["diff_xy"].to_numpy() if n_comp > 0 else np.array([], dtype=float)
            mean_diff = float(np.mean(diffs)) if n_comp > 0 else 0.0
            median_diff = float(np.median(diffs)) if n_comp > 0 else 0.0
            std_diff = float(np.std(diffs)) if n_comp > 0 else 0.0
            min_diff = float(np.min(diffs)) if n_comp > 0 else 0.0
            p95_diff = float(np.quantile(diffs, 0.95)) if n_comp > 0 else 0.0
            max_diff = float(np.max(diffs)) if n_comp > 0 else 0.0

            ok = bool(comp["ok"].all()) if n_comp > 0 else True

            # Debug output
            print(f"{tree_path.name}: nodes={n_nodes}, samples={n_samples}, compared={n_comp}")
            print(
                "  diffs: "
                f"mean={mean_diff:.9f}, median={median_diff:.9f}, std={std_diff:.9f}, "
                f"min={min_diff:.9f}, p95={p95_diff:.9f}, max={max_diff:.9f}"
            )
            status = "OK" if ok else "FAIL"
            print(f"  status: {status}")

            if args.verbose and n_comp > 0 and not ok:
                worst = comp.sort_values("diff_xy", ascending=False).head(5)
                print("  worst nodes (top 5 by diff):")
                for _i, row in worst.iterrows():
                    print(
                        f"    node_id={int(row['node_id'])}, "
                        f"diff={row['diff_xy']:.9f}, "
                        f"R=({row['x_r']:.9f},{row['y_r']:.9f}), "
                        f"Py=({row['x_py']:.9f},{row['y_py']:.9f})"
                    )

            overall.append({
                "tree": tree_path.name,
                "num_nodes": n_nodes,
                "num_samples": n_samples,
                "num_compared": n_comp,
                "all_within_tol": ok,
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "std_diff": std_diff,
                "min_diff": min_diff,
                "p95_diff": p95_diff,
                "max_diff": max_diff,
            })
        except Exception as e:
            overall.append({
                "tree": tree_path.name,
                "num_nodes": None,
                "num_samples": None,
                "num_compared": 0,
                "all_within_tol": False,
                "mean_diff": None,
                "median_diff": None,
                "std_diff": None,
                "min_diff": None,
                "p95_diff": None,
                "max_diff": None,
                "error": str(e),
            })
            print(f"{tree_path.name}: ERROR {e}")

    # Write summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall, f, indent=2)

    any_fail = any((not rec.get("all_within_tol", False)) for rec in overall)
    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()


