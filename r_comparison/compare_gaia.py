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


def run_python_gaiapy(ts: tskit.TreeSequence, samples_xy: np.ndarray, preserve_sample_locations: bool = False) -> np.ndarray:
    # Match R call which sets use_branch_lengths=TRUE
    mpr = gp.quadratic_mpr(ts, samples_xy, use_branch_lengths=True)
    locs = gp.quadratic_mpr_minimize(mpr, preserve_sample_locations=preserve_sample_locations)  # shape (n_nodes, 2)
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


def compare_sample_locations_to_true(
    computed_df: pd.DataFrame, true_samples: np.ndarray, tolerance: float
) -> pd.DataFrame:
    """Compare computed sample locations against true input locations.
    
    Args:
        computed_df: DataFrame with node_id, x, y columns (computed locations)
        true_samples: Array with shape (n_samples, 3) where columns are [node_id, x, y]
        tolerance: Tolerance for distance differences
        
    Returns:
        DataFrame with comparison results
    """
    true_df = pd.DataFrame({
        "node_id": true_samples[:, 0].astype(int),
        "x_true": true_samples[:, 1].astype(float),
        "y_true": true_samples[:, 2].astype(float),
    })
    
    # Merge on node_id
    merged = computed_df.merge(true_df, on="node_id", how="inner")
    if len(merged) == 0:
        return pd.DataFrame(columns=["node_id", "x_computed", "y_computed", "x_true", "y_true", "diff_xy", "ok"])
    
    dx = (merged["x"] - merged["x_true"]).to_numpy()
    dy = (merged["y"] - merged["y_true"]).to_numpy()
    dist = np.sqrt(dx * dx + dy * dy)
    
    out = pd.DataFrame({
        "node_id": merged["node_id"].to_numpy(),
        "x_computed": merged["x"].to_numpy(),
        "y_computed": merged["y"].to_numpy(),
        "x_true": merged["x_true"].to_numpy(),
        "y_true": merged["y_true"].to_numpy(),
        "diff_xy": dist,
    })
    out["ok"] = out["diff_xy"] <= tolerance
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare R gaia and Python gaiapy quadratic MPR outputs.")
    parser.add_argument("--trees-dir", type=str, default="r_comparison/trees", help="Directory containing .trees files")
    parser.add_argument("--out-dir", type=str, default="r_comparison/results", help="Output directory for reports and intermediates")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Tolerance for x,y differences")
    parser.add_argument("--rscript", type=str, default="Rscript", help="Path to Rscript executable")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--preserve-sample-locations", action="store_true", 
                       help="If set, Python will preserve original sample locations in output. "
                            "By default, both R and Python optimize sample locations (intended behavior).")
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
            locs = run_python_gaiapy(ts, samples_xy, preserve_sample_locations=args.preserve_sample_locations)
            py_df = pd.DataFrame({
                "node_id": np.arange(locs.shape[0], dtype=int),
                "x": locs[:, 0],
                "y": locs[:, 1],
                "z": 0.0,
            })
            sample_nodes = set(samples_xy[:, 0].astype(int))

            # Load R output
            r_df = pd.read_csv(r_out_csv)

            # Separate internal nodes and sample nodes
            py_internal = py_df[~py_df["node_id"].isin(sample_nodes)].copy()
            py_samples = py_df[py_df["node_id"].isin(sample_nodes)].copy()
            r_internal = r_df[~r_df["node_id"].isin(sample_nodes)].copy()
            r_samples = r_df[r_df["node_id"].isin(sample_nodes)].copy()

            # Compare internal nodes: R vs Python
            comp_internal = compare_results(r_internal, py_internal, args.tolerance)
            comp_internal_csv = out_dir / f"{tree_path.stem}.comparison_internal.csv"
            comp_internal.to_csv(comp_internal_csv, index=False)

            # Compare sample nodes: R vs Python (cross-validation)
            comp_samples_rvspy = compare_results(r_samples, py_samples, args.tolerance)
            comp_samples_rvspy_csv = out_dir / f"{tree_path.stem}.comparison_samples_rvspy.csv"
            comp_samples_rvspy.to_csv(comp_samples_rvspy_csv, index=False)

            # Compare sample nodes: computed vs true (validation)
            # Note: By default, both R and Python optimize sample locations, so they may differ from input.
            # This is intended behavior. Only test against true locations if preserve_sample_locations is enabled.
            comp_samples_r_vs_true = compare_sample_locations_to_true(r_samples, samples_xy, args.tolerance)
            comp_samples_py_vs_true = compare_sample_locations_to_true(py_samples, samples_xy, args.tolerance)
            comp_samples_r_vs_true_csv = out_dir / f"{tree_path.stem}.comparison_samples_r_vs_true.csv"
            comp_samples_py_vs_true_csv = out_dir / f"{tree_path.stem}.comparison_samples_py_vs_true.csv"
            comp_samples_r_vs_true.to_csv(comp_samples_r_vs_true_csv, index=False)
            comp_samples_py_vs_true.to_csv(comp_samples_py_vs_true_csv, index=False)

            # Legacy combined comparison (internal nodes only, for backward compatibility)
            comp = comp_internal.copy()
            comp_csv = out_dir / f"{tree_path.stem}.comparison.csv"
            comp.to_csv(comp_csv, index=False)

            # Stats for internal nodes
            n_nodes = int(ts.num_nodes)
            n_samples = int(len(ts.samples()))
            n_comp_internal = int(len(comp_internal))
            diffs_internal = comp_internal["diff_xy"].to_numpy() if n_comp_internal > 0 else np.array([], dtype=float)
            mean_diff_internal = float(np.mean(diffs_internal)) if n_comp_internal > 0 else 0.0
            median_diff_internal = float(np.median(diffs_internal)) if n_comp_internal > 0 else 0.0
            std_diff_internal = float(np.std(diffs_internal)) if n_comp_internal > 0 else 0.0
            min_diff_internal = float(np.min(diffs_internal)) if n_comp_internal > 0 else 0.0
            p95_diff_internal = float(np.quantile(diffs_internal, 0.95)) if n_comp_internal > 0 else 0.0
            max_diff_internal = float(np.max(diffs_internal)) if n_comp_internal > 0 else 0.0
            ok_internal = bool(comp_internal["ok"].all()) if n_comp_internal > 0 else True

            # Stats for sample nodes: R vs Python
            n_comp_samples_rvspy = int(len(comp_samples_rvspy))
            diffs_samples_rvspy = comp_samples_rvspy["diff_xy"].to_numpy() if n_comp_samples_rvspy > 0 else np.array([], dtype=float)
            mean_diff_samples_rvspy = float(np.mean(diffs_samples_rvspy)) if n_comp_samples_rvspy > 0 else 0.0
            median_diff_samples_rvspy = float(np.median(diffs_samples_rvspy)) if n_comp_samples_rvspy > 0 else 0.0
            std_diff_samples_rvspy = float(np.std(diffs_samples_rvspy)) if n_comp_samples_rvspy > 0 else 0.0
            min_diff_samples_rvspy = float(np.min(diffs_samples_rvspy)) if n_comp_samples_rvspy > 0 else 0.0
            p95_diff_samples_rvspy = float(np.quantile(diffs_samples_rvspy, 0.95)) if n_comp_samples_rvspy > 0 else 0.0
            max_diff_samples_rvspy = float(np.max(diffs_samples_rvspy)) if n_comp_samples_rvspy > 0 else 0.0
            ok_samples_rvspy = bool(comp_samples_rvspy["ok"].all()) if n_comp_samples_rvspy > 0 else True

            # Stats for sample nodes: R vs true
            n_comp_samples_r_vs_true = int(len(comp_samples_r_vs_true))
            diffs_samples_r_vs_true = comp_samples_r_vs_true["diff_xy"].to_numpy() if n_comp_samples_r_vs_true > 0 else np.array([], dtype=float)
            mean_diff_samples_r_vs_true = float(np.mean(diffs_samples_r_vs_true)) if n_comp_samples_r_vs_true > 0 else 0.0
            max_diff_samples_r_vs_true = float(np.max(diffs_samples_r_vs_true)) if n_comp_samples_r_vs_true > 0 else 0.0
            ok_samples_r_vs_true = bool(comp_samples_r_vs_true["ok"].all()) if n_comp_samples_r_vs_true > 0 else True

            # Stats for sample nodes: Python vs true
            n_comp_samples_py_vs_true = int(len(comp_samples_py_vs_true))
            diffs_samples_py_vs_true = comp_samples_py_vs_true["diff_xy"].to_numpy() if n_comp_samples_py_vs_true > 0 else np.array([], dtype=float)
            mean_diff_samples_py_vs_true = float(np.mean(diffs_samples_py_vs_true)) if n_comp_samples_py_vs_true > 0 else 0.0
            max_diff_samples_py_vs_true = float(np.max(diffs_samples_py_vs_true)) if n_comp_samples_py_vs_true > 0 else 0.0
            ok_samples_py_vs_true = bool(comp_samples_py_vs_true["ok"].all()) if n_comp_samples_py_vs_true > 0 else True
            
            # When preserve_sample_locations is False (default), sample locations are optimized
            # and may differ from input. This is expected and matches R behavior.
            # Only require exact match when preserve_sample_locations=True
            # Note: R always optimizes sample locations, so R vs true will always differ in default mode
            if args.preserve_sample_locations:
                # When preserving, Python should match true locations
                ok_samples_py_vs_true_required = ok_samples_py_vs_true
                # R still optimizes, so R vs true differences are still expected
                ok_samples_r_vs_true_required = True
            else:
                # When optimizing (default), differences from true are expected for both R and Python
                # We still report them but don't fail the test
                ok_samples_py_vs_true_required = True
                ok_samples_r_vs_true_required = True

            # Overall status: all tests must pass
            # Note: Sample locations vs true may differ when optimizing (default behavior)
            ok = ok_internal and ok_samples_rvspy and ok_samples_r_vs_true_required and ok_samples_py_vs_true_required

            # Debug output
            print(f"{tree_path.name}: nodes={n_nodes}, samples={n_samples}")
            print(f"  Internal nodes (R vs Py): compared={n_comp_internal}, status={'OK' if ok_internal else 'FAIL'}")
            if n_comp_internal > 0:
                print(
                    "    diffs: "
                    f"mean={mean_diff_internal:.9f}, median={median_diff_internal:.9f}, std={std_diff_internal:.9f}, "
                    f"min={min_diff_internal:.9f}, p95={p95_diff_internal:.9f}, max={max_diff_internal:.9f}"
                )
            print(f"  Sample nodes (R vs Py): compared={n_comp_samples_rvspy}, status={'OK' if ok_samples_rvspy else 'FAIL'}")
            if n_comp_samples_rvspy > 0:
                print(
                    "    diffs: "
                    f"mean={mean_diff_samples_rvspy:.9f}, median={median_diff_samples_rvspy:.9f}, std={std_diff_samples_rvspy:.9f}, "
                    f"min={min_diff_samples_rvspy:.9f}, p95={p95_diff_samples_rvspy:.9f}, max={max_diff_samples_rvspy:.9f}"
                )
            print(f"  Sample nodes (R vs true): compared={n_comp_samples_r_vs_true}, status={'OK' if ok_samples_r_vs_true else 'FAIL'}")
            if n_comp_samples_r_vs_true > 0:
                print(f"    diffs: mean={mean_diff_samples_r_vs_true:.9f}, max={max_diff_samples_r_vs_true:.9f}")
                if not args.preserve_sample_locations:
                    print(f"    NOTE: Differences are expected - R optimizes sample locations (intended behavior)")
            print(f"  Sample nodes (Py vs true): compared={n_comp_samples_py_vs_true}, status={'OK' if ok_samples_py_vs_true_required else 'FAIL'}")
            if n_comp_samples_py_vs_true > 0:
                print(f"    diffs: mean={mean_diff_samples_py_vs_true:.9f}, max={max_diff_samples_py_vs_true:.9f}")
                if not args.preserve_sample_locations:
                    print(f"    NOTE: Differences are expected - Python optimizes sample locations (intended behavior, matches R)")
            status = "OK" if ok else "FAIL"
            print(f"  Overall status: {status}")

            if args.verbose:
                if n_comp_internal > 0 and not ok_internal:
                    worst = comp_internal.sort_values("diff_xy", ascending=False).head(5)
                    print("  Worst internal nodes (top 5 by diff):")
                    for _i, row in worst.iterrows():
                        print(
                            f"    node_id={int(row['node_id'])}, "
                            f"diff={row['diff_xy']:.9f}, "
                            f"R=({row['x_r']:.9f},{row['y_r']:.9f}), "
                            f"Py=({row['x_py']:.9f},{row['y_py']:.9f})"
                        )
                if n_comp_samples_rvspy > 0 and not ok_samples_rvspy:
                    worst = comp_samples_rvspy.sort_values("diff_xy", ascending=False).head(5)
                    print("  Worst sample nodes R vs Py (top 5 by diff):")
                    for _i, row in worst.iterrows():
                        print(
                            f"    node_id={int(row['node_id'])}, "
                            f"diff={row['diff_xy']:.9f}, "
                            f"R=({row['x_r']:.9f},{row['y_r']:.9f}), "
                            f"Py=({row['x_py']:.9f},{row['y_py']:.9f})"
                        )
                if n_comp_samples_r_vs_true > 0 and not ok_samples_r_vs_true:
                    worst = comp_samples_r_vs_true.sort_values("diff_xy", ascending=False).head(5)
                    print("  Worst sample nodes R vs true (top 5 by diff):")
                    for _i, row in worst.iterrows():
                        print(
                            f"    node_id={int(row['node_id'])}, "
                            f"diff={row['diff_xy']:.9f}, "
                            f"R=({row['x_computed']:.9f},{row['y_computed']:.9f}), "
                            f"true=({row['x_true']:.9f},{row['y_true']:.9f})"
                        )
                if n_comp_samples_py_vs_true > 0 and not ok_samples_py_vs_true:
                    worst = comp_samples_py_vs_true.sort_values("diff_xy", ascending=False).head(5)
                    print("  Worst sample nodes Py vs true (top 5 by diff):")
                    for _i, row in worst.iterrows():
                        print(
                            f"    node_id={int(row['node_id'])}, "
                            f"diff={row['diff_xy']:.9f}, "
                            f"Py=({row['x_computed']:.9f},{row['y_computed']:.9f}), "
                            f"true=({row['x_true']:.9f},{row['y_true']:.9f})"
                        )

            overall.append({
                "tree": tree_path.name,
                "num_nodes": n_nodes,
                "num_samples": n_samples,
                # Internal nodes stats
                "num_compared_internal": n_comp_internal,
                "internal_all_within_tol": ok_internal,
                "internal_mean_diff": mean_diff_internal,
                "internal_median_diff": median_diff_internal,
                "internal_std_diff": std_diff_internal,
                "internal_min_diff": min_diff_internal,
                "internal_p95_diff": p95_diff_internal,
                "internal_max_diff": max_diff_internal,
                # Sample nodes R vs Python stats
                "num_compared_samples_rvspy": n_comp_samples_rvspy,
                "samples_rvspy_all_within_tol": ok_samples_rvspy,
                "samples_rvspy_mean_diff": mean_diff_samples_rvspy,
                "samples_rvspy_median_diff": median_diff_samples_rvspy,
                "samples_rvspy_std_diff": std_diff_samples_rvspy,
                "samples_rvspy_min_diff": min_diff_samples_rvspy,
                "samples_rvspy_p95_diff": p95_diff_samples_rvspy,
                "samples_rvspy_max_diff": max_diff_samples_rvspy,
                # Sample nodes R vs true stats
                "num_compared_samples_r_vs_true": n_comp_samples_r_vs_true,
                "samples_r_vs_true_all_within_tol": ok_samples_r_vs_true,
                "samples_r_vs_true_mean_diff": mean_diff_samples_r_vs_true,
                "samples_r_vs_true_max_diff": max_diff_samples_r_vs_true,
                # Sample nodes Python vs true stats
                "num_compared_samples_py_vs_true": n_comp_samples_py_vs_true,
                "samples_py_vs_true_all_within_tol": ok_samples_py_vs_true,
                "samples_py_vs_true_mean_diff": mean_diff_samples_py_vs_true,
                "samples_py_vs_true_max_diff": max_diff_samples_py_vs_true,
                # Overall status
                "all_within_tol": ok,
                # Legacy fields for backward compatibility
                "num_compared": n_comp_internal,
                "mean_diff": mean_diff_internal,
                "median_diff": median_diff_internal,
                "std_diff": std_diff_internal,
                "min_diff": min_diff_internal,
                "p95_diff": p95_diff_internal,
                "max_diff": max_diff_internal,
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


