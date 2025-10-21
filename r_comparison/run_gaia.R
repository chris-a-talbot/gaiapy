#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

# Declare NSE variables for data.table to appease linters
utils::globalVariables(c("node_id", "x", "y", "z", ".", "V1", "V2", "V3"))

# Parse command-line arguments (simple base R parser)
args <- commandArgs(trailingOnly = TRUE)
parse_args <- function(args) {
  res <- list(tree = NULL, samples_csv = NULL, out_csv = NULL, verbose = FALSE)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--tree", "-t")) { res$tree <- args[[i+1]]; i <- i + 2; next }
    if (key %in% c("--samples_csv", "-s")) { res$samples_csv <- args[[i+1]]; i <- i + 2; next }
    if (key %in% c("--out_csv", "-o")) { res$out_csv <- args[[i+1]]; i <- i + 2; next }
    if (key %in% c("--verbose", "-v")) { res$verbose <- TRUE; i <- i + 1; next }
    stop(sprintf("Unknown argument: %s", key))
  }
  if (is.null(res$tree) || is.null(res$samples_csv) || is.null(res$out_csv)) {
    stop("Usage: run_gaia.R --tree <trees> --samples_csv <samples.csv> --out_csv <out.csv> [--verbose]")
  }
  res
}

main <- function() {
  opts <- parse_args(args)
  base <- basename(opts$tree)
  if (opts$verbose) message(sprintf("[run_gaia] Processing %s", base))

  # Load packages lazily to allow helpful error messages
  if (!requireNamespace("gaia", quietly = TRUE)) {
    stop("R package 'gaia' is required. Install it before running.")
  }

  # Load tree and samples
  tree <- gaia::treeseq_load(opts$tree)
  sample_dt <- data.table::fread(opts$samples_csv)

  # Ensure schema and z=0
  needed_cols <- c("node_id", "x", "y", "z")
  if (!all(needed_cols %in% names(sample_dt))) {
    stop(sprintf("samples_csv must have columns: %s", paste(needed_cols, collapse = ",")))
  }
  sample_dt[, ("z") := 0.0]

  # GAIA expects a numeric matrix (node_id, x, y[, z])
  sample_mat <- as.matrix(sample_dt[, .SD, .SDcols = c("node_id", "x", "y", "z")])

  # Run quadratic MPR and minimize
  mpr_output <- gaia::treeseq_quadratic_mpr(tree, sample_mat, TRUE)
  locations_output <- gaia::treeseq_quadratic_mpr_minimize(mpr_output)

  # Coerce output to data.table with x,y,z columns
  locations_dt <- data.table::as.data.table(locations_output)
  if (ncol(locations_dt) == 3) {
    data.table::setnames(locations_dt, c("V1", "V2", "V3"), c("x", "y", "z"))
  } else if (ncol(locations_dt) == 2) {
    data.table::setnames(locations_dt, c("V1", "V2"), c("x", "y"))
    locations_dt[, ("z") := 0.0]
  } else {
    stop(sprintf("Unexpected number of columns in GAIA output: %d", ncol(locations_dt)))
  }

  # Add node_id (0-based)
  locations_dt[, ("node_id") := .I - 1L]

  # Filter out sample nodes
  sample_nodes <- as.integer(sample_dt$node_id)
  filtered_locations <- locations_dt[!(get("node_id") %in% sample_nodes)]
  data.table::setcolorder(filtered_locations, c("node_id", "x", "y", "z"))

  # Save
  dir.create(dirname(opts$out_csv), recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(filtered_locations, opts$out_csv)
  if (opts$verbose) message(sprintf("[run_gaia] Wrote %s", opts$out_csv))
}

tryCatch({
  main()
}, error = function(e) {
  message(sprintf("[run_gaia] ERROR: %s", e$message))
  q(status = 1, save = "no")
})


