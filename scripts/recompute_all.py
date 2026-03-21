#!/usr/bin/env python3
"""Recompute all topology with scale-normalized distances for all 28 CPC pairs.

This script:
1. Loads patent citation data
2. Computes topology for all 28 CPC section pairs with scale-normalized
   distance matrices (controlling for the density confound)
3. Computes global topology
4. Clears stale null model caches
5. Saves everything to data/topology_cache/

Expected runtime: ~6-8 hours on 16GB MacBook (each window takes seconds,
but building co-citation matrices from ~118M citations takes ~10 min per pair).
"""

import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.topology import (
    compute_all_priority_pairs,
    compute_global_topology,
    ALL_PAIRS,
)
from src.utils import get_logger

logger = get_logger(__name__)

CACHE_DIR = "data/topology_cache"
NULL_CACHE = Path("data/null_cache")


def main():
    start = time.time()

    # Load data
    logger.info("Loading citation data...")
    citations = pd.read_parquet("data/citations.parquet")
    cpc_map = pd.read_parquet("data/cpc_map.parquet")
    citations["citing_year"] = pd.to_datetime(citations["citing_date"]).dt.year
    logger.info(f"  {len(citations):,} citations, {len(cpc_map):,} CPC mappings")

    # Clear stale null cache (was computed with unnormalized distances)
    if NULL_CACHE.exists():
        logger.info("Clearing stale null model cache...")
        shutil.rmtree(NULL_CACHE)
        logger.info("  Done")

    # Compute all 28 pairs
    logger.info(f"\n{'='*60}")
    logger.info(f"Computing topology for all 28 CPC section pairs")
    logger.info(f"Cache: {CACHE_DIR}")
    logger.info(f"Scale normalization: ON (density confound control)")
    logger.info(f"{'='*60}\n")

    pair_results = compute_all_priority_pairs(
        citations,
        cpc_map,
        cache_dir=CACHE_DIR,
        pairs=ALL_PAIRS,
    )

    logger.info(f"\nPair results: {len(pair_results)} total window-pair observations")
    if len(pair_results) > 0:
        pairs_computed = pair_results["section_pair"].nunique()
        logger.info(f"  Pairs computed: {pairs_computed}/28")

    # Compute global topology
    logger.info(f"\n{'='*60}")
    logger.info("Computing global topology (all sections)")
    logger.info(f"{'='*60}\n")

    global_results = compute_global_topology(
        citations,
        cpc_map,
        cache_dir=CACHE_DIR,
    )

    logger.info(f"\nGlobal results: {len(global_results)} windows")

    elapsed = time.time() - start
    hours = elapsed / 3600
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE in {hours:.1f} hours ({elapsed:.0f}s)")
    logger.info(f"Results cached to: {CACHE_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
