#!/usr/bin/env python3
"""Run sliding-window persistent homology for all 28 CPC section pairs.

This is the heavy computation step. Run it overnight on a laptop or on
Google Colab with an A100/T4 runtime. Results are cached per-pair, so
it resumes cleanly if interrupted.

Usage:
    python scripts/run_topology.py                    # ripser backend (default)
    python scripts/run_topology.py --backend flagser  # directed flag complex
    python scripts/run_topology.py --backend both     # run both backends
    python scripts/run_topology.py --max-nodes 50000  # increase for high-RAM machines

Output:
    data/topology_cache/sliding_{A}_{B}_w5_s1.pkl       (ripser)
    data/topology_cache/sliding_{A}_{B}_w5_s1_flagser.pkl (flagser)
"""

from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.topology import sliding_window_topology, TOPOLOGY_CACHE
from src.utils import get_logger, log_memory

logger = get_logger("run_topology")

CPC_SECTIONS = ["A", "B", "C", "D", "E", "F", "G", "H"]
ALL_PAIRS = list(itertools.combinations(CPC_SECTIONS, 2))  # 28 pairs


def already_cached(section_a: str, section_b: str, backend: str,
                   window_years: int = 5, stride_years: int = 1) -> bool:
    """Check if results for this pair/backend are already cached."""
    suffix = f"_{backend}" if backend != "ripser" else ""
    cache_file = TOPOLOGY_CACHE / f"sliding_{section_a}_{section_b}_w{window_years}_s{stride_years}{suffix}.pkl"
    return cache_file.exists()


def run_all(
    backend: str = "ripser",
    max_nodes: int = 30_000,
    window_years: int = 5,
    stride_years: int = 1,
    start_year: int = 1985,
    end_year: int = 2023,
    max_dim: int = 2,
) -> None:
    """Run topology computation for all 28 CPC section pairs."""

    # Load data
    logger.info("Loading data...")
    data_dir = PROJECT_ROOT / "data"
    citations = pd.read_parquet(data_dir / "citations.parquet")
    cpc_map = pd.read_parquet(data_dir / "cpc_map.parquet")
    logger.info(
        "Loaded %d citations, %d CPC mappings",
        len(citations), len(cpc_map),
    )
    log_memory("After loading data")

    # Count how many pairs are already done vs remaining
    remaining = []
    for a, b in ALL_PAIRS:
        if already_cached(a, b, backend, window_years, stride_years):
            logger.info("(%s, %s) [%s] — already cached, skipping", a, b, backend)
        else:
            remaining.append((a, b))

    if not remaining:
        logger.info("All 28 pairs already cached for backend=%s. Nothing to do.", backend)
        return

    logger.info(
        "%d / %d pairs remaining for backend=%s",
        len(remaining), len(ALL_PAIRS), backend,
    )

    total_start = time.perf_counter()
    completed = 0
    times = []

    for i, (section_a, section_b) in enumerate(remaining):
        pair_start = time.perf_counter()
        logger.info(
            "=" * 60 + "\n  Pair %d/%d: (%s, %s) [%s]\n" + "=" * 60,
            i + 1, len(remaining), section_a, section_b, backend,
        )

        try:
            result = sliding_window_topology(
                citations=citations,
                cpc_map=cpc_map,
                section_a=section_a,
                section_b=section_b,
                window_years=window_years,
                stride_years=stride_years,
                start_year=start_year,
                end_year=end_year,
                max_dim=max_dim,
                max_nodes=max_nodes,
                use_cache=True,
                backend=backend,
            )
            pair_elapsed = time.perf_counter() - pair_start
            times.append(pair_elapsed)
            completed += 1

            logger.info(
                "(%s, %s) done in %.1f min — %d windows, β₁ range: [%d, %d]",
                section_a, section_b,
                pair_elapsed / 60,
                len(result),
                result["beta_1"].min() if len(result) > 0 else 0,
                result["beta_1"].max() if len(result) > 0 else 0,
            )

            # Estimate remaining time
            avg_time = sum(times) / len(times)
            remaining_pairs = len(remaining) - (i + 1)
            eta_min = avg_time * remaining_pairs / 60
            logger.info(
                "Progress: %d/%d complete | Avg: %.1f min/pair | ETA: %.0f min",
                completed, len(remaining), avg_time / 60, eta_min,
            )

        except Exception as e:
            logger.error(
                "FAILED on (%s, %s): %s — continuing to next pair",
                section_a, section_b, e,
            )
            continue

        # Force garbage collection between pairs to prevent memory accumulation
        gc.collect()
        log_memory(f"After pair ({section_a}, {section_b})")

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "\nDone! %d/%d pairs completed in %.1f min (backend=%s)",
        completed, len(remaining), total_elapsed / 60, backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sliding-window topology for all 28 CPC section pairs",
    )
    parser.add_argument(
        "--backend", choices=["ripser", "flagser", "both"], default="flagser",
        help="Persistence backend (default: flagser — avoids distance matrix OOM)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=30_000,
        help="Max nodes per subgraph before reduction (default: 30000)",
    )
    parser.add_argument(
        "--window-years", type=int, default=5,
        help="Sliding window width in years (default: 5)",
    )
    parser.add_argument(
        "--stride-years", type=int, default=1,
        help="Stride between windows (default: 1)",
    )
    parser.add_argument(
        "--start-year", type=int, default=1985,
        help="First window end year (default: 1985)",
    )
    parser.add_argument(
        "--end-year", type=int, default=2023,
        help="Last window end year (default: 2023)",
    )
    args = parser.parse_args()

    logger.info("Configuration:")
    logger.info("  Backend:      %s", args.backend)
    logger.info("  Max nodes:    %d", args.max_nodes)
    logger.info("  Window:       %d years, stride %d", args.window_years, args.stride_years)
    logger.info("  Range:        %d - %d", args.start_year, args.end_year)

    backends = ["ripser", "flagser"] if args.backend == "both" else [args.backend]

    for backend in backends:
        logger.info("\n" + "#" * 60)
        logger.info("# Running backend: %s", backend)
        logger.info("#" * 60)
        run_all(
            backend=backend,
            max_nodes=args.max_nodes,
            window_years=args.window_years,
            stride_years=args.stride_years,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    logger.info("All done. Results in %s", TOPOLOGY_CACHE)


if __name__ == "__main__":
    main()
