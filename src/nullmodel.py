"""Null model generation for hypothesis testing.

Provides two types of null distributions:
    1. Random baseline: random CPC section pairs at random times.
    2. Matched null: same CPC pair as a breakthrough, shifted to non-breakthrough periods.

Both reuse the topology pipeline from src/topology.py and cache results.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.breakthroughs import Breakthrough, get_precursor_window
from src.topology import sliding_window_topology, topology_summary, reduce_graph
from src.graph import cpc_subgraph_nx
from src.utils import DATA_DIR, get_logger, timer

logger = get_logger(__name__)

NULL_CACHE = DATA_DIR / "null_cache"

CPC_SECTIONS = list("ABCDEFGH")


# ---------------------------------------------------------------------------
# Random CPC-pair baseline
# ---------------------------------------------------------------------------

@timer
def random_cpc_pair_baseline(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    n_samples: int = 100,
    window_years: int = 5,
    year_range: tuple[int, int] = (1985, 2018),
    max_dim: int = 2,
    max_nodes: int = 30_000,
    seed: int = 42,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Generate null topology measurements from random CPC pairs and years.

    For each sample: pick a random CPC section pair and a random year,
    then compute the same topological metrics as for real breakthroughs.

    Args:
        citations: Full citation DataFrame.
        cpc_map: CPC mapping DataFrame.
        n_samples: Number of random samples.
        window_years: Window width in years.
        year_range: (min_year, max_year) for random sampling.
        max_dim: Maximum homological dimension.
        max_nodes: Max nodes per subgraph.
        seed: Random seed for reproducibility.
        use_cache: Whether to cache results.

    Returns:
        DataFrame with topology metrics for each random sample.
    """
    cache_file = NULL_CACHE / f"random_baseline_n{n_samples}_w{window_years}_s{seed}.pkl"
    if use_cache and cache_file.exists():
        logger.info("Loading cached random baseline")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)

    citations = citations.copy()
    citations["citing_date"] = pd.to_datetime(citations["citing_date"])

    # Generate all unique section pairs
    section_pairs = []
    for i, a in enumerate(CPC_SECTIONS):
        for b in CPC_SECTIONS[i + 1:]:
            section_pairs.append((a, b))

    rows = []
    for i in tqdm(range(n_samples), desc="Random baseline"):
        # Random section pair and year
        pair_idx = rng.integers(0, len(section_pairs))
        sec_a, sec_b = section_pairs[pair_idx]
        year = int(rng.integers(year_range[0], year_range[1] + 1))

        # Build window
        win_start = pd.Timestamp(f"{year - window_years + 1}-01-01")
        win_end = pd.Timestamp(f"{year}-12-31")

        mask = (citations["citing_date"] >= win_start) & (citations["citing_date"] <= win_end)
        window_cites = citations[mask]

        if len(window_cites) == 0:
            continue

        G = cpc_subgraph_nx(window_cites, cpc_map, sec_a, sec_b, max_nodes=max_nodes)

        if G.number_of_nodes() < 3:
            continue

        G = reduce_graph(G, max_nodes=max_nodes)
        summary = topology_summary(G, max_dim=max_dim)
        summary["year"] = year
        summary["section_a"] = sec_a
        summary["section_b"] = sec_b
        summary["sample_idx"] = i
        rows.append(summary)

    result = pd.DataFrame(rows)

    if use_cache and len(result) > 0:
        NULL_CACHE.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

    logger.info("Random baseline: %d/%d samples computed", len(result), n_samples)
    return result


# ---------------------------------------------------------------------------
# Matched null model
# ---------------------------------------------------------------------------

@timer
def matched_null(
    breakthrough: Breakthrough,
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    n_samples: int = 100,
    window_years: int = 5,
    years_before: int = 10,
    exclusion_buffer: int = 3,
    max_dim: int = 2,
    max_nodes: int = 30_000,
    seed: int = 42,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Generate null topology measurements matched to a specific breakthrough.

    Holds the CPC section pair fixed and varies the time window,
    excluding years near the actual breakthrough.

    Args:
        breakthrough: The breakthrough to generate null for.
        citations: Full citation DataFrame.
        cpc_map: CPC mapping DataFrame.
        n_samples: Number of null samples.
        window_years: Window width.
        years_before: Precursor window length.
        exclusion_buffer: Years around the breakthrough to exclude from null.
        max_dim: Maximum homological dimension.
        max_nodes: Max nodes per subgraph.
        seed: Random seed.
        use_cache: Whether to cache.

    Returns:
        DataFrame of null topology measurements.
    """
    bt_name = breakthrough.name.replace(" ", "_").lower()[:30]
    cache_file = NULL_CACHE / f"matched_{bt_name}_n{n_samples}_s{seed}.pkl"
    if use_cache and cache_file.exists():
        logger.info("Loading cached matched null for %s", breakthrough.name)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)

    citations = citations.copy()
    citations["citing_date"] = pd.to_datetime(citations["citing_date"])

    # Determine CPC sections for this breakthrough
    if len(breakthrough.cpc_sections) >= 2:
        sec_a, sec_b = breakthrough.cpc_sections[0], breakthrough.cpc_sections[1]
    else:
        sec_a = breakthrough.cpc_sections[0]
        sec_b = sec_a  # Same-section analysis

    # Exclude years around the breakthrough
    exclude_start = breakthrough.filing_year - years_before - exclusion_buffer
    exclude_end = breakthrough.recognition_year + exclusion_buffer

    # Available years for null sampling
    all_years = list(range(1985, 2019))
    null_years = [y for y in all_years if y < exclude_start or y > exclude_end]

    if len(null_years) == 0:
        logger.warning("No null years available for %s", breakthrough.name)
        return pd.DataFrame()

    rows = []
    for i in tqdm(range(n_samples), desc=f"Matched null: {breakthrough.name}"):
        year = int(rng.choice(null_years))

        win_start = pd.Timestamp(f"{year - window_years + 1}-01-01")
        win_end = pd.Timestamp(f"{year}-12-31")

        mask = (citations["citing_date"] >= win_start) & (citations["citing_date"] <= win_end)
        window_cites = citations[mask]

        if len(window_cites) == 0:
            continue

        G = cpc_subgraph_nx(window_cites, cpc_map, sec_a, sec_b, max_nodes=max_nodes)

        if G.number_of_nodes() < 3:
            continue

        G = reduce_graph(G, max_nodes=max_nodes)
        summary = topology_summary(G, max_dim=max_dim)
        summary["year"] = year
        summary["section_a"] = sec_a
        summary["section_b"] = sec_b
        summary["sample_idx"] = i
        rows.append(summary)

    result = pd.DataFrame(rows)

    if use_cache and len(result) > 0:
        NULL_CACHE.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

    logger.info(
        "Matched null for %s: %d/%d samples computed",
        breakthrough.name, len(result), n_samples,
    )
    return result


# ---------------------------------------------------------------------------
# Superposed epoch analysis
# ---------------------------------------------------------------------------

def superposed_epoch(
    breakthroughs: list[Breakthrough],
    topology_results: dict[str, pd.DataFrame],
    metric: str = "beta_1",
    years_before: int = 10,
    years_after: int = 5,
) -> pd.DataFrame:
    """Align multiple breakthroughs at t=0 and average a topological metric.

    Args:
        breakthroughs: List of Breakthrough objects.
        topology_results: Dict mapping "section_a_section_b" to topology DataFrames.
        metric: Column name to average (e.g., 'beta_1', 'persistence_entropy').
        years_before: Years before breakthrough to include.
        years_after: Years after breakthrough to include.

    Returns:
        DataFrame with epoch_year (relative), mean, std, n_breakthroughs.
    """
    all_series = []

    for bt in breakthroughs:
        # Find the topology result for this breakthrough's CPC pair
        if len(bt.cpc_sections) >= 2:
            key = f"{bt.cpc_sections[0]}_{bt.cpc_sections[1]}"
        else:
            key = f"{bt.cpc_sections[0]}_{bt.cpc_sections[0]}"

        if key not in topology_results:
            # Try reversed pair
            key_rev = f"{key.split('_')[1]}_{key.split('_')[0]}"
            if key_rev not in topology_results:
                logger.warning("No topology data for %s (pair: %s)", bt.name, key)
                continue
            key = key_rev

        topo = topology_results[key]
        if metric not in topo.columns:
            continue

        # Align at filing_year = 0
        series = topo[["year", metric]].copy()
        series["epoch_year"] = series["year"] - bt.filing_year
        series = series[
            (series["epoch_year"] >= -years_before) & (series["epoch_year"] <= years_after)
        ]

        if len(series) > 0:
            all_series.append(series[["epoch_year", metric]])

    if not all_series:
        return pd.DataFrame(columns=["epoch_year", "mean", "std", "n"])

    combined = pd.concat(all_series)
    result = combined.groupby("epoch_year")[metric].agg(["mean", "std", "count"]).reset_index()
    result = result.rename(columns={"count": "n"})

    return result
