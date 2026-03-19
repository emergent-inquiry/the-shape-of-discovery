"""Null model generation for hypothesis testing.

Provides two types of null distributions:
    1. Random baseline: random CPC section pairs at random times.
    2. Matched null: same CPC pair as a breakthrough, shifted to non-breakthrough periods.

Both reuse the co-citation topology pipeline from src/topology.py and cache results.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.breakthroughs import Breakthrough, get_precursor_window
from src.topology import (
    build_cocitation_matrix,
    cocitation_to_distance,
    compute_persistence,
    betti_numbers,
    persistence_entropy,
    max_persistence,
    n_long_lived_features,
)
from src.utils import DATA_DIR, get_logger, timer

logger = get_logger(__name__)

NULL_CACHE = DATA_DIR / "null_cache"

CPC_SECTIONS = list("ABCDEFGH")


def _ensure_citing_year(citations: pd.DataFrame) -> pd.DataFrame:
    """Ensure citing_year column exists as int."""
    if "citing_year" not in citations.columns:
        citations = citations.copy()
        citations["citing_year"] = pd.to_datetime(citations["citing_date"]).dt.year
    return citations


def _compute_topology_for_window(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    sec_a: str,
    sec_b: str,
    year: int,
    window_years: int = 5,
) -> dict | None:
    """Compute topology metrics for a single section pair and time window.

    Returns a dict of metrics or None if computation fails.
    """
    # Filter CPC map to the two sections
    pair_cpc = cpc_map[cpc_map["cpc_section"].isin([sec_a, sec_b])]
    pair_patents = set(pair_cpc["patent_id"].unique())

    # Filter citations
    start_year = year - window_years + 1
    end_year = year

    pair_citations = citations[
        (citations["citing_id"].isin(pair_patents) |
         citations["cited_id"].isin(pair_patents))
    ]

    if len(pair_citations) < 10:
        return None

    # Build co-citation matrix
    cocite_df, labels = build_cocitation_matrix(
        pair_citations, pair_cpc, start_year, end_year, level="subclass"
    )

    if cocite_df.empty or len(labels) < 3:
        return None

    # Convert to distance
    result = cocitation_to_distance(cocite_df.values)
    if isinstance(result, tuple):
        dist_matrix, active_mask = result
    else:
        dist_matrix = result
        active_mask = np.ones(len(labels), dtype=bool)

    if dist_matrix.size == 0:
        return None

    n_active = dist_matrix.shape[0]

    # Compute persistence
    diagrams = compute_persistence(dist_matrix, max_dim=2)

    # Extract metrics
    b0, b1, b2 = betti_numbers(diagrams)
    pe = persistence_entropy(diagrams)
    max_p = max_persistence(diagrams, dim=1)
    n_long = n_long_lived_features(diagrams, dim=1)

    # Distance stats
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    mean_dist = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
    median_dist = float(np.median(upper_tri)) if len(upper_tri) > 0 else 0.0

    return {
        "beta_0": b0,
        "beta_1": b1,
        "beta_2": b2,
        "persistence_entropy": pe,
        "max_persistence_h1": max_p,
        "n_long_lived_h1": n_long,
        "n_active_classes": n_active,
        "mean_distance": mean_dist,
        "median_distance": median_dist,
    }


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
        seed: Random seed for reproducibility.
        use_cache: Whether to cache results.

    Returns:
        DataFrame with topology metrics for each random sample.
    """
    cache_file = NULL_CACHE / f"random_baseline_n{n_samples}_w{window_years}_s{seed}_cocite.pkl"
    if use_cache and cache_file.exists():
        logger.info("Loading cached random baseline")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)
    citations = _ensure_citing_year(citations)

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

        summary = _compute_topology_for_window(
            citations, cpc_map, sec_a, sec_b, year, window_years
        )

        if summary is None:
            continue

        summary["window_end"] = year
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
        seed: Random seed.
        use_cache: Whether to cache.

    Returns:
        DataFrame of null topology measurements.
    """
    bt_name = breakthrough.name.replace(" ", "_").lower()[:30]
    cache_file = NULL_CACHE / f"matched_{bt_name}_n{n_samples}_s{seed}_cocite.pkl"
    if use_cache and cache_file.exists():
        logger.info("Loading cached matched null for %s", breakthrough.name)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)
    citations = _ensure_citing_year(citations)

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

        summary = _compute_topology_for_window(
            citations, cpc_map, sec_a, sec_b, year, window_years
        )

        if summary is None:
            continue

        summary["window_end"] = year
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
        topology_results: Dict mapping pair key to topology DataFrames.
            Keys can be "AxC" (new format) or "A_C" (old format).
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
            sec_a, sec_b = bt.cpc_sections[0], bt.cpc_sections[1]
        else:
            sec_a = sec_b = bt.cpc_sections[0]

        # Try multiple key formats
        candidates = [
            f"{sec_a}x{sec_b}",      # new format
            f"{sec_b}x{sec_a}",      # new format reversed
            f"{sec_a}_{sec_b}",      # old format
            f"{sec_b}_{sec_a}",      # old format reversed
        ]

        topo = None
        for key in candidates:
            if key in topology_results:
                topo = topology_results[key]
                break

        if topo is None:
            logger.warning("No topology data for %s (tried: %s)", bt.name, candidates[:2])
            continue

        if metric not in topo.columns:
            continue

        # Determine the year column (support both old and new format)
        if "window_end" in topo.columns:
            year_col = "window_end"
        elif "year" in topo.columns:
            year_col = "year"
        else:
            continue

        # Align at filing_year = 0
        series = topo[[year_col, metric]].copy()
        series["epoch_year"] = series[year_col] - bt.filing_year
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
