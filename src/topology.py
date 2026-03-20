"""
Topological analysis of the patent citation knowledge space.

APPROACH: Instead of computing persistent homology on the raw citation graph
(intractable — 20K+ node clique complexes have billions of simplices), we
compute it on the KNOWLEDGE SPACE defined by CPC subclass co-citation patterns.

Each CPC subclass (~260 total) becomes a point. The distance between two
subclasses is defined by how differently they cite other subclasses
(1 - cosine_similarity of their citation vectors). Persistent homology on
this ~260-point distance matrix is trivial (seconds per window) and answers
a better question: "What is the shape of the knowledge landscape, and how
does it change over time?"

β₀ dropping = previously disconnected fields merging in citation space
β₁ appearing = circular citation flows forming between field clusters
persistence entropy = overall topological complexity of the knowledge space

Conceived by Claude (Opus 4.6, Anthropic). Implementation by Claude Code.
"""

import logging
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core: Co-citation matrix construction
# ---------------------------------------------------------------------------

def build_cocitation_matrix(
    citations_df: pd.DataFrame,
    cpc_map: pd.DataFrame,
    start_year: int,
    end_year: int,
    level: str = "subclass",
) -> tuple[pd.DataFrame, list[str]]:
    """Build a CPC co-citation matrix for a given time window.

    For each citation (A cites B), look up the CPC classifications of A and B.
    Increment the co-citation count for each (cpc_of_A, cpc_of_B) pair.
    The result is a square matrix where entry (i, j) counts how often
    patents in CPC class i cite patents in CPC class j.

    Args:
        citations_df: DataFrame with columns [citing_id, cited_id, citing_date].
        cpc_map: DataFrame with columns [patent_id, cpc_section, cpc_class, cpc_subclass].
        start_year: Start of time window (inclusive).
        end_year: End of time window (inclusive).
        level: CPC granularity — "section" (8 categories), "class" (~130),
               or "subclass" (~260). Default "subclass".

    Returns:
        Tuple of (co-citation DataFrame indexed by CPC labels, list of CPC labels).
    """
    cpc_col = f"cpc_{level}"
    if cpc_col not in cpc_map.columns:
        raise ValueError(f"Column '{cpc_col}' not found in cpc_map. Available: {list(cpc_map.columns)}")

    # Filter citations to time window
    window_mask = (
        (citations_df["citing_year"] >= start_year) &
        (citations_df["citing_year"] <= end_year)
    )
    window_citations = citations_df.loc[window_mask, ["citing_id", "cited_id"]].copy()

    if len(window_citations) == 0:
        logger.warning(f"No citations in window {start_year}-{end_year}")
        return pd.DataFrame(), []

    logger.info(f"Window {start_year}-{end_year}: {len(window_citations):,} citations")

    # Get primary CPC label per patent (take the first one if multiple)
    patent_to_cpc = (
        cpc_map.groupby("patent_id")[cpc_col]
        .first()
        .to_dict()
    )

    # Map citing and cited patents to their CPC labels
    window_citations["citing_cpc"] = window_citations["citing_id"].map(patent_to_cpc)
    window_citations["cited_cpc"] = window_citations["cited_id"].map(patent_to_cpc)

    # Drop rows where either patent has no CPC mapping
    window_citations = window_citations.dropna(subset=["citing_cpc", "cited_cpc"])

    if len(window_citations) == 0:
        logger.warning(f"No citations with valid CPC mappings in {start_year}-{end_year}")
        return pd.DataFrame(), []

    # Count co-citations
    cocite_counts = (
        window_citations
        .groupby(["citing_cpc", "cited_cpc"])
        .size()
        .reset_index(name="count")
    )

    # Get all unique CPC labels in this window
    all_labels = sorted(
        set(cocite_counts["citing_cpc"].unique()) |
        set(cocite_counts["cited_cpc"].unique())
    )

    # Build square matrix
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=np.float64)

    for _, row in cocite_counts.iterrows():
        i = label_to_idx[row["citing_cpc"]]
        j = label_to_idx[row["cited_cpc"]]
        matrix[i, j] = row["count"]

    cocite_df = pd.DataFrame(matrix, index=all_labels, columns=all_labels)

    logger.info(f"  Co-citation matrix: {n}x{n}, {int(matrix.sum()):,} total citations")

    return cocite_df, all_labels


def cocitation_to_distance(
    cocite_matrix: np.ndarray,
    normalize_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a co-citation matrix to a distance matrix using cosine distance.

    Each row of the co-citation matrix is treated as a vector describing
    how a CPC class cites other classes. The distance between two classes
    is 1 - cosine_similarity(row_i, row_j).

    When normalize_scale=True (default), the distance matrix is divided by
    its mean so that Vietoris-Rips operates on relative structure rather
    than absolute scale. This controls for the density confound: as the
    patent network grows, co-citation vectors fill in zeros and converge,
    mechanically compressing cosine distances. Without normalization, this
    compression reduces topological features over time as an artifact of
    density growth rather than genuine structural change.

    Args:
        cocite_matrix: Square numpy array of co-citation counts.
        normalize_scale: If True, divide distance matrix by its mean to
            remove absolute-scale effects across time windows. Default True.

    Returns:
        Tuple of (distance matrix, boolean active_mask).
        Distance: 0 = identical citation patterns, 1 = orthogonal (before scaling).
        Returns (empty array, mask) if fewer than 3 active classes.
    """
    # Symmetrize: use (A cites B) + (B cites A) as the undirected co-citation
    sym = cocite_matrix + cocite_matrix.T

    # Handle zero rows (classes with no citations in this window)
    row_sums = sym.sum(axis=1)
    nonzero_mask = row_sums > 0

    if nonzero_mask.sum() < 3:
        logger.warning("Fewer than 3 active CPC classes in window — skipping")
        return np.array([]), nonzero_mask

    # Compute cosine distances only for active classes
    active_matrix = sym[nonzero_mask][:, nonzero_mask]

    # Normalize rows to unit vectors
    norms = np.linalg.norm(active_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # prevent division by zero
    normalized = active_matrix / norms

    # Cosine similarity → distance
    cosine_sim = normalized @ normalized.T
    cosine_sim = np.clip(cosine_sim, -1, 1)  # numerical stability
    distance = 1.0 - cosine_sim

    # Ensure diagonal is exactly 0 and matrix is symmetric
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2

    # Ensure non-negative (numerical precision)
    distance = np.maximum(distance, 0)

    # Scale normalization: divide by mean distance so that Vietoris-Rips
    # filtration operates on relative structure, not absolute scale.
    # This removes the density confound where growing networks compress
    # distances and mechanically reduce topological features.
    if normalize_scale:
        upper_tri = distance[np.triu_indices_from(distance, k=1)]
        mean_d = upper_tri.mean() if len(upper_tri) > 0 else 0.0
        if mean_d > 0:
            distance = distance / mean_d
            logger.info(f"  Scale-normalized distances (raw mean={mean_d:.4f} → 1.0)")

    return distance, nonzero_mask


# ---------------------------------------------------------------------------
# Persistent homology computation
# ---------------------------------------------------------------------------

def compute_persistence(
    distance_matrix: np.ndarray,
    max_dim: int = 2,
) -> list[np.ndarray]:
    """Compute persistent homology on a distance matrix using Vietoris-Rips.

    Args:
        distance_matrix: Square distance matrix.
        max_dim: Maximum homological dimension to compute (0, 1, or 2).

    Returns:
        List of persistence diagrams, one per dimension.
        Each diagram is an array of (birth, death) pairs.
    """
    try:
        from ripser import ripser
    except ImportError:
        logger.error("ripser not installed. Run: pip install ripser")
        raise

    n = distance_matrix.shape[0]
    logger.info(f"  Running Vietoris-Rips on {n} points, max_dim={max_dim}")

    result = ripser(
        distance_matrix,
        maxdim=max_dim,
        distance_matrix=True,
    )

    diagrams = result["dgms"]
    for dim, dgm in enumerate(diagrams):
        n_features = len(dgm)
        n_finite = np.isfinite(dgm[:, 1]).sum() if n_features > 0 else 0
        logger.info(f"  H{dim}: {n_features} features ({n_finite} finite)")

    return diagrams


def betti_numbers(
    diagrams: list[np.ndarray],
    threshold: Optional[float] = None,
) -> tuple[int, int, int]:
    """Extract Betti numbers from persistence diagrams.

    Args:
        diagrams: List of persistence diagrams from compute_persistence.
        threshold: If given, only count features with persistence > threshold.
                   If None, count all features alive at the median filtration value.

    Returns:
        Tuple of (β₀, β₁, β₂). If a dimension wasn't computed, returns 0.
    """
    bettis = []
    for dim in range(3):
        if dim >= len(diagrams) or len(diagrams[dim]) == 0:
            bettis.append(0)
            continue

        dgm = diagrams[dim]

        if threshold is not None:
            # Count features with persistence above threshold
            persistence = dgm[:, 1] - dgm[:, 0]
            # Handle infinite death times
            finite_mask = np.isfinite(persistence)
            count = int((persistence[finite_mask] > threshold).sum())
            # Add infinite features (they always exceed any threshold)
            count += int((~finite_mask).sum())
        else:
            # Count all features (excluding trivial ones with zero persistence)
            persistence = dgm[:, 1] - dgm[:, 0]
            finite_mask = np.isfinite(persistence)
            count = int((persistence[finite_mask] > 1e-10).sum())
            count += int((~finite_mask).sum())

        bettis.append(count)

    return tuple(bettis)


def persistence_entropy(diagrams: list[np.ndarray]) -> float:
    """Compute Shannon entropy of the persistence diagram.

    Measures the topological complexity of the space. Higher entropy means
    more diverse mix of feature lifetimes. Lower entropy means features
    are dominated by a few long-lived structures.

    Args:
        diagrams: List of persistence diagrams from compute_persistence.

    Returns:
        Shannon entropy (bits). Returns 0.0 if no finite features exist.
    """
    # Collect all finite persistence values across all dimensions
    all_persistence = []
    for dgm in diagrams:
        if len(dgm) == 0:
            continue
        persistence = dgm[:, 1] - dgm[:, 0]
        finite = persistence[np.isfinite(persistence)]
        positive = finite[finite > 1e-10]
        all_persistence.extend(positive)

    if len(all_persistence) == 0:
        return 0.0

    all_persistence = np.array(all_persistence)

    # Normalize to probability distribution
    total = all_persistence.sum()
    if total == 0:
        return 0.0

    probs = all_persistence / total
    probs = probs[probs > 0]  # remove zeros for log

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)


def max_persistence(diagrams: list[np.ndarray], dim: int = 1) -> float:
    """Get the maximum persistence in a given dimension.

    Args:
        diagrams: List of persistence diagrams.
        dim: Homological dimension (default 1 for loops).

    Returns:
        Maximum persistence value, or 0.0 if no features in that dimension.
    """
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return 0.0

    dgm = diagrams[dim]
    persistence = dgm[:, 1] - dgm[:, 0]
    finite = persistence[np.isfinite(persistence)]

    if len(finite) == 0:
        return 0.0

    return float(finite.max())


def n_long_lived_features(
    diagrams: list[np.ndarray],
    dim: int = 1,
    percentile: float = 90,
) -> int:
    """Count features with persistence above a given percentile.

    "Long-lived" topological features are the persistent structures —
    loops or voids that survive across a wide range of scales.

    Args:
        diagrams: List of persistence diagrams.
        dim: Homological dimension.
        percentile: Threshold percentile (default 90th).

    Returns:
        Count of features above the percentile threshold.
    """
    if dim >= len(diagrams) or len(diagrams[dim]) == 0:
        return 0

    dgm = diagrams[dim]
    persistence = dgm[:, 1] - dgm[:, 0]
    finite = persistence[np.isfinite(persistence)]

    if len(finite) == 0:
        return 0

    threshold = np.percentile(finite, percentile)
    return int((finite > threshold).sum())


# ---------------------------------------------------------------------------
# Sliding window topology computation
# ---------------------------------------------------------------------------

def sliding_window_topology(
    citations_df: pd.DataFrame,
    cpc_map: pd.DataFrame,
    window_years: int = 5,
    stride_years: int = 1,
    start_year: int = 1980,
    end_year: int = 2023,
    level: str = "subclass",
    max_dim: int = 2,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Compute topological metrics across sliding time windows.

    For each window, builds a CPC co-citation matrix, converts to a
    distance matrix, computes persistent homology, and extracts metrics.

    Args:
        citations_df: Citations with [citing_id, cited_id, citing_year].
        cpc_map: CPC mappings with [patent_id, cpc_section, cpc_class, cpc_subclass].
        window_years: Width of each time window in years.
        stride_years: Step between consecutive windows.
        start_year: First window starts here.
        end_year: Last window ends here.
        level: CPC granularity ("section", "class", or "subclass").
        max_dim: Maximum homological dimension.
        cache_dir: If given, cache results per window to this directory.

    Returns:
        DataFrame with columns: window_start, window_end, n_active_classes,
        beta_0, beta_1, beta_2, persistence_entropy, max_persistence_h1,
        n_long_lived_h1, mean_distance, median_distance.
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

    results = []
    window_starts = range(start_year, end_year - window_years + 2, stride_years)

    logger.info(f"Computing topology: {len(list(window_starts))} windows, "
                f"level={level}, max_dim={max_dim}")

    for ws in window_starts:
        we = ws + window_years - 1

        # Check cache
        if cache_dir:
            cache_file = cache_path / f"window_{ws}_{we}_{level}.parquet"
            if cache_file.exists():
                logger.info(f"  [{ws}-{we}] Loaded from cache")
                cached = pd.read_parquet(cache_file)
                results.append(cached.iloc[0].to_dict())
                continue

        logger.info(f"  [{ws}-{we}] Computing...")

        # Build co-citation matrix
        cocite_df, labels = build_cocitation_matrix(
            citations_df, cpc_map, ws, we, level=level
        )

        if cocite_df.empty or len(labels) < 3:
            logger.warning(f"  [{ws}-{we}] Insufficient data, skipping")
            continue

        # Convert to distance matrix
        result = cocitation_to_distance(cocite_df.values)

        if isinstance(result, tuple):
            dist_matrix, active_mask = result
        else:
            dist_matrix = result
            active_mask = np.ones(len(labels), dtype=bool)

        if dist_matrix.size == 0:
            logger.warning(f"  [{ws}-{we}] Distance matrix empty, skipping")
            continue

        n_active = dist_matrix.shape[0]
        logger.info(f"  [{ws}-{we}] {n_active} active CPC {level}es")

        # Compute persistent homology
        diagrams = compute_persistence(dist_matrix, max_dim=max_dim)

        # Extract metrics
        b0, b1, b2 = betti_numbers(diagrams)
        pe = persistence_entropy(diagrams)
        max_p = max_persistence(diagrams, dim=1)
        n_long = n_long_lived_features(diagrams, dim=1)

        # Distance matrix statistics
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        mean_dist = float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0
        median_dist = float(np.median(upper_tri)) if len(upper_tri) > 0 else 0.0

        row = {
            "window_start": ws,
            "window_end": we,
            "n_active_classes": n_active,
            "beta_0": b0,
            "beta_1": b1,
            "beta_2": b2,
            "persistence_entropy": pe,
            "max_persistence_h1": max_p,
            "n_long_lived_h1": n_long,
            "mean_distance": mean_dist,
            "median_distance": median_dist,
        }

        results.append(row)

        # Cache this window
        if cache_dir:
            pd.DataFrame([row]).to_parquet(cache_file, index=False)
            logger.info(f"  [{ws}-{we}] Cached")

        # Aggressive memory cleanup
        del cocite_df, dist_matrix, diagrams
        if 'active_mask' in dir():
            del active_mask
        gc.collect()

    df = pd.DataFrame(results)
    logger.info(f"Topology computation complete: {len(df)} windows")
    return df


def sliding_window_topology_by_section_pair(
    citations_df: pd.DataFrame,
    cpc_map: pd.DataFrame,
    section_a: str,
    section_b: str,
    window_years: int = 5,
    stride_years: int = 1,
    start_year: int = 1980,
    end_year: int = 2023,
    max_dim: int = 2,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Compute topology for a specific CPC section pair.

    Filters patents to those in section A or section B, then computes
    co-citation patterns at the subclass level within and between these
    two sections.

    Args:
        citations_df: Full citations DataFrame.
        cpc_map: Full CPC mappings DataFrame.
        section_a: CPC section code (e.g., "A", "C", "G").
        section_b: CPC section code.
        window_years, stride_years, start_year, end_year: Window parameters.
        max_dim: Maximum homological dimension.
        cache_dir: Cache directory (pair-specific subdirectory will be created).

    Returns:
        DataFrame with topology metrics per time window.
    """
    pair_label = f"{section_a}x{section_b}"
    logger.info(f"=== CPC Section Pair: {pair_label} ===")

    # Filter CPC map to these two sections
    pair_cpc = cpc_map[cpc_map["cpc_section"].isin([section_a, section_b])].copy()
    pair_patents = set(pair_cpc["patent_id"].unique())

    # Filter citations to those where at least one patent is in the pair
    pair_citations = citations_df[
        citations_df["citing_id"].isin(pair_patents) |
        citations_df["cited_id"].isin(pair_patents)
    ].copy()

    logger.info(f"  {pair_label}: {len(pair_patents):,} patents, "
                f"{len(pair_citations):,} citations")

    if len(pair_citations) < 100:
        logger.warning(f"  {pair_label}: Too few citations, skipping")
        return pd.DataFrame()

    # Set up pair-specific cache
    pair_cache = None
    if cache_dir:
        pair_cache = str(Path(cache_dir) / pair_label)

    # Compute sliding window topology at subclass level
    result = sliding_window_topology(
        pair_citations,
        pair_cpc,
        window_years=window_years,
        stride_years=stride_years,
        start_year=start_year,
        end_year=end_year,
        level="subclass",
        max_dim=max_dim,
        cache_dir=pair_cache,
    )

    if not result.empty:
        result["section_pair"] = pair_label

    # Cleanup
    del pair_cpc, pair_patents, pair_citations
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Full analysis: all priority section pairs
# ---------------------------------------------------------------------------

PRIORITY_PAIRS = [
    ("A", "C"),  # Human necessities × Chemistry → biotech/pharma
    ("A", "G"),  # Human necessities × Physics → medical devices
    ("C", "G"),  # Chemistry × Physics → materials/sensors
    ("C", "H"),  # Chemistry × Electricity → batteries/energy
    ("G", "H"),  # Physics × Electricity → semiconductors/computing
    ("B", "G"),  # Operations × Physics → manufacturing tech
    ("B", "H"),  # Operations × Electricity → automation
    ("A", "H"),  # Human necessities × Electricity → health tech
    ("C", "B"),  # Chemistry × Operations → chemical engineering
    ("F", "H"),  # Mechanical engineering × Electricity → electromechanical
]

# All 28 unique CPC section pairs
CPC_SECTIONS = list("ABCDEFGH")
ALL_PAIRS = [
    (CPC_SECTIONS[i], CPC_SECTIONS[j])
    for i in range(len(CPC_SECTIONS))
    for j in range(i + 1, len(CPC_SECTIONS))
]


def compute_all_priority_pairs(
    citations_df: pd.DataFrame,
    cpc_map: pd.DataFrame,
    cache_dir: str = "data/topology_cache",
    pairs: list[tuple[str, str]] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Compute topology for CPC section pairs.

    Args:
        citations_df: Full citations DataFrame.
        cpc_map: Full CPC mappings DataFrame.
        cache_dir: Base cache directory.
        pairs: List of (section_a, section_b) tuples. Defaults to PRIORITY_PAIRS.
            Pass ALL_PAIRS for full 28-pair analysis.
        **kwargs: Additional arguments passed to sliding_window_topology_by_section_pair.

    Returns:
        Combined DataFrame with topology metrics for all pairs.
    """
    if pairs is None:
        pairs = PRIORITY_PAIRS

    all_results = []

    for i, (sa, sb) in enumerate(pairs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Pair {i+1}/{len(pairs)}: {sa}x{sb}")
        logger.info(f"{'='*60}")

        result = sliding_window_topology_by_section_pair(
            citations_df,
            cpc_map,
            section_a=sa,
            section_b=sb,
            cache_dir=cache_dir,
            **kwargs,
        )

        if not result.empty:
            all_results.append(result)
            logger.info(f"  {sa}x{sb}: {len(result)} windows computed")
            logger.info(f"  β₁ range: {result['beta_1'].min()} - {result['beta_1'].max()}")
            logger.info(f"  PE range: {result['persistence_entropy'].min():.3f} - "
                        f"{result['persistence_entropy'].max():.3f}")
        else:
            logger.warning(f"  {sa}x{sb}: No results")

        # Force garbage collection between pairs
        gc.collect()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        logger.info(f"\nAll pairs complete: {len(combined)} total window-pair observations")
        return combined
    else:
        logger.error("No results from any pair!")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Global (full-network) topology computation
# ---------------------------------------------------------------------------

def compute_global_topology(
    citations_df: pd.DataFrame,
    cpc_map: pd.DataFrame,
    cache_dir: str = "data/topology_cache",
    **kwargs,
) -> pd.DataFrame:
    """Compute topology on the full CPC co-citation space (all sections).

    This gives a single topological time series for the entire patent
    knowledge landscape — not filtered to any section pair.

    Args:
        citations_df: Full citations DataFrame.
        cpc_map: Full CPC mappings DataFrame.
        cache_dir: Cache directory.
        **kwargs: Window parameters.

    Returns:
        DataFrame with topology metrics per time window.
    """
    logger.info("=== Global topology (all CPC sections) ===")

    global_cache = str(Path(cache_dir) / "global")

    result = sliding_window_topology(
        citations_df,
        cpc_map,
        level="subclass",
        cache_dir=global_cache,
        **kwargs,
    )

    if not result.empty:
        result["section_pair"] = "GLOBAL"

    return result
