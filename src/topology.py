"""Persistent homology computation for the patent citation network.

This is the novel core of the project. Nobody has applied persistent homology
to the patent citation graph to detect topological precursors of technological
breakthroughs.

Design decisions:
    - Work on CPC section-pair subgraphs, never the full 8M-node graph.
    - Use ripser (fastest available) with sparse distance matrices.
    - Fall back to giotto-tda if ripser fails on specific inputs.
    - Tractability cascade: degree pruning → landmark subsampling → alternative filtration.
    - All expensive results are cached to data/topology_cache/ as pickle.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from src.graph import cpc_subgraph_nx, SparseGraph
from src.utils import DATA_DIR, get_logger, log_memory, timer

logger = get_logger(__name__)

TOPOLOGY_CACHE = DATA_DIR / "topology_cache"


# ---------------------------------------------------------------------------
# Distance matrix construction
# ---------------------------------------------------------------------------

def _adjacency_to_distance(G: nx.Graph, max_dist: float = np.inf) -> np.ndarray:
    """Convert an undirected graph to a distance matrix for ripser.

    Uses shortest-path distance. Disconnected pairs get max_dist.

    Args:
        G: Undirected NetworkX graph.
        max_dist: Value for disconnected pairs. Use np.inf for ripser
            (it ignores infinite entries in sparse mode).

    Returns:
        Square distance matrix as a numpy array.
    """
    adj = nx.to_scipy_sparse_array(G, format="csr")

    # Shortest-path distances on the unweighted graph
    dist = shortest_path(adj, directed=False, unweighted=True)

    # Replace 0 (self-loops) with 0, and inf stays as inf
    np.fill_diagonal(dist, 0.0)

    # Cap at max_dist for computational tractability
    if np.isfinite(max_dist):
        dist = np.minimum(dist, max_dist)

    return dist


def _adjacency_to_sparse_distance(G: nx.Graph, max_hops: int = 3) -> sparse.csr_matrix:
    """Convert graph to sparse distance matrix, keeping only short-range distances.

    For large graphs, computing all-pairs shortest paths is O(n^2).
    This function only keeps distances up to max_hops, making the
    distance matrix sparse and tractable for ripser.

    Uses vectorized sparse operations instead of Python loops for performance.

    Args:
        G: Undirected NetworkX graph.
        max_hops: Maximum distance to keep.

    Returns:
        Sparse CSR distance matrix.
    """
    adj = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    n = adj.shape[0]

    # Binary adjacency (no weights)
    adj_bin = (adj > 0).astype(np.float64)
    adj_bin.setdiag(0)
    adj_bin.eliminate_zeros()

    # Initialize distance with hop-1 neighbors
    dist = adj_bin.copy()
    # Track which pairs have been assigned a distance (int8 to avoid bool underflow)
    reached = adj_bin.astype(np.int8)

    current_power = adj_bin.copy()
    for hop in range(2, max_hops + 1):
        current_power = current_power @ adj_bin
        current_power.setdiag(0)

        # New pairs: reachable at this hop but not yet discovered
        # Cast to int8 before subtraction to avoid bool underflow (False - True wraps)
        new_pairs = (current_power > 0).astype(np.int8) - (reached > 0).astype(np.int8)
        new_pairs = new_pairs.multiply(new_pairs > 0)  # Clip negatives
        new_pairs.eliminate_zeros()  # Remove explicit zeros before setting data

        if new_pairs.nnz == 0:
            break

        # Assign distance = hop to newly discovered pairs only
        new_dist = new_pairs.astype(np.float64)
        new_dist.data[:] = float(hop)
        dist = dist + new_dist

        # Update reached mask
        reached = reached + new_pairs.astype(np.int8)

    return dist.tocsr()


# ---------------------------------------------------------------------------
# Graph reduction for tractability
# ---------------------------------------------------------------------------

def reduce_graph(G: nx.Graph, max_nodes: int = 30_000) -> nx.Graph:
    """Reduce graph size while preserving topological structure.

    Applies a cascade of reduction strategies:
        1. Remove degree-1 nodes (leaves don't contribute to cycles/voids)
        2. If still too large, keep highest-degree nodes (maxmin landmark)

    Args:
        G: Input graph.
        max_nodes: Target maximum node count.

    Returns:
        Reduced graph (may be the same object if already small enough).
    """
    if G.number_of_nodes() <= max_nodes:
        return G

    # Step 1: Remove leaves (degree-1 nodes)
    leaves = [n for n, d in G.degree() if d <= 1]
    if leaves:
        G = G.copy()
        G.remove_nodes_from(leaves)
        logger.info(
            "Removed %d leaves, %d nodes remaining",
            len(leaves), G.number_of_nodes(),
        )

    if G.number_of_nodes() <= max_nodes:
        return G

    # Step 2: Keep highest-degree nodes
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
    G = G.subgraph(sorted_nodes).copy()
    logger.info("Subsampled to %d highest-degree nodes", G.number_of_nodes())

    return G


# ---------------------------------------------------------------------------
# Persistent homology
# ---------------------------------------------------------------------------

def compute_persistence(
    G: nx.Graph,
    max_dim: int = 2,
    max_hops: int = 3,
    sparse_mode: bool = True,
) -> dict:
    """Compute persistent homology of a graph using ripser.

    The graph is converted to a distance matrix (shortest-path),
    then passed to ripser for Vietoris-Rips persistence computation.

    Args:
        G: Undirected NetworkX graph.
        max_dim: Maximum homological dimension (0, 1, or 2).
        max_hops: Maximum hop distance for sparse distance matrix.
        sparse_mode: If True, use sparse distance matrix (faster, recommended).

    Returns:
        Dict with:
            - 'dgms': list of persistence diagrams (birth-death pairs) per dimension
            - 'n_nodes': number of nodes in the input graph
            - 'n_edges': number of edges
    """
    try:
        from ripser import ripser
    except ImportError:
        logger.warning("ripser not installed, falling back to giotto-tda")
        return _compute_persistence_giotto(G, max_dim)

    n = G.number_of_nodes()
    if n == 0:
        return {"dgms": [np.empty((0, 2)) for _ in range(max_dim + 1)], "n_nodes": 0, "n_edges": 0}

    if n == 1:
        dgms = [np.array([[0.0, np.inf]])]
        dgms.extend([np.empty((0, 2)) for _ in range(max_dim)])
        return {"dgms": dgms, "n_nodes": 1, "n_edges": 0}

    log_memory(f"Before persistence ({n} nodes)")

    if sparse_mode and n > 500:
        dist = _adjacency_to_sparse_distance(G, max_hops=max_hops)
        result = ripser(dist, maxdim=max_dim, distance_matrix=True)
    else:
        dist = _adjacency_to_distance(G)
        result = ripser(dist, maxdim=max_dim, distance_matrix=True)

    log_memory(f"After persistence ({n} nodes)")

    return {
        "dgms": result["dgms"],
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
    }


def _compute_persistence_giotto(G: nx.Graph, max_dim: int = 2) -> dict:
    """Fallback: compute persistence using giotto-tda.

    Args:
        G: Undirected NetworkX graph.
        max_dim: Maximum homological dimension.

    Returns:
        Same format as compute_persistence().
    """
    from gtda.homology import VietorisRipsPersistence

    n = G.number_of_nodes()
    if n == 0:
        return {"dgms": [np.empty((0, 2)) for _ in range(max_dim + 1)], "n_nodes": 0, "n_edges": 0}

    dist = _adjacency_to_distance(G)
    dist_3d = dist.reshape(1, n, n)

    vr = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dim + 1)),
        metric="precomputed",
    )
    diagrams = vr.fit_transform(dist_3d)

    # Convert giotto format (n_features, 3) to ripser format (list of (n, 2) per dim)
    dgms = []
    for dim in range(max_dim + 1):
        mask = diagrams[0][:, 2] == dim
        dgms.append(diagrams[0][mask, :2])

    return {"dgms": dgms, "n_nodes": n, "n_edges": G.number_of_edges()}


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------

def betti_numbers(
    dgms: list[np.ndarray],
    threshold: Optional[float] = None,
) -> tuple[int, ...]:
    """Extract Betti numbers from persistence diagrams.

    β₀ = connected components
    β₁ = 1-dimensional loops (circular knowledge flows)
    β₂ = 2-dimensional voids (enclosed cavities)

    Args:
        dgms: List of persistence diagrams per dimension.
        threshold: If given, only count features with persistence > threshold.

    Returns:
        Tuple of Betti numbers (β₀, β₁, ..., β_max_dim).
    """
    bettis = []
    for dgm in dgms:
        if len(dgm) == 0:
            bettis.append(0)
            continue

        persistence = dgm[:, 1] - dgm[:, 0]

        if threshold is not None:
            # Count features with persistence > threshold (excluding infinite features)
            finite_mask = np.isfinite(dgm[:, 1])
            count = int(np.sum((persistence[finite_mask] > threshold)))
        else:
            # Count all features (including infinite)
            count = len(dgm)

        bettis.append(count)

    return tuple(bettis)


# ---------------------------------------------------------------------------
# Persistence entropy
# ---------------------------------------------------------------------------

def persistence_entropy(dgm: np.ndarray) -> float:
    """Shannon entropy of a persistence diagram.

    Measures topological complexity: higher entropy = more diverse
    distribution of feature lifetimes.

    Args:
        dgm: Persistence diagram (N x 2 array of birth-death pairs).

    Returns:
        Entropy in bits. Returns 0.0 for empty or single-feature diagrams.
    """
    if len(dgm) == 0:
        return 0.0

    # Only consider finite features
    finite_mask = np.isfinite(dgm[:, 1])
    dgm_finite = dgm[finite_mask]

    if len(dgm_finite) <= 1:
        return 0.0

    lifetimes = dgm_finite[:, 1] - dgm_finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    # Normalize to probability distribution
    total = lifetimes.sum()
    probs = lifetimes / total

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    return float(entropy)


# ---------------------------------------------------------------------------
# Topology summary for a single graph
# ---------------------------------------------------------------------------

def topology_summary(G: nx.Graph, max_dim: int = 2, max_hops: int = 3) -> dict:
    """Compute a full topological summary of a graph.

    Args:
        G: Undirected NetworkX graph (should already be reduced if needed).
        max_dim: Maximum homological dimension.
        max_hops: Maximum hop distance for distance matrix.

    Returns:
        Dict with β₀, β₁, β₂, persistence_entropy, max_persistence,
        n_long_lived_features, n_nodes, n_edges.
    """
    result = compute_persistence(G, max_dim=max_dim, max_hops=max_hops)
    dgms = result["dgms"]

    bettis = betti_numbers(dgms)

    # Persistence entropy across all dimensions
    all_finite = []
    for dgm in dgms:
        if len(dgm) > 0:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                all_finite.append(finite)

    combined = np.vstack(all_finite) if all_finite else np.empty((0, 2))
    pe = persistence_entropy(combined)

    # Max persistence (finite features only)
    max_pers = 0.0
    for dgm in dgms:
        if len(dgm) > 0:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                max_pers = max(max_pers, float(lifetimes.max()))

    # Long-lived features: persistence > median persistence
    n_long = 0
    if len(combined) > 0:
        lifetimes = combined[:, 1] - combined[:, 0]
        median_pers = np.median(lifetimes)
        n_long = int(np.sum(lifetimes > median_pers))

    return {
        "beta_0": bettis[0] if len(bettis) > 0 else 0,
        "beta_1": bettis[1] if len(bettis) > 1 else 0,
        "beta_2": bettis[2] if len(bettis) > 2 else 0,
        "persistence_entropy": pe,
        "max_persistence": max_pers,
        "n_long_lived_features": n_long,
        "n_nodes": result["n_nodes"],
        "n_edges": result["n_edges"],
    }


# ---------------------------------------------------------------------------
# Sliding window topology
# ---------------------------------------------------------------------------

@timer
def sliding_window_topology(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    section_a: str,
    section_b: str,
    patents: Optional[pd.DataFrame] = None,
    window_years: int = 5,
    stride_years: int = 1,
    start_year: int = 1980,
    end_year: int = 2023,
    max_dim: int = 2,
    max_nodes: int = 30_000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compute persistent homology across sliding time windows for a CPC pair.

    This is the main analysis function. For each time window:
        1. Filter citations to the window
        2. Extract CPC-pair subgraph
        3. Reduce if too large
        4. Compute persistent homology
        5. Summarize topological features

    Results are cached per (section_pair, window_params) to avoid recomputation.

    Args:
        citations: Full citation DataFrame with citing_id, cited_id, citing_date.
        cpc_map: CPC mapping DataFrame.
        section_a: First CPC section letter.
        section_b: Second CPC section letter.
        patents: Optional patent metadata for date filtering.
        window_years: Width of each window in years.
        stride_years: Step between windows.
        start_year: First window end year.
        end_year: Last window end year.
        max_dim: Maximum homological dimension.
        max_nodes: Max nodes per subgraph before reduction.
        use_cache: Whether to use/save cached results.

    Returns:
        DataFrame with one row per window: year, β₀, β₁, β₂,
        persistence_entropy, max_persistence, n_long_lived_features,
        n_nodes, n_edges.
    """
    pair_key = f"{section_a}_{section_b}"
    cache_file = TOPOLOGY_CACHE / f"sliding_{pair_key}_w{window_years}_s{stride_years}.pkl"

    if use_cache and cache_file.exists():
        logger.info("Loading cached topology for (%s, %s)", section_a, section_b)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Computing sliding-window topology for (%s, %s): %d-%d, window=%d, stride=%d",
        section_a, section_b, start_year, end_year, window_years, stride_years,
    )

    # Pre-filter citations to relevant CPC sections ONCE (avoids rescanning
    # the full 118M-row DataFrame on every window iteration)
    section_patents = set(
        cpc_map[cpc_map["cpc_section"].isin([section_a, section_b])]["patent_id"]
    )
    citations = citations[
        citations["citing_id"].isin(section_patents)
        & citations["cited_id"].isin(section_patents)
    ].copy()
    citations["citing_date"] = pd.to_datetime(citations["citing_date"])

    logger.info(
        "Pre-filtered to %d citations for sections (%s, %s)",
        len(citations), section_a, section_b,
    )

    rows = []
    for year in range(start_year, end_year + 1, stride_years):
        win_start = pd.Timestamp(f"{year - window_years + 1}-01-01")
        win_end = pd.Timestamp(f"{year}-12-31")

        # Filter citations to window
        mask = (citations["citing_date"] >= win_start) & (citations["citing_date"] <= win_end)
        window_cites = citations[mask]

        if len(window_cites) == 0:
            logger.info("  Year %d: no citations in window, skipping", year)
            continue

        # Build CPC-pair subgraph (undirected, for topology)
        G = cpc_subgraph_nx(window_cites, cpc_map, section_a, section_b, max_nodes=max_nodes)

        if G.number_of_nodes() < 3:
            logger.info("  Year %d: subgraph too small (%d nodes), skipping", year, G.number_of_nodes())
            continue

        # Reduce if needed
        G = reduce_graph(G, max_nodes=max_nodes)

        # Compute topology
        summary = topology_summary(G, max_dim=max_dim)
        summary["year"] = year
        rows.append(summary)

        logger.info(
            "  Year %d: %d nodes, %d edges, β₁=%d, PE=%.3f",
            year, summary["n_nodes"], summary["n_edges"],
            summary["beta_1"], summary["persistence_entropy"],
        )

    result = pd.DataFrame(rows)

    # Cache results
    if use_cache and len(result) > 0:
        TOPOLOGY_CACHE.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        logger.info("Cached topology results to %s", cache_file)

    return result
