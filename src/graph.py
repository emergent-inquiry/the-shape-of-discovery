"""Citation graph construction, temporal snapshots, and CPC subgraph extraction.

Design decisions:
    - Full graph uses scipy.sparse CSR (NetworkX cannot hold 8M nodes / 100M edges).
    - Small CPC-pair subgraphs (<50K nodes) are converted to NetworkX for topology.
    - Patent IDs are mapped to integer indices via a bijective dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from src.utils import get_logger, log_memory, timer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SparseGraph:
    """Sparse directed citation graph with patent-ID ↔ index mapping.

    Attributes:
        adj: Sparse CSR adjacency matrix (rows cite columns).
        id_to_idx: Dict mapping patent_id (str) to integer index.
        idx_to_id: Array mapping integer index back to patent_id.
        dates: Optional array of patent dates aligned with indices.
    """
    adj: sparse.csr_matrix
    id_to_idx: dict[str, int]
    idx_to_id: np.ndarray
    dates: Optional[np.ndarray] = None

    @property
    def n_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def n_edges(self) -> int:
        return self.adj.nnz


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@timer
def build_citation_graph(
    citations: pd.DataFrame,
    patents: Optional[pd.DataFrame] = None,
) -> SparseGraph:
    """Build a sparse directed citation graph from citation data.

    Args:
        citations: DataFrame with columns ``citing_id``, ``cited_id``.
        patents: Optional DataFrame with ``patent_id`` and ``date`` columns.
            If provided, dates are aligned with node indices.

    Returns:
        SparseGraph with CSR adjacency and ID mappings.
    """
    # Build the ID universe: union of all citing and cited IDs
    all_ids = pd.unique(
        pd.concat([citations["citing_id"], citations["cited_id"]])
    )
    all_ids.sort()

    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}
    idx_to_id = all_ids

    rows = citations["citing_id"].map(id_to_idx).values
    cols = citations["cited_id"].map(id_to_idx).values
    n = len(all_ids)

    adj = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n, n),
    )

    # Align dates if patents provided
    dates = None
    if patents is not None and "date" in patents.columns:
        date_series = patents.set_index("patent_id")["date"]
        dates = np.array([
            date_series.get(pid, pd.NaT)
            for pid in idx_to_id
        ], dtype="datetime64[ns]")

    graph = SparseGraph(adj=adj, id_to_idx=id_to_idx, idx_to_id=idx_to_id, dates=dates)
    logger.info("Built graph: %d nodes, %d edges", graph.n_nodes, graph.n_edges)
    log_memory("After graph construction")
    return graph


# ---------------------------------------------------------------------------
# Temporal snapshots
# ---------------------------------------------------------------------------

@timer
def temporal_snapshots(
    citations: pd.DataFrame,
    patents: pd.DataFrame,
    window_years: int = 5,
    stride_years: int = 1,
    start_year: int = 1980,
    end_year: int = 2023,
) -> list[tuple[int, SparseGraph]]:
    """Generate sliding-window subgraphs over time.

    Each window covers [year - window_years + 1, year] inclusive.

    Args:
        citations: DataFrame with ``citing_id``, ``cited_id``, ``citing_date``.
        patents: DataFrame with ``patent_id``, ``date``.
        window_years: Width of each window in years.
        stride_years: Step between consecutive windows.
        start_year: First window center year.
        end_year: Last window center year.

    Returns:
        List of (year, SparseGraph) pairs sorted chronologically.
    """
    citations = citations.copy()
    citations["citing_date"] = pd.to_datetime(citations["citing_date"])

    snapshots = []
    for year in range(start_year, end_year + 1, stride_years):
        win_start = pd.Timestamp(f"{year - window_years + 1}-01-01")
        win_end = pd.Timestamp(f"{year}-12-31")

        mask = (citations["citing_date"] >= win_start) & (citations["citing_date"] <= win_end)
        window_citations = citations[mask]

        if len(window_citations) == 0:
            continue

        graph = build_citation_graph.__wrapped__(window_citations, patents)
        snapshots.append((year, graph))
        logger.info(
            "Snapshot %d: %d nodes, %d edges",
            year, graph.n_nodes, graph.n_edges,
        )

    logger.info("Generated %d temporal snapshots", len(snapshots))
    return snapshots


# ---------------------------------------------------------------------------
# CPC subgraphs
# ---------------------------------------------------------------------------

def cpc_subgraph(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    section_a: str,
    section_b: str,
    patents: Optional[pd.DataFrame] = None,
) -> SparseGraph:
    """Extract a subgraph containing patents from two CPC sections.

    Includes all edges where at least one endpoint is in section_a or section_b.

    Args:
        citations: Full citation DataFrame.
        cpc_map: DataFrame with ``patent_id`` and ``cpc_section``.
        section_a: First CPC section letter (e.g. 'A').
        section_b: Second CPC section letter (e.g. 'C').
        patents: Optional patents DataFrame for date alignment.

    Returns:
        SparseGraph of the CPC-pair subgraph.
    """
    # Patents in either section
    section_patents = set(
        cpc_map[cpc_map["cpc_section"].isin([section_a, section_b])]["patent_id"]
    )

    # Filter citations: both endpoints in the section pair
    mask = (
        citations["citing_id"].isin(section_patents)
        & citations["cited_id"].isin(section_patents)
    )
    sub_citations = citations[mask]

    logger.info(
        "CPC subgraph (%s, %s): %d patents in sections, %d citations",
        section_a, section_b, len(section_patents), len(sub_citations),
    )

    return build_citation_graph.__wrapped__(sub_citations, patents)


def cpc_subgraph_nx(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    section_a: str,
    section_b: str,
    max_nodes: int = 50_000,
) -> nx.Graph:
    """Extract a CPC-pair subgraph as an undirected NetworkX graph.

    This is the form needed for topology computations (clique complex).
    Returns undirected because persistent homology operates on simplicial
    complexes built from undirected adjacency.

    Args:
        citations: Full citation DataFrame.
        cpc_map: DataFrame with ``patent_id`` and ``cpc_section``.
        section_a: First CPC section letter.
        section_b: Second CPC section letter.
        max_nodes: Maximum nodes before subsampling.

    Returns:
        Undirected NetworkX Graph.
    """
    sg = cpc_subgraph(citations, cpc_map, section_a, section_b)

    if sg.n_nodes > max_nodes:
        logger.warning(
            "CPC subgraph (%s, %s) has %d nodes (> %d). Subsampling.",
            section_a, section_b, sg.n_nodes, max_nodes,
        )
        # Keep highest-degree nodes
        degrees = np.array(sg.adj.sum(axis=0) + sg.adj.sum(axis=1).T).flatten()
        top_idx = np.argsort(degrees)[-max_nodes:]
        sub_adj = sg.adj[top_idx][:, top_idx]
        sub_ids = sg.idx_to_id[top_idx]
    else:
        sub_adj = sg.adj
        sub_ids = sg.idx_to_id

    # Convert to undirected NetworkX graph
    adj_sym = sub_adj + sub_adj.T
    adj_sym = (adj_sym > 0).astype(np.int8)
    G = nx.from_scipy_sparse_array(adj_sym, create_using=nx.Graph)

    # Relabel with patent IDs
    mapping = {i: sub_ids[i] for i in range(len(sub_ids))}
    G = nx.relabel_nodes(G, mapping)

    logger.info(
        "NetworkX graph (%s, %s): %d nodes, %d edges (undirected)",
        section_a, section_b, G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Cross-class edge analysis
# ---------------------------------------------------------------------------

def cpc_cross_class_edges(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
) -> pd.DataFrame:
    """Identify citation edges that cross CPC section boundaries.

    Args:
        citations: Citation DataFrame with ``citing_id``, ``cited_id``.
        cpc_map: CPC mapping with ``patent_id``, ``cpc_section``.

    Returns:
        DataFrame of cross-section citations with section labels.
    """
    # Get primary section per patent
    primary = cpc_map.groupby("patent_id")["cpc_section"].first()

    df = citations[["citing_id", "cited_id"]].copy()
    df["citing_section"] = df["citing_id"].map(primary)
    df["cited_section"] = df["cited_id"].map(primary)
    df = df.dropna(subset=["citing_section", "cited_section"])

    cross = df[df["citing_section"] != df["cited_section"]]
    logger.info(
        "Cross-class edges: %d / %d (%.1f%%)",
        len(cross), len(df), 100 * len(cross) / max(len(df), 1),
    )
    return cross
