"""Network metrics: summaries, degree distributions, CPC mixing, entropy."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from src.graph import SparseGraph
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph summary
# ---------------------------------------------------------------------------

def graph_summary(graph: SparseGraph) -> dict:
    """Compute basic summary statistics for a sparse citation graph.

    Args:
        graph: SparseGraph instance.

    Returns:
        Dict with node_count, edge_count, density, avg_in_degree,
        avg_out_degree, connected_components.
    """
    n = graph.n_nodes
    m = graph.n_edges
    density = m / (n * (n - 1)) if n > 1 else 0.0

    out_degrees = np.array(graph.adj.sum(axis=1)).flatten()
    in_degrees = np.array(graph.adj.sum(axis=0)).flatten()

    # Connected components on the undirected version
    adj_sym = graph.adj + graph.adj.T
    n_components, _ = connected_components(adj_sym, directed=False)

    return {
        "node_count": n,
        "edge_count": m,
        "density": density,
        "avg_in_degree": float(in_degrees.mean()),
        "avg_out_degree": float(out_degrees.mean()),
        "connected_components": n_components,
    }


# ---------------------------------------------------------------------------
# Degree distributions
# ---------------------------------------------------------------------------

def degree_distribution(graph: SparseGraph) -> dict[str, np.ndarray]:
    """Compute in-degree and out-degree arrays.

    Args:
        graph: SparseGraph instance.

    Returns:
        Dict with 'in_degree' and 'out_degree' arrays.
    """
    return {
        "in_degree": np.array(graph.adj.sum(axis=0)).flatten(),
        "out_degree": np.array(graph.adj.sum(axis=1)).flatten(),
    }


# ---------------------------------------------------------------------------
# CPC mixing rate
# ---------------------------------------------------------------------------

def cpc_mixing_rate(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
    year_col: str = "citing_date",
) -> pd.DataFrame:
    """Fraction of citations crossing CPC section boundaries, per year.

    Args:
        citations: Citation DataFrame with citing_id, cited_id, and a date column.
        cpc_map: CPC map with patent_id, cpc_section.
        year_col: Name of the date column to extract year from.

    Returns:
        DataFrame with columns: year, total_citations, cross_section_citations, mixing_rate.
    """
    primary = cpc_map.groupby("patent_id")["cpc_section"].first()

    df = citations[["citing_id", "cited_id", year_col]].copy()
    df["year"] = pd.to_datetime(df[year_col]).dt.year
    df["citing_section"] = df["citing_id"].map(primary)
    df["cited_section"] = df["cited_id"].map(primary)
    df = df.dropna(subset=["citing_section", "cited_section"])

    df["is_cross"] = df["citing_section"] != df["cited_section"]

    annual = df.groupby("year").agg(
        total_citations=("is_cross", "size"),
        cross_section_citations=("is_cross", "sum"),
    ).reset_index()

    annual["mixing_rate"] = annual["cross_section_citations"] / annual["total_citations"]
    return annual


# ---------------------------------------------------------------------------
# Shannon entropy of CPC distribution
# ---------------------------------------------------------------------------

def shannon_entropy_cpc(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
) -> float:
    """Shannon entropy of the CPC section distribution among cited patents.

    Higher entropy = more evenly distributed citations across CPC sections.

    Args:
        citations: Citation DataFrame with cited_id.
        cpc_map: CPC map with patent_id, cpc_section.

    Returns:
        Shannon entropy in bits.
    """
    primary = cpc_map.groupby("patent_id")["cpc_section"].first()
    sections = citations["cited_id"].map(primary).dropna()
    counts = sections.value_counts(normalize=True).values

    # H = -sum(p * log2(p))
    entropy = -np.sum(counts * np.log2(counts + 1e-12))
    return float(entropy)


def cpc_section_flow_matrix(
    citations: pd.DataFrame,
    cpc_map: pd.DataFrame,
) -> pd.DataFrame:
    """Build a section-to-section citation flow matrix.

    Args:
        citations: Citation DataFrame.
        cpc_map: CPC map with patent_id, cpc_section.

    Returns:
        8x8 DataFrame with CPC sections as both index and columns.
        Values are citation counts from row-section to column-section.
    """
    primary = cpc_map.groupby("patent_id")["cpc_section"].first()

    df = citations[["citing_id", "cited_id"]].copy()
    df["from_sec"] = df["citing_id"].map(primary)
    df["to_sec"] = df["cited_id"].map(primary)
    df = df.dropna(subset=["from_sec", "to_sec"])

    flow = pd.crosstab(df["from_sec"], df["to_sec"])

    # Ensure all 8 sections are present
    sections = list("ABCDEFGH")
    flow = flow.reindex(index=sections, columns=sections, fill_value=0)

    return flow
