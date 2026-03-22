"""Unit tests for network metrics: graph summaries, mixing rates, entropy."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.graph import SparseGraph
from src.metrics import (
    graph_summary,
    degree_distribution,
    cpc_mixing_rate,
    shannon_entropy_cpc,
    cpc_section_flow_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def triangle_graph():
    """A 3-node directed graph: P1→P2, P2→P3, P3→P1 (cycle)."""
    adj = sparse.csr_matrix(np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.int8))
    ids = np.array(["P1", "P2", "P3"])
    return SparseGraph(
        adj=adj,
        id_to_idx={"P1": 0, "P2": 1, "P3": 2},
        idx_to_id=ids,
    )


@pytest.fixture
def star_graph():
    """A 4-node star: P1→P2, P1→P3, P1→P4 (hub-and-spoke)."""
    adj = sparse.csr_matrix(np.array([
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8))
    ids = np.array(["P1", "P2", "P3", "P4"])
    return SparseGraph(
        adj=adj,
        id_to_idx={"P1": 0, "P2": 1, "P3": 2, "P4": 3},
        idx_to_id=ids,
    )


@pytest.fixture
def citation_data_with_sections():
    """Citation data with CPC sections for mixing rate tests."""
    citations = pd.DataFrame({
        "citing_id": ["P1", "P2", "P3", "P4", "P5"],
        "cited_id":  ["P2", "P3", "P4", "P5", "P1"],
        "citing_date": [
            "2010-03-15", "2010-06-20", "2011-01-10",
            "2011-08-05", "2012-02-28",
        ],
    })
    cpc_map = pd.DataFrame({
        "patent_id": ["P1", "P2", "P3", "P4", "P5"],
        "cpc_section": ["A", "A", "C", "C", "G"],
    })
    return citations, cpc_map


# ---------------------------------------------------------------------------
# TestGraphSummary
# ---------------------------------------------------------------------------

class TestGraphSummary:
    """Test basic graph summary statistics."""

    def test_node_count(self, triangle_graph):
        summary = graph_summary(triangle_graph)
        assert summary["node_count"] == 3

    def test_edge_count(self, triangle_graph):
        summary = graph_summary(triangle_graph)
        assert summary["edge_count"] == 3

    def test_density_triangle(self, triangle_graph):
        """3 edges in a 3-node graph: density = 3 / (3*2) = 0.5."""
        summary = graph_summary(triangle_graph)
        assert abs(summary["density"] - 0.5) < 1e-10

    def test_density_star(self, star_graph):
        """3 edges in a 4-node graph: density = 3 / (4*3) = 0.25."""
        summary = graph_summary(star_graph)
        assert abs(summary["density"] - 0.25) < 1e-10

    def test_avg_degree(self, triangle_graph):
        """In a 3-cycle, every node has in-degree=1 and out-degree=1."""
        summary = graph_summary(triangle_graph)
        assert abs(summary["avg_in_degree"] - 1.0) < 1e-10
        assert abs(summary["avg_out_degree"] - 1.0) < 1e-10

    def test_star_degree_distribution(self, star_graph):
        """Hub has out-degree 3, spokes have out-degree 0."""
        summary = graph_summary(star_graph)
        assert abs(summary["avg_out_degree"] - 0.75) < 1e-10  # 3/4
        assert abs(summary["avg_in_degree"] - 0.75) < 1e-10   # 3/4

    def test_connected_components_cycle(self, triangle_graph):
        """A 3-cycle is one connected component."""
        summary = graph_summary(triangle_graph)
        assert summary["connected_components"] == 1

    def test_connected_components_star(self, star_graph):
        """A star graph is one connected component."""
        summary = graph_summary(star_graph)
        assert summary["connected_components"] == 1

    def test_disconnected_graph(self):
        """Two isolated components should report 2 connected components."""
        adj = sparse.csr_matrix(np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.int8))
        ids = np.array(["P1", "P2", "P3", "P4"])
        graph = SparseGraph(
            adj=adj,
            id_to_idx={p: i for i, p in enumerate(ids)},
            idx_to_id=ids,
        )
        summary = graph_summary(graph)
        assert summary["connected_components"] == 2


# ---------------------------------------------------------------------------
# TestDegreeDistribution
# ---------------------------------------------------------------------------

class TestDegreeDistribution:
    """Test degree distribution extraction."""

    def test_returns_arrays(self, triangle_graph):
        dist = degree_distribution(triangle_graph)
        assert isinstance(dist["in_degree"], np.ndarray)
        assert isinstance(dist["out_degree"], np.ndarray)

    def test_lengths_match_nodes(self, triangle_graph):
        dist = degree_distribution(triangle_graph)
        assert len(dist["in_degree"]) == 3
        assert len(dist["out_degree"]) == 3

    def test_star_hub_degree(self, star_graph):
        dist = degree_distribution(star_graph)
        assert dist["out_degree"][0] == 3  # Hub
        assert all(dist["out_degree"][1:] == 0)  # Spokes


# ---------------------------------------------------------------------------
# TestCPCMixingRate
# ---------------------------------------------------------------------------

class TestCPCMixingRate:
    """Test cross-section citation mixing rate."""

    def test_mixing_rate_values(self, citation_data_with_sections):
        citations, cpc_map = citation_data_with_sections
        result = cpc_mixing_rate(citations, cpc_map)

        assert "mixing_rate" in result.columns
        assert "year" in result.columns
        # All rates should be between 0 and 1
        assert (result["mixing_rate"] >= 0).all()
        assert (result["mixing_rate"] <= 1).all()

    def test_all_same_section_zero_mixing(self):
        """If all patents are in the same section, mixing rate = 0."""
        citations = pd.DataFrame({
            "citing_id": ["P1", "P2"],
            "cited_id": ["P2", "P3"],
            "citing_date": ["2010-01-01", "2010-06-01"],
        })
        cpc_map = pd.DataFrame({
            "patent_id": ["P1", "P2", "P3"],
            "cpc_section": ["A", "A", "A"],
        })
        result = cpc_mixing_rate(citations, cpc_map)
        assert (result["mixing_rate"] == 0).all()

    def test_all_cross_section_full_mixing(self):
        """If every citation crosses sections, mixing rate = 1."""
        citations = pd.DataFrame({
            "citing_id": ["P1", "P2"],
            "cited_id": ["P3", "P4"],
            "citing_date": ["2010-01-01", "2010-06-01"],
        })
        cpc_map = pd.DataFrame({
            "patent_id": ["P1", "P2", "P3", "P4"],
            "cpc_section": ["A", "A", "C", "C"],
        })
        result = cpc_mixing_rate(citations, cpc_map)
        assert (result["mixing_rate"] == 1.0).all()

    def test_annual_grouping(self, citation_data_with_sections):
        """Results should be grouped by year."""
        citations, cpc_map = citation_data_with_sections
        result = cpc_mixing_rate(citations, cpc_map)
        # Our fixture has citations in 2010, 2011, 2012
        assert set(result["year"]) == {2010, 2011, 2012}


# ---------------------------------------------------------------------------
# TestShannonEntropyCPC
# ---------------------------------------------------------------------------

class TestShannonEntropyCPC:
    """Test Shannon entropy of CPC section distribution."""

    def test_single_section_zero_entropy(self):
        """All citations to one section → entropy ≈ 0."""
        citations = pd.DataFrame({
            "cited_id": ["P1", "P2", "P3"],
        })
        cpc_map = pd.DataFrame({
            "patent_id": ["P1", "P2", "P3"],
            "cpc_section": ["A", "A", "A"],
        })
        entropy = shannon_entropy_cpc(citations, cpc_map)
        assert entropy < 0.01  # Approximately zero (1e-12 guard)

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution across 4 sections → entropy = log2(4) = 2."""
        citations = pd.DataFrame({
            "cited_id": ["P1", "P2", "P3", "P4"] * 25,  # 25 each
        })
        cpc_map = pd.DataFrame({
            "patent_id": ["P1", "P2", "P3", "P4"],
            "cpc_section": ["A", "B", "C", "D"],
        })
        entropy = shannon_entropy_cpc(citations, cpc_map)
        assert abs(entropy - 2.0) < 0.01  # log2(4) = 2

    def test_entropy_nonnegative(self, citation_data_with_sections):
        citations, cpc_map = citation_data_with_sections
        entropy = shannon_entropy_cpc(citations, cpc_map)
        assert entropy >= 0


# ---------------------------------------------------------------------------
# TestCPCSectionFlowMatrix
# ---------------------------------------------------------------------------

class TestCPCSectionFlowMatrix:
    """Test section-to-section citation flow matrix."""

    def test_shape_8x8(self, citation_data_with_sections):
        """Flow matrix should always be 8×8 (all CPC sections)."""
        citations, cpc_map = citation_data_with_sections
        flow = cpc_section_flow_matrix(citations, cpc_map)
        assert flow.shape == (8, 8)

    def test_sections_as_labels(self, citation_data_with_sections):
        citations, cpc_map = citation_data_with_sections
        flow = cpc_section_flow_matrix(citations, cpc_map)
        assert list(flow.index) == list("ABCDEFGH")
        assert list(flow.columns) == list("ABCDEFGH")

    def test_nonnegative_counts(self, citation_data_with_sections):
        citations, cpc_map = citation_data_with_sections
        flow = cpc_section_flow_matrix(citations, cpc_map)
        assert (flow.values >= 0).all()

    def test_total_matches_valid_citations(self, citation_data_with_sections):
        """Total flow should equal number of citations with valid CPC mappings."""
        citations, cpc_map = citation_data_with_sections
        flow = cpc_section_flow_matrix(citations, cpc_map)
        # All 5 citations have valid CPC mappings
        assert flow.values.sum() == 5

    def test_diagonal_is_within_section(self):
        """Diagonal elements are within-section citations."""
        citations = pd.DataFrame({
            "citing_id": ["P1", "P2"],
            "cited_id": ["P2", "P1"],
        })
        cpc_map = pd.DataFrame({
            "patent_id": ["P1", "P2"],
            "cpc_section": ["A", "A"],
        })
        flow = cpc_section_flow_matrix(citations, cpc_map)
        assert flow.loc["A", "A"] == 2
        # All other entries should be 0
        assert flow.values.sum() == 2
