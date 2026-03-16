"""Unit tests for graph construction and temporal snapshots."""

import numpy as np
import pandas as pd
import pytest

from src.graph import build_citation_graph, cpc_subgraph, cpc_cross_class_edges


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_citations():
    """Small synthetic citation dataset."""
    return pd.DataFrame({
        "citing_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "cited_id":  ["P2", "P3", "P4", "P5", "P6", "P1"],
        "citing_date": pd.to_datetime([
            "2000-01-01", "2001-06-15", "2005-03-10",
            "2010-07-20", "2015-01-01", "2020-12-01",
        ]),
    })


@pytest.fixture
def sample_patents():
    """Patent metadata matching the citation dataset."""
    return pd.DataFrame({
        "patent_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "date": pd.to_datetime([
            "1999-01-01", "1999-06-01", "2000-01-01",
            "2004-01-01", "2009-01-01", "2014-01-01",
        ]),
    })


@pytest.fixture
def sample_cpc_map():
    """CPC mapping: P1-P3 in section A, P4-P6 in section G."""
    return pd.DataFrame({
        "patent_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "cpc_section": ["A", "A", "A", "G", "G", "G"],
        "cpc_class": ["A01", "A01", "A61", "G06", "G06", "G01"],
        "cpc_subclass": ["A01B", "A01C", "A61K", "G06F", "G06N", "G01N"],
    })


# ---------------------------------------------------------------------------
# TestCitationGraph
# ---------------------------------------------------------------------------

class TestCitationGraph:
    """Test citation graph construction."""

    def test_directed_graph(self, sample_citations):
        """Citation graph adjacency should be asymmetric (directed)."""
        graph = build_citation_graph(sample_citations)
        # In a directed graph, adj[i,j] != adj[j,i] in general
        adj = graph.adj
        assert adj.nnz > 0
        # Check that the graph has the right number of edges
        assert graph.n_edges == len(sample_citations)

    def test_node_count_matches_patents(self, sample_citations):
        """Number of nodes should match unique patents in citations."""
        graph = build_citation_graph(sample_citations)
        unique_ids = pd.unique(
            pd.concat([sample_citations["citing_id"], sample_citations["cited_id"]])
        )
        assert graph.n_nodes == len(unique_ids)

    def test_no_self_citations(self, sample_citations):
        """Diagonal of adjacency should be zero (no self-citations)."""
        graph = build_citation_graph(sample_citations)
        diag = graph.adj.diagonal()
        assert np.all(diag == 0)

    def test_id_mapping_roundtrip(self, sample_citations):
        """ID mapping should be bijective: id → idx → id."""
        graph = build_citation_graph(sample_citations)
        for pid, idx in graph.id_to_idx.items():
            assert graph.idx_to_id[idx] == pid

    def test_dates_aligned(self, sample_citations, sample_patents):
        """Patent dates should be aligned with node indices."""
        graph = build_citation_graph(sample_citations, sample_patents)
        assert graph.dates is not None
        assert len(graph.dates) == graph.n_nodes


# ---------------------------------------------------------------------------
# TestCPCSubgraph
# ---------------------------------------------------------------------------

class TestCPCSubgraph:
    """Test CPC-filtered subgraph extraction."""

    def test_section_filter(self, sample_citations, sample_cpc_map):
        """Subgraph should only contain patents from specified sections."""
        sg = cpc_subgraph(sample_citations, sample_cpc_map, "A", "G")
        # All node IDs should be in section A or G
        for pid in sg.idx_to_id:
            section = sample_cpc_map[sample_cpc_map["patent_id"] == pid]["cpc_section"].values
            assert len(section) > 0
            assert section[0] in ("A", "G")

    def test_cross_class_edges(self, sample_citations, sample_cpc_map):
        """Cross-class edges should cross section boundaries."""
        cross = cpc_cross_class_edges(sample_citations, sample_cpc_map)
        for _, row in cross.iterrows():
            assert row["citing_section"] != row["cited_section"]

    def test_single_section_subgraph(self, sample_citations, sample_cpc_map):
        """Subgraph with same section twice should contain only that section's patents."""
        sg = cpc_subgraph(sample_citations, sample_cpc_map, "A", "A")
        for pid in sg.idx_to_id:
            section = sample_cpc_map[sample_cpc_map["patent_id"] == pid]["cpc_section"].values
            if len(section) > 0:
                assert section[0] == "A"
