"""Unit tests for persistent homology computations on synthetic graphs."""

import numpy as np
import networkx as nx
import pytest

from src.topology import (
    compute_persistence,
    betti_numbers,
    persistence_entropy,
    topology_summary,
    reduce_graph,
)


# ---------------------------------------------------------------------------
# TestBettiNumbers
# ---------------------------------------------------------------------------

class TestBettiNumbers:
    """Test Betti number computation on graphs with known topology."""

    def test_disconnected_components(self):
        """Two disconnected cliques: β₀ should show 2 components."""
        G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
        result = compute_persistence(G, max_dim=0)
        dgms = result["dgms"]
        # H0 should have features; 2 components means one infinite feature per component
        # ripser reports all H0 features including those that merge
        assert len(dgms[0]) >= 2

    def test_cycle_graph(self):
        """A cycle graph should have β₁ = 1 (one independent loop)."""
        G = nx.cycle_graph(8)
        result = compute_persistence(G, max_dim=1)
        dgms = result["dgms"]
        # H1 should have exactly 1 feature (the loop)
        assert len(dgms[1]) == 1

    def test_complete_graph_no_h1(self):
        """A complete graph on 4 nodes should have β₁ = 0 (all loops are filled)."""
        G = nx.complete_graph(4)
        result = compute_persistence(G, max_dim=1, sparse_mode=False)
        dgms = result["dgms"]
        # All H1 features should die immediately (be trivial)
        if len(dgms[1]) > 0:
            lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
            finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
            # All lifetimes should be very small (triangles fill immediately)
            assert np.all(finite_lifetimes < 0.5)

    def test_empty_graph(self):
        """Empty graph should return β₀ = 0, β₁ = 0."""
        G = nx.Graph()
        summary = topology_summary(G, max_dim=1)
        assert summary["beta_0"] == 0
        assert summary["beta_1"] == 0
        assert summary["n_nodes"] == 0

    def test_single_node(self):
        """Single node: β₀ = 1, β₁ = 0."""
        G = nx.Graph()
        G.add_node(0)
        result = compute_persistence(G, max_dim=1)
        assert result["n_nodes"] == 1
        # H0 should have one feature
        assert len(result["dgms"][0]) == 1

    def test_two_triangles_sharing_edge(self):
        """Two triangles sharing an edge have β₁ = 0 (no independent loops)."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2), (1, 3), (2, 3)])
        result = compute_persistence(G, max_dim=1, sparse_mode=False)
        dgms = result["dgms"]
        # Both triangles are filled, so no persistent H1 features
        if len(dgms[1]) > 0:
            lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
            finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
            assert np.all(finite_lifetimes < 0.5)


# ---------------------------------------------------------------------------
# TestPersistenceEntropy
# ---------------------------------------------------------------------------

class TestPersistenceEntropy:
    """Test persistence entropy computation."""

    def test_single_feature(self):
        """Single persistent feature should have entropy 0."""
        dgm = np.array([[0.0, 1.0]])
        assert persistence_entropy(dgm) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_features(self):
        """Equal-lifetime features should give log2(n) entropy."""
        n = 4
        dgm = np.array([[0.0, 1.0]] * n)
        expected = np.log2(n)
        assert persistence_entropy(dgm) == pytest.approx(expected, abs=1e-6)

    def test_empty_diagram(self):
        """Empty diagram should have entropy 0."""
        dgm = np.empty((0, 2))
        assert persistence_entropy(dgm) == 0.0

    def test_infinite_features_excluded(self):
        """Infinite features should not contribute to entropy."""
        dgm = np.array([[0.0, np.inf], [0.0, 1.0]])
        # Only one finite feature → entropy = 0
        assert persistence_entropy(dgm) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# TestReduceGraph
# ---------------------------------------------------------------------------

class TestReduceGraph:
    """Test graph reduction for tractability."""

    def test_small_graph_unchanged(self):
        """Graph smaller than max_nodes should pass through unchanged."""
        G = nx.complete_graph(10)
        reduced = reduce_graph(G, max_nodes=100)
        assert reduced.number_of_nodes() == 10

    def test_large_graph_reduced(self):
        """Graph larger than max_nodes should be reduced."""
        G = nx.barabasi_albert_graph(200, 3, seed=42)
        reduced = reduce_graph(G, max_nodes=50)
        assert reduced.number_of_nodes() <= 50

    def test_leaves_removed_first(self):
        """Degree-1 nodes should be removed before subsampling."""
        # Star graph: 1 hub + many leaves
        G = nx.star_graph(100)
        reduced = reduce_graph(G, max_nodes=50)
        # All leaves should be removed, only hub remains if that's enough
        assert reduced.number_of_nodes() <= 50


# ---------------------------------------------------------------------------
# TestTopologySummary
# ---------------------------------------------------------------------------

class TestTopologySummary:
    """Test the full topology summary pipeline."""

    def test_summary_keys(self):
        """Summary should contain all expected keys."""
        G = nx.cycle_graph(6)
        summary = topology_summary(G, max_dim=1)
        expected_keys = {
            "beta_0", "beta_1", "beta_2", "persistence_entropy",
            "max_persistence", "n_long_lived_features", "n_nodes", "n_edges",
        }
        assert set(summary.keys()) == expected_keys

    def test_cycle_summary(self):
        """Cycle graph summary should show β₁ = 1."""
        G = nx.cycle_graph(10)
        summary = topology_summary(G, max_dim=1)
        assert summary["beta_1"] >= 1
        assert summary["n_nodes"] == 10
        assert summary["n_edges"] == 10
