"""Unit tests for persistent homology computations on synthetic graphs."""
import pytest


class TestBettiNumbers:
    """Test Betti number computation on graphs with known topology."""

    def test_disconnected_components(self):
        """Two disconnected cliques should have β₀ = 2."""
        # TODO: Implement after src/topology.py is built
        pass

    def test_cycle_graph(self):
        """A cycle graph should have β₁ = 1."""
        pass

    def test_complete_graph(self):
        """A complete graph on 4 nodes (tetrahedron boundary) has known Betti numbers."""
        pass

    def test_empty_graph(self):
        """Empty graph should return β₀ = 0, β₁ = 0, β₂ = 0."""
        pass


class TestPersistenceEntropy:
    """Test persistence entropy computation."""

    def test_single_feature(self):
        """Single persistent feature should have entropy 0."""
        pass

    def test_uniform_features(self):
        """Uniformly distributed lifetimes should maximize entropy."""
        pass


class TestSlidingWindow:
    """Test sliding window topology computation."""

    def test_window_count(self):
        """Correct number of windows for given date range and stride."""
        pass

    def test_window_overlap(self):
        """Adjacent windows should overlap correctly."""
        pass
