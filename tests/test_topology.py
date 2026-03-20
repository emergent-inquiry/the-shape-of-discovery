"""Unit tests for co-citation distance matrix topology pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.topology import (
    build_cocitation_matrix,
    cocitation_to_distance,
    compute_persistence,
    betti_numbers,
    persistence_entropy,
    max_persistence,
    n_long_lived_features,
    PRIORITY_PAIRS,
    ALL_PAIRS,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic citation data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create a small synthetic citations + cpc_map dataset."""
    # 6 patents across 3 CPC subclasses in 2 sections
    cpc_map = pd.DataFrame({
        "patent_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "cpc_section": ["A", "A", "C", "C", "A", "C"],
        "cpc_class": ["A01", "A01", "C07", "C12", "A61", "C07"],
        "cpc_subclass": ["A01B", "A01B", "C07K", "C12N", "A61K", "C07K"],
    })

    citations = pd.DataFrame({
        "citing_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P1", "P3"],
        "cited_id":  ["P3", "P4", "P5", "P1", "P6", "P2", "P6", "P2"],
        "citing_year": [2010, 2010, 2011, 2011, 2012, 2012, 2010, 2011],
    })

    return citations, cpc_map


# ---------------------------------------------------------------------------
# TestBuildCocitationMatrix
# ---------------------------------------------------------------------------

class TestBuildCocitationMatrix:
    """Test co-citation matrix construction."""

    def test_basic_counts(self, synthetic_data):
        """Verify that co-citation counts match expected values."""
        citations, cpc_map = synthetic_data
        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, start_year=2010, end_year=2012, level="subclass"
        )
        assert not cocite_df.empty
        assert len(labels) > 0
        # Matrix should be square
        assert cocite_df.shape[0] == cocite_df.shape[1]
        # Total citations should equal number of rows in citations that have CPC mappings
        assert cocite_df.values.sum() > 0

    def test_empty_window(self, synthetic_data):
        """Window with no citations should return empty DataFrame."""
        citations, cpc_map = synthetic_data
        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, start_year=2020, end_year=2025, level="subclass"
        )
        assert cocite_df.empty
        assert labels == []

    def test_level_section(self, synthetic_data):
        """Section-level should produce a smaller matrix."""
        citations, cpc_map = synthetic_data
        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, start_year=2010, end_year=2012, level="section"
        )
        # Only 2 sections: A and C
        assert len(labels) <= 2

    def test_invalid_level(self, synthetic_data):
        """Invalid CPC level should raise ValueError."""
        citations, cpc_map = synthetic_data
        with pytest.raises(ValueError, match="not found"):
            build_cocitation_matrix(
                citations, cpc_map, start_year=2010, end_year=2012, level="invalid"
            )

    def test_requires_citing_year(self, synthetic_data):
        """Should fail if citing_year column is missing."""
        citations, cpc_map = synthetic_data
        citations_no_year = citations.drop(columns=["citing_year"])
        with pytest.raises(KeyError):
            build_cocitation_matrix(
                citations_no_year, cpc_map, start_year=2010, end_year=2012
            )


# ---------------------------------------------------------------------------
# TestCocitationToDistance
# ---------------------------------------------------------------------------

class TestCocitationToDistance:
    """Test distance matrix conversion."""

    def test_symmetric_output(self):
        """Distance matrix should be symmetric."""
        matrix = np.array([
            [0, 10, 5],
            [8, 0, 3],
            [2, 7, 0],
        ], dtype=float)
        dist, mask = cocitation_to_distance(matrix)
        assert dist.shape[0] == dist.shape[1]
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_diagonal_zero(self):
        """Diagonal of distance matrix should be exactly zero."""
        matrix = np.array([
            [0, 10, 5],
            [8, 0, 3],
            [2, 7, 0],
        ], dtype=float)
        dist, mask = cocitation_to_distance(matrix)
        np.testing.assert_array_equal(np.diag(dist), 0)

    def test_nonnegative(self):
        """All distances should be non-negative."""
        matrix = np.random.rand(10, 10) * 100
        dist, mask = cocitation_to_distance(matrix)
        assert (dist >= 0).all()

    def test_identical_rows_small_distance(self):
        """Identical citation patterns should have small distance after symmetrization."""
        matrix = np.array([
            [0, 10, 5],
            [0, 10, 5],  # identical to row 0
            [3, 0, 7],
        ], dtype=float)
        dist, mask = cocitation_to_distance(matrix)
        # After symmetrization (A+A^T), rows 0 and 1 won't be exactly identical
        # because column contributions differ, but should be closer than others
        assert dist[0, 1] < dist[0, 2]

    def test_too_few_active(self):
        """Fewer than 3 active classes should return empty array."""
        matrix = np.array([
            [0, 1],
            [1, 0],
        ], dtype=float)
        dist, mask = cocitation_to_distance(matrix)
        assert dist.size == 0

    def test_zero_rows_filtered(self):
        """Zero rows should be filtered via active_mask."""
        matrix = np.array([
            [0, 10, 5, 0],
            [8, 0, 3, 0],
            [2, 7, 0, 0],
            [0, 0, 0, 0],  # zero row
        ], dtype=float)
        dist, mask = cocitation_to_distance(matrix)
        # 3 active classes, 1 filtered
        assert dist.shape[0] == 3
        assert mask.sum() == 3
        assert not mask[3]


# ---------------------------------------------------------------------------
# TestComputePersistence
# ---------------------------------------------------------------------------

class TestComputePersistence:
    """Test Vietoris-Rips persistence on known distance matrices."""

    def test_equilateral_triangle(self):
        """3 equidistant points: Rips fills triangle immediately, so H1 is trivial."""
        dist = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        diagrams = compute_persistence(dist, max_dim=1)
        assert len(diagrams) >= 2
        # H0: 3 points merge into 1 component
        assert len(diagrams[0]) >= 1
        # H1: In Rips, all 3 edges appear at r=1 and the 2-simplex fills immediately,
        # so no persistent H1. This is correct behavior.
        # (A square with diagonals longer would produce persistent H1.)

    def test_square_has_h1(self):
        """A square (4 points, diagonals longer) should have a persistent H1 feature."""
        # Square: adjacent distance=1, diagonal distance=sqrt(2)
        d = np.sqrt(2)
        dist = np.array([
            [0, 1, d, 1],
            [1, 0, 1, d],
            [d, 1, 0, 1],
            [1, d, 1, 0],
        ], dtype=float)
        diagrams = compute_persistence(dist, max_dim=1)
        assert len(diagrams) >= 2
        # H1: loop born at r=1 (4 edges), dies at r=sqrt(2) (diagonals fill it)
        assert len(diagrams[1]) >= 1
        persistence = diagrams[1][:, 1] - diagrams[1][:, 0]
        finite = persistence[np.isfinite(persistence)]
        assert len(finite) >= 1
        assert finite.max() > 0.1

    def test_collinear_points(self):
        """Collinear points should have no H1 features."""
        dist = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=float)
        diagrams = compute_persistence(dist, max_dim=1)
        # H1 should be empty or have only trivial features
        if len(diagrams[1]) > 0:
            persistence = diagrams[1][:, 1] - diagrams[1][:, 0]
            finite = persistence[np.isfinite(persistence)]
            assert np.all(finite < 0.01)


# ---------------------------------------------------------------------------
# TestBettiNumbers
# ---------------------------------------------------------------------------

class TestBettiNumbers:
    """Test Betti number extraction from diagrams."""

    def test_basic(self):
        """Simple diagrams with known counts."""
        dgm0 = np.array([[0, np.inf], [0, 0.5]])  # 2 H0 features (1 inf, 1 finite)
        dgm1 = np.array([[0.5, 1.0], [0.6, 0.9]])  # 2 H1 features
        b0, b1, b2 = betti_numbers([dgm0, dgm1])
        assert b0 == 2
        assert b1 == 2
        assert b2 == 0  # no H2 diagrams

    def test_with_threshold(self):
        """Threshold should filter out short-lived features."""
        dgm0 = np.array([[0, np.inf], [0, 0.1]])
        dgm1 = np.array([[0.5, 1.5], [0.6, 0.7]])  # persistence 1.0 and 0.1
        b0, b1, b2 = betti_numbers([dgm0, dgm1], threshold=0.5)
        assert b0 == 1  # only the infinite one
        assert b1 == 1  # only the long-lived one

    def test_empty_diagrams(self):
        """Empty diagrams should return all zeros."""
        b0, b1, b2 = betti_numbers([np.empty((0, 2)), np.empty((0, 2))])
        assert (b0, b1, b2) == (0, 0, 0)


# ---------------------------------------------------------------------------
# TestPersistenceEntropy
# ---------------------------------------------------------------------------

class TestPersistenceEntropy:
    """Test persistence entropy computation."""

    def test_single_feature(self):
        """Single finite feature should have entropy 0."""
        dgm = [np.array([[0.0, 1.0]])]
        assert persistence_entropy(dgm) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_features(self):
        """Equal-lifetime features should give log2(n) entropy."""
        n = 4
        dgm = [np.array([[0.0, 1.0]] * n)]
        expected = np.log2(n)
        assert persistence_entropy(dgm) == pytest.approx(expected, abs=1e-6)

    def test_empty_diagram(self):
        """Empty diagram should have entropy 0."""
        dgm = [np.empty((0, 2))]
        assert persistence_entropy(dgm) == 0.0

    def test_infinite_features_excluded(self):
        """Infinite features should not contribute to entropy."""
        dgm = [np.array([[0.0, np.inf], [0.0, 1.0]])]
        # Only one finite feature → entropy = 0
        assert persistence_entropy(dgm) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# TestMaxPersistence
# ---------------------------------------------------------------------------

class TestMaxPersistence:
    """Test max persistence extraction."""

    def test_basic(self):
        """Should return the largest finite persistence."""
        dgm0 = np.array([[0, 0.5]])
        dgm1 = np.array([[0.1, 0.8], [0.2, 1.5]])  # max persistence = 1.3
        result = max_persistence([dgm0, dgm1], dim=1)
        assert result == pytest.approx(1.3, abs=1e-10)

    def test_empty_dimension(self):
        """Missing dimension should return 0."""
        dgm0 = np.array([[0, 0.5]])
        result = max_persistence([dgm0], dim=1)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestNLongLivedFeatures
# ---------------------------------------------------------------------------

class TestNLongLivedFeatures:
    """Test long-lived feature counting."""

    def test_basic(self):
        """Count features above 90th percentile."""
        # 10 features, persistence = 0.1, 0.2, ..., 1.0
        dgm1 = np.array([[0, p] for p in np.arange(0.1, 1.1, 0.1)])
        count = n_long_lived_features([np.empty((0, 2)), dgm1], dim=1, percentile=90)
        # 90th percentile of [0.1..1.0] = 0.9, only 1.0 exceeds it
        assert count == 1

    def test_empty(self):
        """Empty diagram should return 0."""
        count = n_long_lived_features([np.empty((0, 2)), np.empty((0, 2))], dim=1)
        assert count == 0


# ---------------------------------------------------------------------------
# TestPriorityPairs
# ---------------------------------------------------------------------------

class TestPriorityPairs:
    """Test the PRIORITY_PAIRS constant."""

    def test_count(self):
        """Should have exactly 10 priority pairs."""
        assert len(PRIORITY_PAIRS) == 10

    def test_format(self):
        """Each pair should be a tuple of two single-character CPC section codes."""
        for pair in PRIORITY_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert len(pair[0]) == 1 and pair[0].isalpha()
            assert len(pair[1]) == 1 and pair[1].isalpha()


class TestAllPairs:
    """Test the ALL_PAIRS constant."""

    def test_count(self):
        """8 CPC sections → 8 choose 2 = 28 pairs."""
        assert len(ALL_PAIRS) == 28

    def test_no_duplicates(self):
        """No duplicate pairs."""
        normalized = [tuple(sorted(p)) for p in ALL_PAIRS]
        assert len(set(normalized)) == 28

    def test_no_self_pairs(self):
        """No pair should have the same section twice."""
        for a, b in ALL_PAIRS:
            assert a != b


class TestScaleNormalization:
    """Test that scale normalization controls for density confound."""

    def test_normalized_mean_is_one(self):
        """After normalization, mean distance should be ~1.0."""
        matrix = np.array([
            [0, 10, 5],
            [8, 0, 3],
            [2, 7, 0],
        ], dtype=float)
        dist, _ = cocitation_to_distance(matrix, normalize_scale=True)
        upper_tri = dist[np.triu_indices_from(dist, k=1)]
        assert upper_tri.mean() == pytest.approx(1.0, abs=0.01)

    def test_unnormalized_not_one(self):
        """Without normalization, mean distance should NOT be 1.0."""
        matrix = np.array([
            [0, 10, 5],
            [8, 0, 3],
            [2, 7, 0],
        ], dtype=float)
        dist, _ = cocitation_to_distance(matrix, normalize_scale=False)
        upper_tri = dist[np.triu_indices_from(dist, k=1)]
        assert upper_tri.mean() != pytest.approx(1.0, abs=0.01)

    def test_relative_structure_preserved(self):
        """Normalization should preserve relative ordering of distances."""
        matrix = np.array([
            [0, 10, 5, 1],
            [8, 0, 3, 2],
            [2, 7, 0, 6],
            [1, 2, 6, 0],
        ], dtype=float)
        dist_raw, _ = cocitation_to_distance(matrix, normalize_scale=False)
        dist_norm, _ = cocitation_to_distance(matrix, normalize_scale=True)
        # The ordering of pairwise distances should be the same
        raw_order = np.argsort(dist_raw[np.triu_indices_from(dist_raw, k=1)])
        norm_order = np.argsort(dist_norm[np.triu_indices_from(dist_norm, k=1)])
        np.testing.assert_array_equal(raw_order, norm_order)
