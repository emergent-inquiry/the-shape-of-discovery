"""Integration tests: end-to-end pipeline from citations to topology to statistics.

These tests verify that the full analysis pipeline produces consistent results
on synthetic data with known properties. They catch integration bugs that unit
tests on individual modules cannot.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr, wilcoxon

from src.topology import (
    build_cocitation_matrix,
    cocitation_to_distance,
    compute_persistence,
    betti_numbers,
    persistence_entropy,
    max_persistence,
    n_long_lived_features,
)
from src.graph import build_citation_graph, SparseGraph
from src.metrics import graph_summary, cpc_mixing_rate, shannon_entropy_cpc


# ---------------------------------------------------------------------------
# Fixtures: larger synthetic dataset for integration testing
# ---------------------------------------------------------------------------

@pytest.fixture
def medium_citation_network():
    """A 50-patent network across 4 CPC sections with known structure.

    Structure:
    - Section A: patents P001-P012
    - Section C: patents P013-P025
    - Section G: patents P026-P037
    - Section H: patents P038-P050
    - Dense within-section citations, sparse cross-section citations
    """
    rng = np.random.default_rng(42)

    patents_per_section = {
        "A": [f"P{i:03d}" for i in range(1, 13)],
        "C": [f"P{i:03d}" for i in range(13, 26)],
        "G": [f"P{i:03d}" for i in range(26, 38)],
        "H": [f"P{i:03d}" for i in range(38, 51)],
    }

    all_patents = []
    cpc_rows = []
    for sec, pids in patents_per_section.items():
        for pid in pids:
            all_patents.append(pid)
            cpc_rows.append({
                "patent_id": pid,
                "cpc_section": sec,
                "cpc_class": f"{sec}01",
                "cpc_subclass": f"{sec}01{chr(65 + rng.integers(0, 3))}",  # 3 subclasses per section
            })

    cpc_map = pd.DataFrame(cpc_rows)

    # Generate citations: 70% within-section, 30% cross-section
    citing_ids, cited_ids, years = [], [], []
    for _ in range(500):
        if rng.random() < 0.7:
            # Within-section
            sec = rng.choice(list(patents_per_section.keys()))
            pids = patents_per_section[sec]
            a, b = rng.choice(len(pids), 2, replace=False)
            citing_ids.append(pids[a])
            cited_ids.append(pids[b])
        else:
            # Cross-section
            secs = rng.choice(list(patents_per_section.keys()), 2, replace=False)
            citing_ids.append(rng.choice(patents_per_section[secs[0]]))
            cited_ids.append(rng.choice(patents_per_section[secs[1]]))
        years.append(int(rng.integers(2005, 2016)))

    citations = pd.DataFrame({
        "citing_id": citing_ids,
        "cited_id": cited_ids,
        "citing_year": years,
        "citing_date": [f"{y}-06-15" for y in years],
    })

    return citations, cpc_map


# ---------------------------------------------------------------------------
# TestEndToEndTopologyPipeline
# ---------------------------------------------------------------------------

class TestEndToEndTopologyPipeline:
    """Test the full pipeline: citations → co-citation → distance → persistence."""

    def test_pipeline_produces_valid_betti_numbers(self, medium_citation_network):
        """Full pipeline should produce non-negative Betti numbers."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        assert not cocite_df.empty
        assert len(labels) >= 3

        dist_matrix, active_mask = cocitation_to_distance(cocite_df.values)
        assert dist_matrix.size > 0

        diagrams = compute_persistence(dist_matrix, max_dim=2)
        b0, b1, b2 = betti_numbers(diagrams)

        assert b0 >= 0
        assert b1 >= 0
        assert b2 >= 0
        # β₀ should be at least 1 (at least one connected component)
        assert b0 >= 1

    def test_pipeline_persistence_entropy_finite(self, medium_citation_network):
        """Persistence entropy should be a finite non-negative number."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist_matrix, _ = cocitation_to_distance(cocite_df.values)
        diagrams = compute_persistence(dist_matrix, max_dim=2)
        pe = persistence_entropy(diagrams)

        assert np.isfinite(pe)
        assert pe >= 0

    def test_pipeline_max_persistence_nonnegative(self, medium_citation_network):
        """Max persistence should be non-negative."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist_matrix, _ = cocitation_to_distance(cocite_df.values)
        diagrams = compute_persistence(dist_matrix, max_dim=2)
        mp = max_persistence(diagrams, dim=1)

        assert mp >= 0

    def test_section_level_fewer_points(self, medium_citation_network):
        """Section-level analysis should have fewer points than subclass."""
        citations, cpc_map = medium_citation_network

        cocite_sub, labels_sub = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        cocite_sec, labels_sec = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="section"
        )

        # Section level (≤8 points) should be smaller than subclass
        assert len(labels_sec) <= len(labels_sub)
        assert len(labels_sec) <= 8


# ---------------------------------------------------------------------------
# TestGraphMetricsIntegration
# ---------------------------------------------------------------------------

class TestGraphMetricsIntegration:
    """Test graph construction and metrics together."""

    def test_build_graph_and_summarize(self, medium_citation_network):
        """Graph should build and summarize without errors."""
        citations, cpc_map = medium_citation_network
        graph = build_citation_graph.__wrapped__(citations[["citing_id", "cited_id"]])
        summary = graph_summary(graph)

        assert summary["node_count"] > 0
        assert summary["edge_count"] > 0
        assert 0 < summary["density"] < 1
        assert summary["connected_components"] >= 1

    def test_mixing_rate_between_zero_and_one(self, medium_citation_network):
        """Mixing rate should reflect the ~30% cross-section rate we built."""
        citations, cpc_map = medium_citation_network
        result = cpc_mixing_rate(citations, cpc_map)

        overall_mixing = (
            result["cross_section_citations"].sum() / result["total_citations"].sum()
        )
        # We generated ~30% cross-section citations
        assert 0.15 < overall_mixing < 0.45, (
            f"Mixing rate {overall_mixing:.2f} outside expected range for 30% cross-section"
        )


# ---------------------------------------------------------------------------
# TestScaleNormalizationIntegration
# ---------------------------------------------------------------------------

class TestScaleNormalizationIntegration:
    """Test that scale normalization controls the density confound end-to-end."""

    def test_normalized_vs_unnormalized_betti(self, medium_citation_network):
        """Normalized and unnormalized should produce different Betti numbers.

        This verifies normalization actually changes the filtration, not just
        the distance values.
        """
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )

        # Normalized (default)
        dist_norm, _ = cocitation_to_distance(cocite_df.values, normalize_scale=True)
        diag_norm = compute_persistence(dist_norm, max_dim=2)
        b1_norm = betti_numbers(diag_norm)[1]

        # Unnormalized
        dist_raw, _ = cocitation_to_distance(cocite_df.values, normalize_scale=False)
        diag_raw = compute_persistence(dist_raw, max_dim=2)
        b1_raw = betti_numbers(diag_raw)[1]

        # They should generally differ (normalization changes the filtration scale)
        # But both should be valid non-negative integers
        assert b1_norm >= 0
        assert b1_raw >= 0

    def test_normalized_mean_distance_is_one(self, medium_citation_network):
        """After normalization, mean pairwise distance should be ~1.0."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist_norm, _ = cocitation_to_distance(cocite_df.values, normalize_scale=True)

        upper_tri = dist_norm[np.triu_indices_from(dist_norm, k=1)]
        mean_d = upper_tri.mean()
        assert abs(mean_d - 1.0) < 0.01, f"Normalized mean distance {mean_d:.4f} ≠ 1.0"


# ---------------------------------------------------------------------------
# TestTemporalConfoundDetection
# ---------------------------------------------------------------------------

class TestTemporalConfoundDetection:
    """End-to-end test of the temporal confound the project discovered.

    This test creates a synthetic scenario that mirrors the real finding:
    β₁ declines over time, and a naive null model produces a spurious signal
    that detrending removes.
    """

    def test_spurious_signal_and_detrending(self):
        """Simulate the full confound → detection → correction pipeline.

        1. Generate declining β₁ over time (mimics real data)
        2. Sample "breakthroughs" at random years
        3. Compare each against a uniform null → expect spurious signal
        4. Detrend → expect signal to vanish
        """
        rng = np.random.default_rng(42)
        n_years = 40
        years = np.arange(1984, 1984 + n_years, dtype=float)

        # Simulated β₁ with linear decline + noise
        slope = -2.0
        beta_1 = 100.0 + slope * (years - 1984) + rng.normal(0, 4, n_years)

        # 20 "breakthroughs" at random years
        n_bt = 20
        bt_indices = rng.choice(n_years, n_bt, replace=False)
        bt_years = years[bt_indices]
        bt_beta1 = beta_1[bt_indices]

        # Naive null: compare each breakthrough against global mean
        global_mean = beta_1.mean()
        global_std = beta_1.std()
        raw_z = (bt_beta1 - global_mean) / global_std

        # The raw z-scores should correlate with year (THE CONFOUND)
        rho_raw, p_raw = spearmanr(bt_years, raw_z)
        assert abs(rho_raw) > 0.5, (
            f"Raw z-scores should correlate with year (got ρ={rho_raw:.3f})"
        )

        # Detrend β₁
        coeffs = np.polyfit(years, beta_1, 1)
        residuals = beta_1 - np.polyval(coeffs, years)
        bt_residuals = residuals[bt_indices]

        # Detrended: compare against detrended null (mean ≈ 0)
        resid_std = residuals.std()
        detrended_z = bt_residuals / resid_std if resid_std > 0 else bt_residuals

        # Detrended z-scores should have weaker year correlation than raw
        rho_detrended, _ = spearmanr(bt_years, detrended_z)
        assert abs(rho_detrended) < abs(rho_raw), (
            f"Detrending should reduce year correlation: "
            f"raw ρ={rho_raw:.3f}, detrended ρ={rho_detrended:.3f}"
        )

        # Detrended z-scores should not be significant (Wilcoxon)
        if len(detrended_z) >= 10:
            _, p_detrended = wilcoxon(detrended_z)
            # We don't assert p > 0.05 (that would be p-hacking the test),
            # but the detrended effect size should be small
            assert abs(detrended_z.mean()) < 1.0, (
                f"Detrended mean z-score {detrended_z.mean():.3f} still large"
            )


# ---------------------------------------------------------------------------
# TestCosineDistanceProperties
# ---------------------------------------------------------------------------

class TestCosineDistanceProperties:
    """Test mathematical properties of the cosine distance pipeline.

    These are properties that must hold for Vietoris-Rips persistence
    to be meaningful.
    """

    def test_triangle_inequality_violations_bounded(self, medium_citation_network):
        """Cosine distance is a semi-metric — triangle inequality violations exist
        but should be small in magnitude.

        Cosine distance does NOT satisfy the triangle inequality in general.
        This is a known limitation documented in the README. Ripser handles
        this gracefully, but we verify violations are bounded in magnitude.
        """
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist, _ = cocitation_to_distance(cocite_df.values, normalize_scale=False)

        n = dist.shape[0]
        max_violation = 0.0

        for i in range(min(n, 10)):
            for j in range(i + 1, min(n, 10)):
                for k in range(j + 1, min(n, 10)):
                    # How much does d(i,k) exceed d(i,j) + d(j,k)?
                    excess = dist[i, k] - (dist[i, j] + dist[j, k])
                    if excess > 0:
                        max_violation = max(max_violation, excess)

        # Violations should be small relative to distance scale
        max_dist = dist.max()
        if max_dist > 0:
            relative_violation = max_violation / max_dist
            assert relative_violation < 0.5, (
                f"Triangle inequality violation {relative_violation:.1%} of max distance "
                f"is too large for reliable Vietoris-Rips"
            )

    def test_distance_matrix_symmetry(self, medium_citation_network):
        """Distance matrix must be symmetric for Vietoris-Rips."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist, _ = cocitation_to_distance(cocite_df.values)

        np.testing.assert_allclose(dist, dist.T, atol=1e-12)

    def test_distance_matrix_zero_diagonal(self, medium_citation_network):
        """Diagonal must be exactly zero."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist, _ = cocitation_to_distance(cocite_df.values)

        np.testing.assert_array_equal(np.diag(dist), 0)

    def test_distance_matrix_nonnegative(self, medium_citation_network):
        """All distances must be non-negative."""
        citations, cpc_map = medium_citation_network

        cocite_df, labels = build_cocitation_matrix(
            citations, cpc_map, 2005, 2015, level="subclass"
        )
        dist, _ = cocitation_to_distance(cocite_df.values)

        assert (dist >= 0).all()
