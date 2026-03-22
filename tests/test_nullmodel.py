"""Unit tests for null model generation and superposed epoch analysis.

Tests the statistical machinery that underlies the precursor hypothesis test:
    - Matched null exclusion windows
    - Temporal uniformity of null sampling
    - Superposed epoch alignment and aggregation
    - Detrending validity
"""

import numpy as np
import pandas as pd
import pytest

from src.breakthroughs import Breakthrough
from src.nullmodel import (
    _ensure_citing_year,
    superposed_epoch,
)
from src.topology import ALL_PAIRS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_breakthrough():
    """A breakthrough with known properties for null model testing."""
    return Breakthrough(
        name="Test Breakthrough",
        breakthrough_patents=["US1234567"],
        filing_year=2005,
        recognition_year=2008,
        cpc_primary=["C07K"],
        cpc_sections=["A", "C"],
        category="biotech",
        description="A test breakthrough.",
    )


@pytest.fixture
def single_section_breakthrough():
    """A breakthrough with only one CPC section."""
    return Breakthrough(
        name="Single Section",
        breakthrough_patents=["US9999999"],
        filing_year=2010,
        recognition_year=2012,
        cpc_primary=["G06F"],
        cpc_sections=["G"],
        category="computing",
        description="Single section test.",
    )


@pytest.fixture
def synthetic_topology_results():
    """Topology DataFrames keyed by CPC pair, with known β₁ values.

    Creates a controlled scenario where β₁ declines linearly over time
    to test whether detrending and epoch analysis work correctly.
    """
    results = {}
    rng = np.random.default_rng(42)

    for a, b in ALL_PAIRS:
        key = f"{a}x{b}"
        years = list(range(1984, 2024))
        # β₁ declines at ~2 per year with noise
        beta_1 = [100 - 2 * (y - 1984) + rng.normal(0, 3) for y in years]
        results[key] = pd.DataFrame({
            "window_end": years,
            "beta_1": beta_1,
            "persistence_entropy": [5.0 + rng.normal(0, 0.1) for _ in years],
        })

    return results


# ---------------------------------------------------------------------------
# TestEnsureCitingYear
# ---------------------------------------------------------------------------

class TestEnsureCitingYear:
    """Test citing_year column derivation."""

    def test_adds_year_from_date(self):
        """When citing_year is absent, derive it from citing_date."""
        df = pd.DataFrame({
            "citing_id": ["P1", "P2"],
            "cited_id": ["P3", "P4"],
            "citing_date": ["2010-05-15", "2012-11-30"],
        })
        result = _ensure_citing_year(df)
        assert "citing_year" in result.columns
        assert list(result["citing_year"]) == [2010, 2012]

    def test_preserves_existing_year(self):
        """When citing_year already exists, don't modify it."""
        df = pd.DataFrame({
            "citing_id": ["P1"],
            "cited_id": ["P2"],
            "citing_year": [2015],
        })
        result = _ensure_citing_year(df)
        assert result["citing_year"].iloc[0] == 2015

    def test_does_not_mutate_input(self):
        """Original DataFrame should not be modified."""
        df = pd.DataFrame({
            "citing_id": ["P1"],
            "cited_id": ["P2"],
            "citing_date": ["2010-05-15"],
        })
        _ = _ensure_citing_year(df)
        assert "citing_year" not in df.columns


# ---------------------------------------------------------------------------
# TestMatchedNullExclusion
# ---------------------------------------------------------------------------

class TestMatchedNullExclusion:
    """Test that the matched null model correctly excludes breakthrough years."""

    def test_exclusion_window(self, sample_breakthrough):
        """Verify the exclusion zone around the breakthrough."""
        bt = sample_breakthrough
        exclusion_buffer = 3

        exclude_start = bt.filing_year - exclusion_buffer  # 2002
        exclude_end = bt.recognition_year + exclusion_buffer  # 2011

        all_years = list(range(1984, 2019))
        null_years = [y for y in all_years if y < exclude_start or y > exclude_end]

        # Years 2002-2011 should be excluded
        for y in range(2002, 2012):
            assert y not in null_years, f"Year {y} should be excluded"

        # Years outside should be included
        assert 2001 in null_years
        assert 2012 in null_years
        assert 1984 in null_years
        assert 2018 in null_years

    def test_exclusion_leaves_sufficient_years(self, sample_breakthrough):
        """Null model should have enough years to sample from."""
        bt = sample_breakthrough
        exclusion_buffer = 3

        exclude_start = bt.filing_year - exclusion_buffer
        exclude_end = bt.recognition_year + exclusion_buffer

        all_years = list(range(1984, 2019))
        null_years = [y for y in all_years if y < exclude_start or y > exclude_end]

        # Should have at least 20 years remaining (35 total - ~10 excluded)
        assert len(null_years) >= 20

    def test_matching_pairs_multi_section(self, sample_breakthrough):
        """Multi-section breakthrough should match all pairs containing its sections."""
        bt = sample_breakthrough  # sections A, C
        matching_pairs = [
            (a, b) for a, b in ALL_PAIRS
            if any(s in [a, b] for s in bt.cpc_sections)
        ]

        # A pairs with B,C,D,E,F,G,H = 7 pairs
        # C pairs with A,B,D,E,F,G,H = 7 pairs
        # But (A,C) is counted once, not twice
        # Total = 7 + 7 - 1 = 13
        assert len(matching_pairs) == 13

    def test_matching_pairs_single_section(self, single_section_breakthrough):
        """Single-section breakthrough matches exactly 7 pairs."""
        bt = single_section_breakthrough  # section G
        matching_pairs = [
            (a, b) for a, b in ALL_PAIRS
            if any(s in [a, b] for s in bt.cpc_sections)
        ]

        # G pairs with A,B,C,D,E,F,H = 7 pairs
        assert len(matching_pairs) == 7


# ---------------------------------------------------------------------------
# TestSuperposedEpoch
# ---------------------------------------------------------------------------

class TestSuperposedEpoch:
    """Test superposed epoch alignment and aggregation."""

    def test_epoch_alignment(self, sample_breakthrough, synthetic_topology_results):
        """Epoch years should be relative to filing year."""
        result = superposed_epoch(
            [sample_breakthrough],
            synthetic_topology_results,
            metric="beta_1",
            years_before=5,
            years_after=3,
        )

        assert not result.empty
        assert "epoch_year" in result.columns
        # Epoch years should range from -5 to +3
        assert result["epoch_year"].min() >= -5
        assert result["epoch_year"].max() <= 3

    def test_epoch_zero_is_filing_year(self, sample_breakthrough, synthetic_topology_results):
        """Epoch year 0 corresponds to the filing year."""
        bt = sample_breakthrough  # filing_year=2005
        result = superposed_epoch(
            [bt],
            synthetic_topology_results,
            metric="beta_1",
            years_before=10,
            years_after=5,
        )

        assert 0 in result["epoch_year"].values

    def test_multiple_breakthroughs_aggregate(self, synthetic_topology_results):
        """Multiple breakthroughs should increase sample count."""
        bt1 = Breakthrough(
            name="BT1", breakthrough_patents=["P1"], filing_year=2000,
            recognition_year=2003, cpc_primary=["G06F"],
            cpc_sections=["G", "H"], category="computing", description="Test 1",
        )
        bt2 = Breakthrough(
            name="BT2", breakthrough_patents=["P2"], filing_year=2010,
            recognition_year=2013, cpc_primary=["C07K"],
            cpc_sections=["A", "C"], category="biotech", description="Test 2",
        )

        result_one = superposed_epoch(
            [bt1], synthetic_topology_results,
            metric="beta_1", years_before=5, years_after=3,
        )
        result_two = superposed_epoch(
            [bt1, bt2], synthetic_topology_results,
            metric="beta_1", years_before=5, years_after=3,
        )

        # More breakthroughs → more samples per epoch year
        n_one = result_one["n"].sum()
        n_two = result_two["n"].sum()
        assert n_two > n_one

    def test_empty_breakthroughs(self, synthetic_topology_results):
        """Empty breakthrough list should return empty DataFrame."""
        result = superposed_epoch(
            [],
            synthetic_topology_results,
            metric="beta_1",
        )
        assert result.empty

    def test_no_matching_topology(self):
        """Breakthrough with no matching topology data returns empty."""
        bt = Breakthrough(
            name="Orphan", breakthrough_patents=["P1"], filing_year=2000,
            recognition_year=2003, cpc_primary=["Z99"],
            cpc_sections=["A"], category="test", description="No data",
        )
        result = superposed_epoch([bt], {}, metric="beta_1")
        assert result.empty

    def test_epoch_std_nonnegative(self, sample_breakthrough, synthetic_topology_results):
        """Standard deviation in epoch results should be non-negative."""
        result = superposed_epoch(
            [sample_breakthrough],
            synthetic_topology_results,
            metric="beta_1",
            years_before=10,
            years_after=5,
        )
        assert (result["std"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# TestTemporalDetrending
# ---------------------------------------------------------------------------

class TestTemporalDetrending:
    """Test that per-pair linear detrending removes temporal confounds.

    This is the critical statistical control. If β₁ declines linearly
    over time, detrending should remove that trend and leave residuals
    centered at zero.
    """

    def test_linear_trend_removed(self):
        """Detrending a perfect linear signal should leave zero residuals."""
        years = np.arange(1984, 2024, dtype=float)
        beta_1 = 100.0 - 2.0 * (years - 1984)  # Perfect linear decline

        # Detrend: fit and subtract
        coeffs = np.polyfit(years, beta_1, 1)
        predicted = np.polyval(coeffs, years)
        residuals = beta_1 - predicted

        np.testing.assert_allclose(residuals, 0, atol=1e-10)

    def test_detrended_residuals_centered_at_zero(self):
        """Detrended residuals from noisy linear data should be ~zero mean."""
        rng = np.random.default_rng(42)
        years = np.arange(1984, 2024, dtype=float)
        beta_1 = 100.0 - 2.0 * (years - 1984) + rng.normal(0, 3, len(years))

        coeffs = np.polyfit(years, beta_1, 1)
        predicted = np.polyval(coeffs, years)
        residuals = beta_1 - predicted

        # Mean residual should be close to zero
        assert abs(residuals.mean()) < 1.0, f"Mean residual {residuals.mean():.3f} too far from zero"

    def test_detrended_no_year_correlation(self):
        """After detrending, residuals should not correlate with year."""
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        years = np.arange(1984, 2024, dtype=float)
        # Strong linear decline plus noise
        beta_1 = 100.0 - 2.0 * (years - 1984) + rng.normal(0, 5, len(years))

        # Before detrending: strong correlation
        rho_before, _ = spearmanr(years, beta_1)
        assert abs(rho_before) > 0.8, "Raw data should show strong temporal correlation"

        # After detrending: no correlation
        coeffs = np.polyfit(years, beta_1, 1)
        residuals = beta_1 - np.polyval(coeffs, years)
        rho_after, p_after = spearmanr(years, residuals)

        assert abs(rho_after) < 0.3, f"Detrended residuals still correlated: ρ={rho_after:.3f}"

    def test_detrending_preserves_anomalies(self):
        """A genuine anomaly (spike) should survive detrending."""
        rng = np.random.default_rng(42)
        years = np.arange(1984, 2024, dtype=float)
        beta_1 = 100.0 - 2.0 * (years - 1984) + rng.normal(0, 2, len(years))

        # Inject a large spike at year 2005 (index 21)
        spike_idx = 21
        beta_1[spike_idx] += 30.0  # 15σ anomaly

        coeffs = np.polyfit(years, beta_1, 1)
        residuals = beta_1 - np.polyval(coeffs, years)

        # The spike should be the largest residual
        assert np.argmax(np.abs(residuals)) == spike_idx
        assert residuals[spike_idx] > 20.0  # Spike survives detrending

    def test_zscore_year_correlation_is_the_confound(self):
        """Demonstrate the temporal confound: raw z-scores correlate with year.

        This test encodes the project's central methodological finding:
        if β₁ declines over time, breakthroughs filed early will have
        higher z-scores than those filed late, purely as a temporal artifact.
        """
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        years = np.arange(1984, 2024, dtype=float)
        # Simulate declining β₁
        beta_1 = 100.0 - 2.0 * (years - 1984) + rng.normal(0, 3, len(years))

        # Simulate "breakthrough" z-scores: compare each year's β₁ against
        # the global mean (naive null model)
        global_mean = beta_1.mean()
        global_std = beta_1.std()
        z_scores = (beta_1 - global_mean) / global_std

        # z-scores should correlate strongly with year (the confound)
        rho, p = spearmanr(years, z_scores)
        assert abs(rho) > 0.8, f"Expected strong year-zscore correlation, got ρ={rho:.3f}"
        assert rho < 0, "Earlier years should have higher z-scores"


# ---------------------------------------------------------------------------
# TestNullModelStatisticalProperties
# ---------------------------------------------------------------------------

class TestNullModelStatisticalProperties:
    """Test statistical properties that a valid null model must satisfy."""

    def test_null_samples_span_time_range(self):
        """Null years should cover the full available range minus exclusion."""
        all_years = list(range(1984, 2019))
        exclude_start, exclude_end = 2002, 2011
        null_years = [y for y in all_years if y < exclude_start or y > exclude_end]

        # Should have years from both before and after the exclusion
        assert min(null_years) == 1984
        assert max(null_years) == 2018
        assert any(y < 2000 for y in null_years)
        assert any(y > 2012 for y in null_years)

    def test_uniform_null_creates_temporal_bias(self):
        """Document that uniform null sampling is biased when β₁ has a trend.

        This is the known limitation: if β₁ declines linearly and the null
        samples uniformly across years, the null mean will be biased relative
        to any specific year. This test encodes that understanding.
        """
        rng = np.random.default_rng(42)
        years = np.arange(1984, 2019, dtype=float)
        beta_1 = 100.0 - 2.0 * (years - 1984)

        null_mean = beta_1.mean()  # Average across all years

        # An early breakthrough (1990) will appear above the null mean
        early_val = beta_1[years == 1990][0]
        assert early_val > null_mean, "Early year should be above uniform null mean"

        # A late breakthrough (2015) will appear below the null mean
        late_val = beta_1[years == 2015][0]
        assert late_val < null_mean, "Late year should be below uniform null mean"

        # This asymmetry IS the temporal confound
        early_z = (early_val - null_mean) / beta_1.std()
        late_z = (late_val - null_mean) / beta_1.std()
        assert early_z > 0
        assert late_z < 0
