"""Unit tests for breakthrough catalog loading, validation, and context mapping."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.breakthroughs import (
    Breakthrough,
    load_breakthroughs,
    get_cpc_context,
    get_precursor_window,
    get_citation_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_catalog(tmp_path):
    """Write a small JSON catalog to a temp directory."""
    entries = [
        {
            "name": "CRISPR-Cas9",
            "breakthrough_patents": ["US8697359", "US8889356"],
            "filing_year": 2012,
            "recognition_year": 2015,
            "cpc_primary": ["C12N"],
            "cpc_sections": ["C", "A"],
            "category": "biotech",
            "description": "Programmable genome editing",
        },
        {
            "name": "PageRank",
            "breakthrough_patents": ["US6285999"],
            "filing_year": 1998,
            "recognition_year": 2000,
            "cpc_primary": ["G06F"],
            "cpc_sections": ["G"],
            "category": "computing",
            "description": "Web page ranking algorithm",
        },
    ]
    catalog_file = tmp_path / "test_catalog.json"
    with open(catalog_file, "w") as f:
        json.dump(entries, f)
    return tmp_path


@pytest.fixture
def crispr_breakthrough():
    return Breakthrough(
        name="CRISPR-Cas9",
        breakthrough_patents=["US8697359", "US8889356"],
        filing_year=2012,
        recognition_year=2015,
        cpc_primary=["C12N"],
        cpc_sections=["C", "A"],
        category="biotech",
        description="Programmable genome editing",
    )


@pytest.fixture
def sample_cpc_map():
    return pd.DataFrame({
        "patent_id": ["US8697359", "US8889356", "US9999999", "US1111111"],
        "cpc_section": ["C", "A", "G", "H"],
        "cpc_class": ["C12", "A61", "G06", "H04"],
        "cpc_subclass": ["C12N", "A61K", "G06F", "H04L"],
    })


@pytest.fixture
def sample_citations():
    return pd.DataFrame({
        "citing_id": ["US9000001", "US9000002", "US9000003", "US8697359"],
        "cited_id": ["US8697359", "US8697359", "US8889356", "US1111111"],
        "citing_date": ["2013-06-01", "2014-03-15", "2016-01-20", "2012-01-01"],
    })


@pytest.fixture
def sample_patents():
    return pd.DataFrame({
        "patent_id": ["US8697359", "US8889356", "US9000001", "US9000002",
                       "US9000003", "US9999999", "US1111111"],
        "date": pd.to_datetime([
            "2012-04-01", "2012-09-15", "2013-06-01", "2014-03-15",
            "2016-01-20", "2020-01-01", "2005-01-01",
        ]),
    })


# ---------------------------------------------------------------------------
# TestLoadBreakthroughs
# ---------------------------------------------------------------------------

class TestLoadBreakthroughs:
    """Test breakthrough catalog loading from JSON files."""

    def test_load_from_directory(self, sample_catalog):
        bts = load_breakthroughs(sample_catalog)
        assert len(bts) == 2

    def test_sorted_by_filing_year(self, sample_catalog):
        bts = load_breakthroughs(sample_catalog)
        assert bts[0].filing_year <= bts[1].filing_year

    def test_fields_populated(self, sample_catalog):
        bts = load_breakthroughs(sample_catalog)
        crispr = [b for b in bts if b.name == "CRISPR-Cas9"][0]
        assert crispr.filing_year == 2012
        assert crispr.recognition_year == 2015
        assert "C" in crispr.cpc_sections
        assert crispr.category == "biotech"

    def test_empty_directory(self, tmp_path):
        bts = load_breakthroughs(tmp_path)
        assert bts == []

    def test_single_entry_dict(self, tmp_path):
        """A JSON file with a single dict (not a list) should still work."""
        entry = {
            "name": "Solo",
            "breakthrough_patents": ["US0000001"],
            "filing_year": 2000,
            "recognition_year": 2002,
            "cpc_primary": ["H04L"],
            "cpc_sections": ["H"],
            "category": "telecom",
            "description": "Single entry test",
        }
        with open(tmp_path / "solo.json", "w") as f:
            json.dump(entry, f)

        bts = load_breakthroughs(tmp_path)
        assert len(bts) == 1
        assert bts[0].name == "Solo"


# ---------------------------------------------------------------------------
# TestBreakthroughDataclass
# ---------------------------------------------------------------------------

class TestBreakthroughDataclass:
    """Test Breakthrough dataclass properties."""

    def test_required_fields(self):
        """All fields are required — missing one should raise TypeError."""
        with pytest.raises(TypeError):
            Breakthrough(name="Incomplete", filing_year=2000)

    def test_cpc_sections_is_list(self, crispr_breakthrough):
        assert isinstance(crispr_breakthrough.cpc_sections, list)
        assert all(isinstance(s, str) for s in crispr_breakthrough.cpc_sections)

    def test_recognition_after_filing(self, crispr_breakthrough):
        assert crispr_breakthrough.recognition_year >= crispr_breakthrough.filing_year


# ---------------------------------------------------------------------------
# TestGetCPCContext
# ---------------------------------------------------------------------------

class TestGetCPCContext:
    """Test CPC context mapping for breakthroughs."""

    def test_found_patents(self, crispr_breakthrough, sample_cpc_map):
        ctx = get_cpc_context(crispr_breakthrough, sample_cpc_map)
        assert "US8697359" in ctx["found_patents"]
        assert "US8889356" in ctx["found_patents"]

    def test_missing_patents(self, sample_cpc_map):
        bt = Breakthrough(
            name="Ghost",
            breakthrough_patents=["US0000000"],  # Not in cpc_map
            filing_year=2000,
            recognition_year=2002,
            cpc_primary=["X99"],
            cpc_sections=["X"],
            category="test",
            description="Missing patent test",
        )
        ctx = get_cpc_context(bt, sample_cpc_map)
        assert "US0000000" in ctx["missing_patents"]
        assert len(ctx["found_patents"]) == 0

    def test_cpc_sections_extracted(self, crispr_breakthrough, sample_cpc_map):
        ctx = get_cpc_context(crispr_breakthrough, sample_cpc_map)
        # US8697359 → section C, US8889356 → section A
        assert "C" in ctx["cpc_sections"]
        assert "A" in ctx["cpc_sections"]

    def test_cpc_detail_is_dataframe(self, crispr_breakthrough, sample_cpc_map):
        ctx = get_cpc_context(crispr_breakthrough, sample_cpc_map)
        assert isinstance(ctx["cpc_detail"], pd.DataFrame)
        assert len(ctx["cpc_detail"]) == 2


# ---------------------------------------------------------------------------
# TestGetPrecursorWindow
# ---------------------------------------------------------------------------

class TestGetPrecursorWindow:
    """Test precursor window calculation."""

    def test_default_10_years(self, crispr_breakthrough):
        start, end = get_precursor_window(crispr_breakthrough)
        # filing_year = 2012, end = 2011, start = 2002
        assert end == 2011
        assert start == 2002

    def test_custom_years_before(self, crispr_breakthrough):
        start, end = get_precursor_window(crispr_breakthrough, years_before=5)
        assert end == 2011
        assert start == 2007

    def test_window_excludes_filing_year(self, crispr_breakthrough):
        _, end = get_precursor_window(crispr_breakthrough)
        assert end < crispr_breakthrough.filing_year

    def test_window_length(self, crispr_breakthrough):
        start, end = get_precursor_window(crispr_breakthrough, years_before=10)
        assert end - start + 1 == 10


# ---------------------------------------------------------------------------
# TestGetCitationContext
# ---------------------------------------------------------------------------

class TestGetCitationContext:
    """Test citation context statistics."""

    def test_forward_citations(self, crispr_breakthrough, sample_citations, sample_patents):
        ctx = get_citation_context(crispr_breakthrough, sample_citations, sample_patents)
        # 3 patents cite the breakthrough patents
        assert ctx["forward_citations_total"] == 3

    def test_backward_citations(self, crispr_breakthrough, sample_citations, sample_patents):
        ctx = get_citation_context(crispr_breakthrough, sample_citations, sample_patents)
        # US8697359 cites US1111111 (1 backward citation)
        assert ctx["backward_citations"] == 1

    def test_precursor_patents(self, crispr_breakthrough, sample_citations, sample_patents):
        ctx = get_citation_context(crispr_breakthrough, sample_citations, sample_patents)
        assert "US1111111" in ctx["precursor_patents"]

    def test_forward_window_cutoff(self, crispr_breakthrough, sample_citations, sample_patents):
        """Forward citations within window should respect the cutoff."""
        ctx = get_citation_context(
            crispr_breakthrough, sample_citations, sample_patents,
            forward_years=2,  # 2012 + 2 = 2014
        )
        # Only citations before 2014-12-31:
        # US9000001 citing 2013-06-01 ✓, US9000002 citing 2014-03-15 ✓
        # US9000003 citing 2016-01-20 ✗
        assert ctx["forward_citations_in_window"] == 2


# ---------------------------------------------------------------------------
# TestCatalogIntegrity
# ---------------------------------------------------------------------------

class TestCatalogIntegrity:
    """Validate the actual breakthrough catalog on disk.

    These tests run against the real catalog files and verify
    data quality constraints that a peer reviewer would check.
    """

    @pytest.fixture
    def real_catalog(self):
        """Load the actual catalog. Skip if not available."""
        catalog_dir = Path(__file__).parent.parent / "data" / "breakthroughs"
        if not catalog_dir.exists() or not list(catalog_dir.glob("*.json")):
            pytest.skip("Breakthrough catalog not available")
        return load_breakthroughs(catalog_dir)

    def test_minimum_catalog_size(self, real_catalog):
        """Catalog should have at least 50 entries for statistical power."""
        assert len(real_catalog) >= 50, (
            f"Catalog has {len(real_catalog)} entries; need ≥50 for meaningful statistics"
        )

    def test_filing_year_range(self, real_catalog):
        """All filing years should be within the patent dataset range."""
        for bt in real_catalog:
            assert 1976 <= bt.filing_year <= 2023, (
                f"{bt.name}: filing_year {bt.filing_year} outside dataset range"
            )

    def test_recognition_after_filing(self, real_catalog):
        """Recognition year must be ≥ filing year."""
        for bt in real_catalog:
            assert bt.recognition_year >= bt.filing_year, (
                f"{bt.name}: recognition_year {bt.recognition_year} < filing_year {bt.filing_year}"
            )

    def test_valid_cpc_sections(self, real_catalog):
        """All CPC sections must be valid single-letter codes A-H."""
        valid_sections = set("ABCDEFGH")
        for bt in real_catalog:
            for s in bt.cpc_sections:
                assert s in valid_sections, (
                    f"{bt.name}: invalid CPC section '{s}'"
                )

    def test_has_breakthrough_patents(self, real_catalog):
        """Every entry must list at least one patent."""
        for bt in real_catalog:
            assert len(bt.breakthrough_patents) >= 1, (
                f"{bt.name}: no breakthrough patents listed"
            )

    def test_category_coverage(self, real_catalog):
        """Catalog should span multiple technology domains."""
        categories = set(bt.category for bt in real_catalog)
        assert len(categories) >= 5, (
            f"Only {len(categories)} categories; need ≥5 for domain diversity"
        )

    def test_temporal_coverage(self, real_catalog):
        """Breakthroughs should span multiple decades."""
        decades = set(bt.filing_year // 10 for bt in real_catalog)
        assert len(decades) >= 3, (
            f"Only {len(decades)} decades represented; need ≥3 for temporal diversity"
        )

    def test_multi_section_breakthroughs_exist(self, real_catalog):
        """Some breakthroughs should span multiple CPC sections."""
        multi = [bt for bt in real_catalog if len(bt.cpc_sections) >= 2]
        assert len(multi) >= 10, (
            f"Only {len(multi)} multi-section breakthroughs; "
            "need ≥10 for cross-section topology analysis"
        )

    def test_no_duplicate_names(self, real_catalog):
        """No two breakthroughs should have the same name."""
        names = [bt.name for bt in real_catalog]
        assert len(names) == len(set(names)), "Duplicate breakthrough names found"
