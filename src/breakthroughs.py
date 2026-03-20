"""Breakthrough catalog: definitions, loading, and CPC context mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import DATA_DIR, get_logger

logger = get_logger(__name__)

BREAKTHROUGHS_DIR = DATA_DIR / "breakthroughs"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Breakthrough:
    """A cataloged technological breakthrough.

    Attributes:
        name: Human-readable name of the breakthrough.
        breakthrough_patents: List of key US patent IDs.
        filing_year: Year the foundational patent was filed.
        recognition_year: Year the breakthrough was widely recognized.
        cpc_primary: Primary CPC subclass codes.
        cpc_sections: CPC section letters involved.
        category: Domain category (biotech, computing, etc.).
        description: Brief description of the breakthrough.
    """
    name: str
    breakthrough_patents: list[str]
    filing_year: int
    recognition_year: int
    cpc_primary: list[str]
    cpc_sections: list[str]
    category: str
    description: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_breakthroughs(catalog_dir: Optional[Path] = None) -> list[Breakthrough]:
    """Load all breakthrough definitions from JSON files.

    Args:
        catalog_dir: Directory containing JSON catalog files.
            Defaults to ``data/breakthroughs/``.

    Returns:
        List of Breakthrough objects sorted by filing_year.
    """
    catalog_dir = catalog_dir or BREAKTHROUGHS_DIR
    breakthroughs = []

    json_files = sorted(catalog_dir.glob("*.json"))
    if not json_files:
        logger.warning("No breakthrough catalog files found in %s", catalog_dir)
        return []

    for fp in json_files:
        with open(fp) as f:
            entries = json.load(f)

        if isinstance(entries, dict):
            entries = [entries]

        for entry in entries:
            breakthroughs.append(Breakthrough(**entry))

    breakthroughs.sort(key=lambda b: b.filing_year)
    logger.info("Loaded %d breakthroughs from %d files", len(breakthroughs), len(json_files))
    return breakthroughs


# ---------------------------------------------------------------------------
# CPC context
# ---------------------------------------------------------------------------

def get_cpc_context(
    breakthrough: Breakthrough,
    cpc_map: pd.DataFrame,
) -> dict:
    """Map a breakthrough's patents to their CPC classifications.

    Args:
        breakthrough: Breakthrough object.
        cpc_map: CPC mapping DataFrame with patent_id, cpc_section, cpc_class, cpc_subclass.

    Returns:
        Dict with:
            - found_patents: patents from the catalog that exist in the dataset
            - missing_patents: patents not found
            - cpc_sections: set of CPC sections spanned
            - cpc_classes: set of CPC classes
            - cpc_detail: DataFrame of CPC entries for found patents
    """
    patent_ids = set(breakthrough.breakthrough_patents)
    available = set(cpc_map["patent_id"])

    found = patent_ids & available
    missing = patent_ids - available

    if missing:
        logger.warning(
            "%s: %d/%d patents not found in dataset: %s",
            breakthrough.name, len(missing), len(patent_ids), missing,
        )

    cpc_detail = cpc_map[cpc_map["patent_id"].isin(found)]

    return {
        "found_patents": found,
        "missing_patents": missing,
        "cpc_sections": set(cpc_detail["cpc_section"]) if len(cpc_detail) > 0 else set(),
        "cpc_classes": set(cpc_detail["cpc_class"]) if len(cpc_detail) > 0 else set(),
        "cpc_detail": cpc_detail,
    }


# ---------------------------------------------------------------------------
# Precursor windows
# ---------------------------------------------------------------------------

def get_precursor_window(
    breakthrough: Breakthrough,
    years_before: int = 10,
) -> tuple[int, int]:
    """Compute the time window for pre-breakthrough analysis.

    Args:
        breakthrough: Breakthrough object.
        years_before: How many years before filing to include.

    Returns:
        Tuple of (start_year, end_year) for the precursor window.
    """
    end_year = breakthrough.filing_year - 1  # Exclude filing year itself
    start_year = end_year - years_before + 1
    return (start_year, end_year)


def get_citation_context(
    breakthrough: Breakthrough,
    citations: pd.DataFrame,
    patents: pd.DataFrame,
    forward_years: int = 10,
) -> dict:
    """Compute citation statistics for a breakthrough's patents.

    Args:
        breakthrough: Breakthrough object.
        citations: Citation DataFrame with citing_id, cited_id, citing_date.
        patents: Patent DataFrame with patent_id, date.
        forward_years: Years after filing to count forward citations.

    Returns:
        Dict with forward_citations, backward_citations, and precursor_patents.
    """
    bt_patents = set(breakthrough.breakthrough_patents)
    cutoff = pd.Timestamp(f"{breakthrough.filing_year + forward_years}-12-31")

    # Forward citations: patents that cite the breakthrough
    fwd = citations[citations["cited_id"].isin(bt_patents)]
    fwd_in_window = fwd[pd.to_datetime(fwd["citing_date"]) <= cutoff]

    # Backward citations: patents cited BY the breakthrough (its references)
    bwd = citations[citations["citing_id"].isin(bt_patents)]

    return {
        "forward_citations_total": len(fwd),
        "forward_citations_in_window": len(fwd_in_window),
        "backward_citations": len(bwd),
        "precursor_patents": set(bwd["cited_id"]),
    }
