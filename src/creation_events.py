"""CPC Subclass Creation Events — Objective Breakthrough Catalog (Strategy 3).

DATA LIMITATION (discovered 2026-03-21):
    The CPC system retroactively classifies historical patents. PatentsView's
    g_cpc_current table assigns CPC codes to patents going back to 1976, even
    for subclasses that conceptually represent recent technologies (e.g., H04W
    for wireless communications was created in 2006 but is applied to patents
    from the 1980s). As a result, the 4-character CPC subclass level yields only
    ~1 genuine post-1990 creation event (G16Y, 2009) out of 677 total subclasses.

    This approach requires CPC subgroup-level data (~200K unique codes, like
    "A61K31/704") where creation events are more granular and less affected by
    retroactive classification. That data is available in the raw g_cpc_current.tsv
    but was not preserved in our cpc_map.parquet (which stores only section, class,
    subclass). See 00_data_acquisition.py step_clean_cpc() for where the truncation
    occurs.

    The functions here are preserved for future use with subgroup-level data.
    For the current analysis, §6 of NB04 uses a jackknife leave-one-out approach
    instead (see 04_precursor_test.ipynb §6).



Identifies moments when the USPTO recognized a new technology area by assigning
the first patents to a previously unused CPC subclass. These administrative
creation events serve as an objective, institution-validated signal that a
new technological domain had emerged.

The logic:
    1. Find the earliest patent in each CPC subclass (grant date).
    2. Filter to subclasses that:
       - Were first granted in 1990-2018 (precursor topology available)
       - Eventually accumulate >= MIN_PATENTS patents (non-trivial area)
       - Belong to sections with good topology coverage (A, B, C, G, H)
       - Are plausibly new rather than reclassifications (see heuristics below)
    3. Map each creation event to topology section pairs.
    4. Extract pre-creation topology in the window [creation_year - 10, creation_year - 1].

This yields ~150-400 events — a large, objective test set that replaces
subjective curation and removes selection bias from the main result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, get_logger, timer

logger = get_logger(__name__)

# Minimum total patents a subclass must accumulate to be considered meaningful
MIN_PATENTS = 500

# Sections with sufficient topology coverage in the cache
COVERED_SECTIONS = set("ABCGH")

# Earliest/latest creation year for which we have precursor topology data
# (topology cache starts at window_end=1984, so for a 10-year precursor window
#  we need creation_year >= 1994 to get 10 full years of precursor data, or
#  >= 1990 for at least 6 years of precursor data)
MIN_CREATION_YEAR = 1990
MAX_CREATION_YEAR = 2018  # Leave 5+ years before topology cache ends for null sampling


@timer
def find_subclass_creation_events(
    cpc_map: pd.DataFrame,
    patents: pd.DataFrame,
    min_patents: int = MIN_PATENTS,
    covered_sections: set[str] | None = None,
    min_year: int = MIN_CREATION_YEAR,
    max_year: int = MAX_CREATION_YEAR,
) -> pd.DataFrame:
    """Find CPC subclass creation events from patent data.

    For each CPC subclass, identifies the earliest granted patent and uses
    that date as the "creation" date. Applies quality filters to exclude
    trivial or unmappable events.

    Args:
        cpc_map: DataFrame with patent_id, cpc_section, cpc_class, cpc_subclass.
        patents: DataFrame with patent_id and a date column ('date' or 'patent_date').
        min_patents: Minimum patents to qualify as a meaningful subclass.
        covered_sections: CPC sections with topology coverage. Defaults to ABCGH.
        min_year: Minimum creation year (precursor data available from this year).
        max_year: Maximum creation year (leave room for null sampling).

    Returns:
        DataFrame with columns: cpc_subclass, cpc_section, cpc_class,
        creation_year, creation_date, n_patents, first_patent_id.
    """
    if covered_sections is None:
        covered_sections = COVERED_SECTIONS

    # Normalize date column name
    date_col = "date" if "date" in patents.columns else "patent_date"

    # Filter to covered sections upfront
    cpc_filtered = cpc_map[cpc_map["cpc_section"].isin(covered_sections)].copy()
    logger.info(
        "CPC entries in covered sections: %d / %d",
        len(cpc_filtered), len(cpc_map),
    )

    # Drop Y (cross-sectional index) — not a real technology section
    cpc_filtered = cpc_filtered[cpc_filtered["cpc_section"] != "Y"]

    # Patents: keep id and date
    pat_dates = patents[["patent_id", date_col]].rename(columns={date_col: "grant_date"})
    pat_dates = pat_dates.dropna(subset=["grant_date"])
    pat_dates["grant_date"] = pd.to_datetime(pat_dates["grant_date"])
    pat_dates["grant_year"] = pat_dates["grant_date"].dt.year

    # Merge CPC with patent dates
    merged = cpc_filtered.merge(pat_dates, on="patent_id", how="inner")

    # Count total patents per subclass
    subclass_counts = (
        merged.groupby("cpc_subclass")["patent_id"]
        .nunique()
        .rename("n_patents")
        .reset_index()
    )
    logger.info(
        "Subclasses before size filter: %d", len(subclass_counts)
    )

    # Keep only subclasses with enough patents
    large_subclasses = subclass_counts[subclass_counts["n_patents"] >= min_patents]["cpc_subclass"]
    merged = merged[merged["cpc_subclass"].isin(large_subclasses)]
    logger.info(
        "Subclasses with >= %d patents: %d", min_patents, merged["cpc_subclass"].nunique()
    )

    # Earliest patent per subclass
    creation = (
        merged.sort_values("grant_date")
        .drop_duplicates(subset=["cpc_subclass"], keep="first")
        [["cpc_subclass", "cpc_section", "cpc_class", "grant_date", "grant_year", "patent_id"]]
        .rename(columns={
            "grant_date": "creation_date",
            "grant_year": "creation_year",
            "patent_id": "first_patent_id",
        })
    )

    # Merge in total patent counts
    creation = creation.merge(subclass_counts, on="cpc_subclass", how="left")

    # Apply year filter
    creation = creation[
        (creation["creation_year"] >= min_year)
        & (creation["creation_year"] <= max_year)
    ]
    logger.info(
        "Creation events in %d-%d window: %d",
        min_year, max_year, len(creation),
    )

    # Reclassification heuristic: exclude subclasses where >80% of patents in the
    # first 100 share a single *other* subclass classification on the same patent.
    # If a new subclass X was a reclassification of subclass Y, most patents in X
    # will also be classified in Y on the same patent record.
    creation = _filter_reclassifications(creation, cpc_filtered, threshold=0.80, sample_n=100)

    creation = creation.sort_values("creation_year").reset_index(drop=True)
    logger.info("Final creation events after reclassification filter: %d", len(creation))

    return creation


def _filter_reclassifications(
    creation: pd.DataFrame,
    cpc_filtered: pd.DataFrame,
    threshold: float = 0.80,
    sample_n: int = 100,
) -> pd.DataFrame:
    """Remove likely CPC reclassification events.

    A subclass is flagged as a reclassification if more than `threshold`
    fraction of the first `sample_n` patents also appear in a single other
    subclass. This heuristic catches systematic USPTO reclassification
    campaigns (e.g., when a subclass is split from another).

    Args:
        creation: Creation events DataFrame.
        cpc_filtered: CPC map filtered to covered sections.
        threshold: Fraction threshold for reclassification flag.
        sample_n: Number of early patents to inspect per subclass.

    Returns:
        Creation events with likely reclassifications removed.
    """
    # Build a patent → set_of_subclasses map for quick lookup
    patent_subclasses = (
        cpc_filtered.groupby("patent_id")["cpc_subclass"]
        .apply(set)
        .to_dict()
    )

    # For each creation event, find first sample_n patents in that subclass
    subclass_patents = (
        cpc_filtered.groupby("cpc_subclass")["patent_id"]
        .apply(list)
        .to_dict()
    )

    keep_mask = []
    n_removed = 0

    for _, row in creation.iterrows():
        sc = row["cpc_subclass"]
        patents_in_sc = subclass_patents.get(sc, [])[:sample_n]

        if len(patents_in_sc) < 10:
            keep_mask.append(True)
            continue

        # Count co-occurrence of each *other* subclass across these patents
        co_counts: dict[str, int] = {}
        for pid in patents_in_sc:
            for other_sc in patent_subclasses.get(pid, set()):
                if other_sc != sc:
                    co_counts[other_sc] = co_counts.get(other_sc, 0) + 1

        if not co_counts:
            keep_mask.append(True)
            continue

        max_co = max(co_counts.values())
        frac = max_co / len(patents_in_sc)

        if frac > threshold:
            n_removed += 1
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    logger.info("Reclassification filter removed %d events", n_removed)
    return creation[keep_mask].reset_index(drop=True)


def creation_event_precursor_windows(
    creation_events: pd.DataFrame,
    topology_results: dict[str, pd.DataFrame],
    years_before: int = 10,
    metric: str = "beta_1",
) -> pd.DataFrame:
    """Extract pre-creation topology metrics for each creation event.

    For each creation event, averages the topology metric across all cached
    cross-section pairs that include the event's CPC section.

    Args:
        creation_events: DataFrame from find_subclass_creation_events().
        topology_results: Dict mapping pair key (e.g., 'AxC') to topology DataFrame.
        years_before: Number of years before creation to average over.
        metric: Topology metric column name.

    Returns:
        DataFrame with columns: cpc_subclass, cpc_section, creation_year,
        n_patents, pre_metric, n_precursor_windows, n_matching_pairs.
    """
    rows = []

    for _, ev in creation_events.iterrows():
        section = ev["cpc_section"]
        creation_year = int(ev["creation_year"])
        precursor_start = creation_year - years_before
        precursor_end = creation_year - 1

        # Find topology pairs containing this section
        matching_topos = []
        for key, topo in topology_results.items():
            if "x" not in key:
                continue
            sa, sb = key.split("x")
            if section in (sa, sb):
                matching_topos.append(topo)

        if not matching_topos:
            continue

        # Extract pre-creation topology from each matching pair
        pre_vals = []
        n_precursor = 0
        year_col = "window_end"

        for topo in matching_topos:
            if year_col not in topo.columns or metric not in topo.columns:
                continue
            pre = topo[
                (topo[year_col] >= precursor_start)
                & (topo[year_col] <= precursor_end)
            ]
            if len(pre) > 0:
                pre_vals.append(pre[metric].mean())
                n_precursor = max(n_precursor, len(pre))

        if not pre_vals:
            continue

        rows.append({
            "cpc_subclass": ev["cpc_subclass"],
            "cpc_section": section,
            "creation_year": creation_year,
            "n_patents": ev["n_patents"],
            "pre_metric": float(np.mean(pre_vals)),
            "n_precursor_windows": n_precursor,
            "n_matching_pairs": len(matching_topos),
        })

    return pd.DataFrame(rows)


def creation_event_null_distribution(
    topology_results: dict[str, pd.DataFrame],
    section: str,
    creation_year: int,
    n_samples: int = 100,
    years_before: int = 10,
    exclusion_buffer: int = 3,
    metric: str = "beta_1",
    rng: np.random.Generator | None = None,
    null_year_range: tuple[int, int] = (1984, 2018),
) -> np.ndarray:
    """Generate matched null distribution for a creation event.

    Holds the CPC section fixed, samples random non-creation years,
    and computes the average pre-null topology. Mirrors the logic of
    matched_null() in nullmodel.py but operates directly on
    in-memory topology_results rather than recomputing from citations.

    Args:
        topology_results: Dict of pair key → topology DataFrame.
        section: CPC section of the creation event.
        creation_year: The actual creation year (excluded from null).
        n_samples: Number of null samples to draw.
        years_before: Window size for averaging.
        exclusion_buffer: Years around creation_year to exclude.
        metric: Topology metric to use.
        rng: Random number generator. Created if None.
        null_year_range: (min, max) range for null sampling.

    Returns:
        Array of null metric values (length <= n_samples).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Matching pairs
    matching_topos = [
        (key, topo)
        for key, topo in topology_results.items()
        if "x" in key and section in key.split("x")
        and "window_end" in topo.columns
        and metric in topo.columns
    ]

    if not matching_topos:
        return np.array([])

    exclude_start = creation_year - exclusion_buffer
    exclude_end = creation_year + exclusion_buffer
    all_years = list(range(null_year_range[0], null_year_range[1] + 1))
    null_years = [y for y in all_years if y < exclude_start or y > exclude_end]

    if len(null_years) == 0:
        return np.array([])

    null_vals = []
    for _ in range(n_samples):
        null_year = int(rng.choice(null_years))
        pre_start = null_year - years_before
        pre_end = null_year - 1

        pair_avgs = []
        for _, topo in matching_topos:
            pre = topo[
                (topo["window_end"] >= pre_start)
                & (topo["window_end"] <= pre_end)
            ]
            if len(pre) > 0:
                pair_avgs.append(pre[metric].mean())

        if pair_avgs:
            null_vals.append(float(np.mean(pair_avgs)))

    return np.array(null_vals)
