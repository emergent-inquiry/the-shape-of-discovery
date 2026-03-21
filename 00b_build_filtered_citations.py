"""
00b — Build Filtered Citation Datasets
=======================================
Produces robustness-check citation datasets from the raw PatentsView TSVs:

    data/citations_with_category.parquet  — original pairs + citation_category
    data/citations_applicant_only.parquet — examiner citations removed
    data/patent_assignee.parquet          — patent_id → primary assignee_id
    data/citations_no_self_cite.parquet   — intra-assignee citations removed
    data/patent_filing_dates.parquet      — patent_id → filing_date
    data/citations_filing_date.parquet    — original pairs, citing_date = filing date

Run AFTER 00_data_acquisition.py (requires citations.parquet to exist).

    python 00b_build_filtered_citations.py

Robustness checks these datasets support (from CONFOUNDS.md):
    #1  Examiner vs. applicant citations → citations_applicant_only
    #2  Prosecution lag (filing date)    → citations_filing_date
    #8  Assignee self-citation bias      → citations_no_self_cite

citation_category values in raw data:
    "cited by examiner"   — examiner-added, may not reflect inventor knowledge
    "cited by applicant"  — applicant-cited, more likely genuine knowledge flow
"""

# %% Imports
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR, get_logger, log_memory, timer

logger = get_logger(__name__)

RAW_DIR = DATA_DIR / "raw"
CITATION_CATEGORY_APPLICANT = "cited by applicant"
CITATION_CATEGORY_EXAMINER = "cited by examiner"
CHUNK_SIZE = 5_000_000


# %% Step 1: citations_with_category.parquet
@timer
def step_add_citation_category() -> pd.DataFrame:
    """Join citation_category from raw TSV onto existing citations.parquet.

    Reads g_us_patent_citation.tsv in chunks; filters to valid pairs already
    present in citations.parquet; merges to attach citation_category and
    citing_date.

    Returns:
        DataFrame with columns: citing_id, cited_id, citing_date, citation_category.
    """
    citations_path = DATA_DIR / "citations.parquet"
    out_path = DATA_DIR / "citations_with_category.parquet"

    if out_path.exists():
        logger.info("citations_with_category.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    logger.info("Loading citations.parquet for valid pair lookup...")
    citations = pd.read_parquet(citations_path)
    valid_citing = set(citations["citing_id"].unique())
    valid_cited = set(citations["cited_id"].unique())
    logger.info(
        "Valid citing: %d unique IDs, valid cited: %d unique IDs",
        len(valid_citing), len(valid_cited),
    )
    log_memory("After loading citations.parquet")

    tsv_path = RAW_DIR / "g_us_patent_citation.tsv"
    logger.info("Reading %s in chunks...", tsv_path.name)

    chunks = []
    total_raw = 0

    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype={"patent_id": str, "citation_patent_id": str, "citation_category": str},
        usecols=["patent_id", "citation_patent_id", "citation_category"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Reading citation TSV", unit="chunk"):
        total_raw += len(chunk)
        # Filter to pairs that exist in our valid citation set
        mask = (
            chunk["patent_id"].isin(valid_citing)
            & chunk["citation_patent_id"].isin(valid_cited)
        )
        filtered = chunk[mask].copy()
        filtered = filtered.rename(columns={
            "patent_id": "citing_id",
            "citation_patent_id": "cited_id",
        })
        chunks.append(filtered)

    logger.info("Raw citation rows scanned: %d", total_raw)

    raw_cat = pd.concat(chunks, ignore_index=True)
    logger.info("Rows after valid-pair filter: %d", len(raw_cat))
    log_memory("After reading category TSV")

    # Drop duplicates — raw TSV can repeat (patent_id, citation_patent_id) pairs
    raw_cat = raw_cat.drop_duplicates(subset=["citing_id", "cited_id"])
    logger.info("After dedup: %d rows", len(raw_cat))

    # Inner join with citations.parquet to add citing_date and drop any
    # pairs that didn't make it through the original cleaning pipeline
    result = citations.merge(
        raw_cat[["citing_id", "cited_id", "citation_category"]],
        on=["citing_id", "cited_id"],
        how="inner",
    )
    logger.info(
        "citations_with_category: %d rows (%.1f%% of original %d citations)",
        len(result), 100 * len(result) / len(citations), len(citations),
    )

    # Report category distribution
    dist = result["citation_category"].value_counts(normalize=True)
    for cat, frac in dist.items():
        logger.info("  %s: %.1f%%", cat, 100 * frac)

    result.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path.name)
    log_memory("After saving citations_with_category")

    return result


# %% Step 2: citations_applicant_only.parquet
@timer
def step_applicant_only(citations_with_category: pd.DataFrame) -> pd.DataFrame:
    """Filter to applicant-cited citations only (remove examiner-added).

    Args:
        citations_with_category: DataFrame from step_add_citation_category().

    Returns:
        Filtered DataFrame (citing_id, cited_id, citing_date, citation_category).
    """
    out_path = DATA_DIR / "citations_applicant_only.parquet"

    if out_path.exists():
        logger.info("citations_applicant_only.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    original_n = len(citations_with_category)
    result = citations_with_category[
        citations_with_category["citation_category"] == CITATION_CATEGORY_APPLICANT
    ].copy()

    logger.info(
        "Applicant-only citations: %d / %d (%.1f%% retained)",
        len(result), original_n, 100 * len(result) / original_n,
    )

    # Report retention by decade
    result["decade"] = (result["citing_date"].dt.year // 10 * 10)
    full_decade = (
        citations_with_category["citing_date"].dt.year // 10 * 10
    ).value_counts().sort_index()
    appl_decade = result.groupby("decade").size()
    logger.info("Applicant citation fraction by decade:")
    for dec in sorted(full_decade.index):
        n_full = full_decade.get(dec, 0)
        n_appl = appl_decade.get(dec, 0)
        if n_full > 0:
            logger.info("  %ds: %.1f%% (%d / %d)", dec, 100 * n_appl / n_full, n_appl, n_full)

    result = result.drop(columns=["decade"])
    result.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path.name)

    return result


# %% Step 3: patent_assignee.parquet
@timer
def step_patent_assignee() -> pd.DataFrame:
    """Build patent → primary assignee mapping from g_assignee_disambiguated.tsv.

    Uses sequence=0 (primary assignee) where available; falls back to
    minimum sequence otherwise.

    Returns:
        DataFrame with columns: patent_id, assignee_id.
    """
    out_path = DATA_DIR / "patent_assignee.parquet"

    if out_path.exists():
        logger.info("patent_assignee.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    tsv_path = RAW_DIR / "g_assignee_disambiguated.tsv"
    logger.info("Reading %s...", tsv_path.name)

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype={"patent_id": str, "assignee_id": str, "assignee_sequence": int},
        usecols=["patent_id", "assignee_sequence", "assignee_id"],
        low_memory=False,
    )
    logger.info("Raw assignee rows: %d", len(df))

    # Drop rows with no assignee_id
    df = df.dropna(subset=["assignee_id"])

    # Keep only the primary assignee (minimum sequence per patent)
    df = (
        df.sort_values("assignee_sequence")
          .drop_duplicates(subset=["patent_id"], keep="first")
          [["patent_id", "assignee_id"]]
    )
    logger.info("Patents with assignee: %d", len(df))

    df.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path.name)
    log_memory("After saving patent_assignee")

    return df


# %% Step 4: citations_no_self_cite.parquet
@timer
def step_no_self_cite(
    citations: pd.DataFrame,
    patent_assignee: pd.DataFrame,
) -> pd.DataFrame:
    """Remove intra-assignee citations (same organization, different patents).

    Args:
        citations: Base citation DataFrame (citing_id, cited_id, citing_date).
        patent_assignee: Patent → assignee mapping.

    Returns:
        Filtered DataFrame with intra-assignee edges removed.
    """
    out_path = DATA_DIR / "citations_no_self_cite.parquet"

    if out_path.exists():
        logger.info("citations_no_self_cite.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    original_n = len(citations)

    # Map assignee onto citing patent
    assignee_map = patent_assignee.set_index("patent_id")["assignee_id"]
    citing_assignee = citations["citing_id"].map(assignee_map)
    cited_assignee = citations["cited_id"].map(assignee_map)

    # Keep edges where: assignees differ OR either party has no assignee (non-corporate)
    self_cite_mask = (
        citing_assignee.notna()
        & cited_assignee.notna()
        & (citing_assignee == cited_assignee)
    )

    result = citations[~self_cite_mask].copy()
    n_removed = self_cite_mask.sum()

    logger.info(
        "Intra-assignee citations removed: %d (%.1f%% of %d total)",
        n_removed, 100 * n_removed / original_n, original_n,
    )
    logger.info("Remaining citations: %d", len(result))

    # Report self-cite fraction by decade
    self_cite_df = citations[self_cite_mask].copy()
    self_cite_df["decade"] = self_cite_df["citing_date"].dt.year // 10 * 10
    all_decades = (citations["citing_date"].dt.year // 10 * 10).value_counts().sort_index()
    sc_decades = self_cite_df.groupby("decade").size()
    logger.info("Intra-assignee fraction by decade:")
    for dec in sorted(all_decades.index):
        n_all = all_decades.get(dec, 0)
        n_sc = sc_decades.get(dec, 0)
        if n_all > 0:
            logger.info("  %ds: %.1f%% (%d / %d)", dec, 100 * n_sc / n_all, n_sc, n_all)

    result.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path.name)

    return result


# %% Step 5: patent_filing_dates.parquet
@timer
def step_filing_dates() -> pd.DataFrame:
    """Extract patent_id → filing_date from g_application.tsv.

    Returns:
        DataFrame with columns: patent_id, filing_date.
    """
    out_path = DATA_DIR / "patent_filing_dates.parquet"

    if out_path.exists():
        logger.info("patent_filing_dates.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    tsv_path = RAW_DIR / "g_application.tsv"
    logger.info("Reading %s...", tsv_path.name)

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype={"patent_id": str},
        usecols=["patent_id", "filing_date"],
        parse_dates=["filing_date"],
        low_memory=False,
    )
    logger.info("Raw application rows: %d", len(df))

    # Deduplicate — keep one filing date per patent (take earliest)
    df = df.dropna(subset=["filing_date", "patent_id"])
    df = df.sort_values("filing_date").drop_duplicates(subset=["patent_id"], keep="first")
    logger.info("Patents with filing date: %d", len(df))

    df.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path.name)

    return df


# %% Step 6: citations_filing_date.parquet
@timer
def step_citations_filing_date(
    citations: pd.DataFrame,
    filing_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Replace citing_date with filing_date where available.

    Keeps grant date as fallback for patents without application data.

    Args:
        citations: Base citations DataFrame (citing_id, cited_id, citing_date).
        filing_dates: patent_id → filing_date mapping.

    Returns:
        Citations with citing_date replaced by filing_date where available.
    """
    out_path = DATA_DIR / "citations_filing_date.parquet"

    if out_path.exists():
        logger.info("citations_filing_date.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    filing_map = filing_dates.set_index("patent_id")["filing_date"]
    result = citations.copy()
    result["filing_date"] = result["citing_id"].map(filing_map)

    coverage = result["filing_date"].notna().mean()
    logger.info("Filing date coverage: %.1f%% of citations", 100 * coverage)

    # Compute prosecution lag stats where both dates are available
    both_dates = result[result["filing_date"].notna()].copy()
    both_dates["lag_days"] = (
        both_dates["citing_date"] - both_dates["filing_date"]
    ).dt.days
    # Filter to plausible range (0-15 years)
    valid_lag = both_dates[(both_dates["lag_days"] >= 0) & (both_dates["lag_days"] <= 365 * 15)]
    logger.info(
        "Prosecution lag (grant - filing): median=%.1f days (%.1f yr), "
        "mean=%.1f days (%.1f yr)",
        valid_lag["lag_days"].median(),
        valid_lag["lag_days"].median() / 365.25,
        valid_lag["lag_days"].mean(),
        valid_lag["lag_days"].mean() / 365.25,
    )

    # Replace citing_date with filing_date where available
    result["citing_date"] = result["filing_date"].fillna(result["citing_date"])
    result = result.drop(columns=["filing_date"])

    result.to_parquet(out_path, index=False)
    logger.info("Saved → %s", out_path.name)

    return result


# %% Main pipeline
@timer
def main() -> None:
    """Build all filtered citation datasets."""
    logger.info("=" * 60)
    logger.info("The Shape of Discovery — Filtered Citation Builder")
    logger.info("=" * 60)

    citations_path = DATA_DIR / "citations.parquet"
    if not citations_path.exists():
        logger.error(
            "citations.parquet not found at %s. "
            "Run 00_data_acquisition.py first.",
            citations_path,
        )
        sys.exit(1)

    citations = pd.read_parquet(citations_path)
    logger.info("Base citations: %d rows", len(citations))

    # 1. Add citation_category
    citations_with_cat = step_add_citation_category()

    # 2. Applicant-only filter (Confound #1)
    step_applicant_only(citations_with_cat)

    # 3. Assignee mapping (needed for Confound #8)
    patent_assignee = step_patent_assignee()

    # 4. No-self-cite filter (Confound #8)
    step_no_self_cite(citations, patent_assignee)

    # 5. Filing dates (Confound #2)
    filing_dates = step_filing_dates()

    # 6. Filing-date citations (Confound #2)
    step_citations_filing_date(citations, filing_dates)

    logger.info("=" * 60)
    logger.info("All filtered citation datasets built.")
    logger.info("Output files in %s:", DATA_DIR)
    for fname in [
        "citations_with_category.parquet",
        "citations_applicant_only.parquet",
        "patent_assignee.parquet",
        "citations_no_self_cite.parquet",
        "patent_filing_dates.parquet",
        "citations_filing_date.parquet",
    ]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            logger.info("  %-45s  %.0f MB", fname, size_mb)
        else:
            logger.warning("  %-45s  MISSING", fname)

    log_memory("Final")


if __name__ == "__main__":
    main()
