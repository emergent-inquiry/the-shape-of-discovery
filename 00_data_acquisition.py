"""
00 — Data Acquisition
=====================
Download PatentsView bulk data, clean it, and produce three parquet files:

    data/patents.parquet    — patent_id, date, title, cpc_primary
    data/citations.parquet  — citing_id, cited_id, citing_date
    data/cpc_map.parquet    — patent_id, cpc_section, cpc_class, cpc_subclass

Run:
    python 00_data_acquisition.py
"""

# %% Imports
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.fetch import download_all
from src.utils import DATA_DIR, get_logger, log_memory, timer

logger = get_logger(__name__)

RAW_DIR = DATA_DIR / "raw"


# %% Download raw data
@timer
def step_download() -> dict[str, Path]:
    """Download all PatentsView tables."""
    logger.info("Step 1/4: Downloading PatentsView bulk data...")
    return download_all()


# %% Load and clean patents
@timer
def step_clean_patents(tsv_path: Path) -> pd.DataFrame:
    """Load g_patent, filter to utility patents, parse dates.

    Args:
        tsv_path: Path to the raw g_patent.tsv file.

    Returns:
        Cleaned patent DataFrame.
    """
    logger.info("Step 2/4: Cleaning patent metadata...")

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype={"patent_id": str, "patent_title": str, "patent_type": str},
        parse_dates=["patent_date"],
        low_memory=False,
    )

    logger.info("Raw patents: %d rows", len(df))

    # Filter to utility patents only
    df = df[df["patent_type"] == "utility"].copy()
    logger.info("Utility patents: %d", len(df))

    # Drop rows with missing dates
    df = df.dropna(subset=["patent_date"])

    # Select and rename columns
    df = df.rename(columns={
        "patent_date": "date",
        "patent_title": "title",
    })[["patent_id", "date", "title"]]

    log_memory("After patent cleaning")
    return df


# %% Load and clean citations
@timer
def step_clean_citations(tsv_path: Path, valid_patent_ids: set) -> pd.DataFrame:
    """Load g_us_patent_citation in chunks, filter, deduplicate.

    Args:
        tsv_path: Path to the raw g_us_patent_citation.tsv file.
        valid_patent_ids: Set of patent IDs to keep (utility patents only).

    Returns:
        Cleaned citation DataFrame.
    """
    logger.info("Step 3/4: Cleaning citations (chunked read)...")

    chunks = []
    chunk_size = 5_000_000
    total_raw = 0

    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype={"patent_id": str, "citation_patent_id": str},
        usecols=["patent_id", "citation_patent_id"],
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Reading citations", unit="chunk"):
        total_raw += len(chunk)

        # Filter: both citing and cited must be in our valid set
        mask = (
            chunk["patent_id"].isin(valid_patent_ids)
            & chunk["citation_patent_id"].isin(valid_patent_ids)
        )
        filtered = chunk[mask].copy()

        # Remove self-citations
        filtered = filtered[filtered["patent_id"] != filtered["citation_patent_id"]]

        chunks.append(filtered)

    logger.info("Raw citation rows: %d", total_raw)

    df = pd.concat(chunks, ignore_index=True)

    # Rename columns
    df = df.rename(columns={
        "patent_id": "citing_id",
        "citation_patent_id": "cited_id",
    })

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["citing_id", "cited_id"])
    logger.info("Citations after dedup: %d (removed %d dupes)", len(df), before - len(df))

    log_memory("After citation cleaning")
    return df


# %% Load and clean CPC classifications
@timer
def step_clean_cpc(tsv_path: Path, valid_patent_ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load g_cpc_current, extract section/class/subclass hierarchy.

    Args:
        tsv_path: Path to the raw g_cpc_current.tsv file.
        valid_patent_ids: Set of patent IDs to keep.

    Returns:
        Tuple of (cpc_map DataFrame, patents with primary CPC added).
    """
    logger.info("Step 4/4: Cleaning CPC classifications...")

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype=str,
        usecols=["patent_id", "cpc_sequence", "cpc_section", "cpc_class", "cpc_subclass"],
        low_memory=False,
    )

    logger.info("Raw CPC rows: %d", len(df))

    # Filter to valid patents
    df = df[df["patent_id"].isin(valid_patent_ids)].copy()

    # Build the CPC map: unique (patent_id, section, class, subclass) tuples
    cpc_map = df[["patent_id", "cpc_section", "cpc_class", "cpc_subclass"]].drop_duplicates()

    # Get primary CPC per patent (sequence 0 is the main classification)
    primary_cpc = df[df["cpc_sequence"] == "0"]
    if len(primary_cpc) > 0:
        primary_map = primary_cpc[["patent_id", "cpc_section"]].drop_duplicates("patent_id")
    else:
        primary_map = cpc_map.groupby("patent_id")["cpc_section"].first().reset_index()

    primary_map = primary_map.rename(columns={"cpc_section": "cpc_primary"})

    logger.info("CPC map: %d rows, %d unique patents", len(cpc_map), cpc_map["patent_id"].nunique())
    log_memory("After CPC cleaning")
    return cpc_map, primary_map


# %% Add citing dates to citations
def add_citing_dates(citations: pd.DataFrame, patents: pd.DataFrame) -> pd.DataFrame:
    """Merge patent dates onto the citation table.

    Args:
        citations: Citation DataFrame with citing_id, cited_id.
        patents: Patent DataFrame with patent_id, date.

    Returns:
        Citations with citing_date column added.
    """
    date_map = patents.set_index("patent_id")["date"]
    citations["citing_date"] = citations["citing_id"].map(date_map)
    citations = citations.dropna(subset=["citing_date"])
    citations["citing_date"] = pd.to_datetime(citations["citing_date"])
    return citations


# %% Main pipeline
@timer
def main() -> None:
    """Run the full data acquisition pipeline."""
    logger.info("=" * 60)
    logger.info("The Shape of Discovery — Data Acquisition")
    logger.info("=" * 60)

    # 1. Download
    paths = step_download()

    # 2. Clean patents
    patents = step_clean_patents(paths["g_patent"])
    valid_ids = set(patents["patent_id"])

    # 3. Clean citations
    citations = step_clean_citations(paths["g_us_patent_citation"], valid_ids)

    # 4. Clean CPC
    cpc_map, primary_cpc = step_clean_cpc(paths["g_cpc_current"], valid_ids)

    # 5. Merge primary CPC onto patents
    patents = patents.merge(primary_cpc, on="patent_id", how="left")

    # 6. Add citing dates to citations
    citations = add_citing_dates(citations, patents)

    # 7. Save to parquet
    logger.info("Saving cleaned data to parquet...")
    patents.to_parquet(DATA_DIR / "patents.parquet", index=False)
    citations.to_parquet(DATA_DIR / "citations.parquet", index=False)
    cpc_map.to_parquet(DATA_DIR / "cpc_map.parquet", index=False)

    # 8. Summary statistics
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Patents:   %d", len(patents))
    logger.info("Citations: %d", len(citations))
    logger.info("CPC map:   %d rows (%d unique patents)", len(cpc_map), cpc_map["patent_id"].nunique())
    logger.info(
        "Date range: %s to %s",
        patents["date"].min().strftime("%Y-%m-%d"),
        patents["date"].max().strftime("%Y-%m-%d"),
    )
    logger.info("CPC section distribution:")
    if "cpc_primary" in patents.columns:
        dist = patents["cpc_primary"].value_counts().sort_index()
        for section, count in dist.items():
            logger.info("  %s: %d (%.1f%%)", section, count, 100 * count / len(patents))

    log_memory("Final")
    logger.info("Data acquisition complete.")


if __name__ == "__main__":
    main()
