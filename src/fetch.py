"""PatentsView bulk data downloader with resume logic and fallback strategies."""

import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.utils import DATA_DIR, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known download URLs (verified 2026-03-16, S3 bucket still live)
# ---------------------------------------------------------------------------

S3_BASE = "https://s3.amazonaws.com/data.patentsview.org/download"

BULK_TABLES = {
    "g_patent": {
        "url": f"{S3_BASE}/g_patent.tsv.zip",
        "description": "Patent metadata (patent_id, date, title, type)",
        "expected_size_mb": 230,
    },
    "g_us_patent_citation": {
        "url": f"{S3_BASE}/g_us_patent_citation.tsv.zip",
        "description": "Citation links (citing → cited patent)",
        "expected_size_mb": 2232,
    },
    "g_cpc_current": {
        "url": f"{S3_BASE}/g_cpc_current.tsv.zip",
        "description": "CPC classifications per patent",
        "expected_size_mb": 495,
    },
    "g_application": {
        "url": f"{S3_BASE}/g_application.tsv.zip",
        "description": "Application metadata (filing_date, series_code)",
        "expected_size_mb": 120,
    },
    "g_assignee_disambiguated": {
        "url": f"{S3_BASE}/g_assignee_disambiguated.tsv.zip",
        "description": "Assignee info (assignee_id, organization, type)",
        "expected_size_mb": 180,
    },
}

# Fallback URLs if S3 goes down after PatentsView migration
FALLBACK_BASES = [
    "https://data.uspto.gov/download",
    "https://bulkdata.patentsview.org",
]

STATE_FILE = DATA_DIR / ".download_state.json"


# ---------------------------------------------------------------------------
# Download state management
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    """Load download progress state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    """Persist download progress state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def _download_with_resume(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    """Download a file with resume support via HTTP Range headers.

    Args:
        url: URL to download.
        dest: Local destination path.
        chunk_size: Download chunk size in bytes (default 8 MB).

    Returns:
        True if download completed successfully.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_size = dest.stat().st_size if dest.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info("Resuming download from %d bytes", existing_size)

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)

        if resp.status_code == 416:
            logger.info("File already fully downloaded: %s", dest.name)
            return True

        if resp.status_code not in (200, 206):
            logger.warning("HTTP %d from %s", resp.status_code, url)
            return False

        total = int(resp.headers.get("content-length", 0))
        if resp.status_code == 200:
            existing_size = 0
            mode = "wb"
        else:
            mode = "ab"

        desc = dest.name
        with (
            open(dest, mode) as f,
            tqdm(
                total=total + existing_size,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                desc=desc,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.RequestException as e:
        logger.error("Download failed for %s: %s", url, e)
        return False


def _try_download(table_name: str, table_info: dict, dest_zip: Path) -> bool:
    """Try downloading from primary URL, then fallbacks.

    Args:
        table_name: Name of the table (e.g., 'g_patent').
        table_info: Dict with 'url' and metadata.
        dest_zip: Local path for the zip file.

    Returns:
        True if any URL succeeded.
    """
    # Try primary S3 URL
    logger.info("Downloading %s from S3...", table_name)
    if _download_with_resume(table_info["url"], dest_zip):
        return True

    # Try fallback URLs
    for base in FALLBACK_BASES:
        fallback_url = f"{base}/{table_name}.tsv.zip"
        logger.info("Trying fallback: %s", fallback_url)
        if _download_with_resume(fallback_url, dest_zip):
            return True

    return False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a zip file and return path to the extracted TSV.

    Args:
        zip_path: Path to the zip file.
        dest_dir: Directory to extract into.

    Returns:
        Path to the extracted .tsv file.
    """
    logger.info("Extracting %s...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if not tsv_names:
            raise ValueError(f"No .tsv file found in {zip_path.name}")
        zf.extractall(dest_dir)
        extracted = dest_dir / tsv_names[0]
        logger.info("Extracted: %s (%.0f MB)", extracted.name, extracted.stat().st_size / 1e6)
        return extracted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_table(table_name: str, force: bool = False) -> Path:
    """Download and extract a single PatentsView table.

    Args:
        table_name: One of 'g_patent', 'g_us_patent_citation', 'g_cpc_current'.
        force: If True, re-download even if file exists.

    Returns:
        Path to the extracted TSV file.

    Raises:
        ValueError: If table_name is not recognized.
        RuntimeError: If download fails from all sources.
    """
    if table_name not in BULK_TABLES:
        raise ValueError(
            f"Unknown table '{table_name}'. "
            f"Available: {list(BULK_TABLES.keys())}"
        )

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    table_info = BULK_TABLES[table_name]
    dest_zip = raw_dir / f"{table_name}.tsv.zip"
    dest_tsv = raw_dir / f"{table_name}.tsv"

    state = _load_state()

    # Check if already extracted
    if not force and dest_tsv.exists() and state.get(table_name) == "complete":
        logger.info("Already downloaded and extracted: %s", dest_tsv.name)
        return dest_tsv

    # Download
    if not dest_tsv.exists() or force:
        if not _try_download(table_name, table_info, dest_zip):
            raise RuntimeError(
                f"Failed to download {table_name} from all sources. "
                f"Try manual download: curl -O {table_info['url']}"
            )

        # Extract
        dest_tsv = _extract_zip(dest_zip, raw_dir)

        # Clean up zip to save disk space
        dest_zip.unlink()
        logger.info("Removed zip: %s", dest_zip.name)

    # Update state
    state[table_name] = "complete"
    _save_state(state)

    return dest_tsv


def download_all(force: bool = False) -> dict[str, Path]:
    """Download all required PatentsView tables.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        Dict mapping table name to extracted TSV path.
    """
    results = {}
    for table_name in BULK_TABLES:
        logger.info(
            "=== %s (%s, ~%d MB) ===",
            table_name,
            BULK_TABLES[table_name]["description"],
            BULK_TABLES[table_name]["expected_size_mb"],
        )
        results[table_name] = download_table(table_name, force=force)

    logger.info("All tables downloaded successfully.")
    return results


def check_availability() -> dict[str, bool]:
    """Probe each download URL to verify availability.

    Returns:
        Dict mapping table name to availability status.
    """
    status = {}
    for table_name, info in BULK_TABLES.items():
        try:
            resp = requests.head(info["url"], timeout=10)
            available = resp.status_code == 200
            size_mb = int(resp.headers.get("content-length", 0)) / 1e6
            logger.info(
                "%s: %s (%.0f MB)",
                table_name,
                "available" if available else f"HTTP {resp.status_code}",
                size_mb,
            )
            status[table_name] = available
        except requests.RequestException as e:
            logger.warning("%s: unreachable (%s)", table_name, e)
            status[table_name] = False
    return status
