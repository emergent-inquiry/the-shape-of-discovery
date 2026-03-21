"""Confound characterization helpers for NB04 robustness checks.

Provides functions to measure how much each confounding variable (examiner
citations, intra-assignee citations) contributes to the topological signal in
each CPC section pair and sliding time window.

These are used in NB04 §5 to determine whether the pre-breakthrough topology
result survives after statistically controlling for each confound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def examiner_fraction_by_window(
    citations_with_category: pd.DataFrame,
    cpc_map: pd.DataFrame,
    sec_a: str,
    sec_b: str,
    window_years: int = 5,
    year_range: tuple[int, int] = (1985, 2023),
) -> pd.DataFrame:
    """Fraction of citations that are examiner-added per sliding window.

    Filters to citations involving patents in either CPC section, then
    computes the examiner-citation fraction for each window end-year.

    Args:
        citations_with_category: DataFrame with citing_id, cited_id,
            citing_date, citation_category.
        cpc_map: CPC mapping DataFrame with patent_id, cpc_section.
        sec_a: First CPC section (e.g., "C").
        sec_b: Second CPC section (e.g., "A"). May equal sec_a.
        window_years: Width of the sliding window in years.
        year_range: (start, end) inclusive range of window end-years.

    Returns:
        DataFrame with columns: window_end, examiner_fraction, n_citations.
    """
    if "citing_year" not in citations_with_category.columns:
        citations_with_category = citations_with_category.copy()
        citations_with_category["citing_year"] = pd.to_datetime(
            citations_with_category["citing_date"]
        ).dt.year

    # Patents in either section
    pair_patents = set(
        cpc_map[cpc_map["cpc_section"].isin([sec_a, sec_b])]["patent_id"]
    )

    # Filter citations to the pair
    mask = (
        citations_with_category["citing_id"].isin(pair_patents)
        | citations_with_category["cited_id"].isin(pair_patents)
    )
    pair_cites = citations_with_category[mask]

    rows = []
    start_yr, end_yr = year_range
    for win_end in range(start_yr, end_yr + 1):
        win_start = win_end - window_years + 1
        window_cites = pair_cites[
            (pair_cites["citing_year"] >= win_start)
            & (pair_cites["citing_year"] <= win_end)
        ]
        n = len(window_cites)
        if n == 0:
            rows.append({"window_end": win_end, "examiner_fraction": np.nan, "n_citations": 0})
            continue
        examiner_n = (window_cites["citation_category"] == "cited by examiner").sum()
        rows.append({
            "window_end": win_end,
            "examiner_fraction": examiner_n / n,
            "n_citations": n,
        })

    return pd.DataFrame(rows)


def self_citation_fraction_by_window(
    citations: pd.DataFrame,
    patent_assignee: pd.DataFrame,
    cpc_map: pd.DataFrame,
    sec_a: str,
    sec_b: str,
    window_years: int = 5,
    year_range: tuple[int, int] = (1985, 2023),
) -> pd.DataFrame:
    """Fraction of citations that are intra-assignee per sliding window.

    Args:
        citations: Base citation DataFrame with citing_id, cited_id, citing_date.
        patent_assignee: patent_id → assignee_id mapping.
        cpc_map: CPC mapping DataFrame.
        sec_a: First CPC section.
        sec_b: Second CPC section.
        window_years: Window width in years.
        year_range: (start, end) inclusive range of window end-years.

    Returns:
        DataFrame with columns: window_end, self_cite_fraction, n_citations.
    """
    if "citing_year" not in citations.columns:
        citations = citations.copy()
        citations["citing_year"] = pd.to_datetime(citations["citing_date"]).dt.year

    pair_patents = set(
        cpc_map[cpc_map["cpc_section"].isin([sec_a, sec_b])]["patent_id"]
    )

    mask = (
        citations["citing_id"].isin(pair_patents)
        | citations["cited_id"].isin(pair_patents)
    )
    pair_cites = citations[mask].copy()

    # Map assignees
    assignee_map = patent_assignee.set_index("patent_id")["assignee_id"]
    pair_cites["citing_assignee"] = pair_cites["citing_id"].map(assignee_map)
    pair_cites["cited_assignee"] = pair_cites["cited_id"].map(assignee_map)

    rows = []
    start_yr, end_yr = year_range
    for win_end in range(start_yr, end_yr + 1):
        win_start = win_end - window_years + 1
        window_cites = pair_cites[
            (pair_cites["citing_year"] >= win_start)
            & (pair_cites["citing_year"] <= win_end)
        ]
        n = len(window_cites)
        if n == 0:
            rows.append({"window_end": win_end, "self_cite_fraction": np.nan, "n_citations": 0})
            continue
        self_cite_n = (
            window_cites["citing_assignee"].notna()
            & window_cites["cited_assignee"].notna()
            & (window_cites["citing_assignee"] == window_cites["cited_assignee"])
        ).sum()
        rows.append({
            "window_end": win_end,
            "self_cite_fraction": self_cite_n / n,
            "n_citations": n,
        })

    return pd.DataFrame(rows)


def policy_shock_dates() -> dict[str, str]:
    """Return known USPTO policy event dates.

    Returns:
        Dict mapping event name to ISO date string (YYYY-MM-DD).
    """
    return {
        "America Invents Act (AIA)": "2011-09-16",
        "Alice Corp. v. CLS Bank": "2014-06-19",
    }


def partial_out_confound(
    z_scores: pd.Series,
    confound_values: pd.Series,
) -> pd.Series:
    """Remove linear effect of a confound variable from z-scores using OLS.

    Fits z = a + b * confound + residual and returns the residuals (plus mean
    z_score, to keep the residuals interpretable on the same scale).

    Args:
        z_scores: Pre-breakthrough z-scores for each breakthrough.
        confound_values: The confound measurement (e.g., examiner fraction)
            for the same breakthroughs. NaN values are excluded from the fit
            but will receive NaN residuals.

    Returns:
        Residual z-scores (confound effect removed). Same index as z_scores.
    """
    valid = z_scores.notna() & confound_values.notna()
    if valid.sum() < 3:
        logger.warning("Too few valid pairs (%d) to fit confound model", valid.sum())
        return z_scores.copy()

    x = confound_values[valid].values
    y = z_scores[valid].values

    # OLS: y = a + b*x
    x_design = np.column_stack([np.ones_like(x), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("OLS failed; returning raw z-scores")
        return z_scores.copy()

    a, b = coeffs
    logger.info(
        "Confound partial-out: intercept=%.3f, slope=%.3f (R²=%.3f)",
        a, b, _r_squared(y, x_design @ coeffs),
    )

    # Residuals + mean (centre at same location as raw z-scores)
    residuals = z_scores.copy()
    fitted = a + b * confound_values
    residuals[valid] = z_scores[valid] - (fitted[valid] - fitted[valid].mean())

    return residuals


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def prosecution_lag_by_section(
    citations_filing_date: pd.DataFrame,
    citations_grant_date: pd.DataFrame,
    cpc_map: pd.DataFrame,
) -> pd.DataFrame:
    """Compute median prosecution lag (grant - filing) per CPC section.

    Args:
        citations_filing_date: Citations with filing-date as citing_date.
        citations_grant_date: Original citations with grant-date as citing_date.
        cpc_map: CPC mapping DataFrame.

    Returns:
        DataFrame with columns: cpc_section, median_lag_years, n_patents.
    """
    # Build citing patent → section mapping
    citing_section = (
        cpc_map
        .drop_duplicates(subset=["patent_id"])
        .set_index("patent_id")["cpc_section"]
    )

    # Merge grant and filing dates onto the same citations
    grant_dates = citations_grant_date[["citing_id", "citing_date"]].rename(
        columns={"citing_date": "grant_date"}
    )
    filing_dates = citations_filing_date[["citing_id", "citing_date"]].rename(
        columns={"citing_date": "filing_date"}
    )

    merged = grant_dates.merge(filing_dates, on="citing_id", how="inner")
    merged["lag_years"] = (
        pd.to_datetime(merged["grant_date"]) - pd.to_datetime(merged["filing_date"])
    ).dt.days / 365.25

    # Filter to plausible prosecution duration (0-15 years)
    merged = merged[(merged["lag_years"] >= 0) & (merged["lag_years"] <= 15)]
    merged["cpc_section"] = merged["citing_id"].map(citing_section)
    merged = merged.dropna(subset=["cpc_section"])

    result = (
        merged
        .groupby("cpc_section")["lag_years"]
        .agg(median_lag_years="median", n_patents="count")
        .reset_index()
        .sort_values("cpc_section")
    )

    return result
