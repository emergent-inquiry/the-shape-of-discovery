#!/usr/bin/env python3
"""Diagnose why §5.8 temporally-matched null disagrees with §5.7 detrending.

Key question: is the §5.8 signal real, or a methodological artifact?

Three versions of the temporally-matched null:
  A) Original: single-year null from ±15 years (flawed?)
  B) Sliding-window: 10-year null windows, same aggregation as precursor
  C) Same-direction-only: null from BEFORE the precursor only (no post-breakthrough)
"""
import sys
sys.path.insert(0, '.')

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from src.breakthroughs import load_breakthroughs, get_precursor_window
from src.topology import ALL_PAIRS
from src.plotting import save_figure, PALETTE
from src.utils import DATA_DIR

# Load topology cache
print("Loading topology cache...")
topology_results = {}
cache_dir = DATA_DIR / "topology_cache"
for pair_dir in sorted(cache_dir.iterdir()):
    if not pair_dir.is_dir() or 'x' not in pair_dir.name:
        continue
    windows = sorted(pair_dir.glob("window_*_subclass.parquet"))
    if windows:
        dfs = [pd.read_parquet(w) for w in windows]
        topology_results[pair_dir.name] = pd.concat(dfs, ignore_index=True)

breakthroughs = load_breakthroughs()

def get_pair_beta1_at_year(matching_pairs, year):
    """Get mean β₁ across matching pairs at a single year."""
    vals = []
    for sa, sb in matching_pairs:
        pk = f'{sa}x{sb}'
        topo = topology_results.get(pk)
        if topo is None:
            continue
        row = topo[topo['window_end'] == year]
        if len(row) > 0:
            vals.append(row['beta_1'].values[0])
    return np.mean(vals) if vals else None

def get_pair_beta1_window(matching_pairs, start, end):
    """Get mean β₁ across matching pairs over a multi-year window."""
    vals = []
    for sa, sb in matching_pairs:
        pk = f'{sa}x{sb}'
        topo = topology_results.get(pk)
        if topo is None:
            continue
        mask = (topo['window_end'] >= start) & (topo['window_end'] <= end)
        pre = topo[mask]
        if len(pre) > 0:
            vals.append(pre['beta_1'].mean())
    return np.mean(vals) if vals else None

print("\n" + "="*60)
print("DIAGNOSING §5.8 TEMPORAL-MATCHED NULL")
print("="*60)

# Show the structure of a specific example
example_bt = next(b for b in breakthroughs if 1998 <= b.filing_year <= 2002 and len(b.cpc_sections) >= 2)
print(f"\nExample: {example_bt.name} (filing year: {example_bt.filing_year})")
matching_ex = [(a, b) for a, b in ALL_PAIRS if any(s in [a, b] for s in example_bt.cpc_sections)]
print(f"Matching pairs: {len(matching_ex)}")

# Show β₁ at various years
pre_start = example_bt.filing_year - 10
pre_end = example_bt.filing_year
print(f"Precursor window: {pre_start}-{pre_end}")
pre_val = get_pair_beta1_window(matching_ex, pre_start, pre_end)
print(f"Precursor β₁: {pre_val:.1f}")

print("\nNull year β₁ values (§5.8 original):")
for y in range(example_bt.filing_year - 15, example_bt.filing_year + 11):
    if 1984 <= y <= 2023:
        in_precursor = pre_start <= y <= pre_end
        in_buffer = pre_end < y <= pre_end + 3
        val = get_pair_beta1_at_year(matching_ex, y)
        marker = " ← PRECURSOR" if in_precursor else (" ← EXCLUDED" if in_buffer else "")
        if val is not None:
            print(f"  {y}: β₁ = {val:.1f}{marker}")

# ══════════════════════════════════════════════════════════════════
# VERSION B: Sliding-window null (10-year windows, like precursor)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("VERSION B: SLIDING-WINDOW NULL (10-year windows)")
print("="*60 + "\n")

sw_rows = []
for bt in breakthroughs:
    start, end = get_precursor_window(bt, years_before=10)
    matching_pairs = [(a, b) for a, b in ALL_PAIRS if any(s in [a, b] for s in bt.cpc_sections)]
    if not matching_pairs:
        continue

    pre_beta1 = get_pair_beta1_window(matching_pairs, start, end)
    if pre_beta1 is None:
        continue

    # Null: other 10-year windows from the same matching pairs
    # Exclude windows that overlap with [start, end+3]
    null_window_betas = []
    for null_end in range(1984, 2024):
        null_start = null_end - 10
        # No overlap with [start, end+3]
        if null_end < start or null_start > end + 3:
            val = get_pair_beta1_window(matching_pairs, null_start, null_end)
            if val is not None:
                null_window_betas.append(val)

    if len(null_window_betas) < 3:
        continue

    null_mean = np.mean(null_window_betas)
    null_std = np.std(null_window_betas)
    z_sw = (pre_beta1 - null_mean) / null_std if null_std > 0 else 0

    sw_rows.append({
        'name': bt.name,
        'filing_year': bt.filing_year,
        'pre_beta1': pre_beta1,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_sw': z_sw,
        'n_null_windows': len(null_window_betas),
    })

sw_df = pd.DataFrame(sw_rows)
print(f"N = {len(sw_df)}")
print(f"Mean z: {sw_df['z_sw'].mean():.3f}")
print(f"Positive: {(sw_df['z_sw'] > 0).sum()}/{len(sw_df)} ({100*(sw_df['z_sw'] > 0).mean():.0f}%)")
t_sw, p_t_sw = stats.ttest_1samp(sw_df['z_sw'], 0)
w_sw, p_w_sw = stats.wilcoxon(sw_df['z_sw'])
print(f"t-test: p = {p_t_sw:.4f}")
print(f"Wilcoxon: p = {p_w_sw:.4f}")
r_sw, p_r_sw = pearsonr(sw_df['filing_year'], sw_df['z_sw'])
print(f"z vs year: r = {r_sw:.3f} (p = {p_r_sw:.3f})")

# ══════════════════════════════════════════════════════════════════
# VERSION C: Pre-direction only null (no post-breakthrough years)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("VERSION C: SAME-DIRECTION ONLY (pre-precursor null)")
print("="*60 + "\n")

pre_only_rows = []
for bt in breakthroughs:
    start, end = get_precursor_window(bt, years_before=10)
    matching_pairs = [(a, b) for a, b in ALL_PAIRS if any(s in [a, b] for s in bt.cpc_sections)]
    if not matching_pairs:
        continue

    pre_beta1 = get_pair_beta1_window(matching_pairs, start, end)
    if pre_beta1 is None:
        continue

    # Null: single years from BEFORE the precursor window only
    null_betas = []
    for y in range(1984, start):
        val = get_pair_beta1_at_year(matching_pairs, y)
        if val is not None:
            null_betas.append(val)

    if len(null_betas) < 3:
        continue

    null_mean = np.mean(null_betas)
    null_std = np.std(null_betas)
    z_pre = (pre_beta1 - null_mean) / null_std if null_std > 0 else 0

    pre_only_rows.append({
        'name': bt.name,
        'filing_year': bt.filing_year,
        'z_pre': z_pre,
        'n_null_years': len(null_betas),
    })

pre_df = pd.DataFrame(pre_only_rows)
print(f"N = {len(pre_df)}")
print(f"Mean z: {pre_df['z_pre'].mean():.3f}")
print(f"Positive: {(pre_df['z_pre'] > 0).sum()}/{len(pre_df)} ({100*(pre_df['z_pre'] > 0).mean():.0f}%)")
if len(pre_df) > 0:
    t_pre, p_t_pre = stats.ttest_1samp(pre_df['z_pre'], 0)
    w_pre, p_w_pre = stats.wilcoxon(pre_df['z_pre'])
    print(f"t-test: p = {p_t_pre:.4f}")
    print(f"Wilcoxon: p = {p_w_pre:.4f}")
    r_pre, p_r_pre = pearsonr(pre_df['filing_year'], pre_df['z_pre'])
    print(f"z vs year: r = {r_pre:.3f} (p = {p_r_pre:.3f})")

# ══════════════════════════════════════════════════════════════════
# VERSION D: Proper sliding-window, SAME ERA (±10 years)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("VERSION D: SLIDING-WINDOW, SAME ERA (±10 years)")
print("="*60 + "\n")

era_sw_rows = []
for bt in breakthroughs:
    start, end = get_precursor_window(bt, years_before=10)
    matching_pairs = [(a, b) for a, b in ALL_PAIRS if any(s in [a, b] for s in bt.cpc_sections)]
    if not matching_pairs:
        continue

    pre_beta1 = get_pair_beta1_window(matching_pairs, start, end)
    if pre_beta1 is None:
        continue

    # Null: 10-year windows from ±10 years of filing year, non-overlapping with precursor
    null_window_betas = []
    for null_end in range(bt.filing_year - 10, bt.filing_year + 11):
        null_start = null_end - 10
        if null_start < 1984 or null_end > 2023:
            continue
        # No overlap with [start, end+3]
        if null_end < start or null_start > end + 3:
            val = get_pair_beta1_window(matching_pairs, null_start, null_end)
            if val is not None:
                null_window_betas.append(val)

    if len(null_window_betas) < 2:
        continue

    null_mean = np.mean(null_window_betas)
    null_std = np.std(null_window_betas)
    z_era = (pre_beta1 - null_mean) / null_std if null_std > 0 else 0

    era_sw_rows.append({
        'name': bt.name,
        'filing_year': bt.filing_year,
        'z_era': z_era,
        'n_null_windows': len(null_window_betas),
    })

era_df = pd.DataFrame(era_sw_rows)
print(f"N = {len(era_df)}")
print(f"Mean z: {era_df['z_era'].mean():.3f}")
print(f"Positive: {(era_df['z_era'] > 0).sum()}/{len(era_df)} ({100*(era_df['z_era'] > 0).mean():.0f}%)")
if len(era_df) > 0:
    t_era, p_t_era = stats.ttest_1samp(era_df['z_era'], 0)
    try:
        w_era, p_w_era = stats.wilcoxon(era_df['z_era'])
    except ValueError:
        p_w_era = 1.0
    print(f"t-test: p = {p_t_era:.4f}")
    print(f"Wilcoxon: p = {p_w_era:.4f}")
    r_era, p_r_era = pearsonr(era_df['filing_year'], era_df['z_era'])
    print(f"z vs year: r = {r_era:.3f} (p = {p_r_era:.3f})")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY: FOUR APPROACHES TO TEMPORAL CONTROL")
print("="*60)
print(f"\n{'Approach':<45} {'N':>3} {'Mean z':>8} {'%+':>5} {'p(W)':>8} {'r(year)':>8}")
print("-"*80)
print(f"{'§5.7 Per-pair detrending (residuals)':<45} {'57':>3} {'0.21':>8} {'40%':>5} {'0.846':>8} {'0.007':>8}")
print(f"{'§5.8a Original TM (single-year, flawed)':<45} {'57':>3} {'3.13':>8} {'100%':>5} {'<0.001':>8} {'-0.69':>8}")
if len(sw_df) > 0:
    print(f"{'§5.8b Sliding-window null (all years)':<45} {len(sw_df):>3} {sw_df['z_sw'].mean():>8.3f} {f'{100*(sw_df.z_sw>0).mean():.0f}%':>5} {p_w_sw:>8.4f} {r_sw:>8.3f}")
if len(pre_df) > 0:
    print(f"{'§5.8c Pre-direction only null':<45} {len(pre_df):>3} {pre_df['z_pre'].mean():>8.3f} {f'{100*(pre_df.z_pre>0).mean():.0f}%':>5} {p_w_pre:>8.4f} {r_pre:>8.3f}")
if len(era_df) > 0:
    print(f"{'§5.8d Sliding-window, same era (±10yr)':<45} {len(era_df):>3} {era_df['z_era'].mean():>8.3f} {f'{100*(era_df.z_era>0).mean():.0f}%':>5} {p_w_era:>8.4f} {r_era:>8.3f}")
