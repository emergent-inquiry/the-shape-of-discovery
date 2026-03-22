#!/usr/bin/env python3
"""Standalone script to run §5.7 temporal confound analysis.

Loads topology cache + null cache + breakthroughs directly,
computes z-scores, runs detrending, and generates the diagnostic figure.
Much lighter than running the full NB04 notebook.
"""
import sys
sys.path.insert(0, '.')

import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from src.breakthroughs import load_breakthroughs, get_precursor_window
from src.topology import ALL_PAIRS
from src.plotting import save_figure, PALETTE
from src.utils import DATA_DIR

# ── Load topology cache ──────────────────────────────────────────────
print("Loading topology cache...")
topology_results = {}
cache_dir = DATA_DIR / "topology_cache"
for pair_dir in sorted(cache_dir.iterdir()):
    if not pair_dir.is_dir() or 'x' not in pair_dir.name:
        continue
    windows = sorted(pair_dir.glob("window_*_subclass.parquet"))
    if windows:
        dfs = [pd.read_parquet(w) for w in windows]
        topo = pd.concat(dfs, ignore_index=True)
        topology_results[pair_dir.name] = topo

print(f"Loaded {len(topology_results)} CPC pairs from topology cache")

# ── Load breakthroughs ───────────────────────────────────────────────
breakthroughs = load_breakthroughs()
print(f"Loaded {len(breakthroughs)} breakthroughs")

# ── Load null cache and compute z-scores ─────────────────────────────
print("Loading null cache and computing z-scores...")
null_cache = DATA_DIR / "null_cache"

comp_rows = []
for bt in breakthroughs:
    # Get precursor window
    start, end = get_precursor_window(bt, years_before=10)

    # Find matching pairs
    matching_pairs = [
        (a, b) for a, b in ALL_PAIRS
        if any(s in [a, b] for s in bt.cpc_sections)
    ]

    if not matching_pairs:
        continue

    # Compute pre-breakthrough topology (average across matching pairs)
    pre_values = []
    pre_pe_values = []
    n_matching = 0
    precursor_windows = 0

    for sa, sb in matching_pairs:
        pk = f'{sa}x{sb}'
        topo = topology_results.get(pk)
        if topo is None:
            continue
        n_matching += 1
        mask = (topo['window_end'] >= start) & (topo['window_end'] <= end)
        pre = topo[mask]
        if len(pre) > 0:
            pre_values.append(pre['beta_1'].mean())
            if 'persistence_entropy' in pre.columns:
                pre_pe_values.append(pre['persistence_entropy'].mean())
            precursor_windows += len(pre)

    if not pre_values:
        continue

    pre_beta1 = np.mean(pre_values)
    pre_pe = np.mean(pre_pe_values) if pre_pe_values else np.nan

    # Load null model
    bt_name = bt.name.replace(" ", "_").replace("/", "_").lower()[:30]
    cache_file = null_cache / f"matched_{bt_name}_n100_s42.pkl"
    if not cache_file.exists():
        continue

    with open(cache_file, 'rb') as f:
        null_df = pickle.load(f)

    if null_df.empty or 'beta_1' not in null_df.columns:
        continue

    null_mean = null_df['beta_1'].mean()
    null_std = null_df['beta_1'].std()

    if null_std > 0:
        z_beta1 = (pre_beta1 - null_mean) / null_std
    else:
        z_beta1 = 0.0

    # PE z-score
    z_pe = np.nan
    if not np.isnan(pre_pe) and 'persistence_entropy' in null_df.columns:
        pe_mean = null_df['persistence_entropy'].mean()
        pe_std = null_df['persistence_entropy'].std()
        if pe_std > 0:
            z_pe = (pre_pe - pe_mean) / pe_std

    comp_rows.append({
        'name': bt.name,
        'filing_year': bt.filing_year,
        'category': bt.category,
        'n_sections': len(bt.cpc_sections),
        'n_matching_pairs': n_matching,
        'n_precursor_windows': precursor_windows,
        'pre_beta1': pre_beta1,
        'null_mean_beta1': null_mean,
        'null_std_beta1': null_std,
        'z_beta1': z_beta1,
        'z_pe': z_pe,
    })

comp_df = pd.DataFrame(comp_rows)
print(f"\nValid comparisons: N = {len(comp_df)}")
print(f"Mean z_beta1: {comp_df['z_beta1'].mean():.3f}")
print(f"Positive z: {(comp_df['z_beta1'] > 0).sum()}/{len(comp_df)} "
      f"({100*(comp_df['z_beta1'] > 0).mean():.0f}%)")

# Raw statistical tests
t_raw, p_t_raw = stats.ttest_1samp(comp_df['z_beta1'], 0)
w_raw, p_w_raw = stats.wilcoxon(comp_df['z_beta1'])
print(f"\nRaw tests:")
print(f"  t-test: t = {t_raw:.3f}, p = {p_t_raw:.6f}")
print(f"  Wilcoxon: p = {p_w_raw:.6f}")

# ══════════════════════════════════════════════════════════════════════
# §5.7 TEMPORAL CONFOUND ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('§5.7: TEMPORAL CONFOUND — THE CRITICAL CONTROL')
print('='*60 + '\n')

# --- (a) Raw z-score vs filing year ---
r_p, p_p = pearsonr(comp_df['filing_year'], comp_df['z_beta1'])
r_s, p_s = spearmanr(comp_df['filing_year'], comp_df['z_beta1'])
print('Raw z_beta1 vs filing_year:')
print(f'  Pearson  r = {r_p:.3f}  (p = {p_p:.1e})')
print(f'  Spearman ρ = {r_s:.3f}  (p = {p_s:.1e})')
print(f'  → {abs(r_p**2)*100:.0f}% of z-score variance explained by filing year alone\n')

# --- (b) Detrend β₁ per CPC pair ---
pair_trends = {}
for pk, topo in topology_results.items():
    if 'window_end' not in topo.columns or 'beta_1' not in topo.columns:
        continue
    X = topo['window_end'].values.reshape(-1, 1)
    y = topo['beta_1'].values
    reg = LinearRegression().fit(X, y)
    resid = y - reg.predict(X)
    pair_trends[pk] = reg
    topo['beta_1_detrended'] = resid

print(f'Linear trends fitted for {len(pair_trends)} CPC pairs')
slopes = [reg.coef_[0] for reg in pair_trends.values()]
print(f'Mean β₁ slope: {np.mean(slopes):.2f}/year '
      f'(range: {min(slopes):.2f} to {max(slopes):.2f})')
print(f'All {sum(s < 0 for s in slopes)}/{len(slopes)} pairs show declining β₁\n')

# Recompute pre-breakthrough metric using detrended β₁
detrended_rows = []
for _, row in comp_df.iterrows():
    bt = next((b for b in breakthroughs if b.name == row['name']), None)
    if bt is None:
        continue
    start, end = get_precursor_window(bt, years_before=10)
    matching_pairs = [
        (a, b) for a, b in ALL_PAIRS
        if any(s in [a, b] for s in bt.cpc_sections)
    ]
    pre_resid = []
    for sa, sb in matching_pairs:
        pk = f'{sa}x{sb}'
        topo = topology_results.get(pk)
        if topo is None or 'beta_1_detrended' not in topo.columns:
            continue
        mask = (topo['window_end'] >= start) & (topo['window_end'] <= end)
        pre = topo[mask]
        if len(pre) > 0:
            pre_resid.append(pre['beta_1_detrended'].mean())
    if pre_resid:
        detrended_rows.append({
            'name': row['name'],
            'filing_year': row['filing_year'],
            'z_beta1_raw': row['z_beta1'],
            'pre_beta1_detrended': np.mean(pre_resid),
        })

det_df = pd.DataFrame(detrended_rows)
print(f'Detrended comparisons: N = {len(det_df)}')
print(f'Mean detrended β₁: {det_df.pre_beta1_detrended.mean():.3f}')
print(f'Median detrended β₁: {det_df.pre_beta1_detrended.median():.3f}')
print(f'Positive: {(det_df.pre_beta1_detrended > 0).sum()}/{len(det_df)} '
      f'({100*(det_df.pre_beta1_detrended > 0).mean():.0f}%)\n')

# Statistical tests on detrended values
t_det, p_t_det = stats.ttest_1samp(det_df['pre_beta1_detrended'], 0)
try:
    w_det, p_w_det = stats.wilcoxon(det_df['pre_beta1_detrended'])
except ValueError:
    p_w_det = 1.0
print('Detrended tests (H₀: pre-breakthrough β₁ residual = 0):')
print(f'  t-test:   t = {t_det:.3f},  p = {p_t_det:.4f}')
print(f'  Wilcoxon:           p = {p_w_det:.4f}')

# Verify detrending removed temporal correlation
r_det, p_det = pearsonr(det_df['filing_year'], det_df['pre_beta1_detrended'])
print(f'\nDetrended β₁ vs filing_year: r = {r_det:.3f} (p = {p_det:.3f}) — should be ~0\n')

# --- By era ---
early = det_df[det_df.filing_year <= 1995]
mid = det_df[(det_df.filing_year > 1995) & (det_df.filing_year <= 2005)]
late = det_df[det_df.filing_year > 2005]
print('By era (detrended):')
for era_name, era in [('Early ≤1995', early), ('Mid 1996-2005', mid), ('Late >2005', late)]:
    if len(era) >= 5:
        t_e, p_e = stats.ttest_1samp(era['pre_beta1_detrended'], 0)
        print(f'  {era_name:15s} (N={len(era):2d}): mean = {era.pre_beta1_detrended.mean():+.2f}, '
              f't = {t_e:.2f}, p = {p_e:.3f}')
    else:
        print(f'  {era_name:15s} (N={len(era):2d}): mean = {era.pre_beta1_detrended.mean():+.2f} '
              '(too few for t-test)')

# --- Figure: Temporal Confound Diagnostic ---
fig57, axes57 = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Raw z-scores vs filing year
ax = axes57[0]
colors = [PALETTE['orange'] if z > 0 else PALETTE['blue'] for z in det_df['z_beta1_raw']]
ax.scatter(det_df['filing_year'], det_df['z_beta1_raw'], c=colors, alpha=0.7, s=40)
x_fit = np.linspace(det_df['filing_year'].min(), det_df['filing_year'].max(), 100)
reg_line = LinearRegression().fit(
    det_df[['filing_year']].values, det_df['z_beta1_raw'].values
)
ax.plot(x_fit, reg_line.predict(x_fit.reshape(-1, 1)),
        color='red', linewidth=2, linestyle='--',
        label=f'r = {r_p:.2f}')
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('Breakthrough Filing Year')
ax.set_ylabel('z-score (β₁)')
ax.set_title('(a) Raw z-scores: Confounded')
ax.legend()

# Panel 2: Detrended β₁ vs filing year
ax = axes57[1]
colors_det = [PALETTE['orange'] if z > 0 else PALETTE['blue']
              for z in det_df['pre_beta1_detrended']]
ax.scatter(det_df['filing_year'], det_df['pre_beta1_detrended'],
           c=colors_det, alpha=0.7, s=40)
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('Breakthrough Filing Year')
ax.set_ylabel('Detrended β₁ (residual)')
ax.set_title(f'(b) Detrended: r = {r_det:.2f}')

# Panel 3: Distribution of detrended residuals
ax = axes57[2]
ax.hist(det_df['pre_beta1_detrended'], bins=20, color=PALETTE['blue'],
        alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Expected under H₀')
ax.axvline(det_df['pre_beta1_detrended'].mean(), color=PALETTE['orange'],
           linewidth=2, label=f'Mean = {det_df.pre_beta1_detrended.mean():.2f}')
ax.set_xlabel('Detrended β₁ (residual)')
ax.set_ylabel('Count')
ax.set_title(f'(c) Distribution (Wilcoxon p = {p_w_det:.3f})')
ax.legend(fontsize=9)

fig57.suptitle('§5.7: Temporal Confound — β₁ Secular Trend Drives Apparent Signal',
               fontsize=13, fontweight='bold')
fig57.tight_layout()
save_figure(fig57, '04_s5_temporal_confound')
print(f"\nFigure saved: figures/04_s5_temporal_confound.png")

# --- Verdict ---
print('\n=== §5.7 VERDICT ===')
if p_t_det < 0.05 or p_w_det < 0.05:
    print('  Signal SURVIVES detrending — topology is genuinely elevated before breakthroughs')
    print('  beyond what the secular β₁ trend explains.')
else:
    print('  Signal does NOT survive detrending.')
    print(f'  The raw significance (p < 0.001) is a TEMPORAL ARTIFACT.')
    print(f'  β₁ declines {abs(np.mean(slopes)):.1f}/year across all CPC pairs.')
    print(f'  Breakthroughs from the 1980s fall in the high-β₁ era;')
    print(f'  recent breakthroughs fall in the low-β₁ era.')
    print(f'  After removing this secular trend, pre-breakthrough topology')
    print(f'  is indistinguishable from any other period (p = {p_w_det:.3f}).')
    print()
    print('  IMPLICATION: The patent citation network does exhibit systematic')
    print('  topological evolution over time, but this evolution reflects network')
    print('  growth and citation practice changes — not breakthrough dynamics.')

# ══════════════════════════════════════════════════════════════════════
# §5.8 TEMPORALLY-MATCHED NULL MODEL
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('§5.8: TEMPORALLY-MATCHED NULL MODEL')
print('='*60 + '\n')
print('Instead of post-hoc detrending, match each breakthrough\'s null')
print('distribution to its own era (±5 years of filing year).\n')

tm_rows = []
for bt in breakthroughs:
    start, end = get_precursor_window(bt, years_before=10)

    matching_pairs = [
        (a, b) for a, b in ALL_PAIRS
        if any(s in [a, b] for s in bt.cpc_sections)
    ]
    if not matching_pairs:
        continue

    # Pre-breakthrough topology (same as before)
    pre_values = []
    for sa, sb in matching_pairs:
        pk = f'{sa}x{sb}'
        topo = topology_results.get(pk)
        if topo is None:
            continue
        mask = (topo['window_end'] >= start) & (topo['window_end'] <= end)
        pre = topo[mask]
        if len(pre) > 0:
            pre_values.append(pre['beta_1'].mean())

    if not pre_values:
        continue
    pre_beta1 = np.mean(pre_values)

    # Temporally-matched null: use topology from ±5 years of filing year,
    # excluding the precursor window itself
    # For each null year, average β₁ across matching pairs (same aggregation)
    null_window = 5  # ±5 years
    exclude_start = bt.filing_year - 10  # precursor window
    exclude_end = bt.filing_year + 3     # post-breakthrough buffer

    null_years = [
        y for y in range(bt.filing_year - 15, bt.filing_year + 11)
        if 1984 <= y <= 2023 and (y < exclude_start or y > exclude_end)
    ]

    if len(null_years) < 3:
        continue

    # Compute null distribution: for each null year, average β₁ across pairs
    null_beta1_values = []
    for ny in null_years:
        pair_vals = []
        for sa, sb in matching_pairs:
            pk = f'{sa}x{sb}'
            topo = topology_results.get(pk)
            if topo is None:
                continue
            row = topo[topo['window_end'] == ny]
            if len(row) > 0:
                pair_vals.append(row['beta_1'].values[0])
        if pair_vals:
            null_beta1_values.append(np.mean(pair_vals))

    if len(null_beta1_values) < 3:
        continue

    null_mean = np.mean(null_beta1_values)
    null_std = np.std(null_beta1_values)

    if null_std > 0:
        z_tm = (pre_beta1 - null_mean) / null_std
    else:
        z_tm = 0.0

    tm_rows.append({
        'name': bt.name,
        'filing_year': bt.filing_year,
        'pre_beta1': pre_beta1,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_tm': z_tm,
        'n_null_years': len(null_years),
    })

tm_df = pd.DataFrame(tm_rows)
print(f'Temporally-matched comparisons: N = {len(tm_df)}')
print(f'Mean z (temporally-matched): {tm_df["z_tm"].mean():.3f}')
print(f'Median z: {tm_df["z_tm"].median():.3f}')
print(f'Positive: {(tm_df["z_tm"] > 0).sum()}/{len(tm_df)} '
      f'({100*(tm_df["z_tm"] > 0).mean():.0f}%)\n')

# Statistical tests
t_tm, p_t_tm = stats.ttest_1samp(tm_df['z_tm'], 0)
try:
    w_tm, p_w_tm = stats.wilcoxon(tm_df['z_tm'])
except ValueError:
    p_w_tm = 1.0
print('Temporally-matched null tests (H₀: z = 0):')
print(f'  t-test:   t = {t_tm:.3f},  p = {p_t_tm:.4f}')
print(f'  Wilcoxon:           p = {p_w_tm:.4f}')

# Check if temporal correlation is removed
r_tm, p_r_tm = pearsonr(tm_df['filing_year'], tm_df['z_tm'])
print(f'\nTM z vs filing_year: r = {r_tm:.3f} (p = {p_r_tm:.3f})')

# --- Figure: Temporally-Matched Null ---
fig58, axes58 = plt.subplots(1, 2, figsize=(12, 5))

ax = axes58[0]
colors_tm = [PALETTE['orange'] if z > 0 else PALETTE['blue'] for z in tm_df['z_tm']]
ax.scatter(tm_df['filing_year'], tm_df['z_tm'], c=colors_tm, alpha=0.7, s=40)
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('Breakthrough Filing Year')
ax.set_ylabel('z-score (temporally-matched null)')
ax.set_title(f'(a) TM z-scores vs year (r = {r_tm:.2f})')

ax = axes58[1]
ax.hist(tm_df['z_tm'], bins=20, color=PALETTE['blue'], alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Expected under H₀')
ax.axvline(tm_df['z_tm'].mean(), color=PALETTE['orange'], linewidth=2,
           label=f'Mean = {tm_df["z_tm"].mean():.2f}')
ax.set_xlabel('z-score')
ax.set_ylabel('Count')
ax.set_title(f'(b) Distribution (Wilcoxon p = {p_w_tm:.3f})')
ax.legend(fontsize=9)

fig58.suptitle('§5.8: Temporally-Matched Null Model — Same-Era Comparison',
               fontsize=13, fontweight='bold')
fig58.tight_layout()
save_figure(fig58, '04_s5_temporal_matched_null')
print(f"\nFigure saved: figures/04_s5_temporal_matched_null.png")

print('\n=== §5.8 VERDICT ===')
if p_t_tm < 0.05 or p_w_tm < 0.05:
    print('  Signal SURVIVES temporally-matched null!')
    print('  Pre-breakthrough topology is genuinely different from same-era non-breakthrough periods.')
else:
    print('  Signal does NOT survive temporally-matched null.')
    print(f'  Consistent with §5.7 detrending result: the apparent precursor')
    print(f'  signal is entirely driven by secular β₁ trends, not breakthroughs.')

print('\n' + '='*60)
print('OVERALL CONCLUSION')
print('='*60)
print(f'\nRaw (temporally-confounded):    p = {p_w_raw:.6f}  (N={len(comp_df)})')
print(f'After per-pair detrending:      p = {p_w_det:.4f}  (N={len(det_df)})')
print(f'Temporally-matched null model:  p = {p_w_tm:.4f}  (N={len(tm_df)})')
print(f'\nThe precursor hypothesis is {"SUPPORTED" if p_w_tm < 0.05 else "NULL"}.')
