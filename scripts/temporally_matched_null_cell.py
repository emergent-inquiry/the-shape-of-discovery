# %% §5.8 Temporally-Matched Null Model — Non-Parametric Temporal Control
# §5.7 showed the signal vanishes after LINEAR detrending. But what if the
# β₁ decline is non-linear? A temporally-matched null controls for time
# without assuming any functional form.
#
# For each breakthrough at filing year Y:
#   - Pre-breakthrough metric: same as main analysis (mean β₁ across
#     matching pairs in precursor window)
#   - Null: same pairs' β₁ at years within ±HALFWIDTH of Y, excluding
#     the breakthrough period. No trend assumption — just "nearby" years.

TEMPORAL_HALFWIDTH = 7  # years around filing year for null sampling

print('=== §5.8: TEMPORALLY-MATCHED NULL MODEL ===')
print(f'Null window: ±{TEMPORAL_HALFWIDTH} years around each breakthrough\n')

tmn_rows = []
for _, row in comp_df.iterrows():
    bt = next((b for b in breakthroughs if b.name == row['name']), None)
    if bt is None:
        continue

    Y = bt.filing_year
    start, end = get_precursor_window(bt, years_before=10)

    # Matching pairs (same as main analysis)
    matching_pairs = [
        (a, b) for a, b in ALL_PAIRS
        if any(s in [a, b] for s in bt.cpc_sections)
    ]

    # Pre-breakthrough β₁: mean across matching pairs × precursor years
    pre_vals = []
    for sa, sb in matching_pairs:
        pk = f'{sa}x{sb}'
        topo = topology_results.get(pk)
        if topo is None or 'beta_1' not in topo.columns:
            continue
        mask = (topo['window_end'] >= start) & (topo['window_end'] <= end)
        pre = topo[mask]
        if len(pre) > 0:
            pre_vals.append(pre['beta_1'].mean())
    if not pre_vals:
        continue
    pre_metric = np.mean(pre_vals)

    # Temporally-matched null years: within ±HALFWIDTH of Y,
    # excluding the breakthrough period [Y-3, recognition+3]
    exclude_start = Y - 3
    exclude_end = bt.recognition_year + 3
    null_years = [
        y for y in range(max(1984, Y - TEMPORAL_HALFWIDTH),
                         min(2024, Y + TEMPORAL_HALFWIDTH + 1))
        if y < exclude_start or y > exclude_end
    ]

    if len(null_years) < 3:
        continue

    # Null distribution: for each null year, average β₁ across matching pairs
    null_vals = []
    for ny in null_years:
        pair_vals = []
        for sa, sb in matching_pairs:
            pk = f'{sa}x{sb}'
            topo = topology_results.get(pk)
            if topo is None or 'beta_1' not in topo.columns:
                continue
            row_match = topo[topo['window_end'] == ny]
            if len(row_match) > 0:
                pair_vals.append(row_match['beta_1'].values[0])
        if pair_vals:
            null_vals.append(np.mean(pair_vals))

    if len(null_vals) < 3:
        continue

    null_mean = np.mean(null_vals)
    null_std = np.std(null_vals, ddof=1)
    z_tmn = (pre_metric - null_mean) / null_std if null_std > 0 else 0.0

    tmn_rows.append({
        'name': row['name'],
        'filing_year': Y,
        'z_beta1_original': row['z_beta1'],
        'z_beta1_tmn': z_tmn,
        'pre_beta1': pre_metric,
        'null_mean': null_mean,
        'null_std': null_std,
        'n_null_years': len(null_years),
    })

tmn_df = pd.DataFrame(tmn_rows)
print(f'Temporally-matched comparisons: N = {len(tmn_df)}')
print(f'Mean TMN z-score: {tmn_df.z_beta1_tmn.mean():.3f}')
print(f'Median TMN z-score: {tmn_df.z_beta1_tmn.median():.3f}')
print(f'Positive: {(tmn_df.z_beta1_tmn > 0).sum()}/{len(tmn_df)} '
      f'({100*(tmn_df.z_beta1_tmn > 0).mean():.0f}%)')
print(f'Mean null years per breakthrough: {tmn_df.n_null_years.mean():.1f}\n')

# Statistical tests
_t_tmn, _p_t_tmn = stats.ttest_1samp(tmn_df['z_beta1_tmn'], 0)
try:
    _w_tmn, _p_w_tmn = stats.wilcoxon(tmn_df['z_beta1_tmn'])
except ValueError:
    _p_w_tmn = 1.0

print('Temporally-matched null tests (H₀: z = 0):')
print(f'  t-test:   t = {_t_tmn:.3f},  p = {_p_t_tmn:.4f}')
print(f'  Wilcoxon:              p = {_p_w_tmn:.4f}')

# Verify temporal correlation is removed
_r_tmn, _p_r_tmn = pearsonr(tmn_df['filing_year'], tmn_df['z_beta1_tmn'])
print(f'\nTMN z-score vs filing_year: r = {_r_tmn:.3f} (p = {_p_r_tmn:.3f})')
print(f'(Original z vs filing_year: ρ = {r_s:.3f})\n')

# Compare with original and detrended
print('=== Comparison of three approaches ===')
print(f'  {"Method":<30s} {"Mean z":>8s} {"p (Wilcoxon)":>14s} {"r(year)":>8s}')
print(f'  {"─"*30} {"─"*8} {"─"*14} {"─"*8}')
print(f'  {"Original (uniform null)":<30s} {comp_df.z_beta1.mean():>+8.3f} '
      f'{"< 0.0001":>14s} {r_s:>+8.3f}')
if '_p_w_det' in dir():
    print(f'  {"§5.7 Linear detrending":<30s} '
          f'{_det_df.pre_beta1_detrended.mean():>+8.3f} '
          f'{_p_w_det:>14.4f} {_r_det:>+8.3f}')
print(f'  {"§5.8 Temporal matching":<30s} {tmn_df.z_beta1_tmn.mean():>+8.3f} '
      f'{_p_w_tmn:>14.4f} {_r_tmn:>+8.3f}')

# --- Figure: TMN diagnostic ---
fig58, axes58 = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Original vs TMN z-scores
ax = axes58[0]
ax.scatter(tmn_df['z_beta1_original'], tmn_df['z_beta1_tmn'],
           alpha=0.6, s=40, c=PALETTE['blue'])
lims = [min(tmn_df['z_beta1_original'].min(), tmn_df['z_beta1_tmn'].min()) - 0.5,
        max(tmn_df['z_beta1_original'].max(), tmn_df['z_beta1_tmn'].max()) + 0.5]
ax.plot(lims, [0, 0], 'r--', linewidth=1, alpha=0.5)
ax.plot([0, 0], lims, 'r--', linewidth=1, alpha=0.5)
ax.set_xlabel('Original z-score (uniform null)')
ax.set_ylabel('TMN z-score (era-matched null)')
ax.set_title('(a) Original vs Temporally-Matched')

# Panel 2: TMN z-scores vs filing year
ax = axes58[1]
colors_tmn = [PALETTE['orange'] if z > 0 else PALETTE['blue']
              for z in tmn_df['z_beta1_tmn']]
ax.scatter(tmn_df['filing_year'], tmn_df['z_beta1_tmn'],
           c=colors_tmn, alpha=0.7, s=40)
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('Breakthrough Filing Year')
ax.set_ylabel('TMN z-score')
ax.set_title(f'(b) TMN z vs Year: r = {_r_tmn:.2f}')

# Panel 3: Distribution
ax = axes58[2]
ax.hist(tmn_df['z_beta1_tmn'], bins=20, color=PALETTE['blue'],
        alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Expected under H₀')
ax.axvline(tmn_df['z_beta1_tmn'].mean(), color=PALETTE['orange'],
           linewidth=2, label=f'Mean = {tmn_df.z_beta1_tmn.mean():.2f}')
ax.set_xlabel('TMN z-score')
ax.set_ylabel('Count')
ax.set_title(f'(c) Distribution (Wilcoxon p = {_p_w_tmn:.3f})')
ax.legend(fontsize=9)

fig58.suptitle('§5.8: Temporally-Matched Null — Non-Parametric Temporal Control',
               fontsize=13, fontweight='bold')
fig58.tight_layout()
save_figure(fig58, '04_s5_temporal_matched_null')

# --- Verdict ---
print('\n=== §5.8 VERDICT ===')
if _p_t_tmn < 0.05 or _p_w_tmn < 0.05:
    print('  Signal SURVIVES temporal matching.')
    print('  The linear detrending in §5.7 may have been too aggressive.')
    print('  There may be a genuine non-linear topological precursor.')
else:
    print('  ✗ Signal does NOT survive temporal matching either.')
    print(f'  Consistent with §5.7 linear detrending (p = {_p_w_det:.3f}).')
    print(f'  Both linear and non-parametric controls agree:')
    print(f'  pre-breakthrough topology is indistinguishable from nearby years.')
    print(f'  The precursor hypothesis is robustly null.')
