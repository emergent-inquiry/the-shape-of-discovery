# %% §5.7 Temporal Confound — The Critical Control
# β₁ declines ~2.2/year across ALL CPC pairs (secular trend from network
# maturation, not breakthrough dynamics). The matched null samples uniformly
# from 1984-2018, so early breakthroughs (high-β₁ era) always appear
# "elevated" and late ones always appear "depressed."
#
# This cell controls for the temporal trend two ways:
#   (a) Show the raw z-score vs filing_year correlation
#   (b) Detrend β₁ per CPC pair (remove linear year trend), recompute test

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

print('=== §5.7: TEMPORAL CONFOUND — THE CRITICAL CONTROL ===\n')

# --- (a) Raw z-score vs filing year ---
r_p, p_p = pearsonr(comp_df['filing_year'], comp_df['z_beta1'])
r_s, p_s = spearmanr(comp_df['filing_year'], comp_df['z_beta1'])
print('Raw z_beta1 vs filing_year:')
print(f'  Pearson  r = {r_p:.3f}  (p = {p_p:.1e})')
print(f'  Spearman ρ = {r_s:.3f}  (p = {p_s:.1e})')
print(f'  → {abs(r_p**2)*100:.0f}% of z-score variance explained by filing year alone\n')

# --- (b) Detrend β₁ per CPC pair ---
# For each pair, fit β₁ ~ year and compute residuals
_pair_trends = {}
for pk, topo in topology_results.items():
    if 'window_end' not in topo.columns or 'beta_1' not in topo.columns:
        continue
    _X = topo['window_end'].values.reshape(-1, 1)
    _y = topo['beta_1'].values
    _reg = LinearRegression().fit(_X, _y)
    _resid = _y - _reg.predict(_X)
    _pair_trends[pk] = _reg
    topo['beta_1_detrended'] = _resid

print(f'Linear trends fitted for {len(_pair_trends)} CPC pairs')
_slopes = [reg.coef_[0] for reg in _pair_trends.values()]
print(f'Mean β₁ slope: {np.mean(_slopes):.2f}/year '
      f'(range: {min(_slopes):.2f} to {max(_slopes):.2f})')
print(f'All {sum(s < 0 for s in _slopes)}/{len(_slopes)} pairs show declining β₁\n')

# Recompute pre-breakthrough metric using detrended β₁
_detrended_rows = []
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
        _detrended_rows.append({
            'name': row['name'],
            'filing_year': row['filing_year'],
            'z_beta1_raw': row['z_beta1'],
            'pre_beta1_detrended': np.mean(pre_resid),
        })

_det_df = pd.DataFrame(_detrended_rows)
print(f'Detrended comparisons: N = {len(_det_df)}')
print(f'Mean detrended β₁: {_det_df.pre_beta1_detrended.mean():.3f}')
print(f'Median detrended β₁: {_det_df.pre_beta1_detrended.median():.3f}')
print(f'Positive: {(_det_df.pre_beta1_detrended > 0).sum()}/{len(_det_df)} '
      f'({100*(_det_df.pre_beta1_detrended > 0).mean():.0f}%)\n')

# Statistical tests on detrended values
_t_det, _p_t_det = stats.ttest_1samp(_det_df['pre_beta1_detrended'], 0)
try:
    _w_det, _p_w_det = stats.wilcoxon(_det_df['pre_beta1_detrended'])
except ValueError:
    _p_w_det = 1.0
print('Detrended tests (H₀: pre-breakthrough β₁ residual = 0):')
print(f'  t-test:   t = {_t_det:.3f},  p = {_p_t_det:.4f}')
print(f'  Wilcoxon: W,          p = {_p_w_det:.4f}')

# Verify detrending removed temporal correlation
_r_det, _p_det = pearsonr(_det_df['filing_year'], _det_df['pre_beta1_detrended'])
print(f'\nDetrended β₁ vs filing_year: r = {_r_det:.3f} (p = {_p_det:.3f}) — should be ~0\n')

# --- By era ---
_early = _det_df[_det_df.filing_year <= 1995]
_mid = _det_df[(_det_df.filing_year > 1995) & (_det_df.filing_year <= 2005)]
_late = _det_df[_det_df.filing_year > 2005]
print('By era (detrended):')
for _name, _era in [('Early ≤1995', _early), ('Mid 1996-2005', _mid), ('Late >2005', _late)]:
    if len(_era) >= 5:
        _t_e, _p_e = stats.ttest_1samp(_era['pre_beta1_detrended'], 0)
        print(f'  {_name:15s} (N={len(_era):2d}): mean = {_era.pre_beta1_detrended.mean():+.2f}, '
              f't = {_t_e:.2f}, p = {_p_e:.3f}')
    else:
        print(f'  {_name:15s} (N={len(_era):2d}): mean = {_era.pre_beta1_detrended.mean():+.2f} '
              '(too few for t-test)')

# --- Figure: Temporal Confound Diagnostic ---
fig57, axes57 = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Raw z-scores vs filing year
ax = axes57[0]
colors = [PALETTE['orange'] if z > 0 else PALETTE['blue'] for z in _det_df['z_beta1_raw']]
ax.scatter(_det_df['filing_year'], _det_df['z_beta1_raw'], c=colors, alpha=0.7, s=40)
_x_fit = np.linspace(_det_df['filing_year'].min(), _det_df['filing_year'].max(), 100)
_reg_line = LinearRegression().fit(
    _det_df[['filing_year']].values, _det_df['z_beta1_raw'].values
)
ax.plot(_x_fit, _reg_line.predict(_x_fit.reshape(-1, 1)),
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
              for z in _det_df['pre_beta1_detrended']]
ax.scatter(_det_df['filing_year'], _det_df['pre_beta1_detrended'],
           c=colors_det, alpha=0.7, s=40)
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('Breakthrough Filing Year')
ax.set_ylabel('Detrended β₁ (residual)')
ax.set_title(f'(b) Detrended: r = {_r_det:.2f}')

# Panel 3: Distribution of detrended residuals
ax = axes57[2]
ax.hist(_det_df['pre_beta1_detrended'], bins=20, color=PALETTE['blue'],
        alpha=0.7, edgecolor='white')
ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Expected under H₀')
ax.axvline(_det_df['pre_beta1_detrended'].mean(), color=PALETTE['orange'],
           linewidth=2, label=f'Mean = {_det_df.pre_beta1_detrended.mean():.2f}')
ax.set_xlabel('Detrended β₁ (residual)')
ax.set_ylabel('Count')
ax.set_title(f'(c) Distribution (Wilcoxon p = {_p_w_det:.3f})')
ax.legend(fontsize=9)

fig57.suptitle('§5.7: Temporal Confound — β₁ Secular Trend Drives Apparent Signal',
               fontsize=13, fontweight='bold')
fig57.tight_layout()
save_figure(fig57, '04_s5_temporal_confound')

# --- Verdict ---
print('\n=== §5.7 VERDICT ===')
if _p_t_det < 0.05 or _p_w_det < 0.05:
    print('  Signal SURVIVES detrending — topology is genuinely elevated before breakthroughs')
    print('  beyond what the secular β₁ trend explains.')
else:
    print('  ✗ Signal does NOT survive detrending.')
    print(f'  The raw significance (p < 0.001) is a TEMPORAL ARTIFACT.')
    print(f'  β₁ declines {abs(np.mean(_slopes)):.1f}/year across all CPC pairs.')
    print(f'  Breakthroughs from the 1980s fall in the high-β₁ era;')
    print(f'  recent breakthroughs fall in the low-β₁ era.')
    print(f'  After removing this secular trend, pre-breakthrough topology')
    print(f'  is indistinguishable from any other period (p = {_p_w_det:.3f}).')
    print()
    print('  IMPLICATION: The patent citation network does exhibit systematic')
    print('  topological evolution over time, but this evolution reflects network')
    print('  growth and citation practice changes — not breakthrough dynamics.')
