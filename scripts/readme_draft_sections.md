# README Draft Sections — Post Temporal Confound Discovery
# Apply these changes after NB04 completes and figures are finalized.

## ABSTRACT (replaces lines 13-28)

We apply persistent homology to the U.S. patent citation network (~8M utility patents, ~118M citations, 1976-2023) to test whether topological signatures in the knowledge landscape systematically precede technological breakthroughs.

**The central finding is null.** After correcting a null model bug, expanding the breakthrough catalog from 21 to 57 valid comparisons, and controlling for density via scale normalization, an apparent precursor signal emerges (Wilcoxon p < 0.001). However, this signal is entirely explained by a **temporal confound**: β₁ (H1 feature counts) declines universally at ~2.2 per year across all 28 CPC section pairs, driven by network maturation rather than breakthrough dynamics. The z-scores correlate almost perfectly with filing year (Spearman ρ = -0.921, p < 10⁻²⁰). After detrending β₁ per CPC pair to remove the secular decline, the signal vanishes (Wilcoxon p = 0.85, only 40% of breakthroughs showing positive residuals).

The analysis reveals three findings:

1. **The precursor hypothesis does not survive temporal detrending.** Across 57 breakthroughs with valid comparisons, the raw z-scores (mean = +1.12) are significant only because early breakthroughs (1980s, high-β₁ era) are compared against null models that average across all years including the lower-β₁ 2000s-2010s. Once each CPC pair's linear β₁ trend is removed, the pre-breakthrough topology is indistinguishable from any other period (t-test p = 0.72, Wilcoxon p = 0.85).

2. **The knowledge landscape is systematically flattening.** H1 feature counts decline globally from ~130 to ~70 (per CPC pair average, 1984-2023) while cross-field citation rates increase. After scale normalization eliminated the density confound (r = 0.970 → r = 0.036), this trend persists — it reflects genuine structural change, not a density artifact. The patent network is losing topological complexity as fields merge.

3. **Temporal trends in TDA on evolving networks produce spurious signals.** This project demonstrates that naively comparing topological features at different time periods against a temporally-uniform null model can produce highly significant but entirely artifactual results. This is a methodological cautionary tale for the growing field of temporal TDA.

These results are based on 57 valid comparisons from 65 curated breakthroughs across all 28 cross-section CPC pairs, with six confound robustness checks and temporal detrending.

---

## QUICK RESULTS (replaces lines 31-38)

| Question | Answer | Confidence |
|----------|--------|------------|
| Do topological loops change before breakthroughs? | **No** — apparent signal is a temporal artifact (detrended p=0.85) | High — 57 valid comparisons, temporal confound fully characterized |
| Is the knowledge space flattening? | **Yes** — β₁ declines ~2.2/year across all pairs | High — universal trend, survives density control |
| Does the decline relate to breakthroughs? | **No** — decline is secular, not breakthrough-specific | High — detrending removes all signal |
| Can topology predict breakthroughs? | **Not with current methods** — temporal trend confounds any model | Moderate — NB05 needs reinterpretation |

---

## FIGURE D CAPTION (replaces line 58)

*All 57 breakthroughs aligned at t=0 (filing year), with topology averaged across -10 to +5 years. The red line is the mean H1 feature count across breakthroughs. The blue band is the 95% CI from the matched null model. The red line sits within the null CI throughout — the pre-breakthrough topology is not significantly different from the null. The subtle elevation at t=-5 to t=-2 reflects the temporal confound: breakthroughs with precursor windows in the high-β₁ 1980s pull the mean upward. After per-pair linear detrending, this elevation disappears entirely (see §5.7).*

---

## FIGURE G CAPTION (replaces line 70)

*Z-scores for each breakthrough's pre-filing topology vs its matched null model (57 valid from 65-entry catalog). Left: H1 feature count z-scores. Right: Persistence entropy z-scores. Orange bars = above null, blue bars = below null. The pattern correlates almost perfectly with filing year (Spearman ρ = -0.921): early breakthroughs (1980s) show positive z-scores, recent ones (2010s) show negative z-scores. This is the temporal confound — not a genuine precursor signal. After per-pair detrending, z-scores are centered at zero with no systematic direction.*

---

## NB04 SECTION (replaces lines 131-151)

### Notebook 04: The Precursor Test

**The hypothesis test.** For each of 65 cataloged breakthroughs: (1) identify relevant CPC section pairs, (2) compute topological metrics in the 10 years before filing, (3) compare against matched null models (same CPC pairs, different time windows, 100 samples each). Aggregate via superposed epoch analysis (align all breakthroughs at t=0, average topology).

Statistical tests: one-sample t-test, Wilcoxon signed-rank, KS test, Holm-Bonferroni correction.

**Raw result:** 57 valid comparisons. β₁ z-scores: mean = +1.12, 75% positive. t-test p < 0.0001, Wilcoxon p < 0.0001. ROC AUC = 0.641. This appears to be a strong positive result.

**Temporal confound (§5.7 — THE CRITICAL CONTROL):** β₁ declines ~2.2/year across all 28 CPC pairs (network maturation). The matched null model samples uniformly from 1984-2018, creating a systematic temporal asymmetry: early breakthroughs are compared against later (lower-β₁) null periods, producing positive z-scores; late breakthroughs are compared against earlier (higher-β₁) null periods, producing negative z-scores. Z-score vs filing year: Spearman ρ = -0.921 (p < 10⁻²⁰). After detrending β₁ per CPC pair: t-test p = 0.72, Wilcoxon p = 0.85. **The precursor signal is null.**

**§5 Robustness Checks (Confound Analysis):** Seven confounds are controlled in §5:
- **§5.1 Examiner citations** (confound #1): ~74% of post-2018 citations are examiner-added. OLS partial-out test.
- **§5.2 Assignee self-citations** (confound #8): full topology re-run on 4 key pairs with intra-assignee edges removed.
- **§5.3 Prosecution lag** (confound #2): filing date vs grant date sensitivity test.
- **§5.4 Policy shocks** (confound #3): Alice (2014) and AIA (2011) discontinuity tests.
- **§5.5 Citation culture drift** (confound #5): mean_distance temporal correlation.
- **§5.6 Truncation bias** (confound #9): verified precursor windows unaffected.
- **§5.7 Temporal confound** (THE CRITICAL CONTROL): per-pair β₁ detrending eliminates apparent signal.

**§6 Leave-One-Out Robustness:** Jackknife sensitivity, minimum-window-count analysis, category-level breakdown.

---

## ETHICAL NOTE (replaces "On the journey" paragraph, line 199)

**On the journey from null to positive to null:** The initial analysis (March 2026, N=21) yielded a positive result (p=0.016) with all four tests surviving Holm-Bonferroni correction. During sample expansion, we discovered a **null model bug** that made single-section breakthroughs compare against global topology (~260 subclasses) instead of matching cross-section pairs. After fixing this bug and expanding to 65 breakthroughs (57 valid), the raw result appeared even stronger (p < 0.0001). However, the z-scores correlated almost perfectly with filing year (ρ = -0.921), revealing that the signal was driven by a **universal temporal decline in β₁** across all CPC pairs — not by breakthrough-specific topology. After per-pair detrending, all significance vanished (p = 0.85).

We report this trajectory in full because intellectual honesty requires it. The null result after detrending is the correct finding. The methodological lessons — that temporal trends in TDA on evolving networks can produce highly significant but entirely artifactual results, and that null model temporal matching is critical — are themselves valuable contributions. A null result obtained through rigorous methodology is more useful to the field than a positive result built on a confounded foundation.

---

## LIMITATIONS — ADD (after existing item 7)

8. **Temporal confound (the dominant limitation).** β₁ declines ~2.2/year across all 28 CPC pairs, driven by network maturation (increasing density, evolving citation practices, scale normalization effects). The matched null model samples uniformly from 1984-2018, creating a systematic temporal asymmetry that produces apparent precursor signals. After per-pair linear detrending, the precursor signal vanishes entirely. Any future analysis using temporal TDA on evolving networks must control for secular trends in topological features. This is not specific to patent networks — it applies to any growing network analyzed with persistent homology in sliding windows.
