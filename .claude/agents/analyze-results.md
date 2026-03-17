# analyze-results

Review completed notebook outputs and assess findings.

## Instructions

After notebooks have been executed:

1. Read the notebook to examine cell outputs, figures, and computed values
2. Check that all expected figures were generated in `figures/`
3. Assess whether results are reasonable:
   - **Notebook 01 (Patent Atlas)**: Network growth should show exponential increase. CPC mixing rate should trend upward. Degree distribution should follow power law.
   - **Notebook 02 (Topological Clock)**: Betti numbers should vary across time windows. Look for β₁ spikes that might correlate with known breakthrough periods. Persistence entropy should generally increase over time.
   - **Notebook 03 (Breakthrough Catalog)**: Verify breakthrough patents exist in dataset. Citation fan-out should show characteristic S-curves. Cross-section precursors validate the topology approach.
   - **Notebook 04 (Precursor Test)**: Compare pre-breakthrough topology against null model. Report whether the difference is statistically significant or a null result.
   - **Notebook 05 (Predictability Horizon)**: Compare AUC-ROC for topology-only vs simple-only vs combined models. Report honestly if topology adds no signal.
4. Flag any anomalies: impossible values, empty figures, suspiciously uniform distributions
5. If null results are detected, verify they're genuine (not bugs) — this is a valid finding

## Intellectual Honesty

- Null results are reported prominently, never buried
- Effect sizes are reported alongside p-values
- Limitations of CPC-level subgraph analysis are acknowledged
- The breakthrough catalog is acknowledged as subjective
