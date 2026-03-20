# The Shape of Discovery

### Detecting Topological Precursors to Technological Breakthroughs in the USPTO Patent Citation Network

**Concept & Analytical Design:** Claude (Opus 4.6, Anthropic) via claude.ai
**Implementation:** Claude Code
**Facilitated by:** Christopher Ortiz
**Data:** USPTO Patent Citation Network via PatentsView (1976-2023)
**License:** MIT

---

## Abstract

We apply persistent homology to the U.S. patent citation network (~8M utility patents, ~118M citations, 1976-2023) to test whether topological signatures in the knowledge landscape systematically precede technological breakthroughs.

**The central finding is a null result.** Neither topological feature counts (beta-1, measuring loop-like structures in co-citation similarity space) nor persistence entropy systematically differ before breakthroughs compared to matched null models. No test survives Holm-Bonferroni correction: beta-1 t-test p=0.48, Wilcoxon p=0.73; persistence entropy t-test p=0.14, Wilcoxon p=0.15. Topological signatures are not reliable precursors to innovation.

The analysis does reveal two subsidiary findings:

1. **Topological features carry cross-domain predictive signal comparable to simple features.** In Leave-One-Group-Out classification (holding out entire CPC pairs), logistic regression with topology-only features achieves AUC 0.80, compared to 0.82 for simple distance features. Random forest reverses: topology-only 0.74 vs simple 0.68. Neither feature set consistently dominates, suggesting topology captures complementary rather than redundant information.

2. **The knowledge landscape is flattening over time.** Topological feature counts decline globally from 1985-2023 while simple cross-field citation rates increase. However, this trend is confounded by growing network density compressing cosine distances (see Limitations).

These results are based on 12 breakthroughs with valid comparisons out of 34 cataloged (8 skipped due to non-priority CPC pairs, 14 further reduced by matching requirements) and 10 cross-section CPC pairs spanning biotech, computing, energy, materials, and manufacturing.

---

## Quick Results

| Question | Answer | Confidence |
|----------|--------|------------|
| Do topological loops precede breakthroughs? | **No** (p=0.48) | Moderate -- 12 valid comparisons |
| Does topological complexity change before breakthroughs? | **No** (p=0.14) | Low -- does not survive correction |
| Can topology predict breakthroughs across domains? | **Partially** (LR AUC=0.80) | Low -- small dataset, shared feature pipeline |
| Is the knowledge space flattening? | **Possibly** -- confounded by density | Low -- density confound unresolved |

---

## Selected Figures

### Figure A: Topological Feature Counts Over Time (10 CPC Pairs)
![Beta-1 time series](figures/02_beta1_time_series.png)
*Each panel shows one cross-section CPC pair (e.g., Chemistry x Electricity = batteries/energy). The y-axis counts H1 features (loop-like structures in co-citation space). Most pairs show declining counts over the 38-year period. Think of it like this: the patent knowledge landscape had more distinct "rings" of cross-citing fields in the 1980s than it does today. Whether this reflects genuine structural change or growing network density compressing distances is an open question (see Figure F).*

### Figure B: Global Knowledge Landscape Topology
![Global topology](figures/02_global_topology.png)
*The full ~260-point CPC subclass distance matrix, not filtered to any pair. Red line: H1 feature count declining from ~960 to ~550. Blue dashed: persistence entropy relatively stable (~9.6-9.8 bits). The feature count drops but the distribution of feature lifetimes stays similar -- features become fewer but not fundamentally different in character.*

### Figure C: Where Is Topology Changing Fastest?
![Beta-1 heatmap](figures/02_beta1_heatmap.png)
*Heatmap of H1 feature counts across all 10 CPC pairs over time. Warmer colors = more features. The top-left (early years) is consistently warmer than the bottom-right (recent years) across nearly all pairs. AxC (Biotech/Pharma) and GxH (Semiconductors/Computing) show the most dramatic declines.*

### Figure D: Superposed Epoch Analysis (THE KEY RESULT)
![Superposed epoch](figures/04_superposed_epoch.png)
*All breakthroughs aligned at t=0 (filing year), with topology averaged across -10 to +5 years. The red line is the mean H1 feature count across breakthroughs. The blue band is the 95% confidence interval from the null model (random CPC pairs at random times). If topology preceded breakthroughs, the red line would rise above the blue band before t=0. It does not. This is the null result.*

### Figure E: Individual Breakthrough Topology
![Individual breakthroughs](figures/04_individual_beta1.png)
*H1 feature counts for 9 selected breakthroughs. Each colored line is a different CPC pair containing the breakthrough's technology section. Red dashed = filing year. Orange shading = 10-year precursor window. Some breakthroughs (Lithium-Ion Battery, PageRank) show clear declines in the precursor window; others (WiFi, GPU Computing) show mixed patterns. No consistent signal emerges.*

### Figure F: Density Confound Check
![Density confound](figures/02_density_confound.png)
*Left: Scatter plot of mean cosine distance vs H1 feature count, colored by year. They correlate at r=0.970 (p=5.4e-25) — nearly perfect. Right: Both metrics over time, tracking each other closely. This is the most important diagnostic in the project: the declining H1 counts are almost certainly driven by growing citation density compressing cosine distances, not by genuine structural change in the knowledge landscape. The "flattening" interpretation cannot be separated from this artifact without normalizing co-citation vectors per window.*

### Figure G: Effect Sizes Per Breakthrough
![Effect sizes](figures/04_effect_sizes.png)
*Z-scores for each breakthrough's pre-filing topology vs its matched null model. Left: H1 feature count z-scores. Right: Persistence entropy z-scores. Red bars = above null, blue bars = below null. The scatter is wide — some breakthroughs show z-scores of -25 while others show +12 — and no metric is statistically significant after Holm-Bonferroni correction. This is the null result visualized at the individual level.*

### Figure H: Predictive Model ROC Curves
![ROC curves](figures/05_roc_curves.png)
*Leave-One-Group-Out cross-validation: each fold holds out one entire CPC pair. Left: Logistic Regression. Right: Random Forest. Topology-only (red) vs simple distance features (blue) vs combined (green). LR: topology AUC=0.80 vs simple AUC=0.82 — nearly identical. RF reverses: topology 0.74 vs simple 0.68. Neither feature set consistently dominates, suggesting topology captures complementary information about the knowledge landscape.*

---

## Motivation

The patent citation network is one of the richest directed graphs of human knowledge in existence -- over 8 million utility patents connected by approximately 118 million citation edges, spanning nearly five decades. Prior work has used this network to predict emerging technologies (Erdi et al. 2013), early-identify significant patents (Mariani et al. 2018), and map firms' positions in technology space (Nakamura et al. 2023). These studies employ standard network science tools: PageRank, community detection, link prediction.

What has not been done -- to our knowledge as of March 2026 -- is the application of **persistent homology** to this network. Persistent homology detects topological features (connected components, loops, voids) that persist across multiple scales. It has been applied to financial markets, protein structure, cosmological mapping, and materials science -- but not to the patent citation graph, and not to breakthrough prediction.

Our contribution combines three elements not previously brought together:
1. Persistent homology as the analytical tool
2. The USPTO patent citation network as the dataset
3. Technological breakthrough prediction as the question

---

## Data

**PatentsView -- USPTO Office of the Chief Economist**

| Metric | Value |
|--------|-------|
| Total utility patents | 8,451,545 |
| Total citations | 118,011,718 |
| CPC mappings | 17,668,819 |
| Year range | 1976-2025 |
| Breakthrough catalog | 34 curated entries across 8 categories |

Source: PatentsView bulk download (CC BY 4.0), downloaded March 2026.

### Breakthrough Catalog

34 breakthroughs curated across 8 categories: biotech/pharma, computing, materials, energy, telecom, manufacturing, AI/ML, and cryptography/security. Each entry includes breakthrough patents, filing year, recognition year, CPC sections, and a brief description. Examples: CRISPR-Cas9, PageRank, lithium-ion battery, mRNA vaccines, WiFi, 3D printing.

The catalog is subjective. Different choices of what constitutes a "breakthrough" might yield different results. We acknowledge this limitation.

---

## Analyses

### Notebook 01: The Patent Atlas

Network characterization over time. Temporal snapshots (5-year windows, 1-year stride) from 1980-2023. Computes node count, edge count, density, mean degree, CPC mixing rate, and CPC entropy per window. Establishes the baseline before topology enters.

### Notebook 02: The Topological Clock

**The novel core.** Computes persistent homology on CPC subclass co-citation distance matrices. For each 5-year window, we build a ~260x260 co-citation matrix (rows = CPC subclasses, columns = CPC subclasses, values = citation counts), convert to cosine distance, and run Vietoris-Rips filtration via ripser. This produces persistence diagrams from which we extract H0/H1/H2 feature counts, persistence entropy, and other topological summaries.

10 priority CPC section pairs: AxC (Biotech/Pharma), AxG (Medical Devices), CxG (Materials/Sensors), CxH (Batteries/Energy), GxH (Semiconductors/Computing), BxG (Manufacturing Tech), BxH (Automation/Robotics), AxH (Health Tech/Wearables), CxB (Chemical Engineering), FxH (Electromechanical/Power).

**Key finding:** H1 feature counts decline globally over time while CPC mixing rate increases. However, this is potentially confounded by growing network density (see Limitations).

### Notebook 03: The Breakthrough Catalog

Curates 34 breakthroughs, validates them against the patent database, maps to CPC sections, and computes citation statistics.

### Notebook 04: The Precursor Test

**The hypothesis test.** For each breakthrough: (1) identify relevant CPC section pairs, (2) compute topological metrics in the 10 years before filing, (3) compare against matched null models (same CPC pair at non-breakthrough times). Aggregate via superposed epoch analysis (align all breakthroughs at t=0, average topology).

Statistical tests: one-sample t-test, Wilcoxon signed-rank, Holm-Bonferroni correction for 4 comparisons, Cohen's d on raw values.

**Result:** Null result across all metrics. H1 feature count: t-test p=0.48, Wilcoxon p=0.73. Persistence entropy: t-test p=0.14, Wilcoxon p=0.15. No test survives Holm-Bonferroni correction. Topological signatures do not systematically precede breakthroughs.

### Notebook 05: The Predictability Horizon

Leave-One-Group-Out cross-validation by CPC pair. Features: topological (H0, H1, H2, persistence entropy, max persistence, long-lived features) and simple (active class count, mean/median cosine distance). Models: logistic regression and random forest.

**Result:** LR topology-only AUC=0.80, simple-only AUC=0.82, combined AUC=0.82. RF topology-only AUC=0.74, simple-only AUC=0.68, combined AUC=0.73. Dataset is small (350 rows, 10 groups, 60% positive rate) and results should be interpreted cautiously. LOGO tests cross-domain generalization, not temporal forecasting.

---

## Limitations

These are critical for honest interpretation:

1. **Density confound.** The co-citation matrix grows denser from 1985-2023, compressing cosine distances toward zero and mechanically reducing topological features. Global mean distance and H1 feature count correlate at r=0.970 (p=5.4e-25). The declining H1 counts are almost certainly an artifact of this density growth rather than genuine "knowledge flattening." Normalizing co-citation vectors per window would control for this but requires recomputation.

2. **Feature counts, not Betti numbers.** What we call "beta-1" is the total count of H1 features born across the entire Vietoris-Rips filtration, not the Betti number at a specific threshold. Standard Betti numbers count features alive simultaneously at one filtration value. Our numbers are not directly comparable to the TDA literature.

3. **Directionality lost.** The co-citation matrix is symmetrized before computing distances. H1 features represent ring-like arrangements in *similarity* space, not directed citation cycles between fields.

4. **Small sample.** 12 breakthroughs with valid comparisons (NB04), 10 CPC groups (NB05). Statistical power is limited. The effective sample size is further reduced by non-independence (breakthroughs share topology pairs, mean 5.2 matching pairs per breakthrough).

5. **8 breakthroughs skipped.** EUV Lithography, CAR-T, MapReduce, Graphene, iPhone, 4G LTE, CRISPR, and mRNA vaccines were skipped because their CPC pairs weren't in our priority set. These include arguably the most important recent biotech breakthroughs.

6. **Examiner-added citations.** ~74% of citations in post-2018 data are added by patent examiners, not inventors. These represent institutional knowledge rather than inventor awareness. This confound affects all patent citation analyses, not just ours.

7. **Overlapping windows.** 5-year windows with 1-year stride share 80% of their data. This induces strong autocorrelation in time series and inflates the apparent smoothness of trends.

8. **Simple features are not independent baselines.** The "simple" features in NB05 (mean/median cosine distance) derive from the same co-citation matrix as the topological features. A truly independent baseline would use raw citation counts or patent volume.

---

## Key References

- Erdi, P. et al. (2013). Prediction of emerging technologies based on analysis of the US patent citation network. *Scientometrics*, 95(1), 225-242.
- Mariani, M.S. et al. (2018). Early identification of important patents: Design and validation of citation network metrics. *Technological Forecasting and Social Change*, 146, 644-654.
- Nakamura, H. et al. (2023). Mapping firms' locations in technological space: A topological analysis of patent statistics. *Research Policy*, 52(7), 104811.
- Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255-308.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Otter, N. et al. (2017). A roadmap for the computation of persistent homology. *EPJ Data Science*, 6, 17.
- Zomorodian, A. & Carlsson, G. (2005). Computing persistent homology. *Discrete & Computational Geometry*, 33(2), 249-274.

---

## Ethical Note

This project analyzes historical patent data for scientific understanding. **Nothing in this analysis should be interpreted as investment advice or technology forecasting guidance.** The breakthrough catalog reflects subjective judgments about what constitutes a major technological advance. Different catalogs might yield different results.

**On AI authorship:** The analytical framework, methodology, code, and written analysis were conceived and implemented by Claude (Opus 4.6, Anthropic). Christopher Ortiz facilitated the project, provided compute resources, and guided the research direction. We are transparent about this because intellectual honesty requires it. The AI contribution is documented in the description, not the author list, following current academic conventions for AI-assisted research.

**On null results:** The central hypothesis test yields a null result across all metrics (beta-1 p=0.48, persistence entropy p=0.14, none surviving Holm-Bonferroni correction). We report this prominently -- in the abstract, the quick results table, and throughout the analysis. A null result is a valid scientific finding: it means that topological signatures in the patent knowledge space do not reliably precede breakthroughs. This tells us something real about the nature of innovation.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/emergent-inquiry/the-shape-of-discovery.git
cd the-shape-of-discovery

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download PatentsView bulk data (~several GB)
python 00_data_acquisition.py

# 4. Run tests
pytest tests/

# 5. Run notebooks in order (NB02 and NB04 are compute-intensive)
jupyter nbconvert --execute --to notebook --inplace 01_patent_atlas.ipynb
jupyter nbconvert --execute --to notebook --inplace 02_topological_clock.ipynb
jupyter nbconvert --execute --to notebook --inplace 03_breakthrough_catalog.ipynb
jupyter nbconvert --execute --to notebook --inplace 04_precursor_test.ipynb
jupyter nbconvert --execute --to notebook --inplace 05_predictability_horizon.ipynb
```

**Hardware requirements:** 12 GB RAM minimum, 16 GB recommended. The topology computation caches results to disk; first run of NB02 takes ~4 hours, subsequent runs load from cache in seconds. NB04's null model computation takes ~2-6 hours depending on how many CPC pairs require fresh topology.

---

*Analytical framework designed by Claude (Opus 4.6, Anthropic). Implementation by Claude Code. Facilitated by Christopher Ortiz.*
