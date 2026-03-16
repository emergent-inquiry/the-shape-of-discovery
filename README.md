# The Shape of Discovery

### Detecting Topological Precursors to Technological Breakthroughs in the USPTO Patent Citation Network

**Concept & Analytical Design:** Claude (Opus 4.6, Anthropic) via claude.ai
**Implementation:** Claude Code
**Facilitated by:** Christopher Ortiz
**Data:** USPTO Patent Citation Network via PatentsView (1976–2023)
**License:** MIT

---

## Abstract

When a technological breakthrough occurs, it doesn't emerge from a vacuum. In the years before a major advance, research threads converge, patents begin citing across previously disconnected fields, and the structure of the knowledge graph changes in measurable ways. But do these structural changes follow consistent topological patterns — and can they be detected *before* the breakthrough is recognized?

This project applies persistent homology — a tool from algebraic topology that detects multi-scale structural features like loops, voids, and connected components — to the U.S. patent citation network (~8M utility patents, ~100M citations, 1976–2023). We compute topological invariants (Betti numbers β₀, β₁, β₂) on CPC-class-level subgraphs in sliding time windows and test whether these invariants show systematic changes in the years preceding ~30–50 known technological breakthroughs.

We organize the work around five questions:

1. **How has the structure of the patent citation network evolved over 47 years?** We map basic network properties and cross-field citation patterns over time — the baseline before topology enters.

2. **What does the topological structure of the knowledge graph look like, and how does it change?** We compute persistent homology on CPC section-pair subgraphs in sliding windows, tracking when new topological features (loops, voids) emerge and persist.

3. **What constitutes a "technological breakthrough" in patent data?** We build a curated catalog of ~30–50 breakthroughs with identifiable patents, filing dates, and CPC contexts.

4. **Do topological signatures systematically precede breakthroughs?** This is the core hypothesis test. We compare pre-breakthrough topology against a matched null model.

5. **Can topological features predict where breakthroughs will land?** We train a model on historical breakthroughs and test whether persistent homology adds information beyond simpler citation metrics.

Each analysis stands on its own. Together, they test whether humanity's knowledge graph *anticipates* its own breakthroughs — or whether innovation is topologically invisible until it arrives.

---

## Quick Results

*[To be populated after analysis]*

---

## Selected Figures

*[To be populated after analysis]*

---

## Motivation

The patent citation network is one of the richest directed graphs of human knowledge in existence — over 8 million utility patents connected by approximately 100 million citation edges, spanning nearly five decades. Prior work has used this network to predict emerging technologies (Érdi et al. 2013), early-identify significant patents (Mariani et al. 2018), and map firms' positions in technology space (Nakamura et al. 2023). These studies employ standard network science tools: PageRank, community detection, link prediction.

What has not been done — to our knowledge as of March 2026 — is the application of **persistent homology** to this network. Persistent homology detects topological features (connected components, loops, voids) that persist across multiple scales, providing information about the *shape* of the data that purely metric or spectral methods miss. It has been applied successfully to financial market phase detection, protein structure analysis, cosmological structure mapping, and materials science — but not to the patent citation graph, and not to the problem of breakthrough prediction.

Our contribution is the combination of three elements that have not been brought together before:
1. Persistent homology as the analytical tool
2. The USPTO patent citation network as the dataset
3. Technological breakthrough prediction as the question

We note that the novelty claim rests on our understanding of the literature as of early 2026 and may not capture all prior work. The value of the project does not depend on strict priority — even if individual techniques have precedents, the integration of all five analyses into a single framework applied to the patent network is, to our knowledge, new.

---

## Data

### Source

**PatentsView — USPTO Office of the Chief Economist**

- **URL:** https://patentsview.org/downloads/data-downloads
- **Bulk Download:** Tab-delimited flat files, CC BY 4.0
- **Coverage:** Granted U.S. patents, 1976–present
- **Key Tables:** g_patent, g_us_patent_citation, g_cpc_current
- **Expected Scale:** ~8M utility patents, ~100M citation edges

### CPC Classification System

The Cooperative Patent Classification (CPC) organizes patents into 8 top-level sections:

| Section | Domain | Example Technologies |
|---------|--------|---------------------|
| A | Human Necessities | Agriculture, food, medicine, clothing |
| B | Performing Operations; Transporting | Manufacturing, printing, vehicles |
| C | Chemistry; Metallurgy | Organic chemistry, materials, biotech |
| D | Textiles; Paper | Fibers, weaving, paper production |
| E | Fixed Constructions | Building, mining, earth drilling |
| F | Mechanical Engineering | Engines, pumps, weapons, heating |
| G | Physics | Instruments, computing, optics, nuclear |
| H | Electricity | Electronics, circuits, power, telecom |

Cross-section citations (e.g., a Chemistry patent citing a Physics patent) are the primary signal of interest — these edges bridge previously disconnected knowledge domains.

---

## Analyses

*[Detailed methodology sections for each notebook — to be populated during implementation]*

---

## Project Structure

```
the-shape-of-discovery/
├── CLAUDE.md
├── README.md
├── LICENSE
├── requirements.txt
├── 00_data_acquisition.py
├── 01_patent_atlas.ipynb
├── 02_topological_clock.ipynb
├── 03_breakthrough_catalog.ipynb
├── 04_precursor_test.ipynb
├── 05_predictability_horizon.ipynb
├── src/
│   ├── __init__.py
│   ├── fetch.py
│   ├── graph.py
│   ├── topology.py
│   ├── breakthroughs.py
│   ├── metrics.py
│   ├── nullmodel.py
│   ├── plotting.py
│   └── utils.py
├── data/
│   ├── breakthroughs/
│   └── external/
├── figures/
├── tests/
│   ├── test_topology.py
│   ├── test_graph.py
│   └── test_structure.py
└── scripts/
    └── download_patentsview.sh
```

---

## Key References

- Érdi, P. et al. (2013). Prediction of emerging technologies based on analysis of the US patent citation network. *Scientometrics*, 95(1), 225–242.
- Mariani, M.S. et al. (2018). Early identification of important patents: Design and validation of citation network metrics. *Technological Forecasting and Social Change*, 146, 644–654.
- Nakamura, H. et al. (2023). Mapping firms' locations in technological space: A topological analysis of patent statistics. *Research Policy*, 52(7), 104811.
- Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255–308.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Otter, N. et al. (2017). A roadmap for the computation of persistent homology. *EPJ Data Science*, 6, 17.
- Zomorodian, A. & Carlsson, G. (2005). Computing persistent homology. *Discrete & Computational Geometry*, 33(2), 249–274.

---

## Ethical Note

This project analyzes historical patent data for scientific understanding. **Nothing in this analysis should be interpreted as investment advice or technology forecasting guidance.** The breakthrough catalog reflects subjective judgments about what constitutes a major technological advance. Different catalogs might yield different results. The predictive model in Notebook 5 is an analytical exercise, not an operational forecasting tool.

If the topological precursor test yields a null result — meaning breakthroughs arrive without detectable structural shadows in the citation network — that finding is reported prominently and honestly. A null result would mean that innovation is topologically invisible until it arrives, which is itself a meaningful statement about the nature of discovery.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download PatentsView bulk data (~several GB)
python 00_data_acquisition.py

# 3. Run tests
pytest tests/

# 4. Run notebooks in order
jupyter notebook 01_patent_atlas.ipynb
```

---

*Analytical framework designed by Claude (Opus 4.6, Anthropic). Implementation by Claude Code. Facilitated by Christopher Ortiz.*
