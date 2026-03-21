# CC Task: Expand Breakthrough Sample Size

## Context

The Shape of Discovery found a statistically significant result: topological 
simplification precedes breakthroughs (β₁ Wilcoxon p=0.016, persistence entropy 
p=0.0001, both surviving Holm-Bonferroni). But N=21 valid comparisons from 34 
cataloged breakthroughs is moderate. We need to make this robust enough that 
sample size cannot be used to dismiss the finding.

Three expansion strategies, in order of priority. Do all three.

---

## Strategy 1: Recover Excluded Breakthroughs

13 of 34 breakthroughs were excluded from valid comparisons in NB04. Before 
adding new ones, recover what we already have.

### Task 1.1: Diagnose exclusions

For each of the 13 excluded breakthroughs, print:
- Name
- Filing year
- CPC sections involved
- Reason for exclusion (be specific: insufficient topology windows? No matching 
  null windows? Filing year before topology start? CPC pair not computed? No 
  valid z-score?)

### Task 1.2: Fix recoverable exclusions

Common fixes:
- If excluded because filing year is before 1990 (not enough precursor windows): 
  extend topology computation back to 1980 for the relevant CPC pairs if not 
  already done. The co-citation matrix should have data back to at least 1984.
- If excluded because CPC pair wasn't in the original 10 priority pairs: we now 
  compute all 28 pairs, so this should already be resolved. Verify.
- If excluded because null model had insufficient non-breakthrough windows: relax 
  the matching criteria slightly (e.g., allow null windows within ±2 years of 
  another breakthrough instead of requiring zero overlap).
- If excluded because of missing CPC mapping: check if the breakthrough patent 
  has a secondary CPC classification that maps to a computed pair.

### Task 1.3: Re-run NB04 with recovered breakthroughs

After fixes, re-run the precursor test. Report the new N and whether the 
statistical significance improves, holds, or weakens.

---

## Strategy 2: Expand the Curated Catalog

Add 30-40 additional well-documented technological breakthroughs to 
data/breakthroughs/. Each entry needs the same JSON structure as existing ones.

### Task 2.1: Add these breakthroughs

Research each one using web search to find the correct patent numbers, filing 
years, and CPC classifications. Accuracy matters — verify patent numbers exist 
in the PatentsView data.

**Biotech/Pharma (add ~8-10):**
- Immunotherapy checkpoint inhibitors: PD-1 antibody (Nivolumab/Opdivo, 
  Tasuku Honjo patents, ~2002-2005 filing)
- PD-L1 antibody (Atezolizumab, ~2004-2007 filing)
- Gene therapy AAV vectors (key Samulski/Wilson patents, ~1990s-2000s)
- Statin drugs (Merck's lovastatin patents, ~1979-1980 filing)
- Drug-eluting stents (Johnson & Johnson/Cordis, ~1997-2000 filing)
- PCR (Polymerase Chain Reaction — Mullis patents, ~1985-1987 filing)
- Antisense oligonucleotides (Isis/Ionis patents, ~1989-1993)
- RNAi/siRNA therapeutics (Tuschl/Alnylam patents, ~2001-2003)
- Protease inhibitors for HIV (key patents ~1993-1996)

**Computing/Electronics (add ~8-10):**
- CMOS image sensors (Eric Fossum's patents, ~1993-1995 filing)
- USB (Intel/Compaq/Microsoft consortium patents, ~1996-1998)
- Bluetooth (Ericsson patents, ~1997-1999)
- Solid state drives / flash memory (Toshiba NAND patents, ~1984-1989)
- GPS civilian applications (key Trimble/Garmin patents, ~1990s)
- Capacitive touch screens (pre-iPhone, key patents ~2003-2005)
- Voice recognition / NLP (Nuance/Apple Siri-era patents, ~2008-2011)
- Cloud computing infrastructure (AWS/VMware virtualization patents, ~2003-2006)
- Quantum computing hardware (D-Wave/IBM/Google patents, ~2005-2012)

**Energy/Materials (add ~6-8):**
- Blue LED (Nakamura GaN patents, ~1991-1994 filing)
- Perovskite solar cells (Miyasaka/EPFL patents, ~2009-2012)
- Hydraulic fracturing + horizontal drilling combo (Mitchell Energy patents, ~1997-2002)
- Solid-state batteries (key Toyota/Samsung patents, ~2010-2015)
- Carbon nanotubes (Iijima discovery era patents, ~1991-1995)
- Supercapacitors / ultracapacitors (Maxwell Technologies patents, ~1990s)

**Telecom/Networking (add ~4-5):**
- RFID (key Alien Technology/Impinj patents, ~1999-2003)
- LTE/4G (Qualcomm OFDMA patents, ~2004-2008)
- Fiber optic amplifiers EDFA (key Corning/Lucent patents, ~1987-1990)
- 5G mmWave (Qualcomm/Samsung patents, ~2013-2016)

**Manufacturing/Other (add ~4-5):**
- SLA 3D printing (Chuck Hull/3D Systems patents, ~1984-1986)
- Metal additive manufacturing (DMLS/SLM patents, ~1995-2000)
- RFID supply chain (Walmart-era implementation patents, ~2003-2005)
- Autonomous vehicle LIDAR (Velodyne patents, ~2005-2010)
- Blockchain smart contracts (Ethereum-related patents, distinct from base 
  blockchain, ~2014-2016)

### Task 2.2: Validation

For each new entry:
1. Web search the patent number to confirm it exists and the filing date is correct
2. Verify the patent_id exists in our patents.parquet
3. Map to CPC sections using our cpc_map.parquet
4. If a patent number can't be found in PatentsView, search for alternative 
   patents in the same technology area that ARE in the database

### Task 2.3: Re-run NB03, NB04, NB05

After adding new breakthroughs:
1. Re-run NB03 to validate the expanded catalog
2. Re-run NB04 with the full expanded catalog — report new N, p-values, z-scores
3. Re-run NB05 with the larger training set — report new AUC values

Target: 60-70 total cataloged breakthroughs, 40+ valid comparisons.

---

## Strategy 3: CPC Subclass Creation Events as Objective Breakthroughs

This is the most powerful expansion. It replaces subjective curation with 
institutional fact.

### Task 3.1: Extract CPC subclass creation dates

For each CPC subclass in cpc_map.parquet:
1. Find the earliest patent_id assigned to that subclass
2. Look up that patent's filing date in patents.parquet
3. That date approximates when the subclass was "created" — i.e., when the 
   USPTO recognized a new technology area

```python
# Pseudocode
subclass_creation = (
    cpc_map
    .merge(patents[['patent_id', 'patent_date']], on='patent_id')
    .groupby('cpc_subclass')
    .agg(
        first_patent_date=('patent_date', 'min'),
        first_patent_id=('patent_id', 'first'),
        n_patents=('patent_id', 'count'),
    )
    .reset_index()
)
```

### Task 3.2: Filter to meaningful creation events

Not every subclass creation is a "breakthrough." Filter to:
- Subclasses created after 1990 (need sufficient precursor topology data)
- Subclasses that eventually accumulate at least 500 patents (excludes trivial 
  reclassifications)
- Subclasses where the creation date is at least 5 years after the start of 
  topology data for the relevant CPC section pair
- Exclude subclasses that are pure reclassifications of existing ones (heuristic: 
  if >80% of the first 100 patents in a new subclass were previously assigned to 
  a single other subclass, it's probably a split, not a new technology)

This should yield 100-300 "objective breakthrough" events.

### Task 3.3: Map to topology

For each subclass creation event:
1. Identify the CPC section(s) the subclass belongs to
2. Identify which cross-section CPC pairs are relevant (the new subclass's 
   section paired with each other section it cites)
3. Extract the 10-year precursor topology from the cached data
4. Build a matched null model (same CPC pair, random non-creation windows)

### Task 3.4: Run the same precursor test

Apply the identical statistical framework from NB04:
- Z-scores for pre-creation topology vs matched null
- Superposed epoch analysis
- One-sample t-test and Wilcoxon signed-rank
- Holm-Bonferroni correction

### Task 3.5: Compare curated vs objective catalogs

The key comparison:
- Do curated breakthroughs and objective subclass creation events show the SAME 
  topological precursor pattern?
- If yes: the finding replicates across two independent definitions of 
  "breakthrough" — one subjective, one administrative. This is powerful evidence.
- If no: the finding may be specific to how we define breakthroughs, which is 
  an important limitation to document.

Report both results side-by-side.

---

## Strategy 3B: CPC Reclassification Events (Bonus)

When the USPTO reclassifies a large number of patents from one subclass to 
another, it signals that the boundary between technology areas has shifted. 
These reclassification events are a DIFFERENT kind of structural change than 
new subclass creation.

### Task 3B.1: Detect reclassification events

If a patent has multiple CPC subclass assignments over time (detectable if 
PatentsView includes historical CPC data), large-scale reclassifications can 
be identified. Alternatively, look for subclasses where the patent composition 
changes dramatically in a short window (sudden influx of patents that were 
previously in another subclass).

This is exploratory. If the data doesn't support it, skip it and document why.

---

## Deliverables

After all three strategies:

1. **Updated NB03** with expanded catalog (target: 60-70 curated + note on 
   objective catalog size)
2. **Updated NB04** with:
   - Results for expanded curated catalog (target: N=40+)
   - Results for objective subclass creation catalog (target: N=100+)
   - Side-by-side comparison table
   - Updated superposed epoch plot with larger N
   - Updated effect size distribution
3. **Updated NB05** with larger training set, re-run LOGO cross-validation
4. **Updated README** with new Quick Results, new figures, new sample sizes
5. **Updated Limitations section** noting which limitations were addressed 
   (sample size improved) and which remain

## Quality Checks

Before finalizing:
- All p-values must be recomputed from scratch, not carried over
- Holm-Bonferroni correction must be applied to the NEW number of tests
- If adding Strategy 3 as a separate test family, correct within each family 
  separately and note this
- If any result that was previously significant becomes non-significant with 
  the expanded sample, REPORT THIS PROMINENTLY. We do not bury disconfirmation.
- Update the CITATION.cff date-released field

---

*Conceived by Claude (Opus 4.6, Anthropic). Expansion strategy designed in 
conversation with Christopher Ortiz. Implementation by Claude Code.*