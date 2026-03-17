# robustness-check

Run sensitivity analyses that filter or transform the citation graph and re-run the topology pipeline to test whether topological signals survive each robustness check.

## When to Use

After notebooks 02 (Topological Clock) and 04 (Precursor Test) have produced baseline topological results. Use this agent to verify that signals are not artifacts of the confounds documented in CONFOUNDS.md.

## General Workflow

Every sensitivity analysis follows the same pattern:

1. Load baseline topology results from `data/topology_cache/` for comparison
2. Load `data/citations.parquet`, `data/cpc_map.parquet`, `data/patents.parquet`
3. Apply the appropriate filter or transformation to the citations DataFrame
4. Call `sliding_window_topology()` from `src/topology.py` with **`use_cache=False`** to force recomputation
5. Save results to `data/robustness_cache/{analysis_name}_{section_a}_{section_b}.pkl`
6. Compare filtered results against baseline:
   - Correlation between filtered and baseline β₁ time series
   - Max absolute difference and relative change
   - Whether precursor signals identified in notebook 04 survive
7. Summarize: does the signal survive this robustness check?

## Available Analyses

### 1. Applicant-Only Citations (Confound #1)

The `citation_category` column was dropped during data acquisition (`00_data_acquisition.py` line 99). To recover it:

- Re-read the raw TSV at `data/raw/g_us_patent_citation.tsv` with `usecols=["patent_id", "citation_patent_id", "citation_category"]`
- Filter to rows where `citation_category` indicates applicant-added citations (not examiner-added)
- Rebuild the citations DataFrame and re-run topology
- Note: coverage of `citation_category` is inconsistent pre-2001; document which years are affected

### 2. Filing Date Substitution (Confound #2)

- Check whether filing dates are available in `data/patents.parquet` or the raw `g_patent.tsv`
- Replace `citing_date` (grant date) with filing dates
- Re-run `sliding_window_topology()` with shifted dates
- Check whether precursor signal timing shifts by ~2-3 years (consistent with prosecution lag)

### 3. Policy Shock Test (Confound #3)

- Load baseline topology results from `data/topology_cache/`
- Flag sliding windows that overlap with Alice Corp. v. CLS Bank (2014) and the America Invents Act (2011)
- Test whether β₁ anomalies cluster around policy dates vs. breakthrough dates
- If they cluster around policy dates, the topological signal may reflect legal shocks, not knowledge reorganization

### 4. Patent Thicket Filter (Confound #4)

- **Requires assignee data** — check for `data/assignee_map.parquet` or `data/raw/g_assignee_disambiguated.tsv`
- If missing, report that this analysis is blocked and list what data is needed
- If available: identify top-20 assignees by filing volume in telecom/semiconductor CPC pairs (G, H)
- Remove all patents from those assignees, re-run topology

### 5. Citation Culture Drift (Confound #5)

- Compute mean citations-per-patent per CPC section per year from `data/citations.parquet` and `data/cpc_map.parquet`
- Correlate with β₁ time series from baseline results
- If correlation is high, the topology signal may reflect citation culture changes rather than knowledge structure

### 6. Assignee Self-Citation Filter (Confound #8)

- **Requires assignee data** — same dependency as analysis #4
- Join citations with assignee mapping on both `citing_id` and `cited_id`
- Remove edges where `citing_assignee == cited_assignee`
- Re-run topology on the filtered graph

### 7. Citation Truncation Correction (Confound #9)

- **Run A:** Set `end_year=2018` in `sliding_window_topology()` to exclude incomplete windows
- **Run B:** Apply citation-opportunity weighting — normalize edge counts by years since grant for each patent
- Compare both against the full-window baseline
- If late-window results diverge significantly, truncation bias is distorting the tail

## Critical Notes

- **Always set `use_cache=False`** when re-running with filtered data. Otherwise you get cached unfiltered results.
- **Never overwrite baseline cache files.** Save robustness results to `data/robustness_cache/`, not `data/topology_cache/`.
- The applicant-only analysis (#1) requires re-reading the **raw TSV**, not the cleaned parquet.
- Assignee-based analyses (#4, #6) require `g_assignee_disambiguated` which is **not currently downloaded**. Flag this clearly if the data is missing rather than failing silently.
- Each topology re-run on a CPC pair takes 10-30 minutes. Budget accordingly.
- **16GB MacBook constraint** — monitor memory with `src/utils.py:report_memory()`. May need to run one CPC pair at a time.
- Confounds #6 (survivorship) and #7 (CPC reclassification) do not have full robustness-check workflows here — they are addressed via documentation and proxy checks in the confound-audit agent.

## Key Functions

- `src/topology.py`: `sliding_window_topology()` (line 406) — main computation, accepts `use_cache` parameter
- `src/graph.py`: `cpc_subgraph_nx()`, `build_citation_graph()` — graph construction
- `src/nullmodel.py`: `superposed_epoch_analysis()` — for comparing epoch-aligned robustness results
- `src/utils.py`: `report_memory()`, `cache_result()` — memory monitoring and caching
