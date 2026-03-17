# confound-audit

Audit current results against all documented confounds. This agent diagnoses — it does not fix.

## When to Use

- After notebooks 02 and 04 have been run and results are available
- Before writing the Results or Limitations sections of README.md
- After adding a new confound to CONFOUNDS.md
- As a final check before making any claims about topological precursors

## Instructions

### Step 1: Load Confound Inventory

1. Read `CONFOUNDS.md` to get the full list of identified confounds
2. For each confound, note:
   - What artifact could it produce in the topology?
   - What robustness check was specified?
   - Has the robustness check been run? (check `data/robustness_cache/`)

### Step 2: Check Data Availability

For each confound, verify whether the required data exists:

| Confound | Required Data | Where to Check |
|----------|--------------|----------------|
| #1 Examiner citations | `citation_category` column | Raw TSV at `data/raw/g_us_patent_citation.tsv` (not in cleaned parquet) |
| #2 Prosecution lag | Filing dates | `data/patents.parquet` or raw `data/raw/g_patent.tsv` |
| #4 Patent thickets | Assignee data | `data/raw/g_assignee_disambiguated.tsv` or `data/assignee_map.parquet` |
| #8 Self-citations | Assignee data | Same as #4 |
| #9 Truncation | Analysis end year | Check `sliding_window_topology()` parameters used in notebook 02 |

### Step 3: Run Lightweight Diagnostics

These are quick checks, not full topology re-runs:

1. **Examiner fraction**: Read the raw citation TSV, compute what percentage of citations per CPC section pair are examiner-added. If >50%, flag as high-priority for the robustness-check agent.

2. **Truncation risk**: For each sliding window, compute what fraction of patents were granted in the most recent 2 years of the window. High fractions in late windows indicate truncation risk.

3. **Policy shock overlap**: List which breakthrough precursor windows (10 years before each breakthrough) overlap with Alice Corp. (2014) or AIA (2011). These breakthroughs need extra scrutiny.

4. **Citation culture drift**: Compute mean citations-per-patent per CPC section per decade. Large shifts suggest citation culture may confound topology.

5. **Self-citation concentration**: If assignee data is available, compute what fraction of cross-section citations are intra-assignee for each CPC pair. High fractions flag confound #8 risk.

### Step 4: Produce Audit Report

Generate a summary table:

```
| # | Confound                    | Data Available? | Check Run? | Status              |
|---|-----------------------------|-----------------|------------|---------------------|
| 1 | Examiner citations          | Raw TSV only    | No         | UNCONTROLLED        |
| 2 | Prosecution lag             | TBD             | No         | UNCONTROLLED        |
| 3 | Policy shocks               | Yes (dates)     | No         | PARTIALLY CONTROLLED|
| 4 | Strategic patenting         | No assignee data| No         | UNCONTROLLED        |
| 5 | Citation cultures           | Yes             | No         | UNCONTROLLED        |
| 6 | Survivorship bias           | Limited         | N/A        | IRREDUCIBLE         |
| 7 | CPC reclassification        | Unclear         | No         | UNCONTROLLED        |
| 8 | Assignee self-citation      | No assignee data| No         | UNCONTROLLED        |
| 9 | Citation truncation         | Yes             | No         | UNCONTROLLED        |
```

Status values:
- **CONTROLLED**: Robustness check run, signal survives
- **PARTIALLY CONTROLLED**: Proxy check done, full check pending
- **UNCONTROLLED**: No data or check available — must be stated as limitation
- **IRREDUCIBLE**: Cannot be fully addressed with available data — document honestly

### Step 5: Recommendations

Based on the audit:
- Which robustness checks should be run next? (invoke robustness-check agent)
- Which confounds require new data downloads? (flag for fetch.py modifications)
- Which confounds are irreducible limitations? (must appear in README.md)
- **Critical**: If any key finding could be fully explained by an uncontrolled confound, flag this prominently

## Key File Locations

- `CONFOUNDS.md` — confound definitions and robustness checks
- `data/topology_cache/` — baseline topology results (pickle files)
- `data/robustness_cache/` — robustness check results (if any exist)
- `data/raw/g_us_patent_citation.tsv` — raw citation data with `citation_category`
- `src/topology.py` — `sliding_window_topology()` at line 406
- `src/nullmodel.py` — null model and superposed epoch functions

## Intellectual Honesty

- Be conservative: if a confound is only partially addressed, say so
- The audit table belongs in README.md's limitations section
- A signal that cannot survive robustness checks is not a signal — it's a hope
- If the audit reveals that a key finding could be fully explained by an uncontrolled confound, this must be stated before any claims are made
