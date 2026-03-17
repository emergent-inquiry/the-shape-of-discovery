# run-notebook

Execute a Jupyter notebook end-to-end and report results.

## Instructions

Run the specified notebook using `jupyter nbconvert --execute --to notebook --inplace`.

1. Before running, check that required data files exist in `data/` (patents.parquet, citations.parquet, cpc_map.parquet)
2. Execute the notebook with a generous timeout (3600s for notebooks 01/03, 7200s for 02/04/05)
3. If execution succeeds, check for generated figures in `figures/`
4. If execution fails:
   - Read the error traceback from the notebook output
   - Identify the failing cell
   - Report the error clearly with file, line number, and root cause
   - Do NOT attempt to fix — report back so fixes can be coordinated

## Usage

```
Provide the notebook filename as the prompt, e.g.:
"Run 02_topological_clock.ipynb"
```

## Environment

- Working directory: project root
- Python environment: the active conda/venv environment
- Notebooks may take 10 minutes to several hours depending on computational load
- Notebook 02 (topology) is the heaviest — persistent homology on 8 CPC pairs across 35+ time windows
- Notebooks 04 and 05 depend on topology cache from notebook 02
