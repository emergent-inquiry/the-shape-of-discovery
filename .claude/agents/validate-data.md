# validate-data

Validate the data pipeline outputs and check for integrity issues.

## Instructions

Check that the data pipeline produced correct outputs:

1. Verify files exist: `data/patents.parquet`, `data/citations.parquet`, `data/cpc_map.parquet`
2. Load each file and check:
   - **patents.parquet**: ~8M rows, columns (patent_id, date, title, cpc_primary), no null patent_ids, dates range 1976-2025
   - **citations.parquet**: ~118M rows, columns (citing_id, cited_id, citing_date), no self-citations, all IDs are strings
   - **cpc_map.parquet**: ~17M rows, columns (patent_id, cpc_section, cpc_class, cpc_subclass), sections are single uppercase letters A-H
3. Cross-validate: citation patent_ids should largely overlap with patents.parquet IDs
4. Check for Arrow-backed string types that may cause issues downstream — report dtype of each column
5. Report summary statistics: row counts, null counts, date ranges, CPC section distribution

## Common Issues

- Arrow StringDtype vs object dtype mismatches between parquet and in-memory operations
- Duplicate citations (same citing-cited pair)
- Patent IDs with leading zeros getting stripped
