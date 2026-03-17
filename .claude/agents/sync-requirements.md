# sync-requirements

Audit all Python imports across the repository and synchronize `requirements.txt` with the actual dependencies used in the codebase.

## When to use

Run this agent whenever:
- A new third-party package is added to any `.py` or `.ipynb` file
- A dependency is removed or replaced
- Before setting up a new environment (Colab, cloud VM, fresh install)
- After a `pip install` of something new in a notebook

## Steps

1. **Scan all Python sources** for import statements:
   - `src/*.py`
   - `tests/*.py`
   - `*.py` (root-level scripts)
   - `*.ipynb` (all notebooks, including `colab_setup.ipynb`)
   - Look for both `import X` and `from X import ...` patterns

2. **Filter to third-party packages only** — exclude Python standard library modules:
   - stdlib: `os`, `sys`, `json`, `pathlib`, `typing`, `logging`, `functools`, `hashlib`, `pickle`, `time`, `zipfile`, `glob`, `shutil`, `platform`, `dataclasses`, `__future__`, `collections`, `itertools`, `re`, `io`, `datetime`, `math`, `copy`, `warnings`, `unittest`, `abc`, `contextlib`, `tempfile`, `subprocess`, `textwrap`, `csv`, `struct`, `operator`
   - Also exclude internal project imports (`src.*`, relative imports)

3. **Exclude environment-specific packages** that shouldn't be in requirements.txt:
   - `google.colab` — only available in Colab runtime, not pip-installable
   - Any other platform-specific modules

4. **Compare found packages against current `requirements.txt`**:
   - Identify packages imported but NOT in requirements.txt (missing)
   - Identify packages in requirements.txt but NOT imported anywhere (unused)
   - Check version pins are reasonable (especially `numpy<2.0` for ABI compatibility)

5. **Update `requirements.txt`**:
   - Add any missing packages with appropriate version pins
   - Comment out or flag unused packages (ask before removing — they may be transitive deps)
   - Keep the existing category comments and organization
   - Maintain the `numpy>=1.24,<2.0` pin to prevent ABI breakage on Colab/fresh installs

6. **Report a summary**:
   - Packages added
   - Packages flagged as potentially unused
   - Any version pin concerns
   - Total dependency count

## Notes

- The `numpy<2.0` upper bound is critical — numpy 2.0 changed the C ABI and breaks pre-compiled packages (ripser, scipy, etc.) on environments like Google Colab.
- `joblib` may not be directly imported but is used by scikit-learn and shap internally — keep it.
- `pyarrow` may not appear in imports but is required by pandas for parquet I/O — keep it.
- When in doubt about whether a package is needed, keep it and note it in the report rather than removing it.
