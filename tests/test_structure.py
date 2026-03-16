"""Verify repository structure is complete."""
from pathlib import Path


def test_repo_structure():
    """All required directories and files exist."""
    root = Path(__file__).resolve().parent.parent

    # Core files
    assert (root / "CLAUDE.md").exists()
    assert (root / "README.md").exists()
    assert (root / "LICENSE").exists()
    assert (root / "requirements.txt").exists()

    # Source modules
    assert (root / "src" / "__init__.py").exists()

    # Directories
    assert (root / "data").is_dir()
    assert (root / "data" / "breakthroughs").is_dir()
    assert (root / "figures").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "scripts").is_dir()


def test_source_modules_exist():
    """All planned source modules are present."""
    root = Path(__file__).resolve().parent.parent / "src"

    expected_modules = [
        "fetch.py",
        "graph.py",
        "topology.py",
        "breakthroughs.py",
        "metrics.py",
        "nullmodel.py",
        "plotting.py",
        "utils.py",
    ]
    for module in expected_modules:
        assert (root / module).exists(), f"Missing src/{module}"


def test_notebooks_exist():
    """All planned notebooks are present."""
    root = Path(__file__).resolve().parent.parent

    expected_notebooks = [
        "00_data_acquisition.py",
        "01_patent_atlas.ipynb",
        "02_topological_clock.ipynb",
        "03_breakthrough_catalog.ipynb",
        "04_precursor_test.ipynb",
        "05_predictability_horizon.ipynb",
    ]
    for nb in expected_notebooks:
        assert (root / nb).exists(), f"Missing {nb}"
