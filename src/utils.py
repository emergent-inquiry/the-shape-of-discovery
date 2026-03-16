"""Common utilities: logging, caching, timing, and memory reporting."""

import functools
import hashlib
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = DATA_DIR / "cache"


def ensure_dirs() -> None:
    """Create standard project directories if they don't exist."""
    for d in (DATA_DIR, FIGURES_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Build a deterministic hash key from function name and arguments."""
    raw = f"{func_name}:{args!r}:{sorted(kwargs.items())!r}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_parquet(subdir: str = "cache") -> Callable:
    """Decorator that caches a function returning a DataFrame as parquet.

    Args:
        subdir: Subdirectory under ``data/`` for cache files.

    Returns:
        Decorator function.
    """
    cache_path = DATA_DIR / subdir

    def decorator(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            cache_path.mkdir(parents=True, exist_ok=True)
            key = _cache_key(func.__name__, args, kwargs)
            fp = cache_path / f"{func.__name__}_{key}.parquet"
            if fp.exists():
                logger = get_logger("cache")
                logger.info("Cache hit: %s -> %s", func.__name__, fp.name)
                return pd.read_parquet(fp)
            result = func(*args, **kwargs)
            result.to_parquet(fp, index=False)
            return result
        return wrapper
    return decorator


def cache_pickle(subdir: str = "cache") -> Callable:
    """Decorator that caches arbitrary return values as pickle.

    Args:
        subdir: Subdirectory under ``data/`` for cache files.

    Returns:
        Decorator function.
    """
    cache_path = DATA_DIR / subdir

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_path.mkdir(parents=True, exist_ok=True)
            key = _cache_key(func.__name__, args, kwargs)
            fp = cache_path / f"{func.__name__}_{key}.pkl"
            if fp.exists():
                logger = get_logger("cache")
                logger.info("Cache hit: %s -> %s", func.__name__, fp.name)
                with open(fp, "rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            with open(fp, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timer(func: Callable) -> Callable:
    """Decorator that logs wall-clock execution time.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that logs elapsed time.
    """
    logger = get_logger("timer")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed < 60:
            logger.info("%s completed in %.1fs", func.__name__, elapsed)
        else:
            mins, secs = divmod(elapsed, 60)
            logger.info("%s completed in %dm %.1fs", func.__name__, int(mins), secs)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------

def memory_usage_mb() -> float:
    """Return current process RSS in megabytes (macOS/Linux)."""
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024
    except ImportError:
        return 0.0


def log_memory(label: str = "") -> None:
    """Log current memory usage.

    Args:
        label: Optional label to prefix the log message.
    """
    mb = memory_usage_mb()
    logger = get_logger("memory")
    prefix = f"{label}: " if label else ""
    logger.info("%sRSS = %.0f MB", prefix, mb)
