"""Runtime fetch + on-disk cache of the historical results CSV.

Uses ``requests`` (imported lazily, so the rest of the package — and the tests — never need the
network). A fresh cache is reused; a stale/missing cache triggers a download with exponential
backoff; a total network failure falls back to a stale cache (with a warning) rather than
crashing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .config import EloConfig


def cache_path(cfg: EloConfig) -> Path:
    return Path(cfg.cache_dir).expanduser() / "results.csv"


def _is_fresh(path: Path, max_age_days: float) -> bool:
    if not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days <= max_age_days


def _download(url: str, *, retries: int = 3, timeout: float = 30.0) -> str:
    import requests

    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last_exc}")


def get_results_csv(cfg: EloConfig, *, cache_only: bool = False) -> str:
    """Return the results CSV text, using the on-disk cache when fresh."""
    cache = cache_path(cfg)
    if cache_only:
        if not cache.exists():
            raise FileNotFoundError(f"--cache-only set but no cache at {cache}")
        return cache.read_text(encoding="utf-8")
    if _is_fresh(cache, cfg.cache_max_age_days):
        return cache.read_text(encoding="utf-8")
    try:
        text = _download(cfg.source_url)
    except Exception as exc:  # noqa: BLE001
        if cache.exists():
            print(f"build-elo: fetch failed ({exc}); using stale cache {cache}", file=sys.stderr)
            return cache.read_text(encoding="utf-8")
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
