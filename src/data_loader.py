"""
src/data_loader.py
------------------
Handles downloading and loading the International Football Results dataset.

Dataset source:
  https://github.com/martj42/international_results
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
"""

from __future__ import annotations

import os
import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DATASET_URL: str = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DEFAULT_CACHE_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "results.csv"
)

REQUIRED_COLUMNS: list[str] = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

EXPECTED_COLUMNS: list[str] = REQUIRED_COLUMNS + [
    "tournament",
    "city",
    "country",
    "neutral",
]


# ── Public functions ────────────────────────────────────────────────────────────


def load_dataset(
    cache_path: Optional[Path] = None,
    force_download: bool = False,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the International Football Results dataset.

    Priority:
      1. Use cached local file (if present and force_download is False).
      2. Download from the canonical public URL and cache locally.

    Returns
    -------
    df : pd.DataFrame
        Raw (uncleaned) dataset.
    meta : dict
        Provenance metadata (source, cache path, row count, etc.).
    """
    cache_path = cache_path or DEFAULT_CACHE_PATH
    meta: dict = {
        "source": None,
        "cache_path": str(cache_path),
        "url": DATASET_URL,
        "download_attempted": False,
        "download_succeeded": False,
        "error": None,
    }

    # 1. Try local cache
    if cache_path.exists() and not force_download:
        try:
            df = _read_csv(cache_path)
            meta["source"] = f"local cache: {cache_path}"
            meta["rows_raw"] = len(df)
            logger.info("Loaded dataset from cache: %s (%d rows)", cache_path, len(df))
            return df, meta
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read cache file %s: %s", cache_path, exc)

    # 2. Download
    meta["download_attempted"] = True
    try:
        response = requests.get(DATASET_URL, timeout=timeout)
        response.raise_for_status()
        csv_bytes = response.content
        df = _read_csv(io.BytesIO(csv_bytes))

        # Cache locally
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(csv_bytes)

        meta["source"] = f"downloaded from {DATASET_URL}"
        meta["download_succeeded"] = True
        meta["rows_raw"] = len(df)
        logger.info("Downloaded dataset (%d rows) and cached at %s", len(df), cache_path)
        return df, meta

    except requests.exceptions.RequestException as exc:
        meta["error"] = str(exc)
        logger.error("Download failed: %s", exc)
        raise RuntimeError(
            f"Could not download the dataset from {DATASET_URL}.\n"
            f"Error: {exc}\n\n"
            "Options:\n"
            "  1. Check your internet connection and retry.\n"
            "  2. Manually download results.csv from the GitHub repository and\n"
            "     place it in the data/ directory.\n"
            "  3. Use the 'Upload CSV' option in the dashboard."
        ) from exc


def load_from_upload(file_obj) -> tuple[pd.DataFrame, dict]:
    """
    Load a user-uploaded CSV file and validate its columns.

    Parameters
    ----------
    file_obj : file-like object
        Uploaded file (e.g., from Streamlit's st.file_uploader).

    Returns
    -------
    df : pd.DataFrame
    meta : dict
    """
    df = _read_csv(file_obj)
    meta = {
        "source": "user upload",
        "rows_raw": len(df),
        "url": None,
        "download_attempted": False,
        "download_succeeded": False,
        "error": None,
    }
    _validate_columns(df)
    return df, meta


def _read_csv(source) -> pd.DataFrame:
    """Read CSV from a file path, bytes-IO, or file-like object."""
    df = pd.read_csv(source, low_memory=False)
    # Normalise column names: strip whitespace, lower-case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required columns: {missing}.\n"
            f"Found columns: {list(df.columns)}"
        )


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return basic descriptive information about the raw dataset."""
    info: dict = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "date_range": None,
        "missing_home_score": int(df["home_score"].isna().sum()) if "home_score" in df.columns else None,
        "missing_away_score": int(df["away_score"].isna().sum()) if "away_score" in df.columns else None,
    }
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            info["date_range"] = (str(dates.min().date()), str(dates.max().date()))
    return info
