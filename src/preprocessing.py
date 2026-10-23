"""
src/preprocessing.py
--------------------
Data cleaning and feature engineering for the International Football dataset.

Cleaning steps applied (each documented in the returned audit log):
  1. Parse dates; drop rows where date cannot be parsed.
  2. Convert home_score / away_score to numeric; drop rows with NaN scores.
  3. Drop rows where either score is negative (invalid).
  4. Remove exact duplicate match records (same date + teams + scores).
  5. Filter to the selected date range [start_date, end_date].
  6. Create derived columns: year, total_goals, goal_difference,
     home_win, away_win, draw, is_neutral.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Public API ─────────────────────────────────────────────────────────────────


def clean_dataset(
    df_raw: pd.DataFrame,
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
) -> tuple[pd.DataFrame, dict]:
    """
    Apply the full cleaning pipeline to the raw dataset.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw data as returned by data_loader.load_dataset().
    start_date : str
        Inclusive start of the study period (ISO format YYYY-MM-DD).
    end_date : str
        Inclusive end of the study period (ISO format YYYY-MM-DD).

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataset with derived columns.
    audit : dict
        Step-by-step record counts and notes for reproducibility documentation.
    """
    audit: dict = {
        "start_date": start_date,
        "end_date": end_date,
        "steps": [],
    }

    df = df_raw.copy()
    _log_step(audit, "raw", len(df), "Original dataset loaded")

    # Step 1 — Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    _log_step(audit, "parse_dates", len(df), f"Dropped {n_bad_dates} rows with unparseable dates")

    # Step 2 — Convert scores to numeric
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    n_missing = df[["home_score", "away_score"]].isna().any(axis=1).sum()
    df = df.dropna(subset=["home_score", "away_score"])
    _log_step(audit, "numeric_scores", len(df), f"Dropped {n_missing} rows with missing scores")

    # Step 3 — Remove invalid (negative) scores
    n_neg = ((df["home_score"] < 0) | (df["away_score"] < 0)).sum()
    df = df[(df["home_score"] >= 0) & (df["away_score"] >= 0)]
    _log_step(audit, "negative_scores", len(df), f"Removed {n_neg} rows with negative scores")

    # Step 4 — Deduplicate
    dup_cols = ["date", "home_team", "away_team", "home_score", "away_score"]
    dup_cols_avail = [c for c in dup_cols if c in df.columns]
    n_dup = df.duplicated(subset=dup_cols_avail).sum()
    df = df.drop_duplicates(subset=dup_cols_avail)
    _log_step(audit, "duplicates", len(df), f"Removed {n_dup} exact duplicate records")

    # Step 5 — Date filter
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    _log_step(audit, "date_filter", len(df), f"Filtered to {start_date} – {end_date}")

    # Step 6 — Cast score columns to int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Step 7 — Derived columns
    df = _add_derived_columns(df)
    _log_step(audit, "derived_columns", len(df), "Added derived columns")

    # Step 8 — Handle 'neutral' column
    if "neutral" in df.columns:
        df["is_neutral"] = (
            df["neutral"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
        )
    else:
        df["is_neutral"] = False
        audit["steps"].append("neutral column not present — treating all matches as non-neutral")

    _log_step(audit, "final", len(df), "Cleaning complete")

    # Reset index
    df = df.reset_index(drop=True)

    return df, audit


def filter_data(
    df: pd.DataFrame,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    tournaments: Optional[list[str]] = None,
    include_neutral: bool = True,
    teams: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply user-specified interactive filters to the cleaned dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (output of clean_dataset).
    start_year : int, optional
        Minimum year (inclusive).
    end_year : int, optional
        Maximum year (inclusive).
    tournaments : list[str], optional
        Subset of tournament names to retain. None = all tournaments.
    include_neutral : bool
        If False, exclude matches played at neutral venues.
    teams : list[str], optional
        Retain only matches involving at least one of these teams.

    Returns
    -------
    pd.DataFrame
        Filtered copy of the dataset.
    """
    mask = pd.Series(True, index=df.index)

    if start_year is not None:
        mask &= df["year"] >= start_year
    if end_year is not None:
        mask &= df["year"] <= end_year
    if tournaments:
        mask &= df["tournament"].isin(tournaments)
    if not include_neutral:
        mask &= ~df["is_neutral"]
    if teams:
        mask &= df["home_team"].isin(teams) | df["away_team"].isin(teams)

    return df[mask].copy()


def get_non_neutral(df: pd.DataFrame) -> pd.DataFrame:
    """Return only matches played at non-neutral venues."""
    return df[~df["is_neutral"]].copy()


# ── Internal helpers ───────────────────────────────────────────────────────────


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_difference"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["away_win"] = (df["away_score"] > df["home_score"]).astype(int)
    df["draw"] = (df["home_score"] == df["away_score"]).astype(int)
    return df


def _log_step(audit: dict, key: str, n_rows: int, note: str) -> None:
    audit["steps"].append({"step": key, "rows_remaining": n_rows, "note": note})
    logger.debug("[preprocessing] %s: %d rows — %s", key, n_rows, note)


def audit_to_dataframe(audit: dict) -> pd.DataFrame:
    """Convert audit log to a tidy DataFrame for display."""
    rows = audit["steps"]
    return pd.DataFrame(rows, columns=["step", "rows_remaining", "note"])
