"""
src/statistics.py
-----------------
Experiment 1: Measures of Central Tendency

Computes descriptive statistics for total_goals, home_score, away_score,
and goal_difference.

All functions accept a pandas Series or DataFrame and return plain dicts or
DataFrames so they can be consumed by both the Streamlit dashboard and the
Jupyter notebook without duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Experiment 1: Central Tendency ────────────────────────────────────────────


def central_tendency(series: pd.Series, var_name: str = "variable") -> dict:
    """
    Compute a full set of descriptive statistics for a numeric Series.

    Parameters
    ----------
    series : pd.Series
        Numeric data (NaN values are ignored).
    var_name : str
        Human-readable variable name for labelling.

    Returns
    -------
    dict with keys:
        variable, n, mean, median, mode, mode_count, std, variance,
        min, max, range, q1, q3, iqr, skewness, kurtosis
    """
    s = series.dropna()
    n = len(s)

    if n == 0:
        return _empty_stats(var_name)

    mode_result = stats.mode(s, keepdims=True)
    mode_values = mode_result.mode.tolist()
    mode_count = int(mode_result.count[0]) if len(mode_result.count) > 0 else 0

    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))

    return {
        "variable": var_name,
        "n": n,
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "mode": mode_values,
        "mode_count": mode_count,
        "std": float(np.std(s, ddof=1)),
        "variance": float(np.var(s, ddof=1)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "range": float(np.max(s) - np.min(s)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "skewness": float(stats.skew(s)),
        "kurtosis": float(stats.kurtosis(s)),
    }


def central_tendency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy summary table for total_goals, home_score, away_score,
    and goal_difference.

    Returns
    -------
    pd.DataFrame with one column per variable and statistics as rows.
    """
    variables = {
        "Total Goals": df["total_goals"],
        "Home Score": df["home_score"],
        "Away Score": df["away_score"],
        "Goal Difference": df["goal_difference"],
    }

    records = []
    for label, series in variables.items():
        stat = central_tendency(series, label)
        records.append(stat)

    tbl = pd.DataFrame(records).set_index("variable")
    return tbl


def yearly_goals_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by year and compute mean total goals and match count per year.

    Returns
    -------
    pd.DataFrame with columns: year, mean_total_goals, n_matches
    """
    grp = (
        df.groupby("year")["total_goals"]
        .agg(mean_total_goals="mean", n_matches="count")
        .reset_index()
    )
    grp = grp.sort_values("year").reset_index(drop=True)
    return grp


def score_distribution_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return value counts of total_goals for histogram-style display.
    """
    vc = df["total_goals"].value_counts().sort_index().reset_index()
    vc.columns = ["total_goals", "count"]
    return vc


def home_away_comparison(df: pd.DataFrame) -> dict:
    """
    Return means of home_score and away_score for bar-chart comparison.
    """
    return {
        "home_mean": float(df["home_score"].mean()),
        "away_mean": float(df["away_score"].mean()),
        "home_median": float(df["home_score"].median()),
        "away_median": float(df["away_score"].median()),
        "n": len(df),
    }


def kpi_summary(df: pd.DataFrame, df_non_neutral: pd.DataFrame | None = None) -> dict:
    """
    Compute top-level KPIs for the Overview dashboard page.

    Parameters
    ----------
    df : pd.DataFrame
        Full filtered dataset.
    df_non_neutral : pd.DataFrame, optional
        Non-neutral subset for home-advantage metrics.

    Returns
    -------
    dict of KPI values.
    """
    nn = df_non_neutral if df_non_neutral is not None else df[~df["is_neutral"]]
    return {
        "total_matches": len(df),
        "avg_total_goals": float(df["total_goals"].mean()) if len(df) > 0 else 0.0,
        "avg_home_goals": float(df["home_score"].mean()) if len(df) > 0 else 0.0,
        "avg_away_goals": float(df["away_score"].mean()) if len(df) > 0 else 0.0,
        "pct_draw": float(df["draw"].mean() * 100) if len(df) > 0 else 0.0,
        "mean_goal_diff_nn": float(nn["goal_difference"].mean()) if len(nn) > 0 else 0.0,
        "home_win_pct": float(nn["home_win"].mean() * 100) if len(nn) > 0 else 0.0,
        "away_win_pct": float(nn["away_win"].mean() * 100) if len(nn) > 0 else 0.0,
        "draw_pct_nn": float(nn["draw"].mean() * 100) if len(nn) > 0 else 0.0,
        "n_non_neutral": len(nn),
    }


# ── Internal ───────────────────────────────────────────────────────────────────


def _empty_stats(var_name: str) -> dict:
    return {
        "variable": var_name,
        "n": 0,
        "mean": None,
        "median": None,
        "mode": [],
        "mode_count": None,
        "std": None,
        "variance": None,
        "min": None,
        "max": None,
        "range": None,
        "q1": None,
        "q3": None,
        "iqr": None,
        "skewness": None,
        "kurtosis": None,
    }
