"""
src/bootstrap.py
----------------
Experiment 3: Bootstrap Resampling

Estimates the uncertainty of the mean home-away goal difference using
non-parametric bootstrap resampling (10 000 iterations by default).

The resampling unit is a complete match observation (home_score, away_score
as a pair). This preserves the natural pairing of scores within each match.

Assumptions and limitations:
  - Matches are treated as approximately exchangeable for resampling.
  - National teams appear in multiple matches, so observations are not
    strictly independent. Ordinary match-level bootstrap may underestimate
    uncertainty. A year-block bootstrap is provided as an optional
    sensitivity analysis to partially account for temporal dependence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ── Experiment 3: Bootstrap ────────────────────────────────────────────────────


def bootstrap_mean_goal_diff(
    df: pd.DataFrame,
    n_iterations: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """
    Perform bootstrap resampling for the mean goal difference.

    Uses non-neutral matches. The goal difference per match is:
        D = home_score - away_score

    Each bootstrap sample resamples complete match rows with replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset. Non-neutral matches will be selected internally.
    n_iterations : int
        Number of bootstrap samples (minimum 1000; default 10 000).
    confidence_level : float
        Desired confidence level for the percentile interval (default 0.95).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with bootstrap results.
    """
    nn = df[~df["is_neutral"]].copy()

    if len(nn) < 10:
        return _empty_bootstrap(n_iterations, confidence_level, random_seed)

    diffs = nn["goal_difference"].values
    observed_mean = float(np.mean(diffs))
    n_matches = len(diffs)

    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(n_iterations, dtype=float)

    for i in range(n_iterations):
        sample = rng.choice(diffs, size=n_matches, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    bootstrap_se = float(np.std(bootstrap_means, ddof=1))

    return {
        "n_matches": n_matches,
        "observed_mean": observed_mean,
        "bootstrap_se": bootstrap_se,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "n_iterations": n_iterations,
        "random_seed": random_seed,
        "bootstrap_means": bootstrap_means.tolist(),
        "error": None,
    }


def bootstrap_year_block(
    df: pd.DataFrame,
    n_iterations: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """
    Year-block bootstrap (sensitivity analysis).

    Resamples years with replacement. Within each resampled year, all
    matches from that year are included. This partially accounts for
    temporal dependence among matches.

    Returns the same dict structure as bootstrap_mean_goal_diff.
    """
    nn = df[~df["is_neutral"]].copy()

    if len(nn) < 10:
        return _empty_bootstrap(n_iterations, confidence_level, random_seed)

    observed_mean = float(nn["goal_difference"].mean())
    years = nn["year"].unique()
    grouped = {yr: nn[nn["year"] == yr]["goal_difference"].values for yr in years}
    n_years = len(years)

    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(n_iterations, dtype=float)

    for i in range(n_iterations):
        sampled_years = rng.choice(years, size=n_years, replace=True)
        combined = np.concatenate([grouped[yr] for yr in sampled_years])
        bootstrap_means[i] = np.mean(combined)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    bootstrap_se = float(np.std(bootstrap_means, ddof=1))

    return {
        "n_matches": len(nn),
        "n_years": int(n_years),
        "observed_mean": observed_mean,
        "bootstrap_se": bootstrap_se,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": confidence_level,
        "n_iterations": n_iterations,
        "random_seed": random_seed,
        "bootstrap_means": bootstrap_means.tolist(),
        "error": None,
        "method": "year-block",
    }


def interpret_bootstrap(results: dict) -> str:
    """Generate a plain-language interpretation of bootstrap results."""
    if results.get("error") or results.get("n_matches", 0) < 10:
        return "Insufficient data for bootstrap analysis."

    obs = results["observed_mean"]
    lower = results["ci_lower"]
    upper = results["ci_upper"]
    ci_pct = int(results["confidence_level"] * 100)
    n = results["n_matches"]
    direction = "positive" if obs > 0 else "negative" if obs < 0 else "zero"
    excludes_zero = (lower > 0) or (upper < 0)

    lines = [
        f"Based on {n:,} non-neutral matches, the observed mean home-away goal "
        f"difference is **{obs:+.4f}** goals.",
        "",
        f"The {ci_pct}% percentile bootstrap confidence interval is "
        f"[**{lower:+.4f}**, **{upper:+.4f}**].",
        "",
    ]

    if excludes_zero:
        lines.append(
            f"Because the interval excludes zero and is entirely {direction}, "
            "this suggests that in the sampled population of matches, the average "
            "home team scores more goals than the away team. However, this does "
            "not imply that home venue alone causes the difference."
        )
    else:
        lines.append(
            "Because the interval includes zero, the estimated average home "
            "advantage is not clearly distinguishable from zero at this confidence "
            "level given the variability in the data."
        )

    lines += [
        "",
        "**Limitation:** Matches are not independent (teams appear repeatedly). "
        "The ordinary match-level bootstrap may underestimate uncertainty. "
        "See the year-block sensitivity analysis for a partial correction.",
    ]
    return "\n".join(lines)


# ── Internal ───────────────────────────────────────────────────────────────────


def _empty_bootstrap(n_iterations: int, confidence_level: float, random_seed: int) -> dict:
    return {
        "n_matches": 0,
        "observed_mean": None,
        "bootstrap_se": None,
        "ci_lower": None,
        "ci_upper": None,
        "confidence_level": confidence_level,
        "n_iterations": n_iterations,
        "random_seed": random_seed,
        "bootstrap_means": [],
        "error": "Insufficient non-neutral match data.",
    }
