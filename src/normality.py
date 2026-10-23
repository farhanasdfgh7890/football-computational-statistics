"""
src/normality.py
----------------
Experiment 4: Normality Testing

Tests whether the distribution of goal differences is compatible with
a normal distribution.

Methods used:
  - Histogram with optional fitted normal curve
  - Normal Q-Q plot
  - Shapiro-Wilk test (H₀: data are normally distributed)

Shapiro-Wilk considerations:
  - scipy.stats.shapiro supports up to 5000 samples without warnings.
  - For samples larger than 5000, we draw a reproducible random subsample.
  - The test is performed on the subsample; visualisations use the full sample.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ──────────────────────────────────────────────────────────────────

SHAPIRO_MAX_N: int = 5000   # scipy's practical limit for Shapiro-Wilk
ALPHA: float = 0.05


# ── Experiment 4: Normality ────────────────────────────────────────────────────


def shapiro_wilk_test(
    df: pd.DataFrame,
    random_seed: int = 42,
    max_n: int = SHAPIRO_MAX_N,
) -> dict:
    """
    Run Shapiro-Wilk normality test on goal_difference for non-neutral matches.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset. Non-neutral subset is selected internally.
    random_seed : int
        Seed for reproducible subsampling if n > max_n.
    max_n : int
        Maximum sample size for the Shapiro-Wilk test.

    Returns
    -------
    dict with test results.
    """
    nn = df[~df["is_neutral"]].copy()
    diffs = nn["goal_difference"].dropna().values

    full_n = len(diffs)
    if full_n < 3:
        return _empty_normality()

    # Subsample if needed
    if full_n > max_n:
        rng = np.random.default_rng(random_seed)
        test_diffs = rng.choice(diffs, size=max_n, replace=False)
        subsampled = True
        test_n = max_n
    else:
        test_diffs = diffs
        subsampled = False
        test_n = full_n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value = stats.shapiro(test_diffs)

    reject_h0 = p_value < ALPHA
    decision = (
        f"Reject H₀ (p = {p_value:.6f} < {ALPHA}): The data are NOT consistent "
        "with a normal distribution."
        if reject_h0
        else
        f"Fail to reject H₀ (p = {p_value:.4f} ≥ {ALPHA}): The data are consistent "
        "with a normal distribution."
    )

    # Q-Q quantiles for plotting
    qq_theoretical, qq_sample = _qq_data(diffs)

    # Normal fit parameters for histogram overlay
    mu_fit = float(np.mean(diffs))
    sigma_fit = float(np.std(diffs, ddof=1))

    return {
        "full_n": int(full_n),
        "test_n": int(test_n),
        "subsampled": subsampled,
        "statistic": float(stat),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "reject_h0": bool(reject_h0),
        "decision": decision,
        "diffs_full": diffs.tolist(),
        "qq_theoretical": qq_theoretical.tolist(),
        "qq_sample": qq_sample.tolist(),
        "mu_fit": mu_fit,
        "sigma_fit": sigma_fit,
        "random_seed": random_seed,
        "max_n": max_n,
        "error": None,
    }


def interpret_normality(results: dict) -> str:
    """Generate a plain-language interpretation of normality test results."""
    if results.get("error") or results.get("full_n", 0) < 3:
        return "Insufficient data for normality testing."

    lines = [
        f"**Shapiro-Wilk test** was applied to the goal differences "
        f"(D = home_score − away_score) from {results['full_n']:,} non-neutral matches.",
        "",
    ]

    if results["subsampled"]:
        lines.append(
            f"Because the sample exceeds the recommended limit for Shapiro-Wilk "
            f"({results['max_n']:,} observations), the test was run on a random "
            f"subsample of {results['test_n']:,} matches (seed = {results['random_seed']}).",
        )
        lines.append("")

    lines += [
        f"**W = {results['statistic']:.6f}**, **p = {results['p_value']:.6f}**",
        "",
        results["decision"],
        "",
    ]

    if results["reject_h0"]:
        lines += [
            "The goal-difference distribution is **not** well-described by a "
            "normal distribution. This is expected: goal differences are discrete "
            "integers bounded at -(maximum goals) and strongly peaked at 0 and ±1.",
            "",
            "**Important distinction:** Statistical significance here reflects the "
            "very large sample size. Even small deviations from normality become "
            "significant with thousands of observations. The practical shape of the "
            "distribution may still be approximately bell-shaped.",
        ]
    else:
        lines += [
            "Failure to reject H₀ does not prove normality; it means that at this "
            "sample size and significance level, we cannot distinguish the data from "
            "a normal distribution. Goal differences remain discrete integers.",
        ]

    return "\n".join(lines)


# ── Internal ───────────────────────────────────────────────────────────────────


def _qq_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute theoretical quantiles and ordered sample quantiles for a Q-Q plot."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    # Standardise
    mu, sigma = np.mean(sorted_data), np.std(sorted_data, ddof=1)
    if sigma == 0:
        return np.zeros(n), sorted_data
    theoretical = stats.norm.ppf(
        (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    )
    return theoretical, sorted_data


def _empty_normality() -> dict:
    return {
        "full_n": 0,
        "test_n": 0,
        "subsampled": False,
        "statistic": None,
        "p_value": None,
        "alpha": ALPHA,
        "reject_h0": None,
        "decision": "Insufficient data.",
        "diffs_full": [],
        "qq_theoretical": [],
        "qq_sample": [],
        "mu_fit": None,
        "sigma_fit": None,
        "random_seed": 42,
        "max_n": SHAPIRO_MAX_N,
        "error": "Insufficient data.",
    }
