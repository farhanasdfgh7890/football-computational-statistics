"""
src/regression.py
-----------------
Experiment 2: Simple Linear Regression

Fits the model:
    mean_total_goals ~ year

Uses ordinary least-squares (OLS) via NumPy polyfit (primary) and
statsmodels OLS for full statistical output (p-values, confidence intervals).

An optional weighted regression (weighting each year by its match count)
is provided as a sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Experiment 2: Regression ───────────────────────────────────────────────────


def fit_yearly_regression(
    yearly_df: pd.DataFrame,
    weighted: bool = False,
) -> dict:
    """
    Fit simple linear regression: yearly_mean_total_goals ~ year.

    Parameters
    ----------
    yearly_df : pd.DataFrame
        Output of statistics.yearly_goals_summary().
        Must contain columns: year, mean_total_goals, n_matches.
    weighted : bool
        If True, weight each year by its match count (WLS approximation).

    Returns
    -------
    dict with regression results.
    """
    if len(yearly_df) < 3:
        return _empty_regression(weighted)

    x = yearly_df["year"].values.astype(float)
    y = yearly_df["mean_total_goals"].values.astype(float)
    n = len(x)

    if weighted and "n_matches" in yearly_df.columns:
        w = yearly_df["n_matches"].values.astype(float)
        w = w / w.sum()  # normalise
    else:
        w = None

    # scipy linregress (unweighted OLS)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

    # Predictions over observed range
    x_pred = np.linspace(x.min(), x.max(), 200)
    y_pred = intercept + slope * x_pred

    y_fit = intercept + slope * x
    residuals = y - y_fit
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Optional weighted slope (using numpy polyfit with weights)
    w_slope, w_intercept = None, None
    if weighted and w is not None:
        coeffs = np.polyfit(x, y, deg=1, w=w)
        w_slope = float(coeffs[0])
        w_intercept = float(coeffs[1])

    return {
        "n": int(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "r": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "mae": mae,
        "rmse": rmse,
        "x_obs": x.tolist(),
        "y_obs": y.tolist(),
        "x_pred": x_pred.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
        "weighted": weighted,
        "w_slope": w_slope,
        "w_intercept": w_intercept,
        "year_counts": (
            yearly_df["n_matches"].tolist()
            if "n_matches" in yearly_df.columns
            else None
        ),
    }


def predict_year(results: dict, year: float) -> float:
    """Return predicted mean goals for a given year using the fitted model."""
    return results["intercept"] + results["slope"] * year


def interpret_regression(results: dict) -> str:
    """
    Generate a plain-language interpretation of regression results.
    """
    if results.get("n", 0) < 3:
        return "Insufficient data for regression analysis."

    slope = results["slope"]
    r2 = results["r_squared"]
    p = results["p_value"]
    direction = "increased" if slope > 0 else "decreased"
    magnitude = abs(slope)
    sig = "statistically significant" if p < 0.05 else "not statistically significant"

    lines = [
        f"The regression slope is **{slope:+.4f}** goals per year, meaning that "
        f"the yearly average total goals per match has {direction} by approximately "
        f"{magnitude:.4f} goals per year over the study period.",
        "",
        f"The R² value is **{r2:.4f}**, indicating that calendar year explains "
        f"approximately {r2*100:.1f}% of the variance in yearly mean goals.",
        "",
        f"The slope is {sig} (p = {p:.4f}).",
        "",
        "**Caution:** Year is a proxy for time trends; this regression does not "
        "identify the causes of any trend. Rule changes, tournament expansion, "
        "defensive tactics, and other factors may all contribute.",
    ]
    return "\n".join(lines)


# ── Internal ───────────────────────────────────────────────────────────────────


def _empty_regression(weighted: bool) -> dict:
    return {
        "n": 0,
        "slope": None,
        "intercept": None,
        "r_squared": None,
        "r": None,
        "p_value": None,
        "std_err": None,
        "mae": None,
        "rmse": None,
        "x_obs": [],
        "y_obs": [],
        "x_pred": [],
        "y_pred": [],
        "residuals": [],
        "weighted": weighted,
        "w_slope": None,
        "w_intercept": None,
        "year_counts": None,
    }
