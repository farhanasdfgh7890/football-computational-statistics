"""
src/hypothesis_testing.py
--------------------------
Experiment 5: Hypothesis Testing

Research question:
    Is the mean home-away goal difference in non-neutral international matches
    significantly greater than zero?

Primary method: One-sample t-test on match-level differences D = home_score - away_score
  H₀: μD = 0
  H₁: μD > 0 (one-sided, right-tailed)
  α  = 0.05

This is mathematically equivalent to a paired t-test comparing home_score
and away_score, which is also reported.

Additional sensitivity analysis: Wilcoxon signed-rank test.
  The Wilcoxon test assesses whether the distribution of signed differences
  is symmetric around zero (or has a median of zero), rather than testing
  the mean directly. Zero differences are handled by the Pratt method.

Effect size: Cohen's dz = mean(D) / std(D)
  - |dz| < 0.2  → negligible
  - |dz| ≈ 0.2  → small
  - |dz| ≈ 0.5  → medium
  - |dz| ≈ 0.8  → large
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ──────────────────────────────────────────────────────────────────

ALPHA: float = 0.05


# ── Experiment 5: Hypothesis Testing ───────────────────────────────────────────


def paired_ttest(df: pd.DataFrame) -> dict:
    """
    Perform one-sample t-test on match-level goal differences.

    Uses non-neutral matches only.

    Returns
    -------
    dict with full t-test results.
    """
    nn = df[~df["is_neutral"]].copy()

    if len(nn) < 3:
        return _empty_ttest()

    home = nn["home_score"].values.astype(float)
    away = nn["away_score"].values.astype(float)
    diffs = home - away   # D_i = home_i - away_i

    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    se_d = std_d / np.sqrt(n)
    df_t = n - 1

    # t-statistic
    t_stat = mean_d / se_d if se_d > 0 else 0.0

    # One-sided p-value (right-tailed: H₁: μ > 0)
    p_one_sided = float(stats.t.sf(t_stat, df=df_t))

    # Two-sided p-value and 95% CI
    p_two_sided = float(2 * min(stats.t.sf(abs(t_stat), df=df_t),
                                stats.t.cdf(abs(t_stat), df=df_t)))
    ci_margin = float(stats.t.ppf(1 - ALPHA / 2, df=df_t) * se_d)
    ci_lower = mean_d - ci_margin
    ci_upper = mean_d + ci_margin

    # Cohen's dz
    cohens_dz = mean_d / std_d if std_d > 0 else 0.0
    effect_label = _effect_label(abs(cohens_dz))

    # Statistical decision (one-sided)
    reject_h0 = p_one_sided < ALPHA

    return {
        "n": int(n),
        "mean_home": float(np.mean(home)),
        "mean_away": float(np.mean(away)),
        "mean_diff": float(mean_d),
        "std_diff": float(std_d),
        "se_diff": float(se_d),
        "t_stat": float(t_stat),
        "df": int(df_t),
        "p_one_sided": float(p_one_sided),
        "p_two_sided": float(p_two_sided),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "alpha": ALPHA,
        "cohens_dz": float(cohens_dz),
        "effect_label": effect_label,
        "reject_h0": bool(reject_h0),
        "decision": _decision_text(reject_h0, mean_d, p_one_sided),
        "diffs": diffs.tolist(),
        "error": None,
    }


def wilcoxon_test(df: pd.DataFrame) -> dict:
    """
    Wilcoxon signed-rank test on match-level goal differences (sensitivity analysis).

    H₀: The distribution of signed differences is symmetric around zero.
    Zero differences are handled by the Pratt method (included in ranking,
    then removed after ranking).

    Note: The Wilcoxon test is a sensitivity analysis. It does not test the
    same estimand as the paired t-test (mean), but is more robust to outliers.

    Returns
    -------
    dict with Wilcoxon test results.
    """
    nn = df[~df["is_neutral"]].copy()
    diffs = (nn["home_score"] - nn["away_score"]).values.astype(float)

    if len(diffs) < 10:
        return {"error": "Insufficient data for Wilcoxon test.", "n": len(diffs)}

    # scipy's Wilcoxon test handles zeros with zero_method='pratt' by default
    try:
        stat, p_value = stats.wilcoxon(
            diffs, alternative="greater", zero_method="pratt"
        )
        return {
            "n": len(diffs),
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": ALPHA,
            "reject_h0": bool(p_value < ALPHA),
            "note": (
                "Wilcoxon signed-rank test (one-sided, H₁: median difference > 0). "
                "Zero differences handled using the Pratt method. "
                "This tests the distribution's symmetry around zero, not the mean."
            ),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "n": len(diffs)}


def interpret_ttest(results: dict) -> str:
    """Generate a plain-language interpretation of t-test results."""
    if results.get("error") or results.get("n", 0) < 3:
        return "Insufficient data for hypothesis testing."

    n = results["n"]
    mean_h = results["mean_home"]
    mean_a = results["mean_away"]
    mean_d = results["mean_diff"]
    t = results["t_stat"]
    df_t = results["df"]
    p1 = results["p_one_sided"]
    p2 = results["p_two_sided"]
    lower = results["ci_lower"]
    upper = results["ci_upper"]
    dz = results["cohens_dz"]
    eff = results["effect_label"]
    reject = results["reject_h0"]

    lines = [
        f"**Sample:** {n:,} non-neutral international matches",
        f"**Mean home goals:** {mean_h:.4f}  |  **Mean away goals:** {mean_a:.4f}",
        f"**Mean difference (home − away):** {mean_d:+.4f}",
        "",
        f"**t({df_t}) = {t:.4f}**, one-sided p = {p1:.6f}, two-sided p = {p2:.6f}",
        f"**95% CI for mean difference:** [{lower:+.4f}, {upper:+.4f}]",
        f"**Cohen's dz = {dz:.4f}** ({eff} effect)",
        "",
        "**Statistical Decision:**",
        results["decision"],
        "",
        "**Interpretation:**",
    ]

    if reject:
        lines += [
            f"There is statistically significant evidence (α = {ALPHA}) that the mean "
            "home-away goal difference is greater than zero in non-neutral "
            "international matches.",
            "",
            "However, statistical significance does not establish causation. "
            "Home advantage may reflect factors beyond venue alone, including "
            "team selection, travel fatigue, crowd effects, referee bias, "
            "and tournament structure.",
            "",
            f"The effect size is {eff} (Cohen's dz = {dz:.4f}), suggesting the "
            "advantage, while real, is modest in absolute scoring terms.",
        ]
    else:
        lines += [
            "At the chosen significance level, we cannot reject the null hypothesis "
            "that the mean goal difference is zero. The data do not provide "
            "sufficient evidence of a positive home advantage in this sample.",
        ]

    return "\n".join(lines)


# ── Internal ───────────────────────────────────────────────────────────────────


def _decision_text(reject_h0: bool, mean_d: float, p: float) -> str:
    if reject_h0:
        return (
            f"**Reject H₀** (p = {p:.6f} < α = {ALPHA}). "
            f"The mean goal difference ({mean_d:+.4f}) is significantly greater than zero."
        )
    return (
        f"**Fail to reject H₀** (p = {p:.4f} ≥ α = {ALPHA}). "
        "Insufficient evidence that the mean goal difference exceeds zero."
    )


def _effect_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _empty_ttest() -> dict:
    return {
        "n": 0,
        "mean_home": None,
        "mean_away": None,
        "mean_diff": None,
        "std_diff": None,
        "se_diff": None,
        "t_stat": None,
        "df": None,
        "p_one_sided": None,
        "p_two_sided": None,
        "ci_lower": None,
        "ci_upper": None,
        "alpha": ALPHA,
        "cohens_dz": None,
        "effect_label": None,
        "reject_h0": None,
        "decision": "Insufficient data.",
        "diffs": [],
        "error": "Insufficient data.",
    }
