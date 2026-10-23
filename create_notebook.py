"""
create_notebook.py
Helper script to generate the Jupyter notebook.
Run from the project root: python create_notebook.py
"""
import json
import os

cells = []
_id = [0]

def new_id():
    _id[0] += 1
    return str(_id[0])

def md(source):
    if isinstance(source, list):
        source = "".join(source)
    cells.append({
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source,
    })

def code(source):
    if isinstance(source, list):
        source = "".join(source)
    cells.append({
        "cell_type": "code",
        "id": new_id(),
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    })

# ─── Title ────────────────────────────────────────────────────────────────────
md("""# ⚽ International Football Analytics
## A Computational Statistics Study of Goals, Home Advantage, and Performance Trends

**Subject:** Computational Statistics  
**Dataset:** International Football Results (Jürisoo, 2025)  
**Study Period:** January 1, 2000 – December 31, 2025  
**Repository:** https://github.com/martj42/international_results
""")

# ─── Abstract ─────────────────────────────────────────────────────────────────
md("""## Abstract

This notebook applies five computational statistics experiments to the publicly
available International Football Results dataset. The study investigates home
advantage, scoring patterns, temporal trends, statistical uncertainty, and the
distributional shape of goal differences.

**Five Experiments:**
1. Measures of Central Tendency
2. Linear Regression (yearly mean goals ~ year)
3. Bootstrap Resampling (95% CI for mean goal difference)
4. Normality Testing (Shapiro-Wilk + Q-Q plot)
5. Hypothesis Testing (Paired t-test, Wilcoxon)

All calculations are derived from the actual downloaded dataset.
""")

# ─── Setup ────────────────────────────────────────────────────────────────────
md("## Setup: Imports and Configuration\n")
code("""import sys, os
sys.path.insert(0, os.path.abspath(".."))  # allow src/ imports from notebooks/

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Project modules
from src.data_loader import load_dataset, describe_dataset
from src.preprocessing import clean_dataset, get_non_neutral, audit_to_dataframe
from src import statistics as stats_module
from src import regression as reg_module
from src import bootstrap as boot_module
from src import normality as norm_module
from src import hypothesis_testing as hyp_module

# Plot style
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0D1B2A", "axes.facecolor": "#1E2D3D",
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.edgecolor": "#2A3F55", "grid.color": "#2A3F55",
    "figure.figsize": (10, 5)
})
GREEN, BLUE, AMBER, RED = "#00B04F", "#2196F3", "#FFC107", "#FF4B4B"
print("Setup complete.")
""")

# ─── Data Loading ─────────────────────────────────────────────────────────────
md("""## 1. Dataset Loading

The dataset is downloaded from the GitHub repository and cached in `data/results.csv`.
""")
code("""df_raw, meta = load_dataset()
info = describe_dataset(df_raw)
print("Source :", meta["source"])
print("Rows   :", meta["rows_raw"])
print("Range  :", info["date_range"])
print("Columns:", info["columns"])
df_raw.head()
""")

# ─── Cleaning ─────────────────────────────────────────────────────────────────
md("""## 2. Data Cleaning

The cleaning pipeline applies 7 sequential steps documented in the audit log.
""")
code("""df, audit = clean_dataset(df_raw, start_date="2000-01-01", end_date="2025-12-31")
audit_df = audit_to_dataframe(audit)
print(audit_df.to_string(index=False))
print(f"\\nFinal dataset: {len(df):,} matches")
""")
code("""print("Neutral matches    :", df["is_neutral"].sum())
print("Non-neutral matches:", (~df["is_neutral"]).sum())
df.describe()
""")

# ─── EDA ──────────────────────────────────────────────────────────────────────
md("## 3. Exploratory Data Analysis\n")
code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["total_goals"], bins=range(0, 21), color=GREEN, alpha=0.8, edgecolor="black")
axes[0].axvline(df["total_goals"].mean(), color=AMBER, linestyle="--",
                label=f"Mean {df['total_goals'].mean():.2f}")
axes[0].set_title("Total Goals per Match"); axes[0].legend()
axes[0].set_xlabel("Total Goals"); axes[0].set_ylabel("Count")

yearly = stats_module.yearly_goals_summary(df)
axes[1].plot(yearly["year"], yearly["mean_total_goals"], color=GREEN, marker="o", ms=4)
axes[1].set_title("Yearly Mean Total Goals")
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Mean Goals")

result_counts = [df["home_win"].sum(), df["away_win"].sum(), df["draw"].sum()]
axes[2].bar(["Home Win", "Away Win", "Draw"], result_counts, color=[GREEN, BLUE, AMBER])
axes[2].set_title("Match Outcomes")

plt.tight_layout()
plt.savefig("../reports/figures/eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# ─── Experiment 1 ─────────────────────────────────────────────────────────────
md(r"""## Experiment 1: Measures of Central Tendency

### Objective
Describe the typical scoring behaviour of international football matches.

### Mathematical Formulas
- **Mean:** $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Variance (sample):** $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
- **IQR:** $Q_3 - Q_1$

### Variables Analysed
`total_goals`, `home_score`, `away_score`, `goal_difference`
""")
code("""tbl = stats_module.central_tendency_table(df)
display_cols = ["n","mean","median","std","variance","min","max","range","q1","q3","iqr","skewness"]
print("=" * 80)
print("DESCRIPTIVE STATISTICS TABLE")
print("=" * 80)
print(tbl[[c for c in display_cols if c in tbl.columns]].round(4).to_string())
""")
code("""print("\\nMODE ANALYSIS")
print("-" * 50)
for col_name, series in [
    ("Total Goals", df["total_goals"]),
    ("Home Score",  df["home_score"]),
    ("Away Score",  df["away_score"]),
    ("Goal Diff",   df["goal_difference"]),
]:
    stat = stats_module.central_tendency(series)
    print(f"{col_name:16s} | Mode: {stat['mode']} | Frequency: {stat['mode_count']}")
""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
tg = df["total_goals"].dropna()
axes[0].hist(tg, bins=range(0, 21), color=GREEN, alpha=0.8, edgecolor="black")
axes[0].axvline(tg.mean(), color=AMBER, linestyle="--", label=f"Mean = {tg.mean():.2f}")
axes[0].axvline(tg.median(), color=BLUE, linestyle=":", label=f"Median = {tg.median():.1f}")
axes[0].set_title("Histogram of Total Goals Per Match")
axes[0].set_xlabel("Total Goals"); axes[0].set_ylabel("Frequency")
axes[0].legend()

# Box plot
data_bp = [df["total_goals"].dropna(), df["home_score"].dropna(), df["away_score"].dropna()]
bp = axes[1].boxplot(data_bp, labels=["Total Goals","Home Score","Away Score"], patch_artist=True)
for patch, c in zip(bp["boxes"], [GREEN, BLUE, AMBER]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[1].set_title("Score Distributions (Box Plot)")
axes[1].set_ylabel("Goals")

plt.tight_layout()
plt.savefig("../reports/figures/exp1_central_tendency.png", dpi=150, bbox_inches="tight")
plt.show()
""")
md("""### Interpretation — Experiment 1

The **mean** of total goals per match is the expected scoring level. Because
the distribution is right-skewed, the mean exceeds the median. The **mode**
is the most common combined scoreline total. The **IQR** measures the spread
of the middle 50% of matches. Home teams consistently average more goals than
away teams, consistent with a home advantage hypothesis.
""")

# ─── Experiment 2 ─────────────────────────────────────────────────────────────
md(r"""## Experiment 2: Regression Analysis

### Objective
Investigate whether average goals per match has changed over time.

### Model
$$\hat{y} = b_0 + b_1 x$$

where $x$ = calendar year, $\hat{y}$ = predicted yearly mean total goals.

### Estimation
OLS minimises $\sum(y_i - \hat{y}_i)^2$.

**Caution:** Calendar year is a time proxy. This regression does not
establish causation.
""")
code("""yearly = stats_module.yearly_goals_summary(df)
reg = reg_module.fit_yearly_regression(yearly)

print("=" * 50)
print("REGRESSION RESULTS")
print("=" * 50)
print(f"  n (years)    : {reg['n']}")
print(f"  Slope (b1)   : {reg['slope']:+.6f} goals/year")
print(f"  Intercept b0 : {reg['intercept']:.4f}")
print(f"  R-squared    : {reg['r_squared']:.6f}")
print(f"  p-value      : {reg['p_value']:.6f}")
print(f"  RMSE         : {reg['rmse']:.6f}")
""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(yearly["year"], yearly["mean_total_goals"],
                color=GREEN, s=50, alpha=0.8, label="Yearly Mean")
axes[0].plot(reg["x_pred"], reg["y_pred"], color=RED, linewidth=2,
             label=f"OLS slope={reg['slope']:+.4f}")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Mean Total Goals")
axes[0].set_title(f"Linear Regression (R\u00b2 = {reg['r_squared']:.4f})")
axes[0].legend()

axes[1].scatter(reg["x_obs"], reg["residuals"], color=AMBER, s=40, alpha=0.8)
axes[1].axhline(0, color=RED, linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("../reports/figures/exp2_regression.png", dpi=150, bbox_inches="tight")
plt.show()
""")
md("""### Interpretation — Experiment 2

The slope $b_1$ indicates whether average match scoring has risen or fallen
over the study period. R² quantifies how much year explains year-to-year
variation in average goals. Residuals are inspected for systematic patterns.
""")

# ─── Experiment 3 ─────────────────────────────────────────────────────────────
md(r"""## Experiment 3: Bootstrap Resampling

### Objective
Estimate the 95% confidence interval for the mean home-away goal difference
without relying on normality.

### Method
Percentile bootstrap with B = 10,000 iterations on non-neutral matches.

**Variable:** $D_i = \text{home\_score}_i - \text{away\_score}_i$

**Algorithm:**
1. Compute observed $\bar{D}$.
2. For $b = 1, \ldots, B$: resample $n$ matches with replacement, compute $\bar{D}^*_b$.
3. 95% CI = $[P_{2.5}(\bar{D}^*), P_{97.5}(\bar{D}^*)]$
""")
code("""boot = boot_module.bootstrap_mean_goal_diff(df, n_iterations=10_000, random_seed=42)

print("=" * 50)
print("BOOTSTRAP RESULTS")
print("=" * 50)
print(f"  Eligible matches : {boot['n_matches']:,}")
print(f"  Observed mean D  : {boot['observed_mean']:+.6f}")
print(f"  Bootstrap SE     : {boot['bootstrap_se']:.6f}")
print(f"  95% CI Lower     : {boot['ci_lower']:+.6f}")
print(f"  95% CI Upper     : {boot['ci_upper']:+.6f}")
print(f"  Iterations       : {boot['n_iterations']:,}")
print(f"  Random seed      : {boot['random_seed']}")
""")
code("""fig, ax = plt.subplots(figsize=(10, 5))
boot_means = np.array(boot["bootstrap_means"])
ax.hist(boot_means, bins=80, color=BLUE, alpha=0.75, density=True)
ax.axvline(boot["observed_mean"], color=GREEN, linewidth=2,
           label=f"Observed Mean = {boot['observed_mean']:+.4f}")
ax.axvline(boot["ci_lower"], color=AMBER, linestyle="--", linewidth=1.5,
           label=f"95% CI = [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")
ax.axvline(boot["ci_upper"], color=AMBER, linestyle="--", linewidth=1.5)
ax.set_xlabel("Bootstrap Mean Goal Difference")
ax.set_ylabel("Density")
ax.set_title("Bootstrap Distribution of Mean Home-Away Goal Difference")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/figures/exp3_bootstrap.png", dpi=150, bbox_inches="tight")
plt.show()
""")
md("""### Interpretation — Experiment 3

The 95% bootstrap CI provides a range that, under repeated sampling, would
contain the true mean difference 95% of the time. If the CI excludes zero
on the positive side, this is consistent with home advantage.

**Limitation:** The ordinary match-level bootstrap assumes approximate
exchangeability. Teams appear in multiple matches. The year-block bootstrap
(see `boot_module.bootstrap_year_block`) partially addresses this.
""")

# ─── Experiment 4 ─────────────────────────────────────────────────────────────
md(r"""## Experiment 4: Normality Testing

### Objective
Assess whether the distribution of goal differences is compatible with a
normal distribution.

### Hypotheses
- $H_0$: The goal differences are drawn from a normal distribution.
- $H_1$: The goal differences are not drawn from a normal distribution.
- $\alpha = 0.05$

### Method
Shapiro-Wilk test. If $n > 5000$, a reproducible subsample is used.
""")
code("""norm = norm_module.shapiro_wilk_test(df, random_seed=42)

print("=" * 50)
print("SHAPIRO-WILK NORMALITY TEST")
print("=" * 50)
print(f"  Full sample n : {norm['full_n']:,}")
print(f"  Test sample n : {norm['test_n']:,}")
print(f"  Subsampled    : {norm['subsampled']}")
print(f"  W statistic   : {norm['statistic']:.6f}")
print(f"  p-value       : {norm['p_value']:.6f}")
print(f"  Alpha         : {norm['alpha']}")
print(f"  Decision      : {norm['decision']}")
""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
diffs = np.array(norm["diffs_full"])

# Histogram with normal fit
axes[0].hist(diffs, bins=range(int(diffs.min()), int(diffs.max())+2),
             density=True, color=GREEN, alpha=0.75, label="Observed")
x_norm = np.linspace(diffs.min(), diffs.max(), 300)
axes[0].plot(x_norm, stats.norm.pdf(x_norm, norm["mu_fit"], norm["sigma_fit"]),
             color=RED, linewidth=2, label="Fitted Normal")
axes[0].set_title("Goal Difference vs. Normal Distribution")
axes[0].set_xlabel("Goal Difference (Home - Away)")
axes[0].set_ylabel("Probability Density")
axes[0].legend()

# Q-Q plot
theoretical = np.array(norm["qq_theoretical"])
sample_q = np.array(norm["qq_sample"])
mu_s = np.mean(sample_q); sigma_s = np.std(sample_q, ddof=1)
sample_std = (sample_q - mu_s) / (sigma_s if sigma_s > 0 else 1)
axes[1].scatter(theoretical, sample_std, color=BLUE, s=2, alpha=0.5, label="Sample Quantiles")
q_range = [min(theoretical.min(), sample_std.min()), max(theoretical.max(), sample_std.max())]
axes[1].plot(q_range, q_range, color=RED, linestyle="--", linewidth=1.5, label="Normal Ref.")
axes[1].set_title("Normal Q-Q Plot of Goal Differences")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles (Standardised)")
axes[1].legend()

plt.tight_layout()
plt.savefig("../reports/figures/exp4_normality.png", dpi=150, bbox_inches="tight")
plt.show()
""")
md("""### Interpretation — Experiment 4

Goal differences are discrete integers. With large samples, Shapiro-Wilk
almost always rejects normality because it detects even trivial deviations.
The Q-Q plot reveals whether departures are systematic (e.g., heavy tails)
or minor. Statistical non-normality does not necessarily imply practical
non-normality.
""")

# ─── Experiment 5 ─────────────────────────────────────────────────────────────
md(r"""## Experiment 5: Hypothesis Testing

### Objective
Test whether the mean home-away goal difference is significantly greater than zero.

### Hypotheses
- $H_0: \mu_D = 0$
- $H_1: \mu_D > 0$ (one-sided, right-tailed)
- $\alpha = 0.05$

### Test Statistic
$$t = \frac{\bar{D}}{s_D / \sqrt{n}}, \quad df = n-1$$

### Effect Size
$$d_z = \frac{\bar{D}}{s_D}$$
Benchmarks: negligible < 0.2, small ≈ 0.2, medium ≈ 0.5, large ≈ 0.8
""")
code("""ttest = hyp_module.paired_ttest(df)

print("=" * 55)
print("PAIRED T-TEST RESULTS")
print("=" * 55)
print(f"  Eligible matches   : {ttest['n']:,}")
print(f"  Mean home goals    : {ttest['mean_home']:.6f}")
print(f"  Mean away goals    : {ttest['mean_away']:.6f}")
print(f"  Mean difference D  : {ttest['mean_diff']:+.6f}")
print(f"  Std of D           : {ttest['std_diff']:.6f}")
print(f"  t-statistic        : {ttest['t_stat']:.6f}")
print(f"  Degrees of freedom : {ttest['df']}")
print(f"  One-sided p-value  : {ttest['p_one_sided']:.6f}")
print(f"  Two-sided p-value  : {ttest['p_two_sided']:.6f}")
print(f"  95% CI             : [{ttest['ci_lower']:+.6f}, {ttest['ci_upper']:+.6f}]")
print(f"  Cohen's dz         : {ttest['cohens_dz']:.6f} ({ttest['effect_label']} effect)")
print(f"  Decision           : {ttest['decision']}")
""")
code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Home vs Away bar
axes[0].bar(["Home", "Away"], [ttest["mean_home"], ttest["mean_away"]],
            color=[GREEN, BLUE], alpha=0.8)
for i, v in enumerate([ttest["mean_home"], ttest["mean_away"]]):
    axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", color="white")
axes[0].set_title("Mean Goals: Home vs Away (Non-Neutral Matches)")
axes[0].set_ylabel("Mean Goals per Match")

# Histogram of differences
diffs_t = np.array(ttest["diffs"])
axes[1].hist(diffs_t, bins=range(int(diffs_t.min()), int(diffs_t.max())+2),
             color=GREEN, alpha=0.8)
axes[1].axvline(ttest["mean_diff"], color=AMBER, linestyle="--", linewidth=2,
                label=f"Mean D = {ttest['mean_diff']:+.4f}")
axes[1].axvline(0, color=RED, linestyle=":", linewidth=1.5, label="H0: D = 0")
axes[1].set_title("Distribution of Match-Level Goal Differences")
axes[1].set_xlabel("Goal Difference (Home - Away)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("../reports/figures/exp5_hypothesis.png", dpi=150, bbox_inches="tight")
plt.show()
""")
code("""# Wilcoxon signed-rank sensitivity analysis
wilcox = hyp_module.wilcoxon_test(df)
print("WILCOXON SIGNED-RANK TEST (Sensitivity Analysis)")
print("=" * 55)
if wilcox.get("error"):
    print("Error:", wilcox["error"])
else:
    print(f"  W statistic  : {wilcox['statistic']:.2f}")
    print(f"  One-sided p  : {wilcox['p_value']:.6f}")
    print(f"  Reject H0    : {wilcox['reject_h0']}")
    print(f"  Note: {wilcox['note']}")
""")
md("""### Interpretation — Experiment 5

The paired t-test formally tests whether home teams score significantly more
than away teams. Statistical significance (p < 0.05) means the result is
unlikely by chance alone. Cohen's dz quantifies practical magnitude.

**Caution:** Home advantage in goals does not mean venue alone is the cause.
Confounders include travel fatigue, crowd support, team selection, and
tournament structure.
""")

# ─── Summary ──────────────────────────────────────────────────────────────────
md("## Comparison of Results Across All Experiments\n")
code("""print("SUMMARY ACROSS ALL FIVE EXPERIMENTS")
print("=" * 65)
stat_goals = stats_module.central_tendency(df["total_goals"])
stat_diff  = stats_module.central_tendency(df["goal_difference"])
print(f"Exp 1 | Mean total goals    : {stat_goals['mean']:.4f}")
print(f"Exp 1 | Mean goal diff      : {stat_diff['mean']:+.4f}")
print(f"Exp 2 | Regression slope    : {reg['slope']:+.6f} goals/year")
print(f"Exp 2 | R-squared           : {reg['r_squared']:.4f}")
print(f"Exp 3 | Bootstrap mean D    : {boot['observed_mean']:+.4f}")
print(f"Exp 3 | 95% CI              : [{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]")
print(f"Exp 4 | Shapiro-Wilk W      : {norm['statistic']:.6f}")
print(f"Exp 4 | p-value             : {norm['p_value']:.6f}")
print(f"Exp 4 | Reject normality    : {norm['reject_h0']}")
print(f"Exp 5 | t-statistic         : {ttest['t_stat']:.4f}")
print(f"Exp 5 | One-sided p-value   : {ttest['p_one_sided']:.6f}")
print(f"Exp 5 | Cohen's dz          : {ttest['cohens_dz']:.4f} ({ttest['effect_label']})")
print(f"Exp 5 | Reject H0 (muD=0)  : {ttest['reject_h0']}")
""")

md("""## Limitations

1. **Non-independence:** Teams appear in multiple matches.
2. **Tournament heterogeneity:** Friendlies and competitive matches are pooled.
3. **Shapiro-Wilk power:** Very large n almost always rejects normality.
4. **Causation:** Regression and t-test show association, not causation.
5. **Dataset completeness:** Not verified to be exhaustive.

## Conclusion

Five statistical methods provide a consistent picture of home advantage and
scoring patterns in international football. The bootstrap CI and t-test
agree on the direction of home advantage. All results are derived from the
actual downloaded dataset.

## References

1. Jürisoo, M. (2025). *International Football Results Dataset*. GitHub.
2. Efron, B. (1979). Bootstrap methods. *Annals of Statistics*, 7(1), 1–26.
3. Shapiro, S. S., & Wilk, M. B. (1965). *Biometrika*, 52(3–4), 591–611.
4. Cohen, J. (1988). *Statistical Power Analysis* (2nd ed.). Erlbaum.
5. Virtanen, P., et al. (2020). SciPy 1.0. *Nature Methods*, 17, 261–272.
""")

# ─── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out_path = os.path.join(
    os.path.dirname(__file__),
    "notebooks",
    "football_computational_statistics.ipynb",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"Notebook written to: {out_path}")
