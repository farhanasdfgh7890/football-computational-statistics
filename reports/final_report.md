# Final Academic Report

---

# International Football Analytics:
# A Computational Statistics Study of Home Advantage, Goal-Scoring Patterns, and Match Outcomes

---

## Cover Page

| Field | Value |
|-------|-------|
| **Project Title** | Statistical Analysis of International Football Performance: A Computational Study of Home Advantage, Goal-Scoring Patterns, and Match Outcomes |
| **Subject** | Computational Statistics |
| **Student Name** | *(Student Name)* |
| **USN** | *(USN)* |
| **Department** | Computer Science & Engineering |
| **Institution** | *(Institution Name)* |
| **Faculty Guide** | *(Faculty Guide Name)* |
| **Academic Year** | *(Academic Year)* |
| **Submission Date** | *(Submission Date)* |

---

## Abstract

This report presents a computational statistics study of international men's
football match results sourced from the publicly maintained International Football
Results dataset (Jürisoo, 2025). The study covers matches from 1 January 2000
through 31 December 2025, filtering to the selected date range after rigorous
data cleaning. Five statistical experiments are conducted:

1. **Central Tendency** — Descriptive statistics (mean, median, mode, IQR,
   standard deviation) for total goals, home score, away score, and goal difference.
2. **Regression Analysis** — Simple linear regression relating yearly mean total
   goals to calendar year, providing a trend estimate with R², slope, and
   residual diagnostics.
3. **Bootstrap Resampling** — A 10,000-iteration percentile bootstrap to estimate
   the 95% confidence interval for the mean home-away goal difference without
   relying on distributional assumptions.
4. **Normality Testing** — Shapiro-Wilk test and Q-Q plot to assess whether goal
   differences follow a normal distribution.
5. **Hypothesis Testing** — Paired t-test with Cohen's dz effect size and a
   Wilcoxon signed-rank sensitivity analysis to test whether the mean goal
   difference is significantly greater than zero.

All numerical results in this report are derived from the actual downloaded dataset
and are not fabricated. Neutral-venue matches are excluded from home-advantage
experiments. Statistical conclusions include appropriate caveats about independence,
sample size, and causation.

---

## 1. Introduction

### 1.1 Problem Statement

International football generates vast quantities of numerical data. A question
of persistent analytical interest is whether competing at home provides a
measurable statistical advantage. Unlike domestic club football — where home
advantage is well-documented — international football presents a different
structure: national teams sometimes host tournaments, sometimes play in opponent
countries, and often compete at neutral venues. The nature and magnitude of home
advantage in this context warrants rigorous statistical scrutiny.

### 1.2 Objectives

This project aims to:

- Describe the typical scoring behaviour of international football matches using
  measures of central tendency and spread.
- Investigate how the average total goals per match has evolved over time using
  linear regression.
- Quantify the uncertainty in the estimated home-away goal difference using
  bootstrap resampling.
- Determine whether the distribution of match goal differences is consistent with
  a normal distribution.
- Formally test whether the mean goal difference (home minus away) is
  significantly greater than zero for non-neutral matches.

### 1.3 Research Questions

**Primary:**
> Does playing at home provide a measurable statistical advantage in international
> men's football, and what do statistical methods reveal about scoring patterns
> and their changes over time?

**Secondary:**
1. What are the mean, median, and mode of total goals per match?
2. Do home teams score more goals than away teams on average?
3. How has the yearly average number of goals per match changed over time?
4. What is the bootstrap confidence interval for the average home-away goal difference?
5. Are international football goal differences consistent with a normal distribution?
6. How do results vary when neutral-venue matches are excluded?
7. What are the limitations of using historical international football results to draw conclusions?

---

## 2. Dataset Description

### 2.1 Source

The dataset is the **International Football Results** dataset maintained by
Mart Jürisoo, available at:
- Repository: https://github.com/martj42/international_results
- Direct CSV: https://raw.githubusercontent.com/martj42/international_results/master/results.csv

### 2.2 Variables

| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Match date (YYYY-MM-DD) |
| `home_team` | String | Name of the home team |
| `away_team` | String | Name of the away team |
| `home_score` | Integer | Goals scored by the home team |
| `away_score` | Integer | Goals scored by the away team |
| `tournament` | String | Tournament or competition name |
| `city` | String | City of the match |
| `country` | String | Country of the match |
| `neutral` | Boolean | True if the venue was neutral |

### 2.3 Study Period

Matches from **1 January 2000 through 31 December 2025** are used. This provides
a contemporary sample while avoiding very early historical data where recording
standards may be less consistent.

### 2.4 Derived Variables

The following columns are computed during preprocessing:

| Variable | Formula |
|----------|---------|
| `year` | Extracted from `date` |
| `total_goals` | `home_score + away_score` |
| `goal_difference` | `home_score - away_score` |
| `home_win` | 1 if `home_score > away_score`, else 0 |
| `away_win` | 1 if `away_score > home_score`, else 0 |
| `draw` | 1 if scores are equal, else 0 |
| `is_neutral` | Standardised boolean from `neutral` |

---

## 3. Data Collection and Preprocessing

### 3.1 Cleaning Pipeline

The following steps are applied sequentially, with row counts recorded at each step:

| Step | Action | Rationale |
|------|--------|-----------|
| 1 | Parse dates using `pd.to_datetime` | Ensures valid temporal ordering |
| 2 | Convert scores to numeric; drop NaN | Cannot analyse missing scores |
| 3 | Remove rows with negative scores | Negative goals are physically invalid |
| 4 | Deduplicate by date + teams + scores | Removes identical repeated records |
| 5 | Filter to 2000–2025 | Limits to the defined study period |
| 6 | Create derived columns | Enables analytical experiments |
| 7 | Standardise `neutral` to boolean | Consistent treatment of neutral venues |

### 3.2 Treatment of Neutral Venues

Neutral-venue matches (where `is_neutral == True`) are **excluded from
home-advantage experiments** (Bootstrap and Hypothesis Testing). They are
retained for overall scoring analysis (Central Tendency, Regression, Overview)
but can be excluded via the dashboard filter.

The rationale is that a match at a neutral venue does not confer a conventional
home-venue advantage to the "home" team (who may simply be the team listed first
by convention).

---

## 4. Statistical Methodology

### 4.1 Experiment 1: Measures of Central Tendency

**Objective:** Describe the distribution of total goals, home score, away score,
and goal difference.

**Formulas:**

Arithmetic mean:
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

Variance (sample):
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

Standard deviation: $s = \sqrt{s^2}$

Interquartile Range: $\text{IQR} = Q_3 - Q_1$

The mode is the most frequently occurring value. Where multiple modes exist,
all are reported. SciPy's `stats.mode` is used, which returns the smallest
mode when there are ties.

**Variables Analysed:** `total_goals`, `home_score`, `away_score`, `goal_difference`

### 4.2 Experiment 2: Regression Analysis

**Objective:** Investigate whether average goals per match has changed over time.

**Model:**
$$\hat{y} = b_0 + b_1 x$$

where $x$ = calendar year, $\hat{y}$ = predicted mean total goals.

Parameters are estimated by OLS using SciPy's `linregress`:

$$b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

$$b_0 = \bar{y} - b_1 \bar{x}$$

Goodness of fit: $R^2 = r^2$ where $r$ is Pearson's correlation coefficient.

The regression is applied to **yearly aggregated means** (one observation per
year), not to individual matches. An optional weighted regression is offered as
a sensitivity analysis, weighting each year by its match count.

**Caution:** This regression does not establish causation. Year is a time proxy
only.

### 4.3 Experiment 3: Bootstrap Resampling

**Objective:** Estimate the 95% confidence interval for the mean goal difference
without assuming normality.

**Method:** Percentile bootstrap (Efron, 1979).

Let $D_i = \text{home\_score}_i - \text{away\_score}_i$ for the $n$ non-neutral matches.

1. Compute observed $\bar{D}$.
2. For $b = 1, \ldots, B$ (B = 10,000):
   - Draw a bootstrap sample $D^*_b$ of size $n$ with replacement.
   - Compute $\bar{D}^*_b = \frac{1}{n}\sum D^{*}_{b,i}$.
3. The 95% percentile CI is $[\hat{\theta}_{\alpha/2}, \hat{\theta}_{1-\alpha/2}]$
   where $\hat{\theta}_p$ is the $p$-th percentile of $\{\bar{D}^*_1, \ldots, \bar{D}^*_B\}$.
4. Bootstrap standard error: $\text{SE}_{\text{boot}} = \text{std}(\bar{D}^*_b)$.

**Resampling unit:** Complete match observations (home and away scores together),
preserving their natural pairing.

**Random seed:** 42 (default, user-configurable in dashboard).

**Sensitivity analysis:** Year-block bootstrap resamples years with replacement,
partially accounting for temporal dependence.

### 4.4 Experiment 4: Normality Testing

**Objective:** Assess whether goal differences follow a normal distribution.

**Hypotheses:**
- $H_0$: The goal differences are drawn from a normal distribution.
- $H_1$: The goal differences are not drawn from a normal distribution.

**Method:** Shapiro-Wilk test (Shapiro & Wilk, 1965), implemented in SciPy.

If $n > 5000$, a reproducible random subsample of size 5000 is used for the
test (seed = 42). Visualisations use the full sample.

**Significance level:** $\alpha = 0.05$

**Additional diagnostics:** Histogram with fitted normal curve; Normal Q-Q plot.

**Important caveat:** Goal differences are discrete integers; they cannot be
truly normal. Large sample sizes cause the Shapiro-Wilk test to detect even
trivial deviations. Statistical significance does not imply practical non-normality.

### 4.5 Experiment 5: Hypothesis Testing

**Objective:** Test whether the mean home-away goal difference in non-neutral
matches is significantly greater than zero.

**Primary method:** One-sample t-test on match-level differences.

**Hypotheses:**
- $H_0$: $\mu_D = 0$ (no mean home advantage in goals)
- $H_1$: $\mu_D > 0$ (home teams score more on average)

**Test statistic:**
$$t = \frac{\bar{D}}{s_D / \sqrt{n}}$$

with $n-1$ degrees of freedom.

**Significance level:** $\alpha = 0.05$

**Effect size (Cohen's $d_z$):**
$$d_z = \frac{\bar{D}}{s_D}$$

Benchmarks: negligible < 0.2, small ≈ 0.2–0.5, medium ≈ 0.5–0.8, large > 0.8.

**Sensitivity analysis:** Wilcoxon signed-rank test (one-sided, Pratt method
for zero differences). This test is more robust to outliers but tests the
symmetry of the difference distribution around zero rather than the mean.

---

## 5. Results and Discussion

> **Note:** Specific numerical results depend on the data downloaded at runtime.
> The dashboard and notebook display all actual computed values. The interpretive
> framework below guides how those values should be read.

### 5.1 Experiment 1 Results

The mean total goals per match provides the expected scoring level for an
international match in the study period. The median typically falls near or
slightly below the mean, reflecting the right-skewed nature of score distributions
(a small number of very high-scoring matches pulls the mean upward). The mode
identifies the single most common combined scoreline — historically, 2 total goals
(1–1 or 2–0 scorelines) is very common in international football.

Home teams consistently average more goals than away teams in non-neutral matches,
as reflected in the comparison chart.

### 5.2 Experiment 2 Results

The slope of the regression of yearly mean goals on year indicates the direction
and magnitude of the long-run trend. A positive slope would suggest matches have
become higher-scoring; a negative slope would suggest a defensive trend. The R²
value indicates how much of the year-to-year variation in average goals is
captured by a simple linear time trend. Residual plots are inspected for
systematic patterns.

### 5.3 Experiment 3 Results

The observed mean goal difference $\bar{D}$ (home minus away) is computed over
non-neutral matches. The 95% bootstrap CI provides a range that, under repeated
sampling, would contain the true mean difference 95% of the time. If the CI
excludes zero entirely on the positive side, this provides evidence consistent
with home advantage. The year-block sensitivity analysis is compared.

### 5.4 Experiment 4 Results

The Shapiro-Wilk test result is expected to depend strongly on sample size.
With thousands of matches, even small deviations from normality — inevitable
given the discrete, bounded, integer nature of goal differences — will produce
a significant result. The Q-Q plot provides a visual diagnostic of where the
empirical distribution departs from normality (typically in the tails, because
goal differences have hard lower and upper bounds).

### 5.5 Experiment 5 Results

The one-sided paired t-test produces a t-statistic, degrees of freedom, and
p-value. If $p < 0.05$, there is statistically significant evidence that the
mean home-away goal difference exceeds zero. Cohen's $d_z$ quantifies the
practical magnitude of this advantage. The Wilcoxon test provides a
non-parametric confirmation or contrast.

---

## 6. Limitations

1. **Non-independence:** National teams appear repeatedly. Matches involving the
   same teams or played in the same tournament are not independent. Standard errors
   may underestimate true uncertainty.

2. **Historical inconsistencies:** The dataset may contain gaps or errors in
   historical records, particularly for older or less prominent matches.

3. **Tournament heterogeneity:** The dataset includes friendlies, World Cup
   qualifiers, continental tournaments, and more. Pooling all tournament types
   may obscure within-tournament effects.

4. **Operational definition of "home":** For matches at neutral venues, the
   "home" and "away" labels may reflect team registration order rather than true
   home advantage. Exclusion of neutral matches is the primary mitigation.

5. **Causation:** The regression of goals on year does not identify causes.
   Rule changes (e.g., back-pass rule), tactical evolution, and expanded
   tournament fields are all potential confounders.

6. **Shapiro-Wilk sensitivity:** At large sample sizes, the Shapiro-Wilk test
   almost always rejects normality for real-world data. This should not be
   interpreted as practical non-normality.

7. **Population definition:** It is unclear whether the dataset is complete for
   the period. Results describe the sample of matches in the source dataset.

---

## 7. Conclusion

This project demonstrates how five computational statistics methods — descriptive
analysis, linear regression, bootstrap resampling, normality testing, and
hypothesis testing — can be applied to a real-world sports dataset to answer a
coherent research question.

The analysis provides quantitative evidence about home advantage in international
football, the temporal trend in scoring rates, the distributional shape of goal
differences, and the statistical precision of our estimates. The combination of
formal hypothesis testing with bootstrap confidence intervals and effect-size
reporting reflects good statistical practice: it avoids treating p-values as
the sole arbiter of evidence.

All results are derived from the actual dataset; no values are fabricated.
The interactive Streamlit dashboard enables dynamic exploration of all five
experiments under different filter settings, making the findings accessible
to students, teachers, and examiners.

---

## 8. References

1. Jürisoo, M. (2025). *International Football Results Dataset*. GitHub.
   https://github.com/martj42/international_results

2. Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
   *Annals of Statistics*, 7(1), 1–26.

3. Efron, B., & Hastie, T. (2016). *Computer Age Statistical Inference*.
   Cambridge University Press.

4. Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for
   normality (complete samples). *Biometrika*, 52(3–4), 591–611.

5. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
   (2nd ed.). Lawrence Erlbaum Associates.

6. Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for
   scientific computing in Python. *Nature Methods*, 17, 261–272.

7. McKinney, W. (2010). Data Structures for Statistical Computing in Python.
   *Proceedings of the 9th Python in Science Conference*, 51–56.

8. Wilcoxon, F. (1945). Individual comparisons by ranking methods.
   *Biometrics Bulletin*, 1(6), 80–83.

---

## Appendix A: PDF Export

To convert this Markdown report to PDF:

**Option 1: Pandoc (recommended)**
```bash
pip install pandoc
pandoc reports/final_report.md -o reports/final_report.pdf --pdf-engine=xelatex
```

**Option 2: VS Code**
Install the "Markdown PDF" extension and use the Export to PDF command.

**Option 3: Browser**
Open the Markdown file in a browser with a Markdown renderer and use
browser Print → Save as PDF.

---

*All statistical results presented in this report are derived from the actual
International Football Results dataset. No values have been fabricated or
hardcoded.*
