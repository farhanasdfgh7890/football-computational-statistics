"""
src/visualizations.py
---------------------
All chart-generation functions for the International Football Analytics project.

Charts are generated using Matplotlib/Seaborn (for static export) or
Plotly (for interactive dashboard use). Each function returns a Plotly Figure
so the dashboard can display interactive charts, and callers can call
.write_image() for static export where needed.

Colour palette
--------------
  navy      #0D1B2A
  charcoal  #1E2D3D
  green     #00B04F
  emerald   #00D46A
  blue      #2196F3
  light     #E8F4FD
  white     #FFFFFF
  red       #FF4B4B
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats


# ── Palette ────────────────────────────────────────────────────────────────────

NAVY    = "#0D1B2A"
CHARCOAL = "#1E2D3D"
GREEN   = "#00B04F"
EMERALD = "#00D46A"
BLUE    = "#2196F3"
WHITE   = "#FFFFFF"
RED     = "#FF4B4B"
AMBER   = "#FFC107"
LIGHT   = "#E8F4FD"

DARK_TEMPLATE = dict(
    plot_bgcolor=CHARCOAL,
    paper_bgcolor=NAVY,
    font=dict(color=WHITE, family="Inter, Arial, sans-serif"),
    xaxis=dict(gridcolor="#2A3F55", zerolinecolor="#2A3F55"),
    yaxis=dict(gridcolor="#2A3F55", zerolinecolor="#2A3F55"),
    title_font=dict(size=16, color=WHITE),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2A3F55"),
)


def _apply_dark(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        **DARK_TEMPLATE,
        title=dict(text=title, font=dict(size=16, color=WHITE)),
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


# ── Experiment 1: Central Tendency ────────────────────────────────────────────


def plot_total_goals_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of total goals per match."""
    vals = df["total_goals"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=int(vals.max() - vals.min() + 1) if len(vals) > 0 else 10,
        marker_color=GREEN,
        opacity=0.85,
        name="Matches",
    ))
    mean_val = vals.mean()
    median_val = vals.median()
    fig.add_vline(x=mean_val, line_dash="dash", line_color=AMBER,
                  annotation_text=f"Mean {mean_val:.2f}", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dot", line_color=BLUE,
                  annotation_text=f"Median {median_val:.1f}", annotation_position="top left")
    _apply_dark(fig, "Distribution of Total Goals Per Match")
    fig.update_xaxes(title_text="Total Goals")
    fig.update_yaxes(title_text="Number of Matches")
    return fig


def plot_total_goals_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plot of total goals, home score, and away score."""
    fig = go.Figure()
    variables = [
        ("Total Goals", df["total_goals"], GREEN),
        ("Home Score", df["home_score"], BLUE),
        ("Away Score", df["away_score"], AMBER),
    ]
    for label, data, color in variables:
        fig.add_trace(go.Box(
            y=data.dropna(),
            name=label,
            marker_color=color,
            boxmean=True,
            line_color=color,
        ))
    _apply_dark(fig, "Score Distributions (Box Plot)")
    fig.update_yaxes(title_text="Goals")
    return fig


def plot_home_away_comparison(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing average home and away goals."""
    home_mean = df["home_score"].mean()
    away_mean = df["away_score"].mean()
    fig = go.Figure(data=[
        go.Bar(name="Home Goals", x=["Home", "Away"], y=[home_mean, away_mean],
               marker_color=[GREEN, BLUE], text=[f"{home_mean:.3f}", f"{away_mean:.3f}"],
               textposition="outside"),
    ])
    _apply_dark(fig, "Average Goals: Home vs Away")
    fig.update_yaxes(title_text="Average Goals per Match")
    return fig


def plot_violin_scores(df: pd.DataFrame) -> go.Figure:
    """Violin plot of home and away score distributions."""
    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=df["home_score"].dropna(), name="Home Score",
        box_visible=True, meanline_visible=True,
        line_color=GREEN, fillcolor="rgba(0,176,79,0.3)",
    ))
    fig.add_trace(go.Violin(
        y=df["away_score"].dropna(), name="Away Score",
        box_visible=True, meanline_visible=True,
        line_color=BLUE, fillcolor="rgba(33,150,243,0.3)",
    ))
    _apply_dark(fig, "Score Distribution: Violin Plot")
    fig.update_yaxes(title_text="Goals Scored")
    return fig


# ── Experiment 2: Regression ───────────────────────────────────────────────────


def plot_regression(yearly_df: pd.DataFrame, reg_results: dict) -> go.Figure:
    """Scatter plot of yearly mean goals with OLS regression line."""
    fig = go.Figure()

    # Bubble size ∝ match count
    sizes = yearly_df["n_matches"] if "n_matches" in yearly_df.columns else None
    if sizes is not None:
        marker_size = (sizes / sizes.max() * 30 + 6).tolist()
    else:
        marker_size = 10

    fig.add_trace(go.Scatter(
        x=yearly_df["year"],
        y=yearly_df["mean_total_goals"],
        mode="markers",
        name="Yearly Mean",
        marker=dict(color=GREEN, size=marker_size, opacity=0.85,
                    line=dict(color=WHITE, width=0.5)),
        text=[f"Year: {y}<br>Mean Goals: {g:.3f}<br>Matches: {n}"
              for y, g, n in zip(
                  yearly_df["year"],
                  yearly_df["mean_total_goals"],
                  yearly_df.get("n_matches", [""]*len(yearly_df)),
              )],
        hoverinfo="text",
    ))

    if reg_results.get("x_pred"):
        fig.add_trace(go.Scatter(
            x=reg_results["x_pred"],
            y=reg_results["y_pred"],
            mode="lines",
            name=f"OLS Fit (slope={reg_results['slope']:+.4f})",
            line=dict(color=RED, width=2, dash="solid"),
        ))

    slope = reg_results.get("slope")
    r2 = reg_results.get("r_squared")
    title = "Yearly Mean Total Goals with OLS Regression"
    if slope is not None and r2 is not None:
        title += f"<br><sup>Slope = {slope:+.4f} goals/year  |  R² = {r2:.4f}</sup>"

    _apply_dark(fig, title)
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Mean Total Goals per Match")
    return fig


def plot_regression_residuals(yearly_df: pd.DataFrame, reg_results: dict) -> go.Figure:
    """Residual plot for regression diagnostics."""
    if not reg_results.get("x_obs"):
        return _empty_figure("No regression data available.")

    x_obs = np.array(reg_results["x_obs"])
    residuals = np.array(reg_results["residuals"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_obs, y=residuals,
        mode="markers",
        marker=dict(color=EMERALD, size=8, opacity=0.8),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=AMBER, line_width=1.5)
    _apply_dark(fig, "Regression Residuals (Actual − Predicted)")
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Residual (goals)")
    return fig


# ── Experiment 3: Bootstrap ────────────────────────────────────────────────────


def plot_bootstrap_distribution(boot_results: dict) -> go.Figure:
    """Histogram of bootstrap mean goal differences with CI markers."""
    if not boot_results.get("bootstrap_means"):
        return _empty_figure("Bootstrap has not been run yet.")

    means = np.array(boot_results["bootstrap_means"])
    obs = boot_results["observed_mean"]
    lower = boot_results["ci_lower"]
    upper = boot_results["ci_upper"]
    ci_pct = int(boot_results["confidence_level"] * 100)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=means,
        nbinsx=80,
        marker_color=BLUE,
        opacity=0.8,
        name="Bootstrap Means",
    ))
    fig.add_vline(x=obs, line_dash="solid", line_color=GREEN, line_width=2.5,
                  annotation_text=f"Observed Mean ({obs:+.4f})",
                  annotation_position="top right",
                  annotation_font_color=GREEN)
    fig.add_vrect(x0=lower, x1=upper,
                  fillcolor=f"rgba(33,150,243,0.15)",
                  line_width=0,
                  annotation_text=f"{ci_pct}% CI [{lower:+.4f}, {upper:+.4f}]",
                  annotation_position="top left",
                  annotation_font_color=WHITE)
    fig.add_vline(x=lower, line_dash="dot", line_color=AMBER, line_width=1.5)
    fig.add_vline(x=upper, line_dash="dot", line_color=AMBER, line_width=1.5)

    _apply_dark(fig, f"Bootstrap Distribution of Mean Goal Difference ({boot_results['n_iterations']:,} iterations)")
    fig.update_xaxes(title_text="Bootstrap Mean Goal Difference")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ── Experiment 4: Normality ────────────────────────────────────────────────────


def plot_normality_histogram(norm_results: dict) -> go.Figure:
    """Histogram of goal differences with optional fitted normal curve."""
    if not norm_results.get("diffs_full"):
        return _empty_figure("No normality data available.")

    diffs = np.array(norm_results["diffs_full"])
    mu = norm_results["mu_fit"]
    sigma = norm_results["sigma_fit"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diffs,
        nbinsx=min(60, int(diffs.max() - diffs.min() + 1)),
        histnorm="probability density",
        marker_color=EMERALD,
        opacity=0.75,
        name="Goal Difference",
    ))

    if mu is not None and sigma is not None and sigma > 0:
        x_norm = np.linspace(diffs.min(), diffs.max(), 300)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            mode="lines",
            name="Fitted Normal",
            line=dict(color=RED, width=2),
        ))

    _apply_dark(fig, "Distribution of Goal Differences with Fitted Normal Curve")
    fig.update_xaxes(title_text="Goal Difference (Home − Away)")
    fig.update_yaxes(title_text="Probability Density")
    return fig


def plot_qq(norm_results: dict) -> go.Figure:
    """Normal Q-Q plot for goal differences."""
    if not norm_results.get("qq_theoretical"):
        return _empty_figure("No Q-Q data available.")

    theoretical = np.array(norm_results["qq_theoretical"])
    sample = np.array(norm_results["qq_sample"])

    # Standardise sample for Q-Q
    mu = np.mean(sample)
    sigma = np.std(sample, ddof=1)
    sample_std = (sample - mu) / sigma if sigma > 0 else sample

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=sample_std,
        mode="markers",
        marker=dict(color=BLUE, size=3, opacity=0.5),
        name="Observed Quantiles",
    ))
    # Reference line y = x
    q_range = [min(theoretical.min(), sample_std.min()),
               max(theoretical.max(), sample_std.max())]
    fig.add_trace(go.Scatter(
        x=q_range, y=q_range,
        mode="lines",
        name="Normal Reference",
        line=dict(color=RED, dash="dash", width=1.5),
    ))
    _apply_dark(fig, "Normal Q-Q Plot of Goal Differences")
    fig.update_xaxes(title_text="Theoretical Quantiles")
    fig.update_yaxes(title_text="Sample Quantiles (standardised)")
    return fig


# ── Experiment 5: Hypothesis Testing ──────────────────────────────────────────


def plot_home_away_bar(ttest_results: dict) -> go.Figure:
    """Grouped bar chart of mean home and away goals."""
    if ttest_results.get("error"):
        return _empty_figure(ttest_results["error"])

    fig = go.Figure(data=[
        go.Bar(
            name="Mean Goals",
            x=["Home", "Away"],
            y=[ttest_results["mean_home"], ttest_results["mean_away"]],
            marker_color=[GREEN, BLUE],
            text=[f"{ttest_results['mean_home']:.3f}", f"{ttest_results['mean_away']:.3f}"],
            textposition="outside",
        )
    ])
    _apply_dark(fig, "Mean Goals: Home vs Away (Non-Neutral Matches)")
    fig.update_yaxes(title_text="Average Goals per Match")
    return fig


def plot_goal_diff_histogram(ttest_results: dict) -> go.Figure:
    """Histogram of match-level goal differences with mean line."""
    if ttest_results.get("error") or not ttest_results.get("diffs"):
        return _empty_figure("No hypothesis testing data available.")

    diffs = np.array(ttest_results["diffs"])
    mean_d = ttest_results["mean_diff"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diffs,
        nbinsx=min(50, int(diffs.max() - diffs.min() + 1)),
        marker_color=GREEN,
        opacity=0.8,
        name="Goal Differences",
    ))
    fig.add_vline(x=mean_d, line_dash="dash", line_color=AMBER, line_width=2,
                  annotation_text=f"Mean D = {mean_d:+.4f}",
                  annotation_position="top right",
                  annotation_font_color=AMBER)
    fig.add_vline(x=0, line_dash="dot", line_color=RED, line_width=1.5,
                  annotation_text="H₀: D = 0",
                  annotation_position="top left",
                  annotation_font_color=RED)
    _apply_dark(fig, "Distribution of Match-Level Goal Differences (Home − Away)")
    fig.update_xaxes(title_text="Goal Difference")
    fig.update_yaxes(title_text="Number of Matches")
    return fig


# ── Overview ───────────────────────────────────────────────────────────────────


def plot_yearly_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of yearly mean total goals (Overview page)."""
    grp = (
        df.groupby("year")["total_goals"]
        .agg(mean_goals="mean", n_matches="count")
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grp["year"], y=grp["mean_goals"],
        mode="lines+markers",
        name="Mean Goals",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6, color=GREEN),
        fill="tozeroy",
        fillcolor="rgba(0,176,79,0.1)",
    ))
    _apply_dark(fig, "Yearly Average Goals per Match")
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Mean Total Goals")
    return fig


def plot_result_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of match outcomes (non-neutral)."""
    nn = df[~df["is_neutral"]]
    if len(nn) == 0:
        return _empty_figure("No non-neutral matches in selection.")
    hw = int(nn["home_win"].sum())
    aw = int(nn["away_win"].sum())
    d = int(nn["draw"].sum())
    fig = go.Figure(go.Pie(
        labels=["Home Win", "Away Win", "Draw"],
        values=[hw, aw, d],
        marker_colors=[GREEN, BLUE, AMBER],
        hole=0.4,
        textinfo="label+percent",
    ))
    _apply_dark(fig, "Match Outcomes (Non-Neutral Matches)")
    return fig


# ── Internal ───────────────────────────────────────────────────────────────────


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=AMBER),
    )
    _apply_dark(fig, "")
    return fig
