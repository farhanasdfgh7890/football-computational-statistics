"""
app.py
------
International Football Analytics
A Computational Statistics Study of Goals, Home Advantage, and Performance Trends

8-page Streamlit dashboard connecting all five statistical experiments.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

import io
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="International Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark navy background */
  .stApp { background-color: #0D1B2A; color: #FFFFFF; }
  section[data-testid="stSidebar"] { background-color: #1E2D3D; }
  section[data-testid="stSidebar"] * { color: #FFFFFF; }

  /* KPI cards */
  .kpi-card {
      background: linear-gradient(135deg, #1E2D3D, #2A3F55);
      border-radius: 12px;
      border-left: 4px solid #00B04F;
      padding: 16px 20px;
      margin: 6px 0;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; color: #00D46A; }
  .kpi-label { font-size: 0.85rem; color: #A0ADB9; margin-top: 2px; }

  /* Section headers */
  h1 { color: #00D46A !important; }
  h2, h3 { color: #FFFFFF !important; }

  /* Metric override */
  [data-testid="stMetric"] {
      background: #1E2D3D;
      border-radius: 10px;
      padding: 12px;
      border-left: 3px solid #00B04F;
  }
  [data-testid="stMetricValue"] { color: #00D46A !important; }
  [data-testid="stMetricLabel"] { color: #A0ADB9 !important; }

  /* Dataframes */
  .stDataFrame { background-color: #1E2D3D; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #1E2D3D; border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { color: #A0ADB9; }
  .stTabs [aria-selected="true"] { color: #00D46A !important; border-bottom: 2px solid #00D46A; }

  /* Buttons */
  .stButton > button { background-color: #00B04F; color: white; border: none; border-radius: 8px; }
  .stButton > button:hover { background-color: #00D46A; }

  /* Expander */
  .streamlit-expanderHeader { color: #A0ADB9; }
</style>
""", unsafe_allow_html=True)

# ── Local imports ──────────────────────────────────────────────────────────────
from src.data_loader import load_dataset, load_from_upload, describe_dataset
from src.preprocessing import clean_dataset, filter_data, get_non_neutral, audit_to_dataframe
from src import statistics as stats_module
from src import regression as reg_module
from src import bootstrap as boot_module
from src import normality as norm_module
from src import hypothesis_testing as hyp_module
from src import visualizations as viz


# ── Session state helpers ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚽ Downloading dataset…")
def _load_and_clean(start: str = "2000-01-01", end: str = "2025-12-31"):
    """Load and clean the dataset (cached per date range)."""
    df_raw, meta = load_dataset()
    df_clean, audit = clean_dataset(df_raw, start_date=start, end_date=end)
    return df_clean, audit, meta


def _get_data():
    """Return the cleaned dataframe, audit, and meta from session state."""
    return (
        st.session_state["df"],
        st.session_state["audit"],
        st.session_state["meta"],
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(df_full: pd.DataFrame) -> dict:
    """Render sidebar controls and return filter parameters."""
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <span style="font-size:2.5rem;">⚽</span>
        <h3 style="color:#00D46A; margin:0;">Football Analytics</h3>
        <p style="color:#A0ADB9; font-size:0.8rem;">Computational Statistics</p>
    </div>
    """, unsafe_allow_html=True)

    # Page navigation
    st.sidebar.markdown("### 📋 Navigation")
    page_options = [
        "🏠 Overview",
        "📊 Central Tendency",
        "📈 Regression",
        "🔄 Bootstrap",
        "🔔 Normality Testing",
        "🔬 Hypothesis Testing",
        "🗂️ Data Explorer",
        "ℹ️ About",
    ]
    page = st.sidebar.selectbox("Select Page", page_options, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Filters")

    year_min = int(df_full["year"].min()) if len(df_full) > 0 else 2000
    year_max = int(df_full["year"].max()) if len(df_full) > 0 else 2025

    col_s, col_e = st.sidebar.columns(2)
    start_year = col_s.number_input("From", min_value=year_min, max_value=year_max,
                                     value=year_min, step=1)
    end_year = col_e.number_input("To", min_value=year_min, max_value=year_max,
                                   value=year_max, step=1)

    include_neutral = st.sidebar.checkbox("Include neutral-venue matches", value=True)

    all_tournaments = ["All"] + sorted(df_full["tournament"].dropna().unique().tolist())
    selected_tournaments = st.sidebar.multiselect(
        "Tournament(s)", all_tournaments, default=["All"]
    )
    tournaments = None if "All" in selected_tournaments else selected_tournaments

    all_teams = sorted(
        pd.concat([df_full["home_team"], df_full["away_team"]]).dropna().unique().tolist()
    )
    selected_teams = st.sidebar.multiselect("Team(s) (any involvement)", all_teams)
    teams = selected_teams if selected_teams else None

    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 📁 Data Source")
    uploaded = st.sidebar.file_uploader("Upload custom CSV", type=["csv"])

    return {
        "page": page,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "include_neutral": include_neutral,
        "tournaments": tournaments,
        "teams": teams,
        "uploaded": uploaded,
    }


# ── Pages ──────────────────────────────────────────────────────────────────────


def page_overview(df: pd.DataFrame):
    st.markdown("# 🏠 Overview")
    st.markdown("High-level statistics computed from the currently selected match data.")

    nn = get_non_neutral(df)
    kpi = stats_module.kpi_summary(df, nn)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches", f"{kpi['total_matches']:,}")
    c2.metric("Avg Total Goals", f"{kpi['avg_total_goals']:.3f}")
    c3.metric("Avg Home Goals", f"{kpi['avg_home_goals']:.3f}")
    c4.metric("Avg Away Goals", f"{kpi['avg_away_goals']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Draw %", f"{kpi['pct_draw']:.1f}%")
    c6.metric("Mean Goal Diff (Non-Neutral)", f"{kpi['mean_goal_diff_nn']:+.3f}")
    c7.metric("Home Win % (Non-Neutral)", f"{kpi['home_win_pct']:.1f}%")
    c8.metric("Away Win % (Non-Neutral)", f"{kpi['away_win_pct']:.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_yearly_trend(df), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_result_distribution(df), use_container_width=True)

    st.plotly_chart(viz.plot_total_goals_histogram(df), use_container_width=True)


def page_central_tendency(df: pd.DataFrame):
    st.markdown("# 📊 Experiment 1: Measures of Central Tendency")
    st.markdown(
        "Descriptive statistics for total goals, home score, away score, and goal difference."
    )

    tbl = stats_module.central_tendency_table(df)
    display_cols = ["n", "mean", "median", "std", "variance", "min", "max",
                    "range", "q1", "q3", "iqr", "skewness", "kurtosis"]
    display_tbl = tbl[[c for c in display_cols if c in tbl.columns]].round(4)
    st.dataframe(display_tbl, use_container_width=True)

    # Mode display (separate because mode is a list)
    st.markdown("#### Mode Values")
    mode_info = []
    for col_name, series in [
        ("Total Goals", df["total_goals"]),
        ("Home Score", df["home_score"]),
        ("Away Score", df["away_score"]),
        ("Goal Difference", df["goal_difference"]),
    ]:
        stat = stats_module.central_tendency(series)
        mode_info.append({
            "Variable": col_name,
            "Mode(s)": ", ".join(str(m) for m in stat["mode"]),
            "Frequency": stat["mode_count"],
        })
    st.dataframe(pd.DataFrame(mode_info), use_container_width=True)

    # Download
    csv_bytes = display_tbl.reset_index().to_csv(index=False).encode()
    st.download_button("⬇ Download Summary CSV", csv_bytes, "central_tendency.csv", "text/csv")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_total_goals_histogram(df), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_total_goals_boxplot(df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(viz.plot_home_away_comparison(df), use_container_width=True)
    with col_d:
        st.plotly_chart(viz.plot_violin_scores(df), use_container_width=True)

    with st.expander("📖 Interpretation"):
        stat_goals = stats_module.central_tendency(df["total_goals"])
        if stat_goals["n"] > 0:
            mean_v = stat_goals["mean"]
            median_v = stat_goals["median"]
            mode_v = stat_goals["mode"]
            skew_v = stat_goals["skewness"]
            st.markdown(f"""
The **mean** total goals per match is **{mean_v:.3f}**, representing the arithmetic
average across all selected matches. The **median** is **{median_v:.1f}**, which is
the midpoint of the distribution and is less sensitive to high-scoring outliers.
The most common (modal) scoreline total is **{mode_v}**.

Because the distribution is {"positively skewed" if skew_v > 0 else "approximately symmetric" if abs(skew_v) < 0.3 else "negatively skewed"}
(skewness = {skew_v:.3f}), the mean {"exceeds" if mean_v > median_v else "is below"} the median.
High-scoring matches pull the mean {"upward" if mean_v > median_v else "downward"} relative to the typical match.

The **standard deviation** ({stat_goals["std"]:.3f}) and **IQR**
({stat_goals["iqr"]:.3f}) both measure spread; a large IQR indicates
substantial variability in match scoring.
            """)


def page_regression(df: pd.DataFrame):
    st.markdown("# 📈 Experiment 2: Regression Analysis")
    st.markdown(
        "Simple linear regression of yearly mean total goals on calendar year."
    )

    yearly = stats_module.yearly_goals_summary(df)

    if len(yearly) < 3:
        st.warning("Not enough yearly data points for regression. Adjust the filters.")
        return

    reg = reg_module.fit_yearly_regression(yearly, weighted=False)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slope (goals/year)", f"{reg['slope']:+.4f}" if reg['slope'] is not None else "N/A")
    c2.metric("Intercept", f"{reg['intercept']:.3f}" if reg['intercept'] is not None else "N/A")
    c3.metric("R²", f"{reg['r_squared']:.4f}" if reg['r_squared'] is not None else "N/A")
    c4.metric("RMSE", f"{reg['rmse']:.4f}" if reg['rmse'] is not None else "N/A")

    c5, c6 = st.columns(2)
    c5.metric("p-value (slope)", f"{reg['p_value']:.4f}" if reg['p_value'] is not None else "N/A")
    c6.metric("Years in model", str(reg['n']))

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_regression(yearly, reg), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_regression_residuals(yearly, reg), use_container_width=True)

    # Yearly data table
    with st.expander("📋 Yearly Data Table"):
        st.dataframe(yearly.round(4), use_container_width=True)
        csv_b = yearly.to_csv(index=False).encode()
        st.download_button("⬇ Download Yearly Data", csv_b, "yearly_goals.csv", "text/csv")

    with st.expander("📖 Interpretation"):
        st.markdown(reg_module.interpret_regression(reg))

    with st.expander("🔬 Weighted Regression (Sensitivity Analysis)"):
        st.markdown(
            "Weighted OLS weights each year by its match count, so years with more "
            "matches have greater influence on the fitted slope."
        )
        reg_w = reg_module.fit_yearly_regression(yearly, weighted=True)
        if reg_w["w_slope"] is not None:
            wc1, wc2 = st.columns(2)
            wc1.metric("Weighted Slope", f"{reg_w['w_slope']:+.4f}")
            wc2.metric("Weighted Intercept", f"{reg_w['w_intercept']:.3f}")


def page_bootstrap(df: pd.DataFrame):
    st.markdown("# 🔄 Experiment 3: Bootstrap Resampling")
    st.markdown(
        "Non-parametric estimation of the uncertainty in the mean home-away goal difference."
    )

    col_s, col_n = st.columns(2)
    n_iter = col_n.number_input(
        "Bootstrap iterations", min_value=1000, max_value=100_000,
        value=10_000, step=1000, key="boot_iters"
    )
    seed = col_s.number_input("Random seed", min_value=0, max_value=9999,
                               value=42, step=1, key="boot_seed")

    run_btn = st.button("▶ Run Bootstrap")

    if run_btn or "boot_results" not in st.session_state:
        with st.spinner(f"Running {n_iter:,} bootstrap iterations…"):
            boot = boot_module.bootstrap_mean_goal_diff(
                df, n_iterations=int(n_iter), random_seed=int(seed)
            )
        st.session_state["boot_results"] = boot

    boot = st.session_state.get("boot_results", {})

    if boot.get("error"):
        st.error(boot["error"])
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligible Matches", f"{boot['n_matches']:,}")
    c2.metric("Observed Mean Diff", f"{boot['observed_mean']:+.4f}")
    c3.metric("Bootstrap SE", f"{boot['bootstrap_se']:.4f}")
    c4.metric("Iterations", f"{boot['n_iterations']:,}")

    ci_pct = int(boot["confidence_level"] * 100)
    st.success(
        f"**{ci_pct}% Bootstrap Confidence Interval:** "
        f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}]"
    )

    st.plotly_chart(viz.plot_bootstrap_distribution(boot), use_container_width=True)

    with st.expander("📖 Interpretation"):
        st.markdown(boot_module.interpret_bootstrap(boot))

    with st.expander("🔬 Year-Block Bootstrap (Sensitivity Analysis)"):
        if st.button("Run Year-Block Bootstrap"):
            with st.spinner("Running year-block bootstrap…"):
                yb = boot_module.bootstrap_year_block(
                    df, n_iterations=int(n_iter), random_seed=int(seed)
                )
            st.session_state["yb_results"] = yb

        yb = st.session_state.get("yb_results")
        if yb and not yb.get("error"):
            yb_c1, yb_c2, yb_c3 = st.columns(3)
            yb_c1.metric("Year-Block CI Lower", f"{yb['ci_lower']:+.4f}")
            yb_c2.metric("Year-Block CI Upper", f"{yb['ci_upper']:+.4f}")
            yb_c3.metric("Bootstrap SE (block)", f"{yb['bootstrap_se']:.4f}")
            st.markdown(
                "The year-block bootstrap partially accounts for temporal dependence "
                "by resampling entire years rather than individual matches. "
                "If this interval is wider, it suggests non-negligible year-to-year "
                "dependence in the data."
            )


def page_normality(df: pd.DataFrame):
    st.markdown("# 🔔 Experiment 4: Normality Testing")
    st.markdown(
        "Assessing whether the goal-difference distribution is compatible with a normal distribution."
    )

    seed = st.number_input("Subsampling seed (if n > 5000)", value=42, step=1)

    with st.spinner("Running normality tests…"):
        norm = norm_module.shapiro_wilk_test(df, random_seed=int(seed))

    if norm.get("error"):
        st.error(norm["error"])
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full Sample n", f"{norm['full_n']:,}")
    c2.metric("Test Sample n", f"{norm['test_n']:,}")
    c3.metric("Shapiro-Wilk W", f"{norm['statistic']:.6f}" if norm['statistic'] else "N/A")
    c4.metric("p-value", f"{norm['p_value']:.6f}" if norm['p_value'] else "N/A")

    if norm.get("reject_h0"):
        st.error(f"**{norm['decision']}**")
    else:
        st.success(f"**{norm['decision']}**")

    if norm["subsampled"]:
        st.info(
            f"ℹ️ The full sample ({norm['full_n']:,}) exceeds the recommended "
            f"Shapiro-Wilk limit. Test was run on a random subsample of "
            f"{norm['test_n']:,} observations (seed={norm['random_seed']})."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_normality_histogram(norm), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_qq(norm), use_container_width=True)

    with st.expander("📖 Interpretation"):
        st.markdown(norm_module.interpret_normality(norm))


def page_hypothesis(df: pd.DataFrame):
    st.markdown("# 🔬 Experiment 5: Hypothesis Testing")
    st.markdown(
        "Testing H₀: μD = 0 (mean home-away goal difference is zero) "
        "against H₁: μD > 0 at α = 0.05, using non-neutral matches only."
    )

    with st.spinner("Running hypothesis tests…"):
        ttest = hyp_module.paired_ttest(df)

    if ttest.get("error") and ttest["n"] == 0:
        st.error("No non-neutral matches in the current selection.")
        return

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Eligible Matches", f"{ttest['n']:,}")
    c2.metric("Mean Home Goals", f"{ttest['mean_home']:.4f}")
    c3.metric("Mean Away Goals", f"{ttest['mean_away']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Mean Difference D", f"{ttest['mean_diff']:+.4f}")
    c5.metric("t-statistic", f"{ttest['t_stat']:.4f}")
    c6.metric("One-sided p-value", f"{ttest['p_one_sided']:.6f}")

    c7, c8, c9 = st.columns(3)
    c7.metric("Degrees of Freedom", str(ttest["df"]))
    c8.metric("Cohen's dz", f"{ttest['cohens_dz']:.4f}" if ttest["cohens_dz"] is not None else "N/A")
    c9.metric("Effect Size", ttest["effect_label"] or "N/A")

    st.markdown(f"**95% CI for mean D:** [{ttest['ci_lower']:+.4f}, {ttest['ci_upper']:+.4f}]")

    if ttest["reject_h0"]:
        st.success(f"**Decision:** {ttest['decision']}")
    else:
        st.warning(f"**Decision:** {ttest['decision']}")

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_home_away_bar(ttest), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_goal_diff_histogram(ttest), use_container_width=True)

    with st.expander("📖 Full Interpretation"):
        st.markdown(hyp_module.interpret_ttest(ttest))

    with st.expander("🔬 Wilcoxon Signed-Rank Test (Sensitivity Analysis)"):
        st.markdown(
            "The Wilcoxon signed-rank test is a non-parametric alternative. "
            "It tests whether the distribution of differences is symmetric around zero, "
            "rather than testing the mean directly."
        )
        if st.button("Run Wilcoxon Test"):
            with st.spinner("Running Wilcoxon test…"):
                wilcox = hyp_module.wilcoxon_test(df)
            st.session_state["wilcox"] = wilcox

        wilcox = st.session_state.get("wilcox")
        if wilcox:
            if wilcox.get("error"):
                st.error(wilcox["error"])
            else:
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("Wilcoxon W", f"{wilcox['statistic']:.2f}")
                wc2.metric("p-value (one-sided)", f"{wilcox['p_value']:.6f}")
                wc3.metric("Reject H₀?", "Yes" if wilcox["reject_h0"] else "No")
                st.caption(wilcox["note"])


def page_data_explorer(df_full: pd.DataFrame, df: pd.DataFrame, audit: dict, meta: dict):
    st.markdown("# 🗂️ Data Explorer")
    st.markdown(f"Viewing **{len(df):,}** matches (filtered). Full cleaned dataset: **{len(df_full):,}** matches.")

    # Search
    col_search, col_dl = st.columns([3, 1])
    search_term = col_search.text_input("🔍 Search by team name", "")

    df_view = df.copy()
    if search_term:
        mask = (
            df_view["home_team"].str.contains(search_term, case=False, na=False) |
            df_view["away_team"].str.contains(search_term, case=False, na=False)
        )
        df_view = df_view[mask]

    st.markdown(f"Showing **{len(df_view):,}** matches")

    display_cols = [c for c in [
        "date", "home_team", "home_score", "away_score", "away_team",
        "tournament", "city", "country", "is_neutral",
        "total_goals", "goal_difference", "home_win", "draw", "away_win"
    ] if c in df_view.columns]

    st.dataframe(
        df_view[display_cols].sort_values("date", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    csv_dl = df_view[display_cols].to_csv(index=False).encode()
    col_dl.download_button("⬇ Download CSV", csv_dl, "filtered_matches.csv", "text/csv",
                           use_container_width=True)

    st.divider()
    st.markdown("### 📋 Data Cleaning Audit")
    audit_df = audit_to_dataframe(audit)
    st.dataframe(audit_df, use_container_width=True)

    with st.expander("📄 Dataset Source Info"):
        st.json(meta)


def page_about():
    st.markdown("# ℹ️ About This Project")

    st.markdown("""
## 🎓 International Football Analytics
### A Computational Statistics Study of Goals, Home Advantage, and Performance Trends

---

### 📌 Project Details

| Field | Value |
|-------|-------|
| **Subject** | Computational Statistics |
| **Student Name** | *(Enter your name)* |
| **USN** | *(Enter your USN)* |
| **Class / Semester** | *(Enter class and semester)* |
| **Department** | Computer Science & Engineering |
| **Institution** | *(Enter institution name)* |
| **Faculty Guide** | *(Enter faculty guide name)* |
| **Academic Year** | *(Enter academic year)* |

---

### 🔬 Research Objective

Investigate whether playing at home provides a measurable statistical advantage
in international men's football, and what computational statistical methods
reveal about scoring patterns and their changes over time.

---

### 📊 Five Statistical Experiments

| # | Experiment | Method |
|---|-----------|--------|
| 1 | **Central Tendency** | Mean, Median, Mode, IQR, Variance |
| 2 | **Regression Analysis** | OLS Linear Regression (Year → Goals) |
| 3 | **Bootstrap Resampling** | 10,000-iteration percentile bootstrap CI |
| 4 | **Normality Testing** | Shapiro-Wilk test, Q-Q plot |
| 5 | **Hypothesis Testing** | Paired t-test, Cohen's dz, Wilcoxon |

---

### 📁 Dataset

**Source:** Mart Jürisoo's International Football Results  
**Repository:** https://github.com/martj42/international_results  
**Study Period:** January 1, 2000 – December 31, 2025  
**Attribution:** Please credit the dataset maintainer in any derivative work.

---

### 🛠️ Tools & Libraries

`Python 3.11` · `Pandas` · `NumPy` · `SciPy` · `Statsmodels` · `Streamlit`  
`Plotly` · `Matplotlib` · `Seaborn` · `pytest`

---

### ⚠️ Limitations

1. Matches are not independent; national teams appear in many matches.
2. The dataset may contain inconsistencies in historical records.
3. Home advantage is measured by goal difference only; other factors are not modelled.
4. Regression with year does not establish causation.
5. The Shapiro-Wilk test is sensitive to sample size; large samples almost always reject normality.
6. "Home" advantage in international football differs from club football due to tournament formats.

---

*All statistical results are derived from the actual dataset. No values have been fabricated.*
    """)


# ── Main App ───────────────────────────────────────────────────────────────────


def main():
    # ── Header ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0D1B2A, #1E2D3D);
                border-bottom: 2px solid #00B04F;
                padding: 20px 30px; margin-bottom: 20px; border-radius: 8px;">
        <h1 style="color:#00D46A; margin:0; font-size:2rem;">⚽ International Football Analytics</h1>
        <p style="color:#A0ADB9; margin:4px 0 0;">
            A Computational Statistics Study of Goals, Home Advantage, and Performance Trends
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ────────────────────────────────────────────────────────────────
    if "df" not in st.session_state:
        try:
            with st.spinner("Loading dataset…"):
                df_clean, audit, meta = _load_and_clean()
            st.session_state["df_full"] = df_clean
            st.session_state["df"] = df_clean
            st.session_state["audit"] = audit
            st.session_state["meta"] = meta
        except Exception as exc:
            st.error(f"**Dataset load failed:** {exc}")
            st.info("Please upload a results.csv file using the sidebar uploader.")
            uploaded = st.file_uploader("Upload results.csv", type=["csv"])
            if uploaded:
                try:
                    df_raw, meta = load_from_upload(uploaded)
                    df_clean, audit = clean_dataset(df_raw)
                    st.session_state["df_full"] = df_clean
                    st.session_state["df"] = df_clean
                    st.session_state["audit"] = audit
                    st.session_state["meta"] = meta
                    st.rerun()
                except Exception as exc2:
                    st.error(f"Upload failed: {exc2}")
            return

    df_full = st.session_state["df_full"]
    audit = st.session_state["audit"]
    meta = st.session_state["meta"]

    # ── Sidebar ──────────────────────────────────────────────────────────────────
    filters = render_sidebar(df_full)

    # Handle custom upload from sidebar
    if filters["uploaded"] is not None:
        try:
            df_raw, meta_up = load_from_upload(filters["uploaded"])
            df_clean_up, audit_up = clean_dataset(df_raw)
            st.session_state["df_full"] = df_clean_up
            st.session_state["df"] = df_clean_up
            st.session_state["audit"] = audit_up
            st.session_state["meta"] = meta_up
            df_full = df_clean_up
            audit = audit_up
            meta = meta_up
        except Exception as exc:
            st.sidebar.error(f"Upload error: {exc}")

    # Apply filters
    df = filter_data(
        df_full,
        start_year=filters["start_year"],
        end_year=filters["end_year"],
        tournaments=filters["tournaments"],
        include_neutral=filters["include_neutral"],
        teams=filters["teams"],
    )

    if len(df) == 0:
        st.warning(
            "⚠️ No matches match the current filter settings. "
            "Please adjust the sidebar filters."
        )
        return

    st.sidebar.markdown(f"**Matches in view:** {len(df):,}")

    # ── Route pages ──────────────────────────────────────────────────────────────
    page = filters["page"]

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 Central Tendency":
        page_central_tendency(df)
    elif page == "📈 Regression":
        page_regression(df)
    elif page == "🔄 Bootstrap":
        page_bootstrap(df)
    elif page == "🔔 Normality Testing":
        page_normality(df)
    elif page == "🔬 Hypothesis Testing":
        page_hypothesis(df)
    elif page == "🗂️ Data Explorer":
        page_data_explorer(df_full, df, audit, meta)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
