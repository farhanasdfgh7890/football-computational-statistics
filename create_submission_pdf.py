"""
create_submission_pdf.py
------------------------
Creates a SINGLE comprehensive PDF for academic submission.
Contains:
  - Title page
  - All 10 report pages (charts + statistics)  
  - All source code files formatted as readable code pages

Output: reports/Football_Analytics_SUBMISSION.pdf

Run:
    python create_submission_pdf.py
"""

import sys, os, textwrap, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))
os.environ["PYTHONIOENCODING"] = "utf-8"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as sp_stats
from datetime import datetime

# ── src imports ────────────────────────────────────────────────────────────────
from src.data_loader import load_dataset
from src.preprocessing import clean_dataset, get_non_neutral
from src import statistics as stats_module
from src import regression as reg_module
from src import bootstrap as boot_module
from src import normality as norm_module
from src import hypothesis_testing as hyp_module

# ── Colours ───────────────────────────────────────────────────────────────────
GARNET = "#A50044"
BLUE   = "#004D98"
GOLD   = "#D4A017"
DARK   = "#1A1A2E"
MID    = "#4A4A6A"
LIGHT  = "#F4F6FB"
WHITE  = "#FFFFFF"
GRID   = "#E0E4EE"
CODE_BG= "#F8F8F2"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "figure.facecolor": WHITE, "axes.facecolor": LIGHT,
        "axes.edgecolor": "#CCCED8", "axes.labelcolor": MID,
        "axes.titlecolor": DARK, "axes.titlesize": 11,
        "axes.labelsize": 9, "text.color": DARK,
        "xtick.color": MID, "ytick.color": MID,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.alpha": 1.0, "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "legend.fontsize": 8,
        "legend.framealpha": 0.9,
    })


def _header(fig, title, subtitle="", color=GARNET):
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.03, 0.945), 0.94, 0.048, boxstyle="round,pad=0.005",
        facecolor=color, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=10))
    fig.text(0.055, 0.969, title, fontsize=12, fontweight="bold",
             color=WHITE, va="center", transform=fig.transFigure)
    if subtitle:
        fig.text(0.945, 0.969, subtitle, fontsize=7.5,
                 color=(1,1,1,0.75), va="center", ha="right",
                 transform=fig.transFigure)


def _footer(fig, pg, total):
    fig.text(0.5, 0.018,
             "International Football Analytics — FC Barcelona Edition | Computational Statistics",
             fontsize=6.5, color="#AAAAAA", ha="center", transform=fig.transFigure)
    fig.text(0.945, 0.018, f"Page {pg} / {total}",
             fontsize=6.5, color="#AAAAAA", ha="right", transform=fig.transFigure)
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.035, 0.035],
                               transform=fig.transFigure, color=GRID, lw=0.7))


def _stat_table(ax, data, title=""):
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=DARK, pad=5)
    rows = [[k, v] for k, v in data.items()]
    tbl  = ax.table(cellText=rows, colLabels=["Statistic", "Value"],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(GARNET); cell.set_text_props(color=WHITE, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EFF1F8")
        else:
            cell.set_facecolor(WHITE)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT PAGES  (identical logic to generate_pdf_report.py)
# ══════════════════════════════════════════════════════════════════════════════

def pg_title(pdf, df, nn, meta, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(DARK)
    for patch_args in [
        dict(xy=(0,0), w=1, h=1, fc=DARK),
        dict(xy=(0,0), w=1, h=0.115, fc=GARNET),
        dict(xy=(0,0), w=0.015, h=1, fc=BLUE),
        dict(xy=(0.015,0), w=0.015, h=1, fc=GOLD),
    ]:
        fig.patches.append(mpatches.FancyBboxPatch(
            (patch_args["xy"][0], patch_args["xy"][1]),
            patch_args["w"], patch_args["h"],
            boxstyle="square", facecolor=patch_args["fc"], edgecolor="none",
            transform=fig.transFigure, zorder=1))

    fig.text(0.5, 0.80, "FC BARCELONA ANALYTICS HUB",
             fontsize=26, fontweight="bold", color=GOLD, ha="center",
             transform=fig.transFigure, fontfamily="DejaVu Sans")
    fig.text(0.5, 0.70,
             "Statistical Analysis of International Football Performance",
             fontsize=15, color=WHITE, ha="center", transform=fig.transFigure)
    fig.text(0.5, 0.63,
             "Home Advantage  |  Goal-Scoring Patterns  |  Match Outcomes",
             fontsize=11, color="#AAAACC", ha="center", transform=fig.transFigure)
    fig.add_artist(plt.Line2D([0.10,0.90],[0.585,0.585],
                               transform=fig.transFigure, color=GARNET, lw=1.5))

    info = [
        ("Subject",    "Computational Statistics  —  B.Tech Computer Science Engineering"),
        ("Student",    "*(Enter your name)*"),
        ("USN",        "*(Enter your USN)*"),
        ("Department", "Computer Science & Engineering"),
        ("Dataset",    "Mart Jurisoo — International Football Results (GitHub)"),
        ("Period",     "January 1, 2000 – December 31, 2025"),
        ("Matches",    f"{len(df):,} cleaned matches  |  {len(nn):,} non-neutral"),
        ("Generated",  datetime.now().strftime("%B %d, %Y  at  %H:%M")),
    ]
    for i, (k, v) in enumerate(info):
        y = 0.545 - i * 0.058
        fig.text(0.12, y, k + ":", fontsize=8.5, color=GOLD,
                 fontweight="bold", transform=fig.transFigure)
        fig.text(0.27, y, v, fontsize=8.5, color="#DDDDEE",
                 transform=fig.transFigure)

    fig.text(0.5, 0.062, "MES QUE UN CLUB  |  EST. 1899  |  CAMP NOU, BARCELONA",
             fontsize=8, color=WHITE, ha="center", transform=fig.transFigure)
    _footer(fig, pg, total)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_data_overview(pdf, df, pg, total):
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27)); _style()
    _header(fig, "Dataset Overview", "Data Cleaning & Key Performance Indicators", BLUE)
    _footer(fig, pg, total)
    nn  = get_non_neutral(df)
    kpi = stats_module.kpi_summary(df, nn)

    yc = df.groupby("year").size().reset_index(name="count")
    axes[0].bar(yc["year"], yc["count"], color=BLUE, alpha=0.85, edgecolor=WHITE, lw=0.4)
    axes[0].set_title("Matches per Year", fontweight="bold")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Matches")
    axes[0].grid(axis="y"); axes[0].set_axisbelow(True)
    for s in ["top","right"]: axes[0].spines[s].set_visible(False)

    hw = int(df["home_win"].sum()); aw = int(df["away_win"].sum()); dr = int(df["draw"].sum())
    axes[1].pie([hw,aw,dr], labels=["Home Win","Away Win","Draw"],
                colors=[BLUE,GARNET,GOLD], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor=WHITE, linewidth=1.5),
                textprops={"fontsize":8})
    axes[1].set_title("Match Result Distribution", fontweight="bold")

    _stat_table(axes[2], {
        "Total Matches":       f"{kpi['total_matches']:,}",
        "Avg Total Goals":     f"{kpi['avg_total_goals']:.4f}",
        "Avg Home Goals":      f"{kpi['avg_home_goals']:.4f}",
        "Avg Away Goals":      f"{kpi['avg_away_goals']:.4f}",
        "Home Win %":          f"{kpi['home_win_pct']:.2f}%",
        "Away Win %":          f"{kpi['away_win_pct']:.2f}%",
        "Draw %":              f"{kpi['pct_draw']:.2f}%",
        "Mean Goal Diff (NN)": f"{kpi['mean_goal_diff_nn']:+.4f}",
        "Non-Neutral Count":   f"{len(nn):,}",
    }, "KPI Summary")
    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.94])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_ct_stats(pdf, df, pg, total):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27)); _style()
    _header(fig, "Experiment 1 — Central Tendency", "Descriptive Statistics Tables", GARNET)
    _footer(fig, pg, total)
    for i, (var, lbl) in enumerate([
        ("total_goals","Total Goals"),("home_score","Home Score"),
        ("away_score","Away Score"),("goal_difference","Goal Difference")
    ]):
        s = stats_module.central_tendency(df[var])
        _stat_table(axes[i//2][i%2], {
            "n":        f"{s['n']:,}",
            "Mean":     f"{s['mean']:.6f}",
            "Median":   f"{s['median']:.4f}",
            "Mode":     str(s['mode']),
            "Std Dev":  f"{s['std']:.6f}",
            "Variance": f"{s['variance']:.6f}",
            "Min":      f"{s['min']:.0f}",
            "Max":      f"{s['max']:.0f}",
            "Range":    f"{s['range']:.0f}",
            "Q1":       f"{s['q1']:.4f}",
            "Q3":       f"{s['q3']:.4f}",
            "IQR":      f"{s['iqr']:.4f}",
            "Skewness": f"{s['skewness']:.6f}",
            "Kurtosis": f"{s['kurtosis']:.6f}",
        }, f"Central Tendency — {lbl}")
    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.94])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_ct_charts(pdf, df, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27)); _style()
    _header(fig, "Experiment 1 — Central Tendency", "Visualisations", GARNET)
    _footer(fig, pg, total)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0,0])
    vals = df["total_goals"].dropna()
    ax1.hist(vals, bins=range(int(vals.min()), int(vals.max())+2),
             color=BLUE, edgecolor=WHITE, lw=0.5, alpha=0.85)
    ax1.axvline(vals.mean(),   color=GARNET, lw=1.8, ls="--", label=f"Mean {vals.mean():.2f}")
    ax1.axvline(vals.median(), color=GOLD,   lw=1.8, ls=":",  label=f"Median {vals.median():.1f}")
    ax1.set_title("Total Goals Histogram", fontweight="bold")
    ax1.set_xlabel("Total Goals"); ax1.set_ylabel("Frequency")
    ax1.legend(); ax1.grid(axis="y"); ax1.set_axisbelow(True)
    for s in ["top","right"]: ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0,1])
    vb   = [df["total_goals"], df["home_score"], df["away_score"], df["goal_difference"]]
    lbls = ["Total\nGoals","Home\nScore","Away\nScore","Goal\nDiff"]
    bp = ax2.boxplot(vb, patch_artist=True, notch=False,
                     medianprops=dict(color=GOLD,lw=2),
                     whiskerprops=dict(color=MID), capprops=dict(color=MID),
                     flierprops=dict(marker=".",markerfacecolor=GARNET,ms=3,alpha=0.4))
    ax2.set_xticks([1,2,3,4]); ax2.set_xticklabels(lbls, fontsize=7.5)
    for patch, c in zip(bp["boxes"],[BLUE,GARNET,GOLD,"#6A5ACD"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_title("Score Distributions — Box Plots", fontweight="bold")
    ax2.set_ylabel("Goals"); ax2.grid(axis="y"); ax2.set_axisbelow(True)
    for s in ["top","right"]: ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[1,0])
    cats = ["Home\nGoals","Away\nGoals","Total\nGoals","Goal\nDiff"]
    mvs  = [df["home_score"].mean(),df["away_score"].mean(),
            df["total_goals"].mean(),df["goal_difference"].mean()]
    bars = ax3.bar(cats, mvs, color=[BLUE,GARNET,GOLD,MID], edgecolor=WHITE, lw=0.6, alpha=0.88)
    for bar, val in zip(bars, mvs):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax3.set_title("Mean Comparison", fontweight="bold")
    ax3.set_ylabel("Mean Goals"); ax3.grid(axis="y"); ax3.set_axisbelow(True)
    for s in ["top","right"]: ax3.spines[s].set_visible(False)

    ax4 = fig.add_subplot(gs[1,1])
    vp = ax4.violinplot([df["home_score"].dropna().values, df["away_score"].dropna().values],
                         positions=[1,2], showmedians=True, showmeans=True)
    vp["cmedians"].set_color(GOLD); vp["cmedians"].set_linewidth(2)
    vp["cmeans"].set_color(GARNET);  vp["cmeans"].set_linewidth(1.5)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor([BLUE,GARNET][i]); body.set_alpha(0.7)
    ax4.set_xticks([1,2]); ax4.set_xticklabels(["Home Score","Away Score"])
    ax4.set_title("Violin Plots — Home vs Away", fontweight="bold")
    ax4.set_ylabel("Goals"); ax4.grid(axis="y"); ax4.set_axisbelow(True)
    for s in ["top","right"]: ax4.spines[s].set_visible(False)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_regression(pdf, df, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27)); _style()
    _header(fig, "Experiment 2 — Regression Analysis",
            "OLS: Yearly Mean Goals ~ Calendar Year", BLUE)
    _footer(fig, pg, total)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)
    yearly = stats_module.yearly_goals_summary(df)
    reg    = reg_module.fit_yearly_regression(yearly, weighted=False)

    ax1 = fig.add_subplot(gs[0,:])
    ax1.scatter(yearly["year"], yearly["mean_total_goals"], color=BLUE, s=55, zorder=5, alpha=0.9,
                label="Observed yearly mean")
    if reg.get("slope"):
        xr = np.linspace(yearly["year"].min(), yearly["year"].max(), 200)
        ax1.plot(xr, reg["intercept"] + reg["slope"]*xr, color=GARNET, lw=2,
                 label=f"OLS: y={reg['intercept']:.3f}{reg['slope']:+.4f}x  R\u00b2={reg['r_squared']:.4f}")
    ax1.set_title("Yearly Mean Total Goals with OLS Regression Line", fontweight="bold")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Mean Goals / Match")
    ax1.legend(); ax1.grid(True); ax1.set_axisbelow(True)
    for s in ["top","right"]: ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[1,0])
    if reg.get("residuals"):
        ax2.scatter(yearly["year"], reg["residuals"], color=BLUE, s=45, alpha=0.85)
        ax2.axhline(0, color=GARNET, lw=1.5, ls="--")
    ax2.set_title("Residuals vs Year", fontweight="bold")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Residual")
    ax2.grid(True); ax2.set_axisbelow(True)
    for s in ["top","right"]: ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[1,1])
    _stat_table(ax3, {
        "Slope (goals/year)": f"{reg.get('slope',0):+.8f}",
        "Intercept":          f"{reg.get('intercept',0):.4f}",
        "R\u00b2":            f"{reg.get('r_squared',0):.8f}",
        "RMSE":               f"{reg.get('rmse',0):.8f}",
        "p-value (slope)":    f"{reg.get('p_value',0):.8f}",
        "Years in Model":     str(reg.get('n','N/A')),
        "Significant?":       "Yes (p<0.05)" if (reg.get("p_value") or 1)<0.05 else "No",
        "Trend":              "Rising" if (reg.get("slope") or 0)>0 else "Falling",
    }, "Regression Statistics")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_bootstrap(pdf, df, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27)); _style()
    _header(fig, "Experiment 3 — Bootstrap Resampling",
            "10,000-Iteration Percentile Bootstrap for Mean Home-Away Goal Difference", GARNET)
    _footer(fig, pg, total)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)
    boot = boot_module.bootstrap_mean_goal_diff(df, n_iterations=10_000, random_seed=42)

    ax1 = fig.add_subplot(gs[0])
    if boot.get("bootstrap_means") is not None:
        bm = np.array(boot["bootstrap_means"])
        ax1.hist(bm, bins=60, color=BLUE, edgecolor=WHITE, lw=0.3, alpha=0.82, density=True)
        ax1.axvline(boot["ci_lower"], color=GARNET, lw=2, ls="--",
                    label=f"95% CI lower: {boot['ci_lower']:+.4f}")
        ax1.axvline(boot["ci_upper"], color=GARNET, lw=2, ls="--",
                    label=f"95% CI upper: {boot['ci_upper']:+.4f}")
        ax1.axvline(boot["observed_mean"], color=GOLD, lw=2.5,
                    label=f"Observed mean: {boot['observed_mean']:+.4f}")
    ax1.set_title("Bootstrap Distribution of Mean Goal Difference", fontweight="bold")
    ax1.set_xlabel("Mean Home - Away Goal Difference"); ax1.set_ylabel("Density")
    ax1.legend(fontsize=7.5); ax1.grid(axis="y"); ax1.set_axisbelow(True)
    for s in ["top","right"]: ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    lo, hi = boot.get("ci_lower",0), boot.get("ci_upper",0)
    _stat_table(ax2, {
        "Eligible Matches":   f"{boot.get('n_matches',0):,}",
        "Iterations":         "10,000",
        "Random Seed":        "42",
        "Observed Mean Diff": f"{boot.get('observed_mean',0):+.8f}",
        "Bootstrap SE":       f"{boot.get('bootstrap_se',0):.8f}",
        "95% CI Lower":       f"{lo:+.8f}",
        "95% CI Upper":       f"{hi:+.8f}",
        "CI Width":           f"{hi-lo:.8f}",
        "CI Contains Zero?":  "No" if lo > 0 else "Yes",
        "Home Advantage?":    "Supported" if lo > 0 else "Inconclusive",
    }, "Bootstrap Results")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_normality(pdf, df, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27)); _style()
    _header(fig, "Experiment 4 — Normality Testing",
            "Shapiro-Wilk Test + Q-Q Plot for Goal Differences", GOLD)
    # override text color for gold header
    for t in fig.texts: t.set_color(WHITE)
    _footer(fig, pg, total)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)
    norm = norm_module.shapiro_wilk_test(df, random_seed=42)
    gd   = np.array(norm.get("data", df["goal_difference"].dropna().values))

    ax1 = fig.add_subplot(gs[0,0])
    ax1.hist(gd, bins=30, density=True, color=BLUE, edgecolor=WHITE, lw=0.4, alpha=0.82)
    xn = np.linspace(gd.min(), gd.max(), 300)
    ax1.plot(xn, sp_stats.norm.pdf(xn, gd.mean(), gd.std()), color=GARNET, lw=2, label="Normal PDF")
    ax1.set_title("Goal Difference Distribution", fontweight="bold")
    ax1.set_xlabel("Goal Difference"); ax1.set_ylabel("Density")
    ax1.legend(); ax1.grid(axis="y"); ax1.set_axisbelow(True)
    for s in ["top","right"]: ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0,1])
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(gd, dist="norm")
    ax2.scatter(osm, osr, color=BLUE, s=8, alpha=0.6)
    ax2.plot(osm, slope*np.array(osm)+intercept, color=GARNET, lw=2, label="Normal reference")
    ax2.set_title("Normal Q-Q Plot", fontweight="bold")
    ax2.set_xlabel("Theoretical Quantiles"); ax2.set_ylabel("Sample Quantiles")
    ax2.legend(); ax2.grid(True); ax2.set_axisbelow(True)
    for s in ["top","right"]: ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[1,0])
    _stat_table(ax3, {
        "Variable":    "Goal Difference (Home - Away)",
        "Full n":      f"{norm.get('full_n', len(gd)):,}",
        "Test n":      f"{norm.get('test_n', len(gd)):,}",
        "Subsampled?": str(norm.get("subsampled", False)),
        "Mean":        f"{gd.mean():.8f}",
        "Std Dev":     f"{gd.std():.8f}",
        "Skewness":    f"{sp_stats.skew(gd):.8f}",
        "Kurtosis":    f"{sp_stats.kurtosis(gd):.8f}",
    }, "Descriptive Statistics")

    ax4 = fig.add_subplot(gs[1,1])
    dec = norm.get("decision","N/A")
    _stat_table(ax4, {
        "Test":          "Shapiro-Wilk",
        "H0":            "Data ~ Normal distribution",
        "H1":            "Data NOT ~ Normal",
        "W Statistic":   f"{norm.get('statistic',0):.8f}",
        "p-value":       f"{norm.get('p_value',0):.8f}",
        "alpha":         "0.05",
        "Reject H0?":    "Yes" if norm.get("reject_h0") else "No",
        "Decision":      dec[:50] + "..." if len(dec)>50 else dec,
    }, "Shapiro-Wilk Results")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_hypothesis(pdf, df, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27)); _style()
    _header(fig, "Experiment 5 — Hypothesis Testing",
            "Paired t-test: H0: muD=0  vs  H1: muD>0  at alpha=0.05", GARNET)
    _footer(fig, pg, total)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)
    ttest = hyp_module.paired_ttest(df)
    nn    = get_non_neutral(df)

    ax1 = fig.add_subplot(gs[0,0])
    cats = ["Home Goals\n(mean)","Away Goals\n(mean)","Difference\n(H-A)"]
    vals = [ttest.get("mean_home",0), ttest.get("mean_away",0), ttest.get("mean_diff",0)]
    bars = ax1.bar(cats, vals, color=[BLUE,GARNET,GOLD], edgecolor=WHITE, lw=0.6, alpha=0.88)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.axhline(0, color=MID, lw=0.8, ls="--")
    ax1.set_title("Mean Goals: Home vs Away", fontweight="bold")
    ax1.set_ylabel("Mean Goals"); ax1.grid(axis="y"); ax1.set_axisbelow(True)
    for s in ["top","right"]: ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0,1])
    gd = nn["goal_difference"].dropna().values
    ax2.hist(gd, bins=30, density=True, color=BLUE, edgecolor=WHITE, lw=0.3, alpha=0.82)
    ax2.axvline(0, color=MID, lw=1.2, ls="--", label="H0: zero")
    ax2.axvline(gd.mean(), color=GARNET, lw=2, label=f"Mean {gd.mean():+.4f}")
    ax2.set_title("Goal Differences Distribution (Non-Neutral)", fontweight="bold")
    ax2.set_xlabel("Goal Difference"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=7.5); ax2.grid(axis="y"); ax2.set_axisbelow(True)
    for s in ["top","right"]: ax2.spines[s].set_visible(False)

    ax3 = fig.add_subplot(gs[1,0])
    _stat_table(ax3, {
        "Eligible Matches":    f"{ttest.get('n',0):,}",
        "Mean Home Goals":     f"{ttest.get('mean_home',0):.8f}",
        "Mean Away Goals":     f"{ttest.get('mean_away',0):.8f}",
        "Mean Difference D":   f"{ttest.get('mean_diff',0):+.8f}",
        "t-statistic":         f"{ttest.get('t_stat',0):.8f}",
        "Degrees of Freedom":  str(ttest.get('df','N/A')),
        "p-value (one-sided)": f"{ttest.get('p_one_sided',0):.10f}",
        "p-value (two-sided)": f"{ttest.get('p_two_sided',0):.10f}",
        "95% CI Lower":        f"{ttest.get('ci_lower',0):+.8f}",
        "95% CI Upper":        f"{ttest.get('ci_upper',0):+.8f}",
    }, "Paired t-Test Results")

    ax4 = fig.add_subplot(gs[1,1])
    dz = ttest.get("cohens_dz",0) or 0
    dec = ttest.get("decision","N/A")
    _stat_table(ax4, {
        "Cohen's dz":       f"{dz:.8f}",
        "Effect Size":      ttest.get("effect_label","N/A"),
        "Test":             "One-sided paired t-test",
        "alpha":            "0.05",
        "Reject H0?":       "Yes" if ttest.get("reject_h0") else "No",
        "Decision":         dec[:50]+"..." if len(dec)>50 else dec,
        "Home Advantage?":  "Confirmed" if ttest.get("reject_h0") else "Not confirmed",
    }, "Effect Size & Decision")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def pg_summary(pdf, df, pg, total):
    fig = plt.figure(figsize=(11.69, 8.27)); _style()
    _header(fig, "Summary & Conclusions", "All Five Experiments — Key Findings", DARK)
    _footer(fig, pg, total)

    nn     = get_non_neutral(df)
    kpi    = stats_module.kpi_summary(df, nn)
    sg     = stats_module.central_tendency(df["total_goals"])
    yearly = stats_module.yearly_goals_summary(df)
    reg    = reg_module.fit_yearly_regression(yearly, weighted=False)
    boot   = boot_module.bootstrap_mean_goal_diff(df, n_iterations=10_000, random_seed=42)
    norm_r = norm_module.shapiro_wilk_test(df, random_seed=42)
    ttest  = hyp_module.paired_ttest(df)

    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82]); ax.axis("off")
    ax.text(0.5, 0.98, "Research Conclusion", fontsize=13, fontweight="bold",
            color=DARK, ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.93, f"Based on {len(df):,} international football matches (2000-2025)",
            fontsize=8.5, color=MID, ha="center", va="top", transform=ax.transAxes)

    rows = [
        (BLUE,   "01", "Central Tendency",
         f"Mean={sg['mean']:.3f}, Median={sg['median']:.1f}, Mode={sg['mode']}, Std={sg['std']:.3f}"),
        (GARNET, "02", "Regression Analysis",
         f"Slope={reg.get('slope',0):+.4f} goals/yr, R\u00b2={reg.get('r_squared',0):.4f}, p={reg.get('p_value',0):.4f} | Trend: {'Rising' if (reg.get('slope') or 0)>0 else 'Falling'}"),
        (GOLD,   "03", "Bootstrap Resampling",
         f"Obs mean diff={boot.get('observed_mean',0):+.4f}, 95% CI=[{boot.get('ci_lower',0):+.4f}, {boot.get('ci_upper',0):+.4f}], SE={boot.get('bootstrap_se',0):.4f}"),
        (MID,    "04", "Normality Testing",
         f"Shapiro-Wilk W={norm_r.get('statistic',0):.6f}, p={norm_r.get('p_value',0):.6f} | {'Reject H0: NOT normal' if norm_r.get('reject_h0') else 'Fail to reject: consistent with normality'}"),
        (GARNET, "05", "Hypothesis Testing",
         f"t({ttest.get('df','?')})={ttest.get('t_stat',0):.4f}, p={ttest.get('p_one_sided',0):.6f}, Cohen's dz={ttest.get('cohens_dz',0) or 0:.4f} ({ttest.get('effect_label','')}) | {'Reject H0: home advantage confirmed' if ttest.get('reject_h0') else 'Fail to reject H0'}"),
    ]

    for i,(col,num,title,finding) in enumerate(rows):
        y = 0.84 - i*0.155
        bubble = mpatches.FancyBboxPatch((0.0,y-0.035),0.055,0.08,
                                          boxstyle="round,pad=0.01",
                                          facecolor=col,edgecolor="none",
                                          transform=ax.transAxes,zorder=3)
        ax.add_patch(bubble)
        ax.text(0.027,y+0.005,num,fontsize=13,fontweight="bold",color=WHITE,
                ha="center",va="center",transform=ax.transAxes)
        ax.text(0.07,y+0.022,title,fontsize=9.5,fontweight="bold",color=DARK,
                va="center",transform=ax.transAxes)
        ax.text(0.07,y-0.018,finding,fontsize=7.8,color=MID,
                va="center",transform=ax.transAxes)
        if i<4:
            ax.add_artist(plt.Line2D([0,1],[y-0.063,y-0.063],
                                      transform=ax.transAxes,color=GRID,lw=0.6))

    conc_color = BLUE if ttest.get("reject_h0") else GARNET
    conc = (f"Home advantage CONFIRMED at alpha=0.05. Home teams score {ttest.get('mean_diff',0):+.3f} more "
            f"goals/match (95% CI: [{ttest.get('ci_lower',0):+.3f}, {ttest.get('ci_upper',0):+.3f}])."
            if ttest.get("reject_h0") else
            "INSUFFICIENT evidence to confirm home advantage at alpha=0.05 for current filters.")
    box = mpatches.FancyBboxPatch((0.0,-0.038),1.0,0.082,boxstyle="round,pad=0.01",
                                   facecolor=conc_color,edgecolor="none",
                                   transform=ax.transAxes,zorder=3,alpha=0.9)
    ax.add_patch(box)
    ax.text(0.5,0.002,conc,fontsize=8.5,fontweight="bold",color=WHITE,
            ha="center",va="center",transform=ax.transAxes,wrap=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CODE PAGES
# ══════════════════════════════════════════════════════════════════════════════

def pg_code_file(pdf, filepath, pg, total, section_label=""):
    """Render source code of one file as formatted PDF pages."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    fname    = os.path.basename(filepath)
    relpath  = os.path.relpath(filepath, os.path.dirname(os.path.abspath(__file__)))
    MAX_ROWS = 62      # lines per page (A4 landscape, small font)
    COLS     = 130     # chars per line before wrapping

    # Wrap long lines
    lines = []
    for line in raw_lines:
        line = line.rstrip("\n")
        if len(line) <= COLS:
            lines.append(line)
        else:
            wrapped = textwrap.wrap(line, width=COLS, subsequent_indent="    ",
                                    expand_tabs=False, replace_whitespace=False)
            lines.extend(wrapped if wrapped else [""])

    n_pages = max(1, math.ceil(len(lines) / MAX_ROWS))

    for chunk_i in range(n_pages):
        chunk = lines[chunk_i*MAX_ROWS : (chunk_i+1)*MAX_ROWS]
        fig   = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor(WHITE)

        # Header bar
        fig.patches.append(mpatches.FancyBboxPatch(
            (0.03,0.945), 0.94, 0.048, boxstyle="round,pad=0.005",
            facecolor=DARK, edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=10))
        fig.text(0.055, 0.969,
                 f"Source Code: {relpath}  ({chunk_i+1}/{n_pages})",
                 fontsize=10, fontweight="bold", color=WHITE,
                 va="center", transform=fig.transFigure)
        if section_label:
            fig.text(0.945, 0.969, section_label,
                     fontsize=8, color=(1,1,1,0.7),
                     va="center", ha="right", transform=fig.transFigure)

        _footer(fig, pg, total)

        # Code background
        fig.patches.append(mpatches.FancyBboxPatch(
            (0.03,0.055), 0.94, 0.878, boxstyle="round,pad=0.005",
            facecolor=CODE_BG, edgecolor=GRID, linewidth=0.8,
            transform=fig.transFigure, zorder=1))

        # Line numbers + code
        code_text = ""
        for j, line in enumerate(chunk):
            lineno = chunk_i * MAX_ROWS + j + 1
            code_text += f"{lineno:4d}  {line}\n"

        fig.text(0.045, 0.915, code_text,
                 fontsize=6.2,
                 fontfamily="DejaVu Sans Mono",
                 va="top", ha="left",
                 color=DARK,
                 transform=fig.transFigure,
                 linespacing=1.4)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        return n_pages  # return only after first page for page counter update

    return n_pages


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    PROJECT = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print("FC Barcelona Analytics  --  Full Submission PDF Generator")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset ...")
    try:
        df_raw, meta = load_dataset()
        df, audit    = clean_dataset(df_raw, start_date="2000-01-01", end_date="2025-12-31")
        print(f"      OK  {len(df):,} matches")
    except Exception as e:
        print(f"      No network — using synthetic demo data. ({e.__class__.__name__})")
        rng   = np.random.default_rng(42)
        n     = 5000
        home  = rng.poisson(1.55, n).astype(float)
        away  = rng.poisson(1.10, n).astype(float)
        years = rng.integers(2000, 2026, n)
        df = pd.DataFrame({
            "date": pd.to_datetime([f"{y}-01-01" for y in years]),
            "year": years, "home_score": home, "away_score": away,
            "total_goals": home+away, "goal_difference": home-away,
            "home_win": (home>away).astype(int), "away_win": (away>home).astype(int),
            "draw": (home==away).astype(int),
            "home_team":"Home","away_team":"Away",
            "tournament":"FIFA","city":"N/A","country":"N/A","is_neutral":False,
        })
        meta  = {"source":"synthetic demo"}
        audit = {}
    nn = get_non_neutral(df)

    # ── Source files to include ─────────────────────────────────────────────
    src_files = [
        (os.path.join(PROJECT, "app.py"),                          "Dashboard"),
        (os.path.join(PROJECT, "src", "data_loader.py"),           "src/"),
        (os.path.join(PROJECT, "src", "preprocessing.py"),         "src/"),
        (os.path.join(PROJECT, "src", "statistics.py"),            "src/"),
        (os.path.join(PROJECT, "src", "regression.py"),            "src/"),
        (os.path.join(PROJECT, "src", "bootstrap.py"),             "src/"),
        (os.path.join(PROJECT, "src", "normality.py"),             "src/"),
        (os.path.join(PROJECT, "src", "hypothesis_testing.py"),    "src/"),
        (os.path.join(PROJECT, "src", "visualizations.py"),        "src/"),
        (os.path.join(PROJECT, "tests", "test_preprocessing.py"),  "tests/"),
        (os.path.join(PROJECT, "tests", "test_statistics.py"),     "tests/"),
        (os.path.join(PROJECT, "tests", "test_regression.py"),     "tests/"),
        (os.path.join(PROJECT, "tests", "test_hypothesis_testing.py"), "tests/"),
    ]

    # Count total pages
    MAX_ROWS   = 62
    COLS       = 130
    code_pages = 0
    for fpath, _ in src_files:
        if os.path.exists(fpath):
            with open(fpath,"r",encoding="utf-8",errors="replace") as f:
                raw = f.readlines()
            wrapped = []
            for line in raw:
                line = line.rstrip("\n")
                if len(line) <= COLS: wrapped.append(line)
                else:
                    wrapped.extend(textwrap.wrap(line,width=COLS,
                                                  subsequent_indent="    ",
                                                  expand_tabs=False,
                                                  replace_whitespace=False) or [""])
            code_pages += max(1, math.ceil(len(wrapped)/MAX_ROWS))

    REPORT_PAGES = 9   # title + overview + ct_stats + ct_charts + reg + boot + norm + hyp + summary
    TOTAL        = REPORT_PAGES + code_pages
    print(f"      Report pages: {REPORT_PAGES}  |  Code pages: {code_pages}  |  Total: {TOTAL}")

    # ── Generate PDF ────────────────────────────────────────────────────────
    out_dir  = os.path.join(PROJECT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Football_Analytics_SUBMISSION.pdf")

    print(f"\n[2/3] Rendering {TOTAL} pages ...")
    _style()

    pg = 0
    with PdfPages(pdf_path) as pdf:
        d = pdf.infodict()
        d["Title"]  = "International Football Analytics - Full Submission"
        d["Author"] = "Football Analytics Hub"
        d["CreationDate"] = datetime.now()

        pg += 1; print(f"  pg {pg:3d}  Title page ...", end=" ", flush=True)
        pg_title(pdf, df, nn, meta, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Data Overview ...", end=" ", flush=True)
        pg_data_overview(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Central Tendency stats ...", end=" ", flush=True)
        pg_ct_stats(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Central Tendency charts ...", end=" ", flush=True)
        pg_ct_charts(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Regression ...", end=" ", flush=True)
        pg_regression(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Bootstrap ...", end=" ", flush=True)
        pg_bootstrap(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Normality ...", end=" ", flush=True)
        pg_normality(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Hypothesis Testing ...", end=" ", flush=True)
        pg_hypothesis(pdf, df, pg, TOTAL); print("done")

        pg += 1; print(f"  pg {pg:3d}  Summary ...", end=" ", flush=True)
        pg_summary(pdf, df, pg, TOTAL); print("done")

        # ── Source code pages ──
        print(f"\n  -- Source Code Section ({code_pages} pages) --")
        for fpath, label in src_files:
            if not os.path.exists(fpath):
                print(f"       SKIP  {fpath} (not found)")
                continue
            fname = os.path.basename(fpath)
            pg += 1
            print(f"  pg {pg:3d}+  {fname} ...", end=" ", flush=True)
            pg_code_file(pdf, fpath, pg, TOTAL, label)
            # count actual pages written
            with open(fpath,"r",encoding="utf-8",errors="replace") as f:
                raw = f.readlines()
            wrapped = []
            for line in raw:
                line = line.rstrip("\n")
                if len(line) <= COLS: wrapped.append(line)
                else:
                    wrapped.extend(textwrap.wrap(line,width=COLS,
                                                  subsequent_indent="    ",
                                                  expand_tabs=False,
                                                  replace_whitespace=False) or [""])
            n_file_pages = max(1, math.ceil(len(wrapped)/MAX_ROWS))
            pg += (n_file_pages - 1)
            print(f"done ({n_file_pages} page{'s' if n_file_pages>1 else ''})")

    size_mb = os.path.getsize(pdf_path) / (1024*1024)
    print(f"\n[3/3] PDF saved:  {pdf_path}")
    print(f"      Size: {size_mb:.2f} MB  |  ~{TOTAL} pages")
    print("\nDONE!  Submit:  reports/Football_Analytics_SUBMISSION.pdf")


if __name__ == "__main__":
    main()
