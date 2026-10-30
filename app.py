"""
app.py
------
International Football Analytics
A Computational Statistics Study of Goals, Home Advantage, and Performance Trends

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import io
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Analytics Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── FULL CSS OVERHAUL ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
* { box-sizing: border-box; }
html, body, .stApp {
    background-color: #080E1A !important;
    color: #E8EDF2 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1622 0%, #0F1E30 60%, #0B2818 100%) !important;
    border-right: 1px solid #1A3A2A !important;
}
section[data-testid="stSidebar"] * { color: #D0D8E0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label { color: #8FA8B8 !important; font-size: 0.78rem !important; }

/* ── Sidebar selectbox / input styling ── */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #0D1F2D !important;
    border: 1px solid #1E3A50 !important;
    border-radius: 6px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #050D18 0%, #0A1628 30%, #0D2B1A 70%, #061008 100%);
    border-bottom: 3px solid #00C853;
    padding: 28px 36px 24px;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,200,83,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(0,150,255,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    background: linear-gradient(90deg, #00E676, #69F0AE, #FFFFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px !important;
    line-height: 1.1 !important;
}
.hero-subtitle {
    color: #7A95A8 !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.5px;
    margin: 0 !important;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,200,83,0.12);
    border: 1px solid rgba(0,200,83,0.3);
    color: #00E676 !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}

/* ── Ticker Bar ── */
.ticker-wrap {
    background: linear-gradient(90deg, #00C853, #00897B);
    padding: 7px 20px;
    font-family: 'Oswald', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #fff !important;
    display: flex;
    gap: 40px;
    overflow: hidden;
}
.ticker-item { white-space: nowrap; }
.ticker-sep { color: rgba(255,255,255,0.5); }

/* ── Section Title ── */
.section-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    color: #FFFFFF !important;
    border-left: 4px solid #00C853;
    padding-left: 14px;
    margin: 24px 0 4px !important;
    line-height: 1.2 !important;
}
.section-sub {
    color: #6A8499 !important;
    font-size: 0.88rem !important;
    padding-left: 18px;
    margin-bottom: 20px !important;
}

/* ── Scoreboard KPI Cards ── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.score-card {
    background: linear-gradient(145deg, #0E1E2E, #152335);
    border: 1px solid #1A3045;
    border-radius: 12px;
    padding: 18px 16px 14px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    cursor: default;
}
.score-card:hover {
    transform: translateY(-3px);
    border-color: #00C853;
}
.score-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00C853, #00897B);
}
.score-card .sc-icon {
    font-size: 1.5rem;
    margin-bottom: 8px;
    display: block;
}
.score-card .sc-value {
    font-family: 'Oswald', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00E676;
    line-height: 1;
    margin-bottom: 4px;
}
.score-card .sc-label {
    font-size: 0.72rem;
    color: #5A7A8E;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.score-card .sc-delta {
    font-size: 0.75rem;
    margin-top: 4px;
    font-weight: 600;
}
.delta-up { color: #00E676; }
.delta-down { color: #FF5252; }
.delta-neutral { color: #9E9E9E; }

/* ── Match Result Cards ── */
.match-card {
    background: #0E1E2E;
    border: 1px solid #1A3045;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: background 0.2s;
}
.match-card:hover { background: #132030; }
.match-team { font-weight: 600; font-size: 0.95rem; color: #D0DCE8; }
.match-score {
    font-family: 'Oswald', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
    background: linear-gradient(135deg, #0A3020, #0E2818);
    border: 1px solid #1A5030;
    border-radius: 8px;
    padding: 4px 20px;
    min-width: 90px;
    text-align: center;
}
.match-meta { font-size: 0.75rem; color: #4A6A7E; margin-top: 3px; }

/* ── Experiment Cards ── */
.exp-card {
    background: linear-gradient(145deg, #0C1A28, #101F30);
    border: 1px solid #1A3045;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
    position: relative;
}
.exp-card .exp-number {
    font-family: 'Oswald', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    color: rgba(0,200,83,0.08);
    position: absolute;
    top: 10px; right: 20px;
    line-height: 1;
}
.exp-card h4 {
    font-family: 'Oswald', sans-serif;
    font-size: 1.1rem;
    color: #00E676 !important;
    margin: 0 0 6px !important;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.exp-card p { color: #7A95A8; font-size: 0.85rem; margin: 0; line-height: 1.5; }

/* ── Stat Banner ── */
.stat-banner {
    background: linear-gradient(90deg, rgba(0,200,83,0.1), rgba(0,137,123,0.1));
    border: 1px solid rgba(0,200,83,0.2);
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0;
    display: flex;
    align-items: center;
    gap: 14px;
}
.stat-banner .sb-icon { font-size: 1.8rem; }
.stat-banner .sb-text { font-size: 0.95rem; color: #B0C8D8; line-height: 1.4; }
.stat-banner .sb-text strong { color: #00E676; }

/* ── Decision Box ── */
.decision-box-yes {
    background: linear-gradient(135deg, rgba(0,200,83,0.12), rgba(0,137,123,0.08));
    border: 1px solid #00C853;
    border-radius: 10px;
    padding: 16px 22px;
    margin: 14px 0;
}
.decision-box-no {
    background: linear-gradient(135deg, rgba(255,82,82,0.1), rgba(200,0,0,0.06));
    border: 1px solid #FF5252;
    border-radius: 10px;
    padding: 16px 22px;
    margin: 14px 0;
}
.decision-box-warn {
    background: linear-gradient(135deg, rgba(255,193,7,0.1), rgba(255,152,0,0.06));
    border: 1px solid #FFC107;
    border-radius: 10px;
    padding: 16px 22px;
    margin: 14px 0;
}
.decision-label {
    font-family: 'Oswald', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.decision-text { font-size: 0.88rem; color: #A0B8C8; line-height: 1.5; }

/* ── Fun Fact Box ── */
.funfact {
    background: linear-gradient(135deg, #0A1E30, #071528);
    border-left: 3px solid #FFB300;
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    margin: 14px 0;
}
.funfact .ff-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #FFB300;
    font-weight: 700;
    margin-bottom: 4px;
}
.funfact .ff-text { font-size: 0.9rem; color: #B0C4D4; line-height: 1.4; }

/* ── Pills / Tags ── */
.tag {
    display: inline-block;
    background: rgba(0,200,83,0.1);
    border: 1px solid rgba(0,200,83,0.25);
    color: #69F0AE !important;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 12px;
    margin: 2px;
    letter-spacing: 0.3px;
}

/* ── Divider ── */
.pitch-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1A3A2A, #00C853, #1A3A2A, transparent);
    margin: 24px 0;
}

/* ── Table / Dataframe ── */
.stDataFrame { background: #0C1A28 !important; border-radius: 10px !important; }
.stDataFrame th { background: #0E2030 !important; color: #00E676 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0C1A28 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #5A7A8E !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    color: #00E676 !important;
    background: #0E2030 !important;
    border: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00A846, #007B33) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    font-family: 'Oswald', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00C853, #009A3E) !important;
    box-shadow: 0 4px 15px rgba(0,200,83,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics override ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0E1E2E, #152335) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    border: 1px solid #1A3045 !important;
}
[data-testid="stMetricValue"] { color: #00E676 !important; font-family: 'Oswald', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #5A7A8E !important; font-size: 0.75rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0C1A28 !important;
    border: 1px solid #1A3045 !important;
    border-radius: 8px !important;
    color: #7A95A8 !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: #0A1622 !important;
    border: 1px solid #1A3045 !important;
    border-top: none !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00C853 !important; }

/* ── Selectbox / input ── */
[data-baseweb="select"] > div { background: #0C1A28 !important; border-color: #1A3045 !important; }
[data-baseweb="input"] > div { background: #0C1A28 !important; border-color: #1A3045 !important; }
[data-baseweb="input"] input { color: #E0EAF0 !important; }

/* ── Success / Info / Warning ── */
.stSuccess { background: rgba(0,200,83,0.1) !important; border-color: #00C853 !important; color: #B0E8C0 !important; }
.stWarning { background: rgba(255,193,7,0.1) !important; border-color: #FFC107 !important; }
.stError { background: rgba(255,82,82,0.1) !important; border-color: #FF5252 !important; }
.stInfo { background: rgba(33,150,243,0.1) !important; border-color: #2196F3 !important; }

/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 20px 16px 16px;
    text-align: center;
    border-bottom: 1px solid rgba(0,200,83,0.15);
    margin-bottom: 16px;
}
.sidebar-brand .sb-ball { font-size: 3rem; margin-bottom: 6px; display: block; }
.sidebar-brand h3 {
    font-family: 'Oswald', sans-serif !important;
    color: #00E676 !important;
    font-size: 1.1rem !important;
    margin: 0 0 2px !important;
    letter-spacing: 1px;
}
.sidebar-brand p { color: #4A6A7E !important; font-size: 0.72rem !important; margin: 0 !important; }

/* ── Sidebar nav item ── */
.sidebar-nav-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3A5A6E;
    font-weight: 700;
    margin: 16px 0 6px;
    padding: 0 4px;
}

/* ── Leaderboard ── */
.lb-row {
    display: flex;
    align-items: center;
    background: #0C1A28;
    border: 1px solid #152535;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 5px 0;
    gap: 12px;
    transition: background 0.2s;
}
.lb-row:hover { background: #102030; }
.lb-rank { font-family: 'Oswald', sans-serif; font-size: 1.1rem; color: #3A5A6E; min-width: 30px; }
.lb-rank.gold { color: #FFD700; }
.lb-rank.silver { color: #C0C0C0; }
.lb-rank.bronze { color: #CD7F32; }
.lb-name { flex: 1; font-weight: 600; color: #C8D8E8; font-size: 0.9rem; }
.lb-val { font-family: 'Oswald', sans-serif; font-size: 1.1rem; color: #00E676; font-weight: 700; }

/* ── Hypothesis result ── */
.hyp-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #0E2030;
}
.hyp-label { font-size: 0.82rem; color: #5A7A8E; font-weight: 600; letter-spacing: 0.3px; }
.hyp-value { font-family: 'Oswald', sans-serif; font-size: 1.05rem; color: #D0DCE8; font-weight: 600; }
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


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚽ Fetching live dataset…")
def _load_and_clean(start: str = "2000-01-01", end: str = "2025-12-31"):
    df_raw, meta = load_dataset()
    df_clean, audit = clean_dataset(df_raw, start_date=start, end_date=end)
    return df_clean, audit, meta


def sc(icon, value, label, delta=None, delta_type="neutral"):
    """Render a scoreboard-style KPI card."""
    delta_html = ""
    if delta is not None:
        cls = f"delta-{delta_type}"
        arrow = "▲" if delta_type == "up" else ("▼" if delta_type == "down" else "●")
        delta_html = f'<div class="sc-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="score-card">
        <span class="sc-icon">{icon}</span>
        <div class="sc-value">{value}</div>
        <div class="sc-label">{label}</div>
        {delta_html}
    </div>"""


def stat_banner(icon, text):
    return f'<div class="stat-banner"><span class="sb-icon">{icon}</span><div class="sb-text">{text}</div></div>'


def pitch_divider():
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)


def section_title(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def funfact(text):
    st.markdown(f"""
    <div class="funfact">
        <div class="ff-label">⚡ Did You Know?</div>
        <div class="ff-text">{text}</div>
    </div>""", unsafe_allow_html=True)


def decision_box(text, kind="yes"):
    st.markdown(f"""
    <div class="decision-box-{kind}">
        <div class="decision-label">{"✅ Conclusion" if kind=="yes" else "❌ Conclusion" if kind=="no" else "⚠️ Note"}</div>
        <div class="decision-text">{text}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(df_full: pd.DataFrame) -> dict:
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <span class="sb-ball">⚽</span>
        <h3>FOOTBALL ANALYTICS</h3>
        <p>Computational Statistics Hub</p>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-nav-label">📋 Pages</div>', unsafe_allow_html=True)
    page_options = [
        "🏟️ Home",
        "📊 Central Tendency",
        "📈 Regression",
        "🎲 Bootstrap",
        "🔔 Normality Testing",
        "🔬 Hypothesis Testing",
        "🗂️ Match Explorer",
        "ℹ️ About",
    ]
    page = st.sidebar.selectbox("Navigate", page_options, label_visibility="collapsed")

    st.sidebar.markdown('<div class="sidebar-nav-label">⚙️ Filters</div>', unsafe_allow_html=True)

    year_min = int(df_full["year"].min()) if len(df_full) > 0 else 2000
    year_max = int(df_full["year"].max()) if len(df_full) > 0 else 2025
    col_s, col_e = st.sidebar.columns(2)
    start_year = col_s.number_input("From", min_value=year_min, max_value=year_max, value=year_min, step=1)
    end_year = col_e.number_input("To", min_value=year_min, max_value=year_max, value=year_max, step=1)

    include_neutral = st.sidebar.checkbox("Include neutral-venue matches", value=True)

    all_tournaments = ["All"] + sorted(df_full["tournament"].dropna().unique().tolist())
    selected_tournaments = st.sidebar.multiselect("Tournament(s)", all_tournaments, default=["All"])
    tournaments = None if "All" in selected_tournaments else selected_tournaments

    all_teams = sorted(pd.concat([df_full["home_team"], df_full["away_team"]]).dropna().unique().tolist())
    selected_teams = st.sidebar.multiselect("Team(s)", all_teams)
    teams = selected_teams if selected_teams else None

    st.sidebar.markdown('<div class="sidebar-nav-label">📁 Data</div>', unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader("Upload custom CSV", type=["csv"])

    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun()

    return {
        "page": page,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "include_neutral": include_neutral,
        "tournaments": tournaments,
        "teams": teams,
        "uploaded": uploaded,
    }


# ── Hero Banner (appears on all pages) ─────────────────────────────────────────
def render_hero(df: pd.DataFrame = None, page: str = ""):
    subtitles = {
        "🏟️ Home":            "Season overview — goals, wins, and the home advantage story",
        "📊 Central Tendency": "Experiment 1 — Mean, Median, Mode & spread of football scores",
        "📈 Regression":       "Experiment 2 — How has average scoring changed over the years?",
        "🎲 Bootstrap":        "Experiment 3 — Quantifying home advantage with 10,000 resamples",
        "🔔 Normality Testing":"Experiment 4 — Is goal difference normally distributed?",
        "🔬 Hypothesis Testing":"Experiment 5 — Statistical proof of home advantage",
        "🗂️ Match Explorer":   "Browse every match, search teams, and export data",
        "ℹ️ About":            "Project details, methodology, and dataset attribution",
    }
    subtitle = subtitles.get(page, "A Computational Statistics Study of International Football")
    matches_str = f"{len(df):,} matches" if df is not None else ""

    st.markdown(f"""
    <div class="hero-banner">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
            <div>
                <span class="hero-badge">⚽ LIVE DATA ANALYTICS</span>
                <div class="hero-title">Football Analytics Hub</div>
                <p class="hero-subtitle">{subtitle}</p>
            </div>
            <div style="text-align:right; padding-top:6px;">
                {'<span style="font-family:Oswald,sans-serif;font-size:1.6rem;color:#00E676;font-weight:700;">'+matches_str+'</span><br><span style="font-size:0.72rem;color:#4A6A7E;letter-spacing:1px;text-transform:uppercase;">IN VIEW</span>' if matches_str else ''}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    nn = get_non_neutral(df)
    kpi = stats_module.kpi_summary(df, nn)

    # ── Ticker ──
    hw = kpi['home_win_pct']
    aw = kpi['away_win_pct']
    dr = kpi['pct_draw']
    avg = kpi['avg_total_goals']
    diff = kpi['mean_goal_diff_nn']
    st.markdown(f"""
    <div class="ticker-wrap">
        <span class="ticker-item">🏆 Home Win Rate: {hw:.1f}%</span>
        <span class="ticker-sep">|</span>
        <span class="ticker-item">✈️ Away Win Rate: {aw:.1f}%</span>
        <span class="ticker-sep">|</span>
        <span class="ticker-item">🤝 Draw Rate: {dr:.1f}%</span>
        <span class="ticker-sep">|</span>
        <span class="ticker-item">⚽ Avg Goals/Match: {avg:.2f}</span>
        <span class="ticker-sep">|</span>
        <span class="ticker-item">📊 Home Advantage: {diff:+.3f} goals</span>
    </div>""", unsafe_allow_html=True)

    pitch_divider()
    section_title("📊 Match Statistics", f"Based on {len(df):,} matches · {kpi.get('year_range', '')}")

    # ── KPI Grid ──
    delta_sign = "up" if diff > 0 else "down" if diff < 0 else "neutral"
    cards_html = '<div class="score-grid">'
    cards_html += sc("🏟️", f"{kpi['total_matches']:,}", "Total Matches")
    cards_html += sc("⚽", f"{avg:.2f}", "Avg Goals / Match")
    cards_html += sc("🏠", f"{kpi['avg_home_goals']:.2f}", "Avg Home Goals")
    cards_html += sc("✈️", f"{kpi['avg_away_goals']:.2f}", "Avg Away Goals")
    cards_html += sc("🤝", f"{dr:.1f}%", "Draw Rate")
    cards_html += sc("📈", f"{diff:+.3f}", "Home Advantage", delta=f"{abs(diff):.3f} goals", delta_type=delta_sign)
    cards_html += sc("🏆", f"{hw:.1f}%", "Home Win %")
    cards_html += sc("✈️", f"{aw:.1f}%", "Away Win %")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    pitch_divider()

    # ── Fun Fact ──
    if diff > 0:
        funfact(f"Home teams score on average <strong>{diff:+.3f} more goals</strong> per match than away teams — a measurable statistical advantage backed by {len(nn):,} non-neutral matches.")
    
    # ── Charts ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_yearly_trend(df), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_result_distribution(df), use_container_width=True)

    st.plotly_chart(viz.plot_total_goals_histogram(df), use_container_width=True)

    pitch_divider()
    section_title("🔬 Five Statistical Experiments", "Navigate the sidebar to explore each experiment in depth")

    exp_data = [
        ("01", "Central Tendency", "Mean, Median, Mode, IQR — describing the typical international football match score."),
        ("02", "Regression Analysis", "OLS linear regression of yearly mean goals on calendar year — is scoring increasing?"),
        ("03", "Bootstrap Resampling", "10,000 resamples to build a confidence interval for home advantage without normality assumptions."),
        ("04", "Normality Testing", "Shapiro-Wilk test + Q-Q plot to check if goal differences follow a normal distribution."),
        ("05", "Hypothesis Testing", "Formal paired t-test: H₀: μD = 0 vs H₁: μD > 0 with Cohen's dz effect size."),
    ]
    c1, c2 = st.columns(2)
    for i, (num, title, desc) in enumerate(exp_data):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="exp-card">
                <div class="exp-number">{num}</div>
                <h4>⚽ {title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)


def page_central_tendency(df: pd.DataFrame):
    section_title("📊 Central Tendency", "Descriptive statistics for total goals, home score, away score, and goal difference")

    tbl = stats_module.central_tendency_table(df)
    stat_goals = stats_module.central_tendency(df["total_goals"])

    if stat_goals["n"] == 0:
        st.warning("No data available for the selected filters.")
        return

    # ── KPI Grid ──
    mean_v = stat_goals["mean"]
    median_v = stat_goals["median"]
    mode_v = stat_goals["mode"]
    std_v = stat_goals["std"]
    iqr_v = stat_goals["iqr"]
    skew_v = stat_goals["skewness"]

    cards_html = '<div class="score-grid">'
    cards_html += sc("📐", f"{mean_v:.3f}", "Mean Goals/Match")
    cards_html += sc("📍", f"{median_v:.1f}", "Median Goals")
    cards_html += sc("🎯", str(mode_v[0] if isinstance(mode_v, list) else mode_v), "Modal Goals")
    cards_html += sc("📊", f"{std_v:.3f}", "Std Deviation")
    cards_html += sc("📦", f"{iqr_v:.3f}", "IQR")
    cards_html += sc("〰️", f"{skew_v:.3f}", "Skewness")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    pitch_divider()

    # ── Full Stats Table ──
    section_title("📋 Full Summary Table", "All variables: total goals, home score, away score, goal difference")
    display_cols = ["n","mean","median","std","variance","min","max","range","q1","q3","iqr","skewness","kurtosis"]
    display_tbl = tbl[[c for c in display_cols if c in tbl.columns]].round(4)
    st.dataframe(display_tbl, use_container_width=True)

    # ── Mode Table ──
    section_title("🎯 Mode Values", "Most frequently occurring score totals")
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

    csv_bytes = display_tbl.reset_index().to_csv(index=False).encode()
    st.download_button("⬇ Download Summary CSV", csv_bytes, "central_tendency.csv", "text/csv")

    pitch_divider()
    section_title("📈 Visual Analysis")

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

    pitch_divider()
    with st.expander("📖 Statistical Interpretation"):
        skew_label = "positively skewed" if skew_v > 0 else ("approximately symmetric" if abs(skew_v) < 0.3 else "negatively skewed")
        direction = "exceeds" if mean_v > median_v else "is below"
        pull = "upward" if mean_v > median_v else "downward"
        st.markdown(f"""
The **mean** total goals per match is **{mean_v:.3f}**, the arithmetic average across all selected matches.
The **median** is **{median_v:.1f}** — the middle value, less affected by high-scoring outliers.
The most common (modal) total is **{mode_v}** goals per match.

Because the distribution is **{skew_label}** (skewness = {skew_v:.3f}), the mean **{direction}** the median.
High-scoring matches pull the mean **{pull}** relative to the typical match.

The **standard deviation** ({std_v:.3f}) and **IQR** ({iqr_v:.3f}) both measure spread.
A larger IQR indicates more variability in how goals are distributed across matches.
        """)


def page_regression(df: pd.DataFrame):
    section_title("📈 Regression Analysis", "Simple OLS linear regression: yearly mean goals ~ calendar year")

    yearly = stats_module.yearly_goals_summary(df)
    if len(yearly) < 3:
        st.warning("Not enough yearly data points for regression. Adjust the date range.")
        return

    reg = reg_module.fit_yearly_regression(yearly, weighted=False)

    slope = reg.get("slope")
    cards_html = '<div class="score-grid">'
    cards_html += sc("📐", f"{slope:+.4f}" if slope else "N/A", "Slope (goals/year)",
                     delta=("Rising" if slope and slope > 0 else "Falling"),
                     delta_type=("up" if slope and slope > 0 else "down"))
    cards_html += sc("📍", f"{reg['intercept']:.3f}" if reg['intercept'] else "N/A", "Intercept")
    cards_html += sc("📊", f"{reg['r_squared']:.4f}" if reg['r_squared'] else "N/A", "R² Score")
    cards_html += sc("📏", f"{reg['rmse']:.4f}" if reg['rmse'] else "N/A", "RMSE")
    cards_html += sc("🔬", f"{reg['p_value']:.4f}" if reg['p_value'] else "N/A", "p-value (slope)")
    cards_html += sc("📅", str(reg['n']), "Years in Model")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    pitch_divider()

    if slope is not None:
        if slope > 0.005:
            st.markdown(stat_banner("📈", f"The regression slope is <strong>+{slope:.4f} goals/year</strong> — scoring is gradually <strong>increasing</strong> over the study period."), unsafe_allow_html=True)
        elif slope < -0.005:
            st.markdown(stat_banner("📉", f"The regression slope is <strong>{slope:.4f} goals/year</strong> — scoring has been gradually <strong>declining</strong> over the study period."), unsafe_allow_html=True)
        else:
            st.markdown(stat_banner("➡️", f"The regression slope is <strong>{slope:.4f} goals/year</strong> — scoring is effectively <strong>flat</strong> over the study period."), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_regression(yearly, reg), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_regression_residuals(yearly, reg), use_container_width=True)

    with st.expander("📋 Yearly Data Table"):
        st.dataframe(yearly.round(4), use_container_width=True)
        st.download_button("⬇ Download Yearly Data", yearly.to_csv(index=False).encode(), "yearly_goals.csv", "text/csv")

    with st.expander("📖 Interpretation"):
        st.markdown(reg_module.interpret_regression(reg))

    with st.expander("🔬 Weighted Regression (Sensitivity Analysis)"):
        st.markdown("Weighted OLS assigns greater influence to years with more matches.")
        reg_w = reg_module.fit_yearly_regression(yearly, weighted=True)
        if reg_w.get("w_slope") is not None:
            wc1, wc2 = st.columns(2)
            wc1.metric("Weighted Slope", f"{reg_w['w_slope']:+.4f}")
            wc2.metric("Weighted Intercept", f"{reg_w['w_intercept']:.3f}")


def page_bootstrap(df: pd.DataFrame):
    section_title("🎲 Bootstrap Resampling", "Non-parametric confidence interval for the mean home-away goal difference")

    st.markdown(stat_banner("🎲", "Bootstrap resampling draws <strong>10,000 random samples</strong> from the data to estimate uncertainty without assuming normality. Each sample recalculates the mean home-away goal difference, building a distribution of plausible values."), unsafe_allow_html=True)

    pitch_divider()

    col_s, col_n = st.columns(2)
    n_iter = col_n.number_input("Bootstrap iterations", min_value=1000, max_value=100_000, value=10_000, step=1000)
    seed = col_s.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    run_btn = st.button("▶ Run Bootstrap Analysis")
    if run_btn or "boot_results" not in st.session_state:
        with st.spinner(f"Running {n_iter:,} bootstrap iterations… ⚽"):
            boot = boot_module.bootstrap_mean_goal_diff(df, n_iterations=int(n_iter), random_seed=int(seed))
        st.session_state["boot_results"] = boot

    boot = st.session_state.get("boot_results", {})
    if boot.get("error"):
        st.error(boot["error"])
        return

    cards_html = '<div class="score-grid">'
    cards_html += sc("🏟️", f"{boot['n_matches']:,}", "Eligible Matches")
    cards_html += sc("📊", f"{boot['observed_mean']:+.4f}", "Observed Mean Diff")
    cards_html += sc("📏", f"{boot['bootstrap_se']:.4f}", "Bootstrap Std Error")
    cards_html += sc("🔁", f"{boot['n_iterations']:,}", "Iterations")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    ci_pct = int(boot["confidence_level"] * 100)
    lower, upper = boot["ci_lower"], boot["ci_upper"]

    if lower > 0:
        decision_box(
            f"The <strong>{ci_pct}% bootstrap confidence interval</strong> is "
            f"[{lower:+.4f}, {upper:+.4f}]. Since the entire interval is above zero, "
            f"we have strong non-parametric evidence of a positive home advantage.",
            kind="yes"
        )
    else:
        decision_box(
            f"The <strong>{ci_pct}% bootstrap confidence interval</strong> is "
            f"[{lower:+.4f}, {upper:+.4f}]. The interval includes zero — insufficient evidence to confirm home advantage.",
            kind="no"
        )

    st.plotly_chart(viz.plot_bootstrap_distribution(boot), use_container_width=True)

    with st.expander("📖 Interpretation"):
        st.markdown(boot_module.interpret_bootstrap(boot))

    with st.expander("🔬 Year-Block Bootstrap (Sensitivity Analysis)"):
        if st.button("Run Year-Block Bootstrap"):
            with st.spinner("Running year-block bootstrap…"):
                yb = boot_module.bootstrap_year_block(df, n_iterations=int(n_iter), random_seed=int(seed))
            st.session_state["yb_results"] = yb
        yb = st.session_state.get("yb_results")
        if yb and not yb.get("error"):
            yc1, yc2, yc3 = st.columns(3)
            yc1.metric("Year-Block CI Lower", f"{yb['ci_lower']:+.4f}")
            yc2.metric("Year-Block CI Upper", f"{yb['ci_upper']:+.4f}")
            yc3.metric("Bootstrap SE (block)", f"{yb['bootstrap_se']:.4f}")
            st.info("A wider year-block interval suggests non-negligible year-to-year temporal dependence.")


def page_normality(df: pd.DataFrame):
    section_title("🔔 Normality Testing", "Is the goal-difference distribution consistent with a normal distribution?")

    st.markdown(stat_banner("🔔", "The <strong>Shapiro-Wilk test</strong> checks whether the distribution of match goal differences could plausibly have come from a normal distribution. We also inspect a <strong>Q-Q plot</strong> and histogram for visual confirmation."), unsafe_allow_html=True)

    pitch_divider()
    seed = st.number_input("Subsampling seed (if n > 5000)", value=42, step=1)

    with st.spinner("Running normality tests…"):
        norm = norm_module.shapiro_wilk_test(df, random_seed=int(seed))

    if norm.get("error"):
        st.error(norm["error"])
        return

    cards_html = '<div class="score-grid">'
    cards_html += sc("📊", f"{norm['full_n']:,}", "Full Sample n")
    cards_html += sc("🔬", f"{norm['test_n']:,}", "Test Sample n")
    cards_html += sc("📐", f"{norm['statistic']:.6f}" if norm['statistic'] else "N/A", "Shapiro-Wilk W")
    cards_html += sc("🎯", f"{norm['p_value']:.6f}" if norm['p_value'] else "N/A", "p-value")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    if norm.get("subsampled"):
        st.markdown(stat_banner("ℹ️", f"Full sample of <strong>{norm['full_n']:,} matches</strong> exceeds the Shapiro-Wilk limit (n=5,000). Test run on a random subsample of <strong>{norm['test_n']:,}</strong> observations (seed={norm['random_seed']})."), unsafe_allow_html=True)

    if norm.get("reject_h0"):
        decision_box(
            f"<strong>{norm['decision']}</strong><br>"
            "The p-value is below 0.05 — we reject H₀ of normality. "
            "Note: With large samples, the Shapiro-Wilk test almost always rejects normality even for minor deviations. "
            "The Q-Q plot and histogram are more practically informative.",
            kind="no"
        )
    else:
        decision_box(f"<strong>{norm['decision']}</strong>", kind="yes")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_normality_histogram(norm), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_qq(norm), use_container_width=True)

    with st.expander("📖 Interpretation"):
        st.markdown(norm_module.interpret_normality(norm))

    funfact("Large datasets (n > 5,000) almost always fail the Shapiro-Wilk test even when deviations from normality are negligibly small — this is a known limitation of the test for big data.")


def page_hypothesis(df: pd.DataFrame):
    section_title("🔬 Hypothesis Testing", "Formal test: does home advantage exist in international football?")

    st.markdown(stat_banner("🔬",
        "We test <strong>H₀: μD = 0</strong> (no home advantage) against <strong>H₁: μD > 0</strong> "
        "(home teams score more) using a <strong>paired t-test</strong> on non-neutral matches at α = 0.05."
    ), unsafe_allow_html=True)

    pitch_divider()

    with st.spinner("Running hypothesis tests…"):
        ttest = hyp_module.paired_ttest(df)

    if ttest.get("error") and ttest["n"] == 0:
        st.error("No non-neutral matches in current selection.")
        return

    # ── Scoreboard grid ──
    cards_html = '<div class="score-grid">'
    cards_html += sc("🏟️", f"{ttest['n']:,}", "Eligible Matches")
    cards_html += sc("🏠", f"{ttest['mean_home']:.4f}", "Mean Home Goals")
    cards_html += sc("✈️", f"{ttest['mean_away']:.4f}", "Mean Away Goals")
    cards_html += sc("📊", f"{ttest['mean_diff']:+.4f}", "Mean Difference D",
                     delta=("Home lead" if ttest['mean_diff'] > 0 else "Away lead"),
                     delta_type=("up" if ttest['mean_diff'] > 0 else "down"))
    cards_html += sc("📐", f"{ttest['t_stat']:.4f}", "t-statistic")
    cards_html += sc("🎯", f"{ttest['p_one_sided']:.6f}", "p-value (one-sided)")
    cards_html += sc("📏", f"{ttest['cohens_dz']:.4f}" if ttest['cohens_dz'] else "N/A", "Cohen's dz")
    cards_html += sc("🏆", ttest['effect_label'] or "N/A", "Effect Size")
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0C1A28;border:1px solid #1A3045;border-radius:8px;padding:12px 18px;margin:12px 0;font-size:0.9rem;color:#8AAABB;">
        📐 <strong style="color:#D0E0EA;">95% CI for mean D:</strong>
        <span style="font-family:'Oswald',sans-serif;color:#00E676;font-size:1.1rem;margin-left:8px;">
            [{ttest['ci_lower']:+.4f}, {ttest['ci_upper']:+.4f}]
        </span>
    </div>""", unsafe_allow_html=True)

    if ttest["reject_h0"]:
        decision_box(
            f"<strong>{ttest['decision']}</strong><br>"
            f"t({ttest['df']}) = {ttest['t_stat']:.4f}, p = {ttest['p_one_sided']:.6f} &lt; 0.05. "
            f"Effect size: Cohen's dz = {ttest['cohens_dz']:.4f} ({ttest['effect_label']}). "
            f"Home teams score significantly more goals than away teams in non-neutral matches.",
            kind="yes"
        )
    else:
        decision_box(
            f"<strong>{ttest['decision']}</strong> — Fail to reject H₀ at α = 0.05.",
            kind="no"
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_home_away_bar(ttest), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_goal_diff_histogram(ttest), use_container_width=True)

    with st.expander("📖 Full Interpretation"):
        st.markdown(hyp_module.interpret_ttest(ttest))

    with st.expander("🔬 Wilcoxon Signed-Rank Test (Non-Parametric Sensitivity)"):
        st.markdown("The Wilcoxon test is a non-parametric alternative — no normality assumption required.")
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
                wc3.metric("Reject H₀?", "✅ Yes" if wilcox["reject_h0"] else "❌ No")
                st.caption(wilcox["note"])

    funfact("The paired t-test is robust to non-normality for large samples (n > 1,000) due to the Central Limit Theorem — making it a valid choice even if individual goal differences aren't normally distributed.")


def page_data_explorer(df_full: pd.DataFrame, df: pd.DataFrame, audit: dict, meta: dict):
    section_title("🗂️ Match Explorer", f"Browse {len(df):,} filtered matches from {len(df_full):,} total")

    tab1, tab2, tab3 = st.tabs(["📋 Match Table", "🏆 Leaderboards", "🔍 Audit & Source"])

    with tab1:
        col_search, col_dl = st.columns([3, 1])
        search_term = col_search.text_input("🔍 Search by team name", "")
        df_view = df.copy()
        if search_term:
            mask = (
                df_view["home_team"].str.contains(search_term, case=False, na=False) |
                df_view["away_team"].str.contains(search_term, case=False, na=False)
            )
            df_view = df_view[mask]
        st.caption(f"Showing **{len(df_view):,}** matches")
        display_cols = [c for c in [
            "date", "home_team", "home_score", "away_score", "away_team",
            "tournament", "city", "country", "is_neutral",
            "total_goals", "goal_difference", "home_win", "draw", "away_win"
        ] if c in df_view.columns]
        st.dataframe(
            df_view[display_cols].sort_values("date", ascending=False).reset_index(drop=True),
            use_container_width=True, height=420
        )
        col_dl.download_button("⬇ Export CSV", df_view[display_cols].to_csv(index=False).encode(),
                               "filtered_matches.csv", "text/csv", use_container_width=True)

    with tab2:
        section_title("🏆 Top Scoring Matches", "Highest total goals in filtered data")
        top_matches = df.nlargest(15, "total_goals")[
            ["date", "home_team", "home_score", "away_score", "away_team", "total_goals", "tournament"]
        ].reset_index(drop=True)

        for i, row in top_matches.iterrows():
            rank_cls = "gold" if i == 0 else "silver" if i == 1 else "bronze" if i == 2 else ""
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            st.markdown(f"""
            <div class="match-card">
                <span class="lb-rank {rank_cls}" style="min-width:36px">{medal}</span>
                <div style="flex:1">
                    <div class="match-team">{row['home_team']} vs {row['away_team']}</div>
                    <div class="match-meta">📅 {str(row['date'])[:10]} &nbsp;·&nbsp; 🏆 {row['tournament']}</div>
                </div>
                <div class="match-score">{int(row['home_score'])} – {int(row['away_score'])}</div>
                <div style="min-width:60px;text-align:right">
                    <span class="lb-val">⚽ {int(row['total_goals'])}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        pitch_divider()
        section_title("📊 Most Active Teams", "By total appearances in filtered matches")
        all_teams_series = pd.concat([df["home_team"], df["away_team"]])
        top_teams = all_teams_series.value_counts().head(10).reset_index()
        top_teams.columns = ["Team", "Matches"]
        for i, row in top_teams.iterrows():
            rank_cls = "gold" if i == 0 else "silver" if i == 1 else "bronze" if i == 2 else ""
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            st.markdown(f"""
            <div class="lb-row">
                <span class="lb-rank {rank_cls}">{medal}</span>
                <span class="lb-name">{row['Team']}</span>
                <span class="lb-val">{row['Matches']:,} matches</span>
            </div>""", unsafe_allow_html=True)

    with tab3:
        section_title("🔍 Data Cleaning Audit")
        audit_df = audit_to_dataframe(audit)
        st.dataframe(audit_df, use_container_width=True)
        with st.expander("📄 Dataset Source Info"):
            st.json(meta)


def page_about():
    section_title("ℹ️ About This Project", "International Football Analytics — Computational Statistics Mini-Project")

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0A1E30,#071528);border:1px solid #1A3045;border-radius:14px;padding:24px 28px;margin:16px 0;">
        <div style="font-family:'Oswald',sans-serif;font-size:1.3rem;color:#00E676;margin-bottom:4px;font-weight:600;">
            ⚽ Statistical Analysis of International Football Performance
        </div>
        <div style="color:#6A8499;font-size:0.85rem;margin-bottom:18px;">A Computational Statistics Study of Home Advantage, Goal-Scoring Patterns, and Match Outcomes</div>
        <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
            <tr><td style="color:#4A6A7E;padding:6px 0;width:160px;font-weight:600;">Subject</td><td style="color:#C0D0E0;">Computational Statistics</td></tr>
            <tr><td style="color:#4A6A7E;padding:6px 0;font-weight:600;">Student Name</td><td style="color:#C0D0E0;font-style:italic;">*(Enter your name)*</td></tr>
            <tr><td style="color:#4A6A7E;padding:6px 0;font-weight:600;">USN</td><td style="color:#C0D0E0;font-style:italic;">*(Enter your USN)*</td></tr>
            <tr><td style="color:#4A6A7E;padding:6px 0;font-weight:600;">Department</td><td style="color:#C0D0E0;">Computer Science &amp; Engineering</td></tr>
            <tr><td style="color:#4A6A7E;padding:6px 0;font-weight:600;">Institution</td><td style="color:#C0D0E0;font-style:italic;">*(Enter institution)*</td></tr>
            <tr><td style="color:#4A6A7E;padding:6px 0;font-weight:600;">Academic Year</td><td style="color:#C0D0E0;font-style:italic;">*(Enter academic year)*</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

    pitch_divider()
    section_title("🔬 Five Statistical Experiments")

    exp_rows = [
        ("01", "Central Tendency",    "Mean · Median · Mode · IQR · Variance · Skewness",         "Describes typical scoring behaviour across all matches."),
        ("02", "Regression Analysis", "OLS Linear Regression · R² · p-value · Residuals",         "Investigates year-on-year trend in average total goals."),
        ("03", "Bootstrap Resampling","10,000-iteration Percentile Bootstrap · 95% CI",            "Non-parametric confidence interval for home advantage."),
        ("04", "Normality Testing",   "Shapiro-Wilk W · p-value · Q-Q Plot · Histogram",          "Tests if goal differences follow a normal distribution."),
        ("05", "Hypothesis Testing",  "Paired t-test · Cohen's dz · Wilcoxon Signed-Rank",        "Formal statistical test: H₀: μD=0 vs H₁: μD>0 at α=0.05."),
    ]
    for num, title, methods, purpose in exp_rows:
        st.markdown(f"""
        <div class="exp-card" style="margin:8px 0;">
            <div class="exp-number">{num}</div>
            <h4>Experiment {num}: {title}</h4>
            <p><strong style="color:#A0B8C8;">Methods:</strong> {methods}</p>
            <p style="margin-top:4px;"><strong style="color:#A0B8C8;">Purpose:</strong> {purpose}</p>
        </div>""", unsafe_allow_html=True)

    pitch_divider()
    section_title("📁 Dataset & Attribution")
    st.markdown("""
    <div class="stat-banner">
        <span class="sb-icon">📊</span>
        <div class="sb-text">
            <strong>Source:</strong> Mart Jürisoo's International Football Results<br>
            <strong>Repository:</strong> <a href="https://github.com/martj42/international_results" target="_blank" style="color:#00E676;">github.com/martj42/international_results</a><br>
            <strong>Study Period:</strong> January 1, 2000 – December 31, 2025<br>
            <strong>Attribution:</strong> Please credit the dataset maintainer in any derivative work.
        </div>
    </div>""", unsafe_allow_html=True)

    section_title("⚠️ Limitations")
    limitations = [
        "Matches are not independent — national teams appear repeatedly across many fixtures.",
        "The dataset may have historical gaps or inconsistencies in older records.",
        "Home advantage is measured by goal difference only; tactical, psychological, and travel factors are not modelled.",
        "Regression with year does not establish causation — it is a time proxy only.",
        "The Shapiro-Wilk test is highly sensitive to sample size; large samples almost always reject normality.",
        "\"Home\" in international football differs from club football — tournament formats vary widely.",
    ]
    for lim in limitations:
        st.markdown(f'<div style="display:flex;gap:10px;padding:7px 0;border-bottom:1px solid #0E2030;"><span style="color:#FF5252;min-width:16px;">●</span><span style="color:#8AAABB;font-size:0.88rem;">{lim}</span></div>', unsafe_allow_html=True)

    pitch_divider()
    st.markdown('<div style="text-align:center;color:#3A5A6E;font-size:0.78rem;padding:10px;">All statistical outputs are derived from the actual downloaded dataset. No values are fabricated or hardcoded.</div>', unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if "df" not in st.session_state:
        try:
            with st.spinner("⚽ Loading football dataset…"):
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
    audit   = st.session_state["audit"]
    meta    = st.session_state["meta"]

    filters = render_sidebar(df_full)

    if filters["uploaded"] is not None:
        try:
            df_raw, meta_up = load_from_upload(filters["uploaded"])
            df_clean_up, audit_up = clean_dataset(df_raw)
            st.session_state["df_full"] = df_clean_up
            st.session_state["df"] = df_clean_up
            st.session_state["audit"] = audit_up
            st.session_state["meta"] = meta_up
            df_full = df_clean_up
            audit   = audit_up
            meta    = meta_up
        except Exception as exc:
            st.sidebar.error(f"Upload error: {exc}")

    df = filter_data(
        df_full,
        start_year=filters["start_year"],
        end_year=filters["end_year"],
        tournaments=filters["tournaments"],
        include_neutral=filters["include_neutral"],
        teams=filters["teams"],
    )

    page = filters["page"]
    render_hero(df, page)

    if len(df) == 0:
        st.warning("⚠️ No matches match the current filter settings. Please adjust the sidebar filters.")
        return

    st.sidebar.markdown(f"""
    <div style="background:rgba(0,200,83,0.08);border:1px solid rgba(0,200,83,0.2);
                border-radius:8px;padding:10px;margin-top:12px;text-align:center;">
        <div style="font-family:'Oswald',sans-serif;font-size:1.4rem;color:#00E676;">{len(df):,}</div>
        <div style="font-size:0.7rem;color:#3A5A6E;text-transform:uppercase;letter-spacing:1px;">Matches in View</div>
    </div>""", unsafe_allow_html=True)

    if page == "🏟️ Home":
        page_overview(df)
    elif page == "📊 Central Tendency":
        page_central_tendency(df)
    elif page == "📈 Regression":
        page_regression(df)
    elif page == "🎲 Bootstrap":
        page_bootstrap(df)
    elif page == "🔔 Normality Testing":
        page_normality(df)
    elif page == "🔬 Hypothesis Testing":
        page_hypothesis(df)
    elif page == "🗂️ Match Explorer":
        page_data_explorer(df_full, df, audit, meta)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
