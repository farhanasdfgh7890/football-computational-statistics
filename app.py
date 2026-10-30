"""
app.py
------
International Football Analytics — FC Barcelona Edition
Blaugrana theme: deep navy, garnet, gold

Run with:
    streamlit run app.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FCB Analytics | Més que un Club",
    page_icon="🔵🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

BARCA_LOGO = "https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/200px-FC_Barcelona_%28crest%29.svg.png"

# ── BARCELONA CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
* { box-sizing: border-box; }
html, body, .stApp {
    background-color: #06080F !important;
    color: #E8EDF5 !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1400px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060810 0%, #0A0C1A 40%, #100614 100%) !important;
    border-right: 1px solid #1A0A20 !important;
}
section[data-testid="stSidebar"] * { color: #C8D0E0 !important; }
section[data-testid="stSidebar"] label { color: #6070A0 !important; font-size: 0.75rem !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #0C0E1C !important; border-color: #1E1A40 !important; border-radius: 6px !important;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    padding: 0 0 18px; text-align: center;
    border-bottom: 1px solid rgba(165,0,68,0.25);
    margin-bottom: 16px;
}
.sidebar-brand .sb-logo { margin-bottom: 8px; }
.sidebar-brand h3 {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1rem !important; letter-spacing: 2px;
    background: linear-gradient(90deg, #A50044, #EDBB00, #004D98);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 2px !important;
}
.sidebar-brand p { color: #3A4060 !important; font-size: 0.68rem !important; letter-spacing: 1px; margin: 0 !important; text-transform: uppercase; }
.sidebar-nav-label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px;
    color: #2A3060; font-weight: 700; margin: 14px 0 5px; padding: 0 2px;
}

/* ── Stripe Top Bar ── */
.barca-stripe {
    height: 6px;
    background: repeating-linear-gradient(
        90deg,
        #A50044 0px, #A50044 40px,
        #004D98 40px, #004D98 80px,
        #A50044 80px, #A50044 120px,
        #004D98 120px, #004D98 160px
    );
}

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, #06080F 0%, #0A0C1E 40%, #120614 80%, #06080F 100%);
    border-bottom: 2px solid #A50044;
    padding: 28px 36px 22px;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,77,152,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute; bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(165,0,68,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(165,0,68,0.2), rgba(0,77,152,0.2));
    border: 1px solid rgba(165,0,68,0.4);
    color: #EDBB00 !important;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; padding: 3px 12px; border-radius: 20px;
    margin-bottom: 10px; display: inline-block;
}
.hero-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 2.8rem !important; font-weight: 700 !important;
    letter-spacing: 2px !important;
    background: linear-gradient(90deg, #EDBB00, #FFFFFF, #EDBB00);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 6px !important; line-height: 1.1 !important;
}
.hero-subtitle { color: #60708A !important; font-size: 0.88rem !important; font-weight: 300; letter-spacing: 0.5px; margin: 0 !important; }
.hero-moto {
    font-family: 'Oswald', sans-serif;
    font-size: 0.8rem; letter-spacing: 3px; text-transform: uppercase;
    color: #A50044 !important; margin-top: 14px;
}

/* ── Blaugrana Ticker ── */
.ticker-wrap {
    background: linear-gradient(90deg, #A50044, #8B003A, #003D7A, #004D98, #A50044);
    background-size: 300% 100%; animation: tickerShift 8s linear infinite;
    padding: 7px 24px; font-family: 'Oswald', sans-serif;
    font-size: 0.88rem; font-weight: 500; letter-spacing: 0.8px;
    color: #FFFFFF !important; display: flex; gap: 36px; align-items: center;
}
@keyframes tickerShift { 0%{background-position:0% 50%} 100%{background-position:100% 50%} }
.ticker-item { white-space: nowrap; }
.ticker-sep { color: rgba(237,187,0,0.6); font-size: 1rem; }

/* ── Section Title ── */
.section-title {
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.6rem !important; font-weight: 600 !important;
    color: #FFFFFF !important;
    border-left: 4px solid #A50044;
    padding-left: 14px; margin: 22px 0 3px !important; line-height: 1.2 !important;
}
.section-title .gold { color: #EDBB00 !important; }
.section-sub { color: #445070 !important; font-size: 0.85rem !important; padding-left: 18px; margin-bottom: 18px !important; }

/* ── KPI / Score Cards ── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 10px; margin: 14px 0;
}
.score-card {
    background: linear-gradient(145deg, #0A0C1C, #0E1028);
    border: 1px solid #181A38;
    border-radius: 12px; padding: 16px 14px 12px;
    position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.score-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(165,0,68,0.2);
    border-color: #A50044;
}
.score-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #A50044, #EDBB00, #004D98);
}
.score-card .sc-icon { font-size: 1.3rem; margin-bottom: 8px; display: block; }
.score-card .sc-value {
    font-family: 'Oswald', sans-serif;
    font-size: 1.9rem; font-weight: 700; color: #EDBB00; line-height: 1; margin-bottom: 4px;
}
.score-card .sc-label { font-size: 0.68rem; color: #3A4070; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.score-card .sc-delta { font-size: 0.72rem; margin-top: 4px; font-weight: 600; }
.delta-up { color: #00D46A; }
.delta-down { color: #FF5252; }
.delta-neutral { color: #7080A0; }

/* ── Pitch Divider ── */
.pitch-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #1A0820, #A50044, #004D98, #A50044, #1A0820, transparent);
    margin: 22px 0;
}

/* ── Stat Banner ── */
.stat-banner {
    background: linear-gradient(90deg, rgba(165,0,68,0.08), rgba(0,77,152,0.06));
    border: 1px solid rgba(165,0,68,0.2);
    border-radius: 10px; padding: 13px 18px; margin: 10px 0;
    display: flex; align-items: center; gap: 12px;
}
.stat-banner .sb-icon { font-size: 1.6rem; }
.stat-banner .sb-text { font-size: 0.88rem; color: #8090B8; line-height: 1.4; }
.stat-banner .sb-text strong { color: #EDBB00; }

/* ── Decision Boxes ── */
.decision-box-yes {
    background: linear-gradient(135deg, rgba(0,77,152,0.15), rgba(165,0,68,0.08));
    border: 1px solid #004D98; border-radius: 10px; padding: 16px 20px; margin: 12px 0;
}
.decision-box-no {
    background: rgba(165,0,68,0.08); border: 1px solid #A50044;
    border-radius: 10px; padding: 16px 20px; margin: 12px 0;
}
.decision-box-warn {
    background: rgba(237,187,0,0.07); border: 1px solid rgba(237,187,0,0.4);
    border-radius: 10px; padding: 16px 20px; margin: 12px 0;
}
.decision-label { font-family: 'Oswald', sans-serif; font-size: 1rem; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
.decision-text { font-size: 0.86rem; color: #8090B8; line-height: 1.5; }

/* ── Fun Fact ── */
.funfact {
    background: linear-gradient(135deg, #0A0C1E, #06080F);
    border-left: 3px solid #EDBB00; border-radius: 0 10px 10px 0;
    padding: 12px 18px; margin: 12px 0;
}
.funfact .ff-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; color: #EDBB00; font-weight: 700; margin-bottom: 4px; }
.funfact .ff-text { font-size: 0.88rem; color: #8090B8; line-height: 1.4; }

/* ── Match Cards ── */
.match-card {
    background: #0A0C1C; border: 1px solid #151830;
    border-radius: 10px; padding: 12px 18px; margin: 6px 0;
    display: flex; align-items: center; justify-content: space-between;
    transition: background 0.2s;
}
.match-card:hover { background: #0E1028; border-color: #A50044; }
.match-team { font-weight: 600; font-size: 0.9rem; color: #C0D0E8; }
.match-score {
    font-family: 'Oswald', sans-serif; font-size: 1.5rem; font-weight: 700;
    color: #FFFFFF;
    background: linear-gradient(135deg, #100618, #0A0418);
    border: 1px solid #3A1040; border-radius: 8px;
    padding: 3px 18px; min-width: 85px; text-align: center;
}
.match-meta { font-size: 0.72rem; color: #303858; margin-top: 2px; }

/* ── Experiment Cards ── */
.exp-card {
    background: linear-gradient(145deg, #0A0C1C, #0C0E20);
    border: 1px solid #181A38; border-radius: 12px;
    padding: 18px; margin: 8px 0; position: relative; overflow: hidden;
}
.exp-card:hover { border-color: #A50044; }
.exp-card .exp-number {
    font-family: 'Oswald', sans-serif; font-size: 3.5rem; font-weight: 700;
    color: rgba(165,0,68,0.07); position: absolute; top: 8px; right: 16px; line-height: 1;
}
.exp-card h4 { font-family: 'Oswald', sans-serif; font-size: 1.05rem; color: #EDBB00 !important; margin: 0 0 5px !important; font-weight: 600; }
.exp-card p { color: #506080; font-size: 0.83rem; margin: 0; line-height: 1.4; }

/* ── Leaderboard ── */
.lb-row {
    display: flex; align-items: center;
    background: #0A0C1C; border: 1px solid #131530;
    border-radius: 8px; padding: 10px 16px; margin: 5px 0; gap: 12px;
    transition: background 0.2s;
}
.lb-row:hover { background: #0E1028; }
.lb-rank { font-family: 'Oswald', sans-serif; font-size: 1.05rem; color: #2A3058; min-width: 30px; }
.lb-rank.gold { color: #EDBB00; }
.lb-rank.silver { color: #C0C8D8; }
.lb-rank.bronze { color: #CD7F32; }
.lb-name { flex: 1; font-weight: 600; color: #A0B0C8; font-size: 0.88rem; }
.lb-val { font-family: 'Oswald', sans-serif; font-size: 1.05rem; color: #EDBB00; font-weight: 700; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0A0C1C !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] { color: #304060 !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.84rem !important; }
.stTabs [aria-selected="true"] { color: #EDBB00 !important; background: #0E1030 !important; border-bottom: 2px solid #A50044 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #A50044, #7A0030) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-family: 'Oswald', sans-serif !important;
    letter-spacing: 1px !important; padding: 8px 22px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #CC0055, #A50044) !important;
    box-shadow: 0 4px 18px rgba(165,0,68,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0A0C1C, #0E1028) !important;
    border-radius: 10px !important; padding: 14px !important;
    border: 1px solid #181A38 !important;
}
[data-testid="stMetricValue"] { color: #EDBB00 !important; font-family: 'Oswald', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #3A4070 !important; font-size: 0.72rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: #0A0C1C !important; border: 1px solid #181A38 !important; border-radius: 8px !important; color: #506080 !important; font-weight: 600 !important; }
.streamlit-expanderContent { background: #080A18 !important; border: 1px solid #181A38 !important; border-top: none !important; }

/* ── DataFrame ── */
.stDataFrame { background: #0A0C1C !important; border-radius: 10px !important; }

/* ── Misc ── */
[data-baseweb="select"] > div { background: #0A0C1C !important; border-color: #181A38 !important; }
[data-baseweb="input"] > div { background: #0A0C1C !important; border-color: #181A38 !important; }
.stSuccess { background: rgba(0,77,152,0.12) !important; border-color: #004D98 !important; }
.stWarning { background: rgba(237,187,0,0.08) !important; border-color: rgba(237,187,0,0.5) !important; }
.stError { background: rgba(165,0,68,0.1) !important; border-color: #A50044 !important; }
.stInfo { background: rgba(0,77,152,0.1) !important; border-color: #004D98 !important; }

/* ── Crest glow ── */
.crest-glow {
    filter: drop-shadow(0 0 12px rgba(237,187,0,0.4)) drop-shadow(0 0 24px rgba(165,0,68,0.2));
    transition: filter 0.3s;
}
.crest-glow:hover { filter: drop-shadow(0 0 20px rgba(237,187,0,0.7)) drop-shadow(0 0 40px rgba(165,0,68,0.4)); }

/* ── Camp Nou strip ── */
.campnou-strip {
    background: linear-gradient(90deg,
        #A50044 0%,  #A50044 14.28%,
        #004D98 14.28%, #004D98 28.57%,
        #A50044 28.57%, #A50044 42.85%,
        #004D98 42.85%, #004D98 57.14%,
        #A50044 57.14%, #A50044 71.42%,
        #004D98 71.42%, #004D98 85.71%,
        #A50044 85.71%, #A50044 100%
    );
    height: 4px; margin-bottom: 0;
}

/* ── Sidebar match count ── */
.sidebar-count {
    background: linear-gradient(135deg, rgba(165,0,68,0.12), rgba(0,77,152,0.12));
    border: 1px solid rgba(165,0,68,0.25);
    border-radius: 8px; padding: 10px; margin-top: 12px; text-align: center;
}
.sidebar-count-val { font-family: 'Oswald', sans-serif; font-size: 1.5rem; color: #EDBB00; }
.sidebar-count-lbl { font-size: 0.65rem; color: #2A3058; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────────
from src.data_loader import load_dataset, load_from_upload, describe_dataset
from src.preprocessing import clean_dataset, filter_data, get_non_neutral, audit_to_dataframe
from src import statistics as stats_module
from src import regression as reg_module
from src import bootstrap as boot_module
from src import normality as norm_module
from src import hypothesis_testing as hyp_module
from src import visualizations as viz


# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔵🔴 Loading data from Camp Nou servers…")
def _load_and_clean(start: str = "2000-01-01", end: str = "2025-12-31"):
    df_raw, meta = load_dataset()
    df_clean, audit = clean_dataset(df_raw, start_date=start, end_date=end)
    return df_clean, audit, meta


# ── UI helpers ─────────────────────────────────────────────────────────────────
def sc(icon, value, label, delta=None, delta_type="neutral"):
    delta_html = ""
    if delta is not None:
        cls = f"delta-{delta_type}"
        arrow = "▲" if delta_type == "up" else ("▼" if delta_type == "down" else "●")
        delta_html = f'<div class="sc-delta {cls}">{arrow} {delta}</div>'
    return f"""<div class="score-card">
        <span class="sc-icon">{icon}</span>
        <div class="sc-value">{value}</div>
        <div class="sc-label">{label}</div>
        {delta_html}
    </div>"""

def pitch_divider():
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)

def section_title(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def stat_banner(icon, text):
    return f'<div class="stat-banner"><span class="sb-icon">{icon}</span><div class="sb-text">{text}</div></div>'

def funfact(text):
    st.markdown(f'<div class="funfact"><div class="ff-label">⚡ Barca Insight</div><div class="ff-text">{text}</div></div>', unsafe_allow_html=True)

def decision_box(text, kind="yes"):
    icon = "🔵" if kind == "yes" else ("🔴" if kind == "no" else "🟡")
    label = "Conclusion" if kind != "warn" else "Note"
    st.markdown(f'<div class="decision-box-{kind}"><div class="decision-label">{icon} {label}</div><div class="decision-text">{text}</div></div>', unsafe_allow_html=True)


# ── Top stripe ─────────────────────────────────────────────────────────────────
st.markdown('<div class="campnou-strip"></div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(df_full: pd.DataFrame) -> dict:
    st.sidebar.markdown(f"""
    <div class="sidebar-brand">
        <div class="sb-logo">
            <img src="{BARCA_LOGO}" width="64" class="crest-glow" alt="FC Barcelona">
        </div>
        <h3>FCB ANALYTICS</h3>
        <p>Més que un Club · Est. 1899</p>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-nav-label">📋 Navigate</div>', unsafe_allow_html=True)
    page_options = [
        "🏟️ Camp Nou Overview",
        "📊 Central Tendency",
        "📈 Regression Analysis",
        "🎲 Bootstrap",
        "🔔 Normality Testing",
        "🔬 Hypothesis Testing",
        "🗂️ Match Explorer",
        "🔵🔴 About",
    ]
    page = st.sidebar.selectbox("Navigate", page_options, label_visibility="collapsed")

    st.sidebar.markdown('<div class="sidebar-nav-label">⚙️ Filters</div>', unsafe_allow_html=True)
    year_min = int(df_full["year"].min()) if len(df_full) > 0 else 2000
    year_max = int(df_full["year"].max()) if len(df_full) > 0 else 2025
    col_s, col_e = st.sidebar.columns(2)
    start_year = col_s.number_input("From", min_value=year_min, max_value=year_max, value=year_min, step=1)
    end_year   = col_e.number_input("To",   min_value=year_min, max_value=year_max, value=year_max, step=1)
    include_neutral = st.sidebar.checkbox("Include neutral venues", value=True)

    all_tournaments = ["All"] + sorted(df_full["tournament"].dropna().unique().tolist())
    sel_t = st.sidebar.multiselect("Tournament(s)", all_tournaments, default=["All"])
    tournaments = None if "All" in sel_t else sel_t

    all_teams = sorted(pd.concat([df_full["home_team"], df_full["away_team"]]).dropna().unique().tolist())
    sel_teams = st.sidebar.multiselect("Team(s)", all_teams)
    teams = sel_teams if sel_teams else None

    st.sidebar.markdown('<div class="sidebar-nav-label">📁 Data</div>', unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if st.sidebar.button("🔄 Reset"):
        st.rerun()

    return {"page": page, "start_year": int(start_year), "end_year": int(end_year),
            "include_neutral": include_neutral, "tournaments": tournaments,
            "teams": teams, "uploaded": uploaded}


# ── Hero ────────────────────────────────────────────────────────────────────────
def render_hero(df: pd.DataFrame = None, page: str = ""):
    subtitles = {
        "🏟️ Camp Nou Overview":  "Season overview — goals, wins, and the home advantage story",
        "📊 Central Tendency":   "Experiment 1 — Mean, Median, Mode & the spread of football scores",
        "📈 Regression Analysis":"Experiment 2 — Tracking goal-scoring trends across the years",
        "🎲 Bootstrap":          "Experiment 3 — 10,000 resamples to quantify home advantage",
        "🔔 Normality Testing":  "Experiment 4 — Is goal difference normally distributed?",
        "🔬 Hypothesis Testing": "Experiment 5 — Statistical proof of home advantage at α = 0.05",
        "🗂️ Match Explorer":     "Browse every match in the dataset — search, filter, export",
        "🔵🔴 About":            "Project details, methodology, and dataset attribution",
    }
    subtitle = subtitles.get(page, "A Computational Statistics Study of International Football")
    count_html = ""
    if df is not None:
        count_html = f"""
        <div style="text-align:right;padding-top:4px;">
            <div style="font-family:'Oswald',sans-serif;font-size:1.8rem;color:#EDBB00;font-weight:700;line-height:1;">{len(df):,}</div>
            <div style="font-size:0.65rem;color:#2A3050;letter-spacing:2px;text-transform:uppercase;">Matches in View</div>
        </div>"""

    st.markdown(f"""
    <div class="hero-banner">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
            <div style="display:flex;align-items:center;gap:20px;">
                <img src="{BARCA_LOGO}" width="72" class="crest-glow" alt="FC Barcelona Crest">
                <div>
                    <span class="hero-badge">🔵🔴 FC BARCELONA · ANALYTICS HUB</span>
                    <div class="hero-title">MÉS QUE UN CLUB</div>
                    <p class="hero-subtitle">{subtitle}</p>
                    <p class="hero-moto">⚽ Computational Statistics · International Football · Est. 1899</p>
                </div>
            </div>
            {count_html}
        </div>
    </div>""", unsafe_allow_html=True)


# ── Pages ───────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    nn  = get_non_neutral(df)
    kpi = stats_module.kpi_summary(df, nn)
    hw  = kpi['home_win_pct']
    aw  = kpi['away_win_pct']
    dr  = kpi['pct_draw']
    avg = kpi['avg_total_goals']
    diff= kpi['mean_goal_diff_nn']

    # Ticker
    st.markdown(f"""
    <div class="ticker-wrap">
        <span class="ticker-item">🏆 Home Win: {hw:.1f}%</span>
        <span class="ticker-sep">✦</span>
        <span class="ticker-item">✈️ Away Win: {aw:.1f}%</span>
        <span class="ticker-sep">✦</span>
        <span class="ticker-item">🤝 Draws: {dr:.1f}%</span>
        <span class="ticker-sep">✦</span>
        <span class="ticker-item">⚽ Avg Goals: {avg:.2f}</span>
        <span class="ticker-sep">✦</span>
        <span class="ticker-item">📊 Home Advantage: {diff:+.3f} goals</span>
        <span class="ticker-sep">✦</span>
        <span class="ticker-item">🔵🔴 Visca el Barça!</span>
    </div>""", unsafe_allow_html=True)

    pitch_divider()
    section_title("📊 Match Statistics", f"Based on {len(df):,} international matches · {kpi.get('year_range','')}")

    delta_sign = "up" if diff > 0 else ("down" if diff < 0 else "neutral")
    cards = '<div class="score-grid">'
    cards += sc("🏟️", f"{kpi['total_matches']:,}", "Total Matches")
    cards += sc("⚽", f"{avg:.2f}",              "Avg Goals/Match")
    cards += sc("🏠", f"{kpi['avg_home_goals']:.2f}", "Avg Home Goals")
    cards += sc("✈️", f"{kpi['avg_away_goals']:.2f}", "Avg Away Goals")
    cards += sc("🤝", f"{dr:.1f}%",             "Draw Rate")
    cards += sc("📈", f"{diff:+.3f}",           "Home Advantage",
                delta=f"{abs(diff):.3f} goals", delta_type=delta_sign)
    cards += sc("🏆", f"{hw:.1f}%",             "Home Win %")
    cards += sc("✈️", f"{aw:.1f}%",             "Away Win %")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    pitch_divider()

    if diff > 0:
        funfact(f"Playing at home gives teams a statistical edge of <strong>{diff:+.3f} goals per match</strong> — "
                f"just like how Camp Nou turns into a fortress for Barça, home advantage is real across {len(nn):,} international non-neutral matches.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(viz.plot_yearly_trend(df), use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_result_distribution(df), use_container_width=True)
    st.plotly_chart(viz.plot_total_goals_histogram(df), use_container_width=True)

    pitch_divider()
    section_title("🔬 Five Statistical Experiments", "Navigate using the sidebar to explore each in detail")

    exps = [
        ("01", "Central Tendency",    "Mean, Median, Mode, IQR — the typical international football scoreline."),
        ("02", "Regression Analysis", "OLS linear regression — is average scoring rising or falling over time?"),
        ("03", "Bootstrap Resampling","10,000 resamples build a distribution of home advantage estimates."),
        ("04", "Normality Testing",   "Shapiro-Wilk + Q-Q plot — does goal difference follow a bell curve?"),
        ("05", "Hypothesis Testing",  "Formal paired t-test: H₀: μD = 0 — is home advantage statistically real?"),
    ]
    c1, c2 = st.columns(2)
    for i, (num, title, desc) in enumerate(exps):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"""
            <div class="exp-card">
                <div class="exp-number">{num}</div>
                <h4>⚽ {title}</h4><p>{desc}</p>
            </div>""", unsafe_allow_html=True)


def page_central_tendency(df: pd.DataFrame):
    section_title("📊 Central Tendency",
                  "Descriptive statistics — total goals, home score, away score, goal difference")

    tbl        = stats_module.central_tendency_table(df)
    stat_goals = stats_module.central_tendency(df["total_goals"])
    if stat_goals["n"] == 0:
        st.warning("No data. Adjust filters.")
        return

    mean_v, median_v = stat_goals["mean"], stat_goals["median"]
    mode_v, std_v    = stat_goals["mode"], stat_goals["std"]
    iqr_v,  skew_v   = stat_goals["iqr"],  stat_goals["skewness"]

    cards = '<div class="score-grid">'
    cards += sc("📐", f"{mean_v:.3f}",  "Mean Goals/Match")
    cards += sc("📍", f"{median_v:.1f}","Median Goals")
    cards += sc("🎯", str(mode_v[0] if isinstance(mode_v, list) else mode_v), "Modal Goals")
    cards += sc("📊", f"{std_v:.3f}",   "Std Deviation")
    cards += sc("📦", f"{iqr_v:.3f}",   "IQR")
    cards += sc("〰️", f"{skew_v:.3f}", "Skewness")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    pitch_divider()
    section_title("📋 Full Summary Table")
    display_cols = ["n","mean","median","std","variance","min","max","range","q1","q3","iqr","skewness","kurtosis"]
    display_tbl  = tbl[[c for c in display_cols if c in tbl.columns]].round(4)
    st.dataframe(display_tbl, use_container_width=True)

    section_title("🎯 Mode Values", "Most frequently occurring score totals")
    mode_info = []
    for col_name, series in [("Total Goals", df["total_goals"]), ("Home Score", df["home_score"]),
                              ("Away Score",  df["away_score"]),  ("Goal Difference", df["goal_difference"])]:
        s = stats_module.central_tendency(series)
        mode_info.append({"Variable": col_name, "Mode(s)": ", ".join(str(m) for m in s["mode"]), "Frequency": s["mode_count"]})
    st.dataframe(pd.DataFrame(mode_info), use_container_width=True)
    st.download_button("⬇ Download CSV", display_tbl.reset_index().to_csv(index=False).encode(), "central_tendency.csv", "text/csv")

    pitch_divider()
    section_title("📈 Visual Analysis")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(viz.plot_total_goals_histogram(df), use_container_width=True)
    with c2: st.plotly_chart(viz.plot_total_goals_boxplot(df),   use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(viz.plot_home_away_comparison(df),  use_container_width=True)
    with c4: st.plotly_chart(viz.plot_violin_scores(df),         use_container_width=True)

    with st.expander("📖 Statistical Interpretation"):
        skew_label = "positively skewed" if skew_v > 0 else ("approximately symmetric" if abs(skew_v) < 0.3 else "negatively skewed")
        direction  = "exceeds" if mean_v > median_v else "is below"
        pull       = "upward"  if mean_v > median_v else "downward"
        st.markdown(f"""
The **mean** total goals per match is **{mean_v:.3f}**. The **median** is **{median_v:.1f}** — less sensitive to high-scoring outliers.
The most common total is **{mode_v}** goals per match.

The distribution is **{skew_label}** (skewness = {skew_v:.3f}), so the mean **{direction}** the median —
high-scoring matches pull it **{pull}**.

Standard deviation ({std_v:.3f}) and IQR ({iqr_v:.3f}) both quantify spread across matches.
        """)


def page_regression(df: pd.DataFrame):
    section_title("📈 Regression Analysis",
                  "OLS linear regression: yearly mean goals ~ calendar year")

    yearly = stats_module.yearly_goals_summary(df)
    if len(yearly) < 3:
        st.warning("Not enough yearly data points. Adjust the date range.")
        return

    reg   = reg_module.fit_yearly_regression(yearly, weighted=False)
    slope = reg.get("slope")

    cards = '<div class="score-grid">'
    cards += sc("📐", f"{slope:+.4f}" if slope else "N/A", "Slope (goals/year)",
                delta=("📈 Rising" if slope and slope > 0 else "📉 Falling"),
                delta_type=("up" if slope and slope > 0 else "down"))
    cards += sc("📍", f"{reg['intercept']:.3f}"  if reg['intercept'] else "N/A", "Intercept")
    cards += sc("📊", f"{reg['r_squared']:.4f}"  if reg['r_squared'] else "N/A", "R²")
    cards += sc("📏", f"{reg['rmse']:.4f}"        if reg['rmse']      else "N/A", "RMSE")
    cards += sc("🔬", f"{reg['p_value']:.4f}"    if reg['p_value']   else "N/A", "p-value")
    cards += sc("📅", str(reg['n']), "Years in Model")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    pitch_divider()
    if slope is not None:
        if slope > 0.005:
            st.markdown(stat_banner("📈", f"Slope = <strong>+{slope:.4f} goals/year</strong> — scoring is gradually <strong>increasing</strong> across the study period."), unsafe_allow_html=True)
        elif slope < -0.005:
            st.markdown(stat_banner("📉", f"Slope = <strong>{slope:.4f} goals/year</strong> — scoring has been <strong>declining</strong> over time."), unsafe_allow_html=True)
        else:
            st.markdown(stat_banner("➡️", f"Slope ≈ <strong>{slope:.4f} goals/year</strong> — scoring is essentially <strong>flat</strong> over the study period."), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(viz.plot_regression(yearly, reg),           use_container_width=True)
    with c2: st.plotly_chart(viz.plot_regression_residuals(yearly, reg), use_container_width=True)

    with st.expander("📋 Yearly Data Table"):
        st.dataframe(yearly.round(4), use_container_width=True)
        st.download_button("⬇ Download", yearly.to_csv(index=False).encode(), "yearly_goals.csv", "text/csv")
    with st.expander("📖 Interpretation"):
        st.markdown(reg_module.interpret_regression(reg))
    with st.expander("🔬 Weighted Regression (Sensitivity)"):
        st.markdown("Years with more matches get higher weight in the OLS fit.")
        reg_w = reg_module.fit_yearly_regression(yearly, weighted=True)
        if reg_w.get("w_slope") is not None:
            wc1, wc2 = st.columns(2)
            wc1.metric("Weighted Slope", f"{reg_w['w_slope']:+.4f}")
            wc2.metric("Weighted Intercept", f"{reg_w['w_intercept']:.3f}")


def page_bootstrap(df: pd.DataFrame):
    section_title("🎲 Bootstrap Resampling",
                  "Non-parametric 95% CI for the mean home-away goal difference")

    st.markdown(stat_banner("🎲",
        "We draw <strong>10,000 random bootstrap samples</strong> from non-neutral matches, "
        "computing the mean home-away goal difference each time. "
        "The 2.5th and 97.5th percentiles define the <strong>95% confidence interval</strong> — "
        "no normality assumption needed."), unsafe_allow_html=True)

    pitch_divider()
    c1, c2 = st.columns(2)
    n_iter = c2.number_input("Iterations", min_value=1000, max_value=100_000, value=10_000, step=1000)
    seed   = c1.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    if st.button("▶ Run Bootstrap Analysis") or "boot_results" not in st.session_state:
        with st.spinner(f"Running {n_iter:,} bootstrap iterations… 🔵🔴"):
            boot = boot_module.bootstrap_mean_goal_diff(df, n_iterations=int(n_iter), random_seed=int(seed))
        st.session_state["boot_results"] = boot

    boot = st.session_state.get("boot_results", {})
    if boot.get("error"):
        st.error(boot["error"]); return

    cards = '<div class="score-grid">'
    cards += sc("🏟️", f"{boot['n_matches']:,}",       "Eligible Matches")
    cards += sc("📊", f"{boot['observed_mean']:+.4f}", "Observed Mean Diff")
    cards += sc("📏", f"{boot['bootstrap_se']:.4f}",  "Bootstrap SE")
    cards += sc("🔁", f"{boot['n_iterations']:,}",    "Iterations")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    ci_pct = int(boot["confidence_level"] * 100)
    lo, hi = boot["ci_lower"], boot["ci_upper"]
    if lo > 0:
        decision_box(
            f"The <strong>{ci_pct}% bootstrap CI</strong> is [{lo:+.4f}, {hi:+.4f}]. "
            f"The entire interval lies above zero — strong non-parametric evidence of home advantage.",
            kind="yes")
    else:
        decision_box(
            f"The <strong>{ci_pct}% bootstrap CI</strong> is [{lo:+.4f}, {hi:+.4f}] — includes zero, inconclusive.",
            kind="no")

    st.plotly_chart(viz.plot_bootstrap_distribution(boot), use_container_width=True)
    with st.expander("📖 Interpretation"):
        st.markdown(boot_module.interpret_bootstrap(boot))
    with st.expander("🔬 Year-Block Bootstrap"):
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


def page_normality(df: pd.DataFrame):
    section_title("🔔 Normality Testing",
                  "Is the goal-difference distribution compatible with a normal distribution?")

    st.markdown(stat_banner("🔔",
        "The <strong>Shapiro-Wilk test</strong> tests H₀: the data come from a normal distribution. "
        "We also inspect the <strong>Q-Q plot</strong> and histogram for visual confirmation."), unsafe_allow_html=True)

    pitch_divider()
    seed = st.number_input("Subsampling seed (if n > 5000)", value=42, step=1)
    with st.spinner("Running normality test… 🔵🔴"):
        norm = norm_module.shapiro_wilk_test(df, random_seed=int(seed))
    if norm.get("error"):
        st.error(norm["error"]); return

    cards = '<div class="score-grid">'
    cards += sc("📊", f"{norm['full_n']:,}",              "Full Sample n")
    cards += sc("🔬", f"{norm['test_n']:,}",              "Test Sample n")
    cards += sc("📐", f"{norm['statistic']:.6f}" if norm['statistic'] else "N/A", "Shapiro-Wilk W")
    cards += sc("🎯", f"{norm['p_value']:.6f}"   if norm['p_value']   else "N/A", "p-value")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    if norm.get("subsampled"):
        st.markdown(stat_banner("ℹ️",
            f"Full sample ({norm['full_n']:,}) exceeds the Shapiro-Wilk limit. "
            f"Test run on a random subsample of <strong>{norm['test_n']:,}</strong> (seed={norm['random_seed']})."),
            unsafe_allow_html=True)

    if norm.get("reject_h0"):
        decision_box(f"<strong>{norm['decision']}</strong><br>"
                     "p < 0.05 — reject H₀ of normality. "
                     "Note: large samples almost always reject normality for minor deviations. "
                     "Inspect the Q-Q plot for practical assessment.", kind="no")
    else:
        decision_box(f"<strong>{norm['decision']}</strong>", kind="yes")

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(viz.plot_normality_histogram(norm), use_container_width=True)
    with c2: st.plotly_chart(viz.plot_qq(norm),                  use_container_width=True)

    with st.expander("📖 Interpretation"):
        st.markdown(norm_module.interpret_normality(norm))
    funfact("Large samples (n > 5,000) almost always fail Shapiro-Wilk — the test detects even trivially small deviations. Statistical significance ≠ practical non-normality.")


def page_hypothesis(df: pd.DataFrame):
    section_title("🔬 Hypothesis Testing",
                  "Formal test: does home advantage exist in international football?")

    st.markdown(stat_banner("🔬",
        "H₀: μD = 0 (no home advantage) vs H₁: μD > 0 (home teams score more). "
        "Paired t-test on <strong>non-neutral matches only</strong>, α = 0.05. "
        "Cohen's dz quantifies practical significance."), unsafe_allow_html=True)

    pitch_divider()
    with st.spinner("Running hypothesis tests… 🔵🔴"):
        ttest = hyp_module.paired_ttest(df)
    if ttest.get("error") and ttest["n"] == 0:
        st.error("No non-neutral matches. Adjust filters."); return

    cards = '<div class="score-grid">'
    cards += sc("🏟️", f"{ttest['n']:,}",            "Eligible Matches")
    cards += sc("🏠", f"{ttest['mean_home']:.4f}",   "Mean Home Goals")
    cards += sc("✈️", f"{ttest['mean_away']:.4f}",   "Mean Away Goals")
    cards += sc("📊", f"{ttest['mean_diff']:+.4f}",  "Mean Diff D",
                delta=("Home lead" if ttest['mean_diff'] > 0 else "Away lead"),
                delta_type=("up" if ttest['mean_diff'] > 0 else "down"))
    cards += sc("📐", f"{ttest['t_stat']:.4f}",      "t-statistic")
    cards += sc("🎯", f"{ttest['p_one_sided']:.6f}", "p-value (one-sided)")
    cards += sc("📏", f"{ttest['cohens_dz']:.4f}" if ttest['cohens_dz'] else "N/A", "Cohen's dz")
    cards += sc("🏆", ttest['effect_label'] or "N/A", "Effect Size")
    cards += '</div>'
    st.markdown(cards, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#0A0C1C;border:1px solid #181A38;border-radius:8px;
                padding:12px 18px;margin:12px 0;font-size:0.88rem;color:#506080;">
        📐 <strong style="color:#C0D0E8;">95% CI for mean D:</strong>
        <span style="font-family:'Oswald',sans-serif;color:#EDBB00;font-size:1.1rem;margin-left:10px;">
            [{ttest['ci_lower']:+.4f}, {ttest['ci_upper']:+.4f}]
        </span>
    </div>""", unsafe_allow_html=True)

    if ttest["reject_h0"]:
        decision_box(
            f"<strong>{ttest['decision']}</strong><br>"
            f"t({ttest['df']}) = {ttest['t_stat']:.4f}, p = {ttest['p_one_sided']:.6f} &lt; 0.05. "
            f"Cohen's dz = {ttest['cohens_dz']:.4f} ({ttest['effect_label']}). "
            f"Home teams score significantly more — just like Barça at Camp Nou! 🔵🔴", kind="yes")
    else:
        decision_box(f"<strong>{ttest['decision']}</strong> — fail to reject H₀ at α = 0.05.", kind="no")

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(viz.plot_home_away_bar(ttest),       use_container_width=True)
    with c2: st.plotly_chart(viz.plot_goal_diff_histogram(ttest), use_container_width=True)

    with st.expander("📖 Full Interpretation"):
        st.markdown(hyp_module.interpret_ttest(ttest))
    with st.expander("🔬 Wilcoxon Signed-Rank Test (Non-Parametric)"):
        st.markdown("Non-parametric alternative — no normality assumption required.")
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
                wc1.metric("Wilcoxon W",         f"{wilcox['statistic']:.2f}")
                wc2.metric("p-value (one-sided)", f"{wilcox['p_value']:.6f}")
                wc3.metric("Reject H₀?", "✅ Yes" if wilcox["reject_h0"] else "❌ No")
                st.caption(wilcox["note"])

    funfact("The paired t-test is robust for large n thanks to the Central Limit Theorem — "
            "even if individual goal differences aren't perfectly normal, the test remains valid for n > 1,000.")


def page_data_explorer(df_full: pd.DataFrame, df: pd.DataFrame, audit: dict, meta: dict):
    section_title("🗂️ Match Explorer", f"Browsing {len(df):,} filtered matches · {len(df_full):,} total cleaned")

    tab1, tab2, tab3 = st.tabs(["📋 Match Table", "🏆 Leaderboards", "🔍 Audit & Source"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        search  = c1.text_input("🔍 Search by team name", "")
        df_view = df.copy()
        if search:
            df_view = df_view[
                df_view["home_team"].str.contains(search, case=False, na=False) |
                df_view["away_team"].str.contains(search, case=False, na=False)
            ]
        st.caption(f"Showing **{len(df_view):,}** matches")
        display_cols = [c for c in ["date","home_team","home_score","away_score","away_team",
            "tournament","city","country","is_neutral","total_goals","goal_difference",
            "home_win","draw","away_win"] if c in df_view.columns]
        st.dataframe(df_view[display_cols].sort_values("date", ascending=False).reset_index(drop=True),
                     use_container_width=True, height=420)
        c2.download_button("⬇ Export", df_view[display_cols].to_csv(index=False).encode(),
                           "matches.csv", "text/csv", use_container_width=True)

    with tab2:
        section_title("🏆 Highest-Scoring Matches", "Top 15 by total goals")
        top = df.nlargest(15, "total_goals")[
            ["date","home_team","home_score","away_score","away_team","total_goals","tournament"]
        ].reset_index(drop=True)
        for i, row in top.iterrows():
            medal = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"
            rank_cls = "gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""
            st.markdown(f"""
            <div class="match-card">
                <span class="lb-rank {rank_cls}" style="min-width:38px;font-size:1.1rem">{medal}</span>
                <div style="flex:1">
                    <div class="match-team">{row['home_team']} vs {row['away_team']}</div>
                    <div class="match-meta">📅 {str(row['date'])[:10]} &nbsp;·&nbsp; 🏆 {row['tournament']}</div>
                </div>
                <div class="match-score">{int(row['home_score'])} – {int(row['away_score'])}</div>
                <div style="min-width:55px;text-align:right">
                    <span class="lb-val">⚽ {int(row['total_goals'])}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        pitch_divider()
        section_title("📊 Most Active Teams", "Top 10 by total match appearances")
        top_teams = pd.concat([df["home_team"], df["away_team"]]).value_counts().head(10).reset_index()
        top_teams.columns = ["Team", "Matches"]
        for i, row in top_teams.iterrows():
            medal    = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"
            rank_cls = "gold" if i==0 else "silver" if i==1 else "bronze" if i==2 else ""
            st.markdown(f"""
            <div class="lb-row">
                <span class="lb-rank {rank_cls}">{medal}</span>
                <span class="lb-name">{row['Team']}</span>
                <span class="lb-val">{row['Matches']:,} matches</span>
            </div>""", unsafe_allow_html=True)

    with tab3:
        section_title("🔍 Data Cleaning Audit")
        st.dataframe(audit_to_dataframe(audit), use_container_width=True)
        with st.expander("📄 Source Info"):
            st.json(meta)


def page_about():
    # Big crest hero
    st.markdown(f"""
    <div style="text-align:center;padding:30px 0 10px;">
        <img src="{BARCA_LOGO}" width="110" class="crest-glow" alt="FC Barcelona Crest">
        <div style="font-family:'Oswald',sans-serif;font-size:2.2rem;font-weight:700;
                    background:linear-gradient(90deg,#A50044,#EDBB00,#004D98,#EDBB00,#A50044);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;margin:10px 0 4px;letter-spacing:3px;">
            FC BARCELONA ANALYTICS
        </div>
        <div style="color:#2A3060;font-size:0.8rem;letter-spacing:4px;text-transform:uppercase;">
            Més que un Club · Est. 1899 · Camp Nou, Barcelona
        </div>
    </div>""", unsafe_allow_html=True)

    pitch_divider()

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0A0C1C,#060810);
                border:1px solid #1A1A38;border-radius:14px;padding:24px 28px;margin:14px 0;">
        <div style="display:flex;align-items:center;gap:20px;margin-bottom:18px;">
            <img src="{BARCA_LOGO}" width="50" class="crest-glow">
            <div>
                <div style="font-family:'Oswald',sans-serif;font-size:1.2rem;color:#EDBB00;font-weight:600;">
                    Statistical Analysis of International Football Performance
                </div>
                <div style="color:#304060;font-size:0.8rem;">Computational Statistics Mini-Project · B.Tech CSE</div>
            </div>
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.87rem;">
            <tr><td style="color:#2A3060;padding:6px 0;width:150px;font-weight:600;">Subject</td><td style="color:#C0D0E0;">Computational Statistics</td></tr>
            <tr><td style="color:#2A3060;padding:6px 0;font-weight:600;">Student Name</td><td style="color:#808090;font-style:italic;">*(Enter your name)*</td></tr>
            <tr><td style="color:#2A3060;padding:6px 0;font-weight:600;">USN</td><td style="color:#808090;font-style:italic;">*(Enter your USN)*</td></tr>
            <tr><td style="color:#2A3060;padding:6px 0;font-weight:600;">Department</td><td style="color:#C0D0E0;">Computer Science &amp; Engineering</td></tr>
            <tr><td style="color:#2A3060;padding:6px 0;font-weight:600;">Institution</td><td style="color:#808090;font-style:italic;">*(Enter institution)*</td></tr>
            <tr><td style="color:#2A3060;padding:6px 0;font-weight:600;">Academic Year</td><td style="color:#808090;font-style:italic;">*(Enter academic year)*</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)

    pitch_divider()
    section_title("🔬 Five Experiments")
    exps = [
        ("01","Central Tendency",   "Mean · Median · Mode · IQR · Variance · Skewness",          "Describes typical scoring behaviour across all matches."),
        ("02","Regression Analysis","OLS Linear Regression · R² · p-value · Residuals",           "Year-on-year trend in average total goals per match."),
        ("03","Bootstrap Resampling","10,000-iteration Percentile Bootstrap · 95% CI",             "Non-parametric CI for mean home-away goal difference."),
        ("04","Normality Testing",  "Shapiro-Wilk W · p-value · Q-Q Plot · Histogram",            "Tests if goal differences follow a normal distribution."),
        ("05","Hypothesis Testing", "Paired t-test · Cohen's dz · Wilcoxon Signed-Rank",          "H₀: μD=0 vs H₁: μD>0 at α=0.05 on non-neutral matches."),
    ]
    for num, title, methods, purpose in exps:
        st.markdown(f"""
        <div class="exp-card" style="margin:7px 0;">
            <div class="exp-number">{num}</div>
            <h4>⚽ Experiment {num}: {title}</h4>
            <p><strong style="color:#8090B8;">Methods:</strong> {methods}</p>
            <p style="margin-top:3px;"><strong style="color:#8090B8;">Purpose:</strong> {purpose}</p>
        </div>""", unsafe_allow_html=True)

    pitch_divider()
    section_title("📁 Dataset & Attribution")
    st.markdown("""
    <div class="stat-banner">
        <span class="sb-icon">📊</span>
        <div class="sb-text">
            <strong>Source:</strong> Mart Jürisoo's International Football Results<br>
            <strong>Repository:</strong> <a href="https://github.com/martj42/international_results" target="_blank" style="color:#EDBB00;">github.com/martj42/international_results</a><br>
            <strong>Study Period:</strong> January 1, 2000 – December 31, 2025<br>
            <strong>Note:</strong> This dataset covers international / national team matches, not club football.
        </div>
    </div>""", unsafe_allow_html=True)

    section_title("⚠️ Limitations")
    for lim in [
        "Matches are not independent — national teams appear repeatedly.",
        "The dataset may have historical gaps in older records.",
        "Home advantage is measured by goal difference only — tactics and travel are not modelled.",
        "Regression on year does not establish causation.",
        "The Shapiro-Wilk test is oversensitive for large samples.",
        "International home advantage differs from club-level home advantage (e.g. Camp Nou).",
    ]:
        st.markdown(f'<div style="display:flex;gap:10px;padding:7px 0;border-bottom:1px solid #0E1028;"><span style="color:#A50044;min-width:16px;">◆</span><span style="color:#607090;font-size:0.86rem;">{lim}</span></div>', unsafe_allow_html=True)

    pitch_divider()
    st.markdown(f"""
    <div style="text-align:center;padding:14px 0;">
        <img src="{BARCA_LOGO}" width="36" style="opacity:0.4;">
        <div style="color:#1A2040;font-size:0.72rem;margin-top:6px;letter-spacing:1px;">
            ALL STATISTICAL OUTPUTS ARE DERIVED FROM THE ACTUAL DATASET · NO VALUES ARE FABRICATED
        </div>
        <div style="color:#A50044;font-size:0.75rem;font-family:'Oswald',sans-serif;letter-spacing:3px;margin-top:4px;">
            VISCA EL BARÇA · VISCA CATALUNYA
        </div>
    </div>""", unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    if "df" not in st.session_state:
        try:
            with st.spinner("🔵🔴 Loading data from Camp Nou servers…"):
                df_clean, audit, meta = _load_and_clean()
            st.session_state.update({"df_full": df_clean, "df": df_clean, "audit": audit, "meta": meta})
        except Exception as exc:
            st.error(f"**Dataset load failed:** {exc}")
            st.info("Please upload a results.csv file using the sidebar uploader.")
            up = st.file_uploader("Upload results.csv", type=["csv"])
            if up:
                try:
                    df_raw, meta = load_from_upload(up)
                    df_clean, audit = clean_dataset(df_raw)
                    st.session_state.update({"df_full": df_clean, "df": df_clean, "audit": audit, "meta": meta})
                    st.rerun()
                except Exception as e2:
                    st.error(f"Upload failed: {e2}")
            return

    df_full = st.session_state["df_full"]
    audit   = st.session_state["audit"]
    meta    = st.session_state["meta"]
    filters = render_sidebar(df_full)

    if filters["uploaded"] is not None:
        try:
            df_raw, meta_up = load_from_upload(filters["uploaded"])
            df_clean_up, audit_up = clean_dataset(df_raw)
            st.session_state.update({"df_full": df_clean_up, "df": df_clean_up, "audit": audit_up, "meta": meta_up})
            df_full, audit, meta = df_clean_up, audit_up, meta_up
        except Exception as exc:
            st.sidebar.error(f"Upload error: {exc}")

    df = filter_data(df_full, start_year=filters["start_year"], end_year=filters["end_year"],
                     tournaments=filters["tournaments"], include_neutral=filters["include_neutral"],
                     teams=filters["teams"])

    page = filters["page"]
    render_hero(df, page)

    if len(df) == 0:
        st.warning("⚠️ No matches match the current filters. Adjust the sidebar settings.")
        return

    st.sidebar.markdown(f"""
    <div class="sidebar-count">
        <div class="sidebar-count-val">{len(df):,}</div>
        <div class="sidebar-count-lbl">Matches in View</div>
    </div>""", unsafe_allow_html=True)

    pages = {
        "🏟️ Camp Nou Overview":  lambda: page_overview(df),
        "📊 Central Tendency":   lambda: page_central_tendency(df),
        "📈 Regression Analysis":lambda: page_regression(df),
        "🎲 Bootstrap":          lambda: page_bootstrap(df),
        "🔔 Normality Testing":  lambda: page_normality(df),
        "🔬 Hypothesis Testing": lambda: page_hypothesis(df),
        "🗂️ Match Explorer":     lambda: page_data_explorer(df_full, df, audit, meta),
        "🔵🔴 About":            page_about,
    }
    pages.get(page, lambda: st.info("Page not found."))()


if __name__ == "__main__":
    main()
