# ⚽ International Football Analytics
### A Computational Statistics Study of Goals, Home Advantage, and Performance Trends

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green)](#)

---

## 📋 Project Overview

This mini-project applies five computational statistics experiments to the
publicly available **International Football Results** dataset to investigate:

- **Home advantage** — do home teams score more goals than away teams?
- **Scoring trends** — how has average goals per match changed over time?
- **Statistical uncertainty** — bootstrap confidence intervals for goal differences.
- **Distribution shape** — are goal differences normally distributed?
- **Formal hypothesis test** — is the mean home advantage statistically significant?

## 🔬 Five Statistical Experiments

| # | Experiment | Key Output |
|---|-----------|------------|
| 1 | **Central Tendency** | Mean, Median, Mode, IQR, Variance for goals |
| 2 | **Linear Regression** | OLS fit: yearly mean goals ∼ year |
| 3 | **Bootstrap Resampling** | 95% CI for mean home-away goal difference |
| 4 | **Normality Testing** | Shapiro-Wilk test on goal differences |
| 5 | **Hypothesis Testing** | Paired t-test: H₀ μD = 0 vs H₁ μD > 0 |

---

## 📁 Project Structure

```
football-computational-statistics/
│
├── app.py                          # Streamlit dashboard (8 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore
│
├── data/
│   └── README.md                   # Dataset documentation
│
├── notebooks/
│   └── football_computational_statistics.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset download & validation
│   ├── preprocessing.py            # Cleaning & derived columns
│   ├── statistics.py               # Experiment 1: Central Tendency
│   ├── regression.py               # Experiment 2: Regression
│   ├── bootstrap.py                # Experiment 3: Bootstrap
│   ├── normality.py                # Experiment 4: Normality Testing
│   ├── hypothesis_testing.py       # Experiment 5: Hypothesis Testing
│   └── visualizations.py           # All chart functions (Plotly)
│
├── reports/
│   ├── final_report.md             # Academic report
│   └── figures/                    # Saved chart images
│
└── tests/
    ├── test_preprocessing.py
    ├── test_statistics.py
    ├── test_regression.py
    └── test_hypothesis_testing.py
```

---

## 📦 Prerequisites

- Python 3.11+ (3.10 also works)
- pip
- Internet connection (for first-time dataset download)

---

## 🚀 Installation

### 1. Create a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** in your browser.

On first launch, the app downloads `results.csv` from GitHub and caches it in `data/`.
If the download fails, upload your own copy via the sidebar uploader.

---

## 📓 Running the Jupyter Notebook

```bash
jupyter notebook notebooks/football_computational_statistics.ipynb
```

Or with JupyterLab:
```bash
jupyter lab
```

Run all cells from top to bottom. The notebook uses the same `src/` modules as the
dashboard, so results will be consistent.

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

To run a specific test file:
```bash
pytest tests/test_statistics.py -v
```

All tests use small, explicitly labeled **synthetic fixtures** and do not
depend on network access or the real dataset.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, yearly trend, result distribution |
| 📊 Central Tendency | Mean, median, mode, spread, score histograms |
| 📈 Regression | Scatter + OLS line, residuals, interpretation |
| 🔄 Bootstrap | 10,000-iteration bootstrap CI for home advantage |
| 🔔 Normality Testing | Shapiro-Wilk, Q-Q plot, histogram |
| 🔬 Hypothesis Testing | Paired t-test, effect size, Wilcoxon |
| 🗂️ Data Explorer | Filterable table, team search, CSV export |
| ℹ️ About | Project details, dataset attribution |

### Sidebar Filters

- **Date range** (year) — configurable within the dataset's range
- **Tournament** — multi-select
- **Neutral venue** — include or exclude
- **Team** — any match involving selected teams
- **Upload CSV** — supply your own results.csv

---

## 📁 Dataset

**Source:** Mart Jürisoo's International Football Results  
**Repository:** https://github.com/martj42/international_results  
**Study Period:** 2000-01-01 to 2025-12-31

The dataset is downloaded automatically at first run. It is not committed to
this repository. Please see `data/README.md` for column documentation and
attribution guidance.

---

## 📐 Statistical Assumptions

1. **Experiment 3 (Bootstrap):** Ordinary match-level bootstrap assumes approximate exchangeability. The year-block sensitivity analysis partially relaxes this.
2. **Experiment 5 (t-test):** The paired t-test assumes approximate normality of the mean of differences (CLT holds well at n > 1000).
3. **Neutral venues excluded** from home-advantage experiments (3 and 5).
4. **Experiment 2 (Regression):** Calendar year is a time proxy; the slope does not establish causation.
5. **Experiment 4 (Normality):** Large samples almost always fail the Shapiro-Wilk test. Statistical significance ≠ practical non-normality.

---

## ⚠️ Data Limitations

- Matches are not independent (teams appear repeatedly).
- The dataset may have historical gaps or inconsistencies.
- Some tournaments may be over-represented.
- "Home" in international football depends on tournament and venue; "home" advantage is different from club football.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Dataset download fails | Check internet connection; manually place `results.csv` in `data/` |
| `ModuleNotFoundError` | Ensure the virtual environment is activated and `pip install -r requirements.txt` was run |
| Streamlit shows blank page | Run from the project root directory: `streamlit run app.py` |
| Tests fail with import errors | Add project root to PYTHONPATH: `$env:PYTHONPATH="."` (Windows) or `export PYTHONPATH=.` (Linux/Mac) |
| Notebook kernel not found | Run `python -m ipykernel install --user` |

---

## 📚 References

1. Jürisoo, M. (2025). *International Football Results Dataset*. GitHub. https://github.com/martj42/international_results
2. Efron, B., & Hastie, T. (2016). *Computer Age Statistical Inference*. Cambridge University Press.
3. SciPy Documentation: https://docs.scipy.org/doc/scipy/
4. Streamlit Documentation: https://docs.streamlit.io/

---

*All statistical outputs are derived from the actual downloaded dataset.
No values are fabricated or hardcoded.*
