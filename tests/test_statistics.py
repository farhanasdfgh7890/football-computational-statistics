"""
tests/test_statistics.py
-------------------------
Tests for Experiment 1: Measures of Central Tendency.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.statistics import central_tendency, central_tendency_table, kpi_summary


# ── Fixtures ────────────────────────────────────────────────────────────────────

def make_clean_df():
    """Minimal synthetic clean dataset for statistics tests."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2005-01-01", "2010-01-01", "2015-01-01",
                                 "2020-01-01", "2020-06-01"]),
        "year": [2005, 2010, 2015, 2020, 2020],
        "home_score": [3, 1, 2, 0, 4],
        "away_score": [0, 1, 1, 0, 2],
        "total_goals": [3, 2, 3, 0, 6],
        "goal_difference": [3, 0, 1, 0, 2],
        "home_win": [1, 0, 1, 0, 1],
        "away_win": [0, 0, 0, 0, 0],
        "draw": [0, 1, 0, 1, 0],
        "is_neutral": [False, False, True, False, False],
        "tournament": ["Friendly"] * 5,
    })


@pytest.fixture
def df():
    return make_clean_df()


# ── Tests ────────────────────────────────────────────────────────────────────────

class TestCentralTendency:

    def test_known_mean(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = central_tendency(s)
        assert abs(result["mean"] - 3.0) < 1e-9

    def test_known_median_odd(self):
        s = pd.Series([1, 3, 5, 7, 9])
        result = central_tendency(s)
        assert result["median"] == 5.0

    def test_known_median_even(self):
        s = pd.Series([2, 4, 6, 8])
        result = central_tendency(s)
        assert result["median"] == 5.0

    def test_known_mode(self):
        s = pd.Series([1, 2, 2, 3, 3, 3])
        result = central_tendency(s)
        assert 3 in result["mode"]

    def test_known_std(self):
        # Use a series whose sample std (ddof=1) is exactly known via numpy
        s = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = central_tendency(s)
        expected_std = float(np.std(s, ddof=1))  # sample std, ddof=1
        assert abs(result["std"] - expected_std) < 1e-9

    def test_known_variance(self):
        s = pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = central_tendency(s)
        expected_var = float(np.var(s, ddof=1))  # sample variance, ddof=1
        assert abs(result["variance"] - expected_var) < 1e-9

    def test_q1_q3_iqr(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        result = central_tendency(s)
        assert result["q3"] > result["q1"]
        assert abs(result["iqr"] - (result["q3"] - result["q1"])) < 1e-9

    def test_range(self):
        s = pd.Series([10, 20, 30])
        result = central_tendency(s)
        assert result["range"] == 20.0

    def test_sample_size(self, df):
        result = central_tendency(df["total_goals"])
        assert result["n"] == 5

    def test_ignores_nan(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        result = central_tendency(s)
        assert result["n"] == 3
        assert abs(result["mean"] - (1 + 2 + 4) / 3) < 1e-9

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        result = central_tendency(s)
        assert result["n"] == 0
        assert result["mean"] is None


class TestCentralTendencyTable:

    def test_returns_dataframe(self, df):
        tbl = central_tendency_table(df)
        assert isinstance(tbl, pd.DataFrame)

    def test_has_expected_index(self, df):
        tbl = central_tendency_table(df)
        assert "Total Goals" in tbl.index

    def test_mean_column_exists(self, df):
        tbl = central_tendency_table(df)
        assert "mean" in tbl.columns


class TestKPISummary:

    def test_total_matches(self, df):
        kpi = kpi_summary(df)
        assert kpi["total_matches"] == 5

    def test_avg_total_goals_correct(self, df):
        kpi = kpi_summary(df)
        expected = df["total_goals"].mean()
        assert abs(kpi["avg_total_goals"] - expected) < 1e-9

    def test_draw_pct_valid(self, df):
        kpi = kpi_summary(df)
        assert 0 <= kpi["pct_draw"] <= 100

    def test_empty_df_does_not_raise(self):
        df_empty = pd.DataFrame(columns=[
            "total_goals", "home_score", "away_score", "goal_difference",
            "draw", "home_win", "away_win", "is_neutral"
        ])
        kpi = kpi_summary(df_empty)
        assert kpi["total_matches"] == 0
        assert kpi["avg_total_goals"] == 0.0
