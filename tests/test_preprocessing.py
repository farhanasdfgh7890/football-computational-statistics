"""
tests/test_preprocessing.py
----------------------------
Tests for data loading, cleaning, and derived column creation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean_dataset, filter_data, get_non_neutral


# ── Fixtures ────────────────────────────────────────────────────────────────────

def make_raw_df():
    """Synthetic fixture that mimics the International Football Results dataset."""
    return pd.DataFrame({
        "date": [
            "2000-01-15", "2005-06-10", "2010-07-11", "2015-09-05",
            "2020-11-17", "2020-11-17",  # duplicate
            "1999-03-01",                 # outside range
            "bad-date",                   # unparseable date
            "2003-04-12",                 # missing score
            "2007-08-22",                 # negative score
        ],
        "home_team": ["Brazil", "Germany", "Spain", "France", "England",
                      "England", "Italy", "Argentina", "Japan", "Portugal"],
        "away_team": ["Argentina", "Brazil", "Germany", "Portugal", "France",
                      "France", "Spain", "Brazil", "Korea", "Spain"],
        "home_score": [3, 1, 1, 2, 2, 2, 0, 1, None, -1],
        "away_score": [0, 0, 0, 1, 1, 1, 0, 2, 1, 1],
        "tournament": ["Friendly"] * 10,
        "city": ["Rio"] * 10,
        "country": ["Brazil"] * 10,
        "neutral": ["False", "False", "False", "False", "False",
                    "False", "True", "False", "False", "False"],
    })


@pytest.fixture
def raw_df():
    return make_raw_df()


@pytest.fixture
def clean_df(raw_df):
    df, audit = clean_dataset(raw_df, start_date="2000-01-01", end_date="2025-12-31")
    return df, audit


# ── Tests ────────────────────────────────────────────────────────────────────────

class TestCleaning:

    def test_returns_dataframe(self, clean_df):
        df, audit = clean_df
        assert isinstance(df, pd.DataFrame)

    def test_audit_has_steps(self, clean_df):
        df, audit = clean_df
        assert "steps" in audit
        assert len(audit["steps"]) > 0

    def test_drops_bad_dates(self, clean_df):
        df, _ = clean_df
        # "bad-date" row should be gone
        assert df["date"].isna().sum() == 0

    def test_drops_missing_scores(self, clean_df):
        df, _ = clean_df
        assert df["home_score"].isna().sum() == 0
        assert df["away_score"].isna().sum() == 0

    def test_drops_negative_scores(self, clean_df):
        df, _ = clean_df
        assert (df["home_score"] < 0).sum() == 0
        assert (df["away_score"] < 0).sum() == 0

    def test_drops_duplicates(self, clean_df):
        df, audit = clean_df
        dup_step = next((s for s in audit["steps"] if s["step"] == "duplicates"), None)
        assert dup_step is not None
        assert "1" in dup_step["note"] or "Removed 1" in dup_step["note"]

    def test_date_range_filter(self, clean_df):
        df, _ = clean_df
        assert (df["date"] >= "2000-01-01").all()
        assert (df["date"] <= "2025-12-31").all()

    def test_out_of_range_date_excluded(self, clean_df):
        df, _ = clean_df
        # 1999-03-01 should be excluded
        dates = pd.to_datetime(df["date"])
        assert (dates.dt.year < 2000).sum() == 0


class TestDerivedColumns:

    def test_total_goals(self, clean_df):
        df, _ = clean_df
        assert "total_goals" in df.columns
        expected = df["home_score"] + df["away_score"]
        pd.testing.assert_series_equal(
            df["total_goals"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_goal_difference(self, clean_df):
        df, _ = clean_df
        assert "goal_difference" in df.columns
        expected = df["home_score"] - df["away_score"]
        pd.testing.assert_series_equal(
            df["goal_difference"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_home_win(self, clean_df):
        df, _ = clean_df
        for _, row in df.iterrows():
            if row["home_score"] > row["away_score"]:
                assert row["home_win"] == 1
            else:
                assert row["home_win"] == 0

    def test_draw(self, clean_df):
        df, _ = clean_df
        for _, row in df.iterrows():
            if row["home_score"] == row["away_score"]:
                assert row["draw"] == 1
            else:
                assert row["draw"] == 0

    def test_year_column(self, clean_df):
        df, _ = clean_df
        assert "year" in df.columns
        assert df["year"].dtype in [int, np.int32, np.int64]

    def test_is_neutral_column(self, clean_df):
        df, _ = clean_df
        assert "is_neutral" in df.columns
        assert df["is_neutral"].dtype == bool


class TestFiltering:

    def test_year_filter(self, clean_df):
        df, _ = clean_df
        filtered = filter_data(df, start_year=2005, end_year=2010)
        assert (filtered["year"] >= 2005).all()
        assert (filtered["year"] <= 2010).all()

    def test_exclude_neutral(self, clean_df):
        df, _ = clean_df
        filtered = filter_data(df, include_neutral=False)
        assert filtered["is_neutral"].sum() == 0

    def test_get_non_neutral(self, clean_df):
        df, _ = clean_df
        nn = get_non_neutral(df)
        assert nn["is_neutral"].sum() == 0
