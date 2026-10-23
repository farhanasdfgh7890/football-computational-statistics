"""
tests/test_regression.py
-------------------------
Tests for Experiment 2: Regression Analysis.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regression import fit_yearly_regression, predict_year, interpret_regression
from src.statistics import yearly_goals_summary


# ── Fixtures ────────────────────────────────────────────────────────────────────

def make_yearly_df(slope: float = 0.0, intercept: float = 3.0):
    """Synthetic yearly summary with a known true slope."""
    years = list(range(2000, 2021))
    means = [intercept + slope * y for y in years]
    counts = [100] * len(years)
    return pd.DataFrame({
        "year": years,
        "mean_total_goals": means,
        "n_matches": counts,
    })


@pytest.fixture
def flat_yearly():
    """Zero-slope data — perfectly horizontal."""
    return make_yearly_df(slope=0.0, intercept=2.8)


@pytest.fixture
def positive_slope_yearly():
    """Positive-slope data."""
    return make_yearly_df(slope=0.02, intercept=-37.0)


# ── Tests ────────────────────────────────────────────────────────────────────────

class TestRegressionOnKnownData:

    def test_zero_slope_detected(self, flat_yearly):
        result = fit_yearly_regression(flat_yearly)
        assert abs(result["slope"]) < 1e-6

    def test_positive_slope_detected(self, positive_slope_yearly):
        result = fit_yearly_regression(positive_slope_yearly)
        assert result["slope"] > 0

    def test_r_squared_perfect(self, positive_slope_yearly):
        result = fit_yearly_regression(positive_slope_yearly)
        # Perfect linear data → R² should be 1.0
        assert abs(result["r_squared"] - 1.0) < 1e-6

    def test_intercept_approximate(self, positive_slope_yearly):
        result = fit_yearly_regression(positive_slope_yearly)
        # y = 0.02x - 37, so b0 = -37
        assert abs(result["intercept"] - (-37.0)) < 0.1

    def test_prediction(self, positive_slope_yearly):
        result = fit_yearly_regression(positive_slope_yearly)
        pred_2010 = predict_year(result, 2010)
        expected = 0.02 * 2010 + (-37.0)
        assert abs(pred_2010 - expected) < 0.1

    def test_residuals_near_zero(self, flat_yearly):
        result = fit_yearly_regression(flat_yearly)
        residuals = np.array(result["residuals"])
        assert np.allclose(residuals, 0, atol=1e-6)

    def test_rmse_near_zero(self, flat_yearly):
        result = fit_yearly_regression(flat_yearly)
        assert result["rmse"] < 1e-6

    def test_sample_size(self, flat_yearly):
        result = fit_yearly_regression(flat_yearly)
        assert result["n"] == 21

    def test_x_pred_length(self, flat_yearly):
        result = fit_yearly_regression(flat_yearly)
        assert len(result["x_pred"]) == 200

    def test_empty_returns_none_slope(self):
        empty = pd.DataFrame(columns=["year", "mean_total_goals", "n_matches"])
        result = fit_yearly_regression(empty)
        assert result["slope"] is None

    def test_too_few_years(self):
        tiny = make_yearly_df().head(2)
        result = fit_yearly_regression(tiny)
        assert result["slope"] is None


class TestInterpretation:

    def test_interpret_not_empty(self, positive_slope_yearly):
        result = fit_yearly_regression(positive_slope_yearly)
        text = interpret_regression(result)
        assert len(text) > 50

    def test_interpret_mentions_slope(self, positive_slope_yearly):
        result = fit_yearly_regression(positive_slope_yearly)
        text = interpret_regression(result)
        assert "slope" in text.lower() or "year" in text.lower()
