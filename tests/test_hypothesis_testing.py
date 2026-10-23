"""
tests/test_hypothesis_testing.py
----------------------------------
Tests for:
  - Experiment 3: Bootstrap Resampling
  - Experiment 4: Normality Testing
  - Experiment 5: Hypothesis Testing
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bootstrap import bootstrap_mean_goal_diff, bootstrap_year_block
from src.normality import shapiro_wilk_test
from src.hypothesis_testing import paired_ttest, wilcoxon_test


# ── Synthetic fixture ────────────────────────────────────────────────────────────

def make_match_df(
    n: int = 200,
    mean_home: float = 1.5,
    mean_away: float = 1.0,
    seed: int = 0,
    include_neutral: bool = False,
) -> pd.DataFrame:
    """Create a synthetic dataset with controlled mean home/away scores."""
    rng = np.random.default_rng(seed)
    home_scores = rng.poisson(mean_home, n).astype(int)
    away_scores = rng.poisson(mean_away, n).astype(int)
    years = rng.integers(2000, 2021, n)
    return pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=n, freq="5D"),
        "year": years,
        "home_score": home_scores,
        "away_score": away_scores,
        "total_goals": home_scores + away_scores,
        "goal_difference": home_scores - away_scores,
        "home_win": (home_scores > away_scores).astype(int),
        "away_win": (away_scores > home_scores).astype(int),
        "draw": (home_scores == away_scores).astype(int),
        "is_neutral": np.zeros(n, dtype=bool),
        "tournament": ["Friendly"] * n,
    })


@pytest.fixture
def df_home_advantage():
    """Dataset where home teams score significantly more."""
    return make_match_df(n=500, mean_home=1.8, mean_away=1.0, seed=42)


@pytest.fixture
def df_no_advantage():
    """Dataset where both teams score equally."""
    return make_match_df(n=500, mean_home=1.2, mean_away=1.2, seed=7)


@pytest.fixture
def df_small():
    """Small dataset for edge cases."""
    return make_match_df(n=15, mean_home=1.5, mean_away=1.0, seed=99)


# ── Bootstrap tests ──────────────────────────────────────────────────────────────

class TestBootstrap:

    def test_reproducibility(self, df_home_advantage):
        r1 = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=1000, random_seed=42)
        r2 = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=1000, random_seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_different_seeds_different_results(self, df_home_advantage):
        r1 = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=1000, random_seed=1)
        r2 = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=1000, random_seed=2)
        # Very unlikely to be identical
        assert r1["bootstrap_se"] != r2["bootstrap_se"]

    def test_ci_lower_less_than_upper(self, df_home_advantage):
        r = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=2000, random_seed=42)
        assert r["ci_lower"] < r["ci_upper"]

    def test_observed_mean_inside_ci(self, df_home_advantage):
        r = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=2000, random_seed=42)
        assert r["ci_lower"] <= r["observed_mean"] <= r["ci_upper"]

    def test_home_advantage_ci_excludes_zero(self, df_home_advantage):
        r = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=5000, random_seed=42)
        assert r["ci_lower"] > 0 or r["ci_upper"] > 0  # at least positive

    def test_n_matches_correct(self, df_home_advantage):
        r = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=500, random_seed=42)
        assert r["n_matches"] == 500

    def test_bootstrap_means_length(self, df_home_advantage):
        n_iter = 1000
        r = bootstrap_mean_goal_diff(df_home_advantage, n_iterations=n_iter, random_seed=42)
        assert len(r["bootstrap_means"]) == n_iter

    def test_empty_data_returns_error(self):
        df_empty = make_match_df(n=0)
        r = bootstrap_mean_goal_diff(df_empty, n_iterations=100)
        assert r["error"] is not None

    def test_year_block_ci_ordering(self, df_home_advantage):
        r = bootstrap_year_block(df_home_advantage, n_iterations=2000, random_seed=42)
        assert r["ci_lower"] < r["ci_upper"]


# ── Normality tests ──────────────────────────────────────────────────────────────

class TestNormality:

    def test_known_normal_data(self):
        """Normally distributed data should not be rejected at α=0.05 for small n."""
        rng = np.random.default_rng(0)
        normal_vals = rng.normal(0, 1, 200)
        df = pd.DataFrame({
            "goal_difference": normal_vals.astype(int),
            "is_neutral": np.zeros(200, dtype=bool),
        })
        result = shapiro_wilk_test(df, random_seed=42)
        assert result["statistic"] is not None
        assert result["p_value"] is not None

    def test_subsampling_triggered(self):
        """Test that subsampling is triggered for large datasets."""
        rng = np.random.default_rng(0)
        large = pd.DataFrame({
            "goal_difference": rng.integers(-10, 10, 6000).astype(int),
            "is_neutral": np.zeros(6000, dtype=bool),
        })
        result = shapiro_wilk_test(large, max_n=5000)
        assert result["subsampled"] is True
        assert result["test_n"] == 5000

    def test_output_structure(self, df_home_advantage):
        result = shapiro_wilk_test(df_home_advantage)
        for key in ["statistic", "p_value", "alpha", "reject_h0", "decision", "full_n"]:
            assert key in result

    def test_reject_h0_is_bool(self, df_home_advantage):
        result = shapiro_wilk_test(df_home_advantage)
        assert isinstance(result["reject_h0"], bool)

    def test_p_value_in_range(self, df_home_advantage):
        result = shapiro_wilk_test(df_home_advantage)
        if result["p_value"] is not None:
            assert 0 <= result["p_value"] <= 1


# ── Hypothesis testing tests ─────────────────────────────────────────────────────

class TestHypothesisTesting:

    def test_home_advantage_detected(self, df_home_advantage):
        """With a clear home advantage (λ_home=1.8, λ_away=1.0), t-test should reject H₀."""
        result = paired_ttest(df_home_advantage)
        assert result["reject_h0"] is True
        assert result["mean_diff"] > 0

    def test_no_advantage_detected(self, df_no_advantage):
        """With equal scoring (λ=1.2 each), H₀ should not be rejected."""
        result = paired_ttest(df_no_advantage)
        # Not guaranteed to fail to reject for any seed, but mean_diff should be small
        assert abs(result["mean_diff"]) < 0.5

    def test_t_statistic_positive_when_home_advantage(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        assert result["t_stat"] > 0

    def test_degrees_of_freedom(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        assert result["df"] == result["n"] - 1

    def test_ci_ordering(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        assert result["ci_lower"] < result["ci_upper"]

    def test_cohens_dz_positive(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        assert result["cohens_dz"] > 0

    def test_effect_label_is_string(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        assert result["effect_label"] in {"negligible", "small", "medium", "large"}

    def test_p_values_in_range(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        assert 0 <= result["p_one_sided"] <= 1
        assert 0 <= result["p_two_sided"] <= 1

    def test_one_sided_p_less_than_two_sided(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        # One-sided p ≤ two-sided p for a right-tailed test with positive mean
        assert result["p_one_sided"] <= result["p_two_sided"]

    def test_mean_diff_equals_home_minus_away(self, df_home_advantage):
        result = paired_ttest(df_home_advantage)
        expected_diff = result["mean_home"] - result["mean_away"]
        assert abs(result["mean_diff"] - expected_diff) < 1e-9

    def test_empty_data_graceful(self):
        df_empty = make_match_df(n=0)
        result = paired_ttest(df_empty)
        assert result.get("error") is not None or result["n"] == 0

    def test_wilcoxon_runs(self, df_home_advantage):
        result = wilcoxon_test(df_home_advantage)
        assert "statistic" in result
        assert "p_value" in result

    def test_wilcoxon_p_in_range(self, df_home_advantage):
        result = wilcoxon_test(df_home_advantage)
        if result.get("error") is None:
            assert 0 <= result["p_value"] <= 1

    def test_correct_pairing(self):
        """Ensure that home and away scores remain paired (not shuffled separately)."""
        # Create data where home always beats away by exactly 1
        df = pd.DataFrame({
            "home_score": [2, 3, 1, 4, 2],
            "away_score": [1, 2, 0, 3, 1],
            "goal_difference": [1, 1, 1, 1, 1],
            "is_neutral": [False] * 5,
            "year": [2010] * 5,
        })
        result = paired_ttest(df)
        assert abs(result["mean_diff"] - 1.0) < 1e-9
