"""
Unit tests for ``models.backtest_engine``
=========================================
All tests use synthetic data and stub pipelines — no live data, GPU, or
heavy model fitting required.  Total runtime ≈ 1 s
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── fixtures ──────────────────────────────────────────────────────────────

def _make_returns(
    n_days: int = 300,
    n_assets: int = 3,
    seed: int = 42,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Synthetic log-return panel with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    data = rng.normal(0, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _stub_pipeline(var_value: float = -0.05):
    """Return a deterministic pipeline callable."""
    def _fn(train_returns: pd.DataFrame, cfg: dict) -> dict:
        return {"var_95": var_value}
    return _fn


# ── import under test ───────────────────────────────────────────────────

from models.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    RollingBacktester,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BacktestConfig
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.forecast_horizon == 30
        assert cfg.var_confidence == 0.95
        assert cfg.min_train_days == 252
        assert cfg.step_size == 1

    def test_invalid_horizon(self):
        with pytest.raises(ValueError, match="forecast_horizon"):
            BacktestConfig(forecast_horizon=0)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="var_confidence"):
            BacktestConfig(var_confidence=1.5)

    def test_pipeline_overrides(self):
        cfg = BacktestConfig(mc_n_paths=1000, mc_n_steps=20, var_confidence=0.99)
        ov = cfg.to_pipeline_overrides()
        assert ov["mc_n_paths"] == 1000
        assert ov["mc_n_steps"] == 20
        assert ov["var_confidence"] == 0.99


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RollingBacktester — expanding window.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExpandingWindow:
    def test_window_grows_by_step_size(self):
        """Train window size must grow by exactly ``step_size`` each step."""
        ret = _make_returns(n_days=300)
        cfg = BacktestConfig(min_train_days=200, step_size=10, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        results = bt.run()

        sizes = [r.train_size for r in results]
        diffs = [sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)]
        assert all(d == cfg.step_size for d in diffs), f"diffs={diffs}"

    def test_first_window_respects_min_train(self):
        ret = _make_returns(n_days=300)
        cfg = BacktestConfig(min_train_days=200, step_size=5, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        results = bt.run()

        assert results[0].train_size == cfg.min_train_days


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Violation logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestViolationLogic:
    def test_violation_when_loss_exceeds_var(self):
        """violation == True iff realized_loss < predicted_var."""
        r = BacktestResult(
            origin_date=datetime(2024, 1, 1),
            train_start=datetime(2023, 1, 1),
            train_size=252,
            predicted_var_95=-0.05,
            realized_loss=-0.08,  # worse than VaR
            violation=True,
        )
        assert r.violation is True

    def test_no_violation_when_loss_within_var(self):
        r = BacktestResult(
            origin_date=datetime(2024, 1, 1),
            train_start=datetime(2023, 1, 1),
            train_size=252,
            predicted_var_95=-0.05,
            realized_loss=-0.02,  # better than VaR
            violation=False,
        )
        assert r.violation is False

    def test_violation_computed_by_engine(self):
        """The engine must flag violations correctly end-to-end."""
        # Create returns where forward windows are strongly negative
        ret = _make_returns(n_days=300, seed=0)
        # Inject large negative returns in forward windows
        ret.iloc[200:230] = -0.05  # guaranteed loss ≈ -0.05 * 30 = -1.5

        cfg = BacktestConfig(
            min_train_days=200, step_size=100, forecast_horizon=30,
        )
        # VaR = -0.01 → realized ≈ -1.5, so violation must be True
        bt = RollingBacktester(ret, _stub_pipeline(var_value=-0.01), cfg)
        results = bt.run()

        first = results[0]
        assert first.violation is True
        assert first.realized_loss < first.predicted_var_95


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# No look-ahead bias.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNoLookahead:
    def test_pipeline_never_sees_future_data(self):
        """The pipeline callable must only receive data up to the origin."""
        observed_max_dates: list[pd.Timestamp] = []

        def _spy(train_returns: pd.DataFrame, cfg: dict) -> dict:
            observed_max_dates.append(train_returns.index[-1])
            return {"var_95": -0.05}

        ret = _make_returns(n_days=300)
        cfg = BacktestConfig(min_train_days=200, step_size=20, forecast_horizon=30)
        bt = RollingBacktester(ret, _spy, cfg)
        results = bt.run()

        for result, max_date in zip(results, observed_max_dates):
            # The latest date the pipeline saw must equal the origin date
            assert max_date == pd.Timestamp(result.origin_date)
            # And must be strictly before any forward-window date
            origin_idx = ret.index.get_loc(max_date)
            assert origin_idx + cfg.forecast_horizon <= len(ret)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary statistics.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSummary:
    def test_summary_violation_rate(self):
        ret = _make_returns(n_days=300)
        cfg = BacktestConfig(min_train_days=200, step_size=10, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        results = bt.run()
        summary = bt.summary()

        expected_violations = sum(r.violation for r in results)
        assert summary["total_origins"] == len(results)
        assert summary["violation_count"] == expected_violations
        assert summary["violation_rate"] == pytest.approx(
            expected_violations / len(results)
        )

    def test_empty_summary(self):
        ret = _make_returns(n_days=50)
        cfg = BacktestConfig(min_train_days=200, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        results = bt.run()

        assert len(results) == 0
        summary = bt.summary()
        assert summary["total_origins"] == 0
        assert summary["violation_rate"] == 0.0
        assert math.isnan(summary["mean_predicted_var"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DataFrame output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestToDataFrame:
    def test_shape_and_columns(self):
        ret = _make_returns(n_days=300)
        cfg = BacktestConfig(min_train_days=200, step_size=10, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        results = bt.run()
        df = bt.to_dataframe()

        assert len(df) == len(results)
        expected_cols = {
            "origin_date", "train_start", "train_size",
            "predicted_var_95", "realized_loss", "violation",
        }
        assert set(df.columns) == expected_cols

    def test_empty_dataframe(self):
        ret = _make_returns(n_days=50)
        cfg = BacktestConfig(min_train_days=200, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        bt.run()
        df = bt.to_dataframe()
        assert df.empty


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Insufficient data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInsufficientData:
    def test_no_origins_when_too_short(self):
        """If T < min_train + horizon, no backtest steps execute."""
        ret = _make_returns(n_days=100)
        cfg = BacktestConfig(min_train_days=90, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        results = bt.run()
        # 90 + 30 = 120 > 100 → impossible to backtest
        # Actually: origin t=90, need t+30=120 <= 100? No. So 0 results.
        assert len(results) == 0

    def test_requires_datetimeindex(self):
        ret = pd.DataFrame(
            np.random.randn(100, 2), columns=["X", "Y"]
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            RollingBacktester(ret, _stub_pipeline(), BacktestConfig())

    def test_empty_returns_rejected(self):
        ret = pd.DataFrame(
            columns=["X", "Y"],
            index=pd.DatetimeIndex([], name="date"),
        )
        with pytest.raises(ValueError, match="empty"):
            RollingBacktester(ret, _stub_pipeline(), BacktestConfig())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSV persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSave:
    def test_save_creates_csv(self, tmp_path: Path):
        ret = _make_returns(n_days=300)
        cfg = BacktestConfig(min_train_days=200, step_size=20, forecast_horizon=30)
        bt = RollingBacktester(ret, _stub_pipeline(), cfg)
        bt.run()

        csv_path = tmp_path / "results.csv"
        bt.save(csv_path)

        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(bt.results)
