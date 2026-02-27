"""
Rolling Backtest Engine — Expanding-Window Walk-Forward Validation
==================================================================
Orchestrates regime-switching systemic-risk Monte Carlo backtests
*without* implementing any simulation logic internally.

Design principles:
  • Pipeline-agnostic — delegates all model fitting + forecasting to a
    user-supplied ``pipeline_fn`` callable.
  • Expanding window — at each origin date the training set grows by
    ``step_size`` days; no future data is ever exposed.
  • Memory-efficient — only lightweight ``BacktestResult`` records are
    retained; full price paths are never accumulated.
  • Modular — ``BacktestConfig``, ``BacktestResult``, and
    ``RollingBacktester`` are fully decoupled for reuse.

Usage
-----
>>> from models.backtest_engine import (
...     RollingBacktester, BacktestConfig, make_pipeline_fn,
... )
>>> cfg = BacktestConfig(min_train_days=252, step_size=5)
>>> bt  = RollingBacktester(returns, make_pipeline_fn(pipeline_cfg), cfg)
>>> results = bt.run()
>>> print(bt.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

log = logging.getLogger("backtest")

# ─────────────────────────────────────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────────────────────────────────────

class PipelineFn(Protocol):
    """Signature expected by ``RollingBacktester.pipeline_fn``."""

    def __call__(
        self,
        train_returns: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Fit model on *train_returns* and return at minimum
        ``{"var_95": float}``.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    """Immutable configuration for a rolling backtest run."""

    forecast_horizon: int = 30
    """Trading-day look-ahead for VaR evaluation."""

    var_confidence: float = 0.95
    """Quantile level for VaR (only informational; the pipeline computes it)."""

    min_train_days: int = 252
    """Minimum expanding-window size before the first forecast (~1 year)."""

    step_size: int = 1
    """Days to advance between consecutive forecast origins."""

    mc_n_paths: int = 5_000
    """Monte Carlo paths per forecast (forwarded to the pipeline callable)."""

    mc_n_steps: int = 30
    """Steps per MC path (should match ``forecast_horizon``)."""

    output_dir: Path | None = None
    """Optional directory for persisting CSV results."""

    def __post_init__(self) -> None:
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")
        if self.min_train_days < 1:
            raise ValueError("min_train_days must be >= 1")
        if self.step_size < 1:
            raise ValueError("step_size must be >= 1")
        if not (0.0 < self.var_confidence < 1.0):
            raise ValueError("var_confidence must be in (0, 1)")

    def to_pipeline_overrides(self) -> dict[str, Any]:
        """Return keys suitable for merging into a pipeline config dict."""
        return {
            "var_confidence": self.var_confidence,
            "mc_n_paths": self.mc_n_paths,
            "mc_n_steps": self.mc_n_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-window result record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """One forecast-origin snapshot produced by the rolling backtest."""

    origin_date: datetime
    """Last day of the training window (forecast is made *after* this day)."""

    train_start: datetime
    """First day of the expanding training window."""

    train_size: int
    """Number of trading days in the training window."""

    predicted_var_95: float
    """Portfolio-level VaR(95%) from the Monte Carlo simulation."""

    realized_loss: float
    """Actual cumulative equal-weight portfolio return over the forward
    ``forecast_horizon`` window.  Negative ⇒ loss."""

    violation: bool
    """``True`` when ``realized_loss < predicted_var_95`` (a VaR breach)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Core orchestration engine
# ─────────────────────────────────────────────────────────────────────────────

class RollingBacktester:
    """
    Expanding-window walk-forward backtester.

    Parameters
    ----------
    returns : pd.DataFrame
        Full historical log-return panel  (T × N_assets).
        Index must be a ``DatetimeIndex``.
    pipeline_fn : PipelineFn
        Callable ``(train_returns, cfg) → {"var_95": float, ...}``.
        The backtester **never** inspects internals—only reads ``var_95``.
    config : BacktestConfig
        Backtest hyper-parameters.
    pipeline_cfg : dict | None
        Base configuration dict forwarded (with backtest overrides) to
        ``pipeline_fn`` at each step.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        pipeline_fn: Callable[..., dict[str, Any]],
        config: BacktestConfig | None = None,
        pipeline_cfg: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError(
                "returns.index must be a DatetimeIndex, "
                f"got {type(returns.index).__name__}"
            )
        if returns.empty:
            raise ValueError("returns DataFrame must not be empty")

        self.returns = returns
        self.pipeline_fn = pipeline_fn
        self.config = config or BacktestConfig()
        self._pipeline_cfg = pipeline_cfg or {}
        self._results: list[BacktestResult] = []

    # ── public API ────────────────────────────────────────────────────────

    def run(self) -> list[BacktestResult]:
        """
        Execute the expanding-window walk-forward backtest.

        Returns
        -------
        list[BacktestResult]
            One record per valid forecast origin.
        """
        self._results.clear()

        T = len(self.returns)
        cfg = self.config
        origins = self._compute_origins(T, cfg)

        if not origins:
            log.warning(
                "No valid forecast origins: T=%d, min_train=%d, horizon=%d",
                T, cfg.min_train_days, cfg.forecast_horizon,
            )
            return self._results

        log.info(
            "Backtest started: %d origins, horizon=%d, min_train=%d, step=%d",
            len(origins), cfg.forecast_horizon, cfg.min_train_days, cfg.step_size,
        )

        merged_cfg = {**self._pipeline_cfg, **cfg.to_pipeline_overrides()}

        for step_num, t in enumerate(origins, start=1):
            result = self._evaluate_origin(t, merged_cfg)
            self._results.append(result)

            if step_num % max(1, len(origins) // 10) == 0 or step_num == len(origins):
                violation_rate = (
                    sum(r.violation for r in self._results) / len(self._results)
                )
                log.info(
                    "  [%3d/%d]  origin=%s  VaR=%.4f  realized=%.4f  "
                    "violation=%s  cumul_rate=%.2f%%",
                    step_num,
                    len(origins),
                    result.origin_date.strftime("%Y-%m-%d"),
                    result.predicted_var_95,
                    result.realized_loss,
                    result.violation,
                    violation_rate * 100,
                )

        final_summary = self.summary()
        log.info(
            "Backtest complete: %d origins, %d violations (%.2f%%)",
            final_summary["total_origins"],
            final_summary["violation_count"],
            final_summary["violation_rate"] * 100,
        )

        if cfg.output_dir is not None:
            self.save(cfg.output_dir / "backtest_results.csv")

        return self._results

    def summary(self) -> dict[str, Any]:
        """
        Aggregate statistics over all completed backtest windows.

        Returns
        -------
        dict with keys:
            total_origins, violation_count, violation_rate,
            mean_predicted_var, mean_realized_loss,
            first_origin, last_origin
        """
        if not self._results:
            return {
                "total_origins": 0,
                "violation_count": 0,
                "violation_rate": 0.0,
                "mean_predicted_var": float("nan"),
                "mean_realized_loss": float("nan"),
                "first_origin": None,
                "last_origin": None,
            }

        violations = sum(r.violation for r in self._results)
        return {
            "total_origins": len(self._results),
            "violation_count": violations,
            "violation_rate": violations / len(self._results),
            "mean_predicted_var": float(
                np.mean([r.predicted_var_95 for r in self._results])
            ),
            "mean_realized_loss": float(
                np.mean([r.realized_loss for r in self._results])
            ),
            "first_origin": self._results[0].origin_date,
            "last_origin": self._results[-1].origin_date,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert backtest results to a tidy ``pd.DataFrame``.

        Columns: origin_date, train_start, train_size, predicted_var_95,
                 realized_loss, violation
        """
        if not self._results:
            return pd.DataFrame(
                columns=[
                    "origin_date", "train_start", "train_size",
                    "predicted_var_95", "realized_loss", "violation",
                ]
            )
        return pd.DataFrame([r.to_dict() for r in self._results])

    def save(self, path: Path | str) -> None:
        """Persist results to CSV at *path* (parent dirs created if needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        log.info("Results saved → %s  (%d rows)", path, len(df))

    @property
    def results(self) -> list[BacktestResult]:
        """Read-only access to the collected backtest results."""
        return list(self._results)

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_origins(
        total_days: int,
        cfg: BacktestConfig,
    ) -> list[int]:
        """
        Return the integer indices of valid forecast origins.

        An origin ``t`` is valid when:
          • ``t >= cfg.min_train_days``  (enough history)
          • ``t + cfg.forecast_horizon <= total_days``  (enough future)
        """
        first = cfg.min_train_days
        last = total_days - cfg.forecast_horizon
        if first > last:
            return []
        return list(range(first, last + 1, cfg.step_size))

    def _evaluate_origin(
        self,
        t: int,
        merged_cfg: dict[str, Any],
    ) -> BacktestResult:
        """
        Run the pipeline on ``returns[:t]``, then compare predicted VaR
        against the realised portfolio return over
        ``returns[t : t + forecast_horizon]``.
        """
        cfg = self.config

        # ── Expanding training window (no future data leak) ───────────
        train_returns = self.returns.iloc[:t]

        # ── Call user-supplied pipeline ───────────────────────────────
        forecast = self.pipeline_fn(train_returns, merged_cfg)
        predicted_var = float(forecast["var_95"])

        # ── Realised forward return ──────────────────────────────────
        forward_returns = self.returns.iloc[t : t + cfg.forecast_horizon]
        # Cumulative equal-weight portfolio return over the window
        daily_portfolio = forward_returns.mean(axis=1)  # mean across assets
        realized_loss = float(daily_portfolio.sum())     # cumulative return

        # ── Violation flag ───────────────────────────────────────────
        violation = realized_loss < predicted_var

        return BacktestResult(
            origin_date=train_returns.index[-1].to_pydatetime(),
            train_start=train_returns.index[0].to_pydatetime(),
            train_size=len(train_returns),
            predicted_var_95=predicted_var,
            realized_loss=realized_loss,
            violation=violation,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory — wraps main.py stages into a backtest-compatible callable
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline_fn(base_cfg: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """
    Create a ``pipeline_fn`` closure that chains the existing pipeline stages
    from ``main.py``.

    The returned callable has the signature::

        def pipeline_fn(train_returns: pd.DataFrame, cfg: dict) -> dict

    It performs stages 2–7 (regime → network → params → simulation → risk)
    on the supplied ``train_returns`` slice, returning the risk-metrics dict.

    Parameters
    ----------
    base_cfg : dict
        Baseline pipeline configuration (e.g. ``CONFIG`` from ``main.py``).

    Returns
    -------
    Callable
        A function suitable for ``RollingBacktester(pipeline_fn=...)``.
    """
    # Lazy imports — keep the module importable without heavy deps
    def _pipeline_fn(
        train_returns: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        # Late import to avoid circular deps & allow standalone testing
        from main import (
            detect_regime,
            build_network,
            adjust_parameters,
            run_simulation,
            compute_risk_metrics,
        )

        run_cfg = {**base_cfg, **cfg}

        # Stage 2 — Regime detection
        regime_result = detect_regime(train_returns, run_cfg)

        # Stage 3 — Correlation network
        network_result = build_network(train_returns, run_cfg)

        # Stage 4 — Sentiment (skip in backtest — default to neutral)
        sentiment_score = 0.0

        # Stage 5 — Parameter adjustment
        params = adjust_parameters(
            train_returns,
            sentiment_score,
            regime_result,
            run_cfg,
        )

        # Stage 6 — Monte Carlo simulation
        S0 = train_returns.iloc[-1].values  # latest available "prices" proxy
        # NOTE: In a full integration the caller should pass actual prices.
        # Here we use exp(cumulative returns) as a price-level proxy.
        S0_proxy = np.exp(train_returns.cumsum().iloc[-1].values)

        paths, _, _ = run_simulation(S0_proxy, params, network_result, run_cfg)

        # Stage 7 — Risk metrics
        risk = compute_risk_metrics(paths, S0_proxy, run_cfg)

        return {
            "var_95": risk["var_95"],
            "expected_shortfall": risk["expected_shortfall"],
            "regime": regime_result["current_regime_name"],
        }

    return _pipeline_fn
