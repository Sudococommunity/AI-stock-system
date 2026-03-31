"""
Risk metrics library — every metric a serious trader needs.
All functions are pure: numpy arrays in, scalars/dicts out.
No side effects, fully vectorized, easy to test.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy import stats as scipy_stats

from stock_recommender.config import CONFIG


@dataclass
class RiskProfile:
    """Complete risk profile for a stock or portfolio."""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # Risk-adjusted performance
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0

    # Absolute risk
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Tail risk
    var_95: float = 0.0       # Value at Risk (95% confidence)
    var_99: float = 0.0
    cvar_95: float = 0.0      # Conditional VaR / Expected Shortfall
    cvar_99: float = 0.0

    # Market relationship
    beta: float = 0.0
    alpha: float = 0.0        # Jensen's Alpha (annualized)
    correlation: float = 0.0  # vs benchmark

    # Win/loss
    win_rate: float = 0.0
    profit_factor: float = 0.0   # gross profit / gross loss
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_to_reward: float = 0.0  # avg_win / abs(avg_loss)

    # Computed fields
    risk_score: float = 0.0   # 0 (low risk) to 100 (extreme risk)
    opportunity_score: float = 0.0  # risk-adjusted opportunity


# ── Core statistical helpers ──────────────────────────────────────────────────

def _annualize(daily_val: float, trading_days: int = 252) -> float:
    return daily_val * np.sqrt(trading_days)


def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    return num / denom if abs(denom) > 1e-10 else fallback


# ── Individual metric functions ───────────────────────────────────────────────

def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = CONFIG.risk.risk_free_rate,
    trading_days: int = CONFIG.risk.trading_days_per_year,
) -> float:
    """Annualized Sharpe ratio. Higher is better (> 1 is good, > 2 is excellent)."""
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(trading_days))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = CONFIG.risk.risk_free_rate,
    trading_days: int = CONFIG.risk.trading_days_per_year,
) -> float:
    """
    Annualized Sortino ratio — like Sharpe but penalizes only downside volatility.
    Better metric for asymmetric return distributions.
    """
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    downside = returns[returns < daily_rf] - daily_rf
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.0
    return float(_safe_div(excess.mean() * trading_days, downside_std * np.sqrt(trading_days)))


def calmar_ratio(
    returns: np.ndarray,
    trading_days: int = CONFIG.risk.trading_days_per_year,
) -> float:
    """
    Calmar ratio = annualized return / max drawdown magnitude.
    Measures return per unit of worst-case loss.
    """
    ann_ret = returns.mean() * trading_days
    mdd = max_drawdown(returns)[0]
    return float(_safe_div(ann_ret, abs(mdd)))


def value_at_risk(
    returns: np.ndarray,
    confidence: float = CONFIG.risk.var_confidence,
    method: str = "historical",
) -> float:
    """
    Value at Risk — maximum expected loss at the given confidence level.
    Returns a negative number (it's a loss).

    Methods:
        historical  — empirical quantile (non-parametric)
        parametric  — assumes normal distribution
        cornish_fisher — adjusts for skewness and kurtosis
    """
    if method == "historical":
        return float(np.percentile(returns, (1 - confidence) * 100))

    elif method == "parametric":
        z = scipy_stats.norm.ppf(1 - confidence)
        return float(returns.mean() + z * returns.std())

    elif method == "cornish_fisher":
        z = scipy_stats.norm.ppf(1 - confidence)
        s = scipy_stats.skew(returns)
        k = scipy_stats.kurtosis(returns)  # excess kurtosis
        # Cornish-Fisher expansion
        z_cf = (z + (z**2 - 1) * s / 6
                + (z**3 - 3*z) * k / 24
                - (2*z**3 - 5*z) * s**2 / 36)
        return float(returns.mean() + z_cf * returns.std())

    raise ValueError(f"Unknown VaR method: {method}")


def conditional_var(
    returns: np.ndarray,
    confidence: float = CONFIG.risk.var_confidence,
) -> float:
    """
    Conditional VaR (CVaR) / Expected Shortfall.
    Average loss in the worst (1-confidence)% of scenarios.
    More informative than VaR — tells you what to expect when VaR is breached.
    """
    var = value_at_risk(returns, confidence, method="historical")
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def max_drawdown(returns: np.ndarray) -> Tuple[float, int, int, int]:
    """
    Maximum peak-to-trough decline.

    Returns:
        (magnitude, peak_idx, trough_idx, recovery_idx)
        magnitude is negative (a loss), e.g. -0.35 means -35% drawdown.
    """
    cum_ret = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max

    if len(drawdown) == 0:
        return 0.0, 0, 0, 0

    trough_idx = int(np.argmin(drawdown))
    mdd = float(drawdown[trough_idx])

    # Peak before the trough
    peak_idx = int(np.argmax(cum_ret[:trough_idx + 1])) if trough_idx > 0 else 0

    # Recovery: first time after trough where cum_ret recovers to peak level
    recovery_idx = trough_idx
    peak_val = cum_ret[peak_idx]
    for i in range(trough_idx, len(cum_ret)):
        if cum_ret[i] >= peak_val:
            recovery_idx = i
            break

    return mdd, peak_idx, trough_idx, recovery_idx


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Full drawdown time series — useful for plotting."""
    cum_ret = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_ret)
    return (cum_ret - running_max) / running_max


def beta_alpha(
    stock_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = CONFIG.risk.risk_free_rate,
    trading_days: int = CONFIG.risk.trading_days_per_year,
) -> Tuple[float, float, float]:
    """
    Compute Beta, Jensen's Alpha, and correlation vs benchmark.

    Beta  > 1 → more volatile than market
    Alpha > 0 → outperformed on risk-adjusted basis (annualized)
    """
    if len(stock_returns) != len(benchmark_returns) or len(stock_returns) < 2:
        return 1.0, 0.0, 0.0

    cov_matrix = np.cov(stock_returns, benchmark_returns)
    beta_val = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] > 0 else 1.0

    daily_rf = risk_free_rate / trading_days
    # Jensen's Alpha = annualized(stock_return) - rf - beta*(annualized(bench) - rf)
    alpha_val = float(
        (stock_returns.mean() - daily_rf) * trading_days
        - beta_val * (benchmark_returns.mean() - daily_rf) * trading_days
    )

    corr_val = float(np.corrcoef(stock_returns, benchmark_returns)[0, 1])
    return beta_val, alpha_val, corr_val


def information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    trading_days: int = CONFIG.risk.trading_days_per_year,
) -> float:
    """Information Ratio = excess return / tracking error (annualized)."""
    active = portfolio_returns - benchmark_returns
    tracking_error = active.std() * np.sqrt(trading_days)
    return float(_safe_div(active.mean() * trading_days, tracking_error))


def win_loss_stats(returns: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Win rate, profit factor, average win, average loss, risk-to-reward ratio.

    Profit factor > 1 means the strategy is profitable overall.
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / max(len(returns), 1)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    profit_factor = float(_safe_div(wins.sum(), abs(losses.sum())))
    rr = float(_safe_div(avg_win, abs(avg_loss)))

    return win_rate, profit_factor, avg_win, avg_loss, rr


def risk_to_profit_ratio(
    expected_return: float,
    var_95: float,
) -> float:
    """
    Risk-to-profit ratio = expected gain / potential loss (VaR).
    Higher is better. > 2 is generally considered attractive.
    """
    return float(_safe_div(expected_return, abs(var_95)))


# ── Composite risk score ──────────────────────────────────────────────────────

def _normalize_score(val: float, low: float, high: float, invert: bool = False) -> float:
    """Map val into [0, 100] given expected range [low, high]."""
    score = np.clip((val - low) / max(high - low, 1e-8), 0, 1) * 100
    return float(100 - score if invert else score)


def compute_risk_score(profile: "RiskProfile") -> float:
    """
    Composite risk score (0 = lowest risk, 100 = highest risk).
    Combines volatility, drawdown, and tail risk into one number.
    """
    vol_score = _normalize_score(profile.annualized_volatility, 0.05, 0.80)
    mdd_score = _normalize_score(abs(profile.max_drawdown), 0.0, 0.80)
    var_score = _normalize_score(abs(profile.var_95), 0.005, 0.10)
    beta_score = _normalize_score(abs(profile.beta), 0.0, 3.0)

    return float(0.35 * vol_score + 0.35 * mdd_score + 0.20 * var_score + 0.10 * beta_score)


def compute_opportunity_score(profile: "RiskProfile") -> float:
    """
    Composite opportunity score (0 = poor, 100 = excellent).
    Combines Sharpe, win rate, and risk-to-reward into one signal.
    """
    sharpe_score = _normalize_score(profile.sharpe_ratio, -1.0, 3.0)
    wr_score = _normalize_score(profile.win_rate, 0.3, 0.7)
    rr_score = _normalize_score(profile.risk_to_reward, 0.5, 4.0)
    calmar_score = _normalize_score(profile.calmar_ratio, 0.0, 5.0)

    return float(0.35 * sharpe_score + 0.25 * wr_score + 0.25 * rr_score + 0.15 * calmar_score)


# ── Master function ───────────────────────────────────────────────────────────

def compute_full_risk_profile(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = CONFIG.risk.risk_free_rate,
    trading_days: int = CONFIG.risk.trading_days_per_year,
) -> RiskProfile:
    """
    Compute the complete risk profile from a daily returns series.

    Args:
        returns: 1-D numpy array of daily fractional returns (e.g. 0.01 = 1%).
        benchmark_returns: benchmark (e.g. SPY) daily returns, same length.
                           If None, beta/alpha/IR are set to defaults.
        risk_free_rate: annual risk-free rate.
        trading_days: number of trading days per year for annualization.
    """
    if len(returns) == 0:
        return RiskProfile()

    returns = np.asarray(returns, dtype=float)
    p = RiskProfile()

    # ── Return metrics ────────────────────────────────────────────────────────
    cum = float((1 + returns).prod() - 1)
    p.total_return = cum
    p.annualized_return = float(returns.mean() * trading_days)
    n_years = len(returns) / trading_days
    p.cagr = float((1 + cum) ** (1 / max(n_years, 1e-6)) - 1)

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    p.annualized_volatility = float(returns.std() * np.sqrt(trading_days))
    p.sharpe_ratio = sharpe_ratio(returns, risk_free_rate, trading_days)
    p.sortino_ratio = sortino_ratio(returns, risk_free_rate, trading_days)

    downside = returns[returns < 0]
    p.downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(trading_days)) if len(downside) > 0 else 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    mdd, peak_i, trough_i, recovery_i = max_drawdown(returns)
    p.max_drawdown = mdd
    p.max_drawdown_duration_days = int(recovery_i - peak_i)
    dd_series = drawdown_series(returns)
    p.avg_drawdown = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0

    p.calmar_ratio = calmar_ratio(returns, trading_days)

    # ── Tail risk ─────────────────────────────────────────────────────────────
    p.var_95 = value_at_risk(returns, 0.95)
    p.var_99 = value_at_risk(returns, 0.99)
    p.cvar_95 = conditional_var(returns, 0.95)
    p.cvar_99 = conditional_var(returns, 0.99)

    # ── Market relationship ───────────────────────────────────────────────────
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        p.beta, p.alpha, p.correlation = beta_alpha(returns, benchmark_returns, risk_free_rate, trading_days)
        p.treynor_ratio = float(_safe_div(p.annualized_return - risk_free_rate, abs(p.beta)))
        p.information_ratio = information_ratio(returns, benchmark_returns, trading_days)
    else:
        p.beta, p.alpha, p.correlation = 1.0, 0.0, 0.0

    # ── Win/loss ──────────────────────────────────────────────────────────────
    p.win_rate, p.profit_factor, p.avg_win, p.avg_loss, p.risk_to_reward = win_loss_stats(returns)

    # ── Composite scores ──────────────────────────────────────────────────────
    p.risk_score = compute_risk_score(p)
    p.opportunity_score = compute_opportunity_score(p)

    return p
