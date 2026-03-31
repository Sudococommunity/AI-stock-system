"""
Technical indicator library — pure numpy/pandas functions with no side effects.
Every indicator operates on pandas Series or DataFrames and returns the same type.
"""
import numpy as np
import pandas as pd
from typing import Tuple

try:
    import talib
except ImportError:
    talib = None


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0–100). Overbought > 70, oversold < 30."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename("macd")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("macd_signal")
    histogram = (macd_line - signal_line).rename("macd_hist")
    return macd_line, signal_line, histogram


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """%K and %D stochastic oscillator."""
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = (100 * (close - low_min) / denom).rename("stoch_k")
    d = k.rolling(d_period).mean().rename("stoch_d")
    return k, d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R (-100 to 0). Below -80 oversold, above -20 overbought."""
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return (-100 * (high_max - close) / denom).rename("williams_r")


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change — momentum as percentage."""
    return (close.pct_change(period) * 100).rename(f"roc_{period}")


# ── Trend ─────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean().rename(f"sma_{period}")


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Upper, middle (SMA), lower bands, bandwidth, and %B position."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = (mid + std_dev * std).rename("bb_upper")
    lower = (mid - std_dev * std).rename("bb_lower")
    bandwidth = ((upper - lower) / mid.replace(0, np.nan)).rename("bb_width")
    pct_b = ((close - lower) / (upper - lower).replace(0, np.nan)).rename("bb_pct")
    return upper, mid.rename("bb_mid"), lower, bandwidth, pct_b


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index plus ±DI. ADX > 25 signals strong trend."""
    up_move = high.diff()
    down_move = (-low.diff())
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = (100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan)).rename("plus_di")
    minus_di = (100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan)).rename("minus_di")
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_series = dx.ewm(span=period, adjust=False).mean().rename("adx")
    return adx_series, plus_di, minus_di


def parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """Parabolic SAR — trailing stop and reversal indicator."""
    n = len(high)
    sar = np.zeros(n)
    ep = np.zeros(n)
    af = np.zeros(n)
    trend = np.ones(n, dtype=int)  # 1 = up, -1 = down

    sar[0] = low.iloc[0]
    ep[0] = high.iloc[0]
    af[0] = step

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prev_ep = ep[i - 1]
        prev_af = af[i - 1]
        prev_trend = trend[i - 1]

        sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)

        if prev_trend == 1:
            if low.iloc[i] < sar[i]:           # reversal to downtrend
                trend[i] = -1
                sar[i] = prev_ep
                ep[i] = low.iloc[i]
                af[i] = step
            else:
                trend[i] = 1
                sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[max(0, i - 2)])
                ep[i] = max(prev_ep, high.iloc[i])
                af[i] = min(prev_af + step if ep[i] > prev_ep else prev_af, max_af)
        else:
            if high.iloc[i] > sar[i]:          # reversal to uptrend
                trend[i] = 1
                sar[i] = prev_ep
                ep[i] = high.iloc[i]
                af[i] = step
            else:
                trend[i] = -1
                sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[max(0, i - 2)])
                ep[i] = min(prev_ep, low.iloc[i])
                af[i] = min(prev_af + step if ep[i] < prev_ep else prev_af, max_af)

    return pd.Series(sar, index=high.index, name="parabolic_sar")


def ichimoku_cloud(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku Cloud components in leak-free current-time form."""
    tenkan = ((high.rolling(9).max() + low.rolling(9).min()) / 2.0).rename("ichimoku_tenkan")
    kijun = ((high.rolling(26).max() + low.rolling(26).min()) / 2.0).rename("ichimoku_kijun")
    # Use the current value of the leading spans instead of the chart-forward projection.
    senkou_a = ((tenkan + kijun) / 2.0).rename("ichimoku_senkou_a")
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).rename("ichimoku_senkou_b")
    # Lagging span as a backward-looking comparison, not a future-shifted leak.
    chikou = close.shift(26).rename("ichimoku_chikou")
    return tenkan, kijun, senkou_a, senkou_b, chikou


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """Supertrend line and direction (+1 uptrend, -1 downtrend)."""
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    st = pd.Series(index=close.index, dtype=float, name="supertrend")
    direction = pd.Series(index=close.index, dtype=float, name="supertrend_dir")

    if len(close) == 0:
        return st, direction

    st.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1.0

    for i in range(1, len(close)):
        prev_close = close.iloc[i - 1]
        if prev_close <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i - 1])
        if prev_close >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i - 1])

        if close.iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] > 0 else final_upper.iloc[i]

    return st, direction


def keltner_channel(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(close, ema_period).rename("keltner_mid")
    atr_val = atr(high, low, close, atr_period)
    upper = (mid + multiplier * atr_val).rename("keltner_upper")
    lower = (mid - multiplier * atr_val).rename("keltner_lower")
    return upper, mid, lower


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(period).max().rename("donchian_upper")
    lower = low.rolling(period).min().rename("donchian_lower")
    mid = ((upper + lower) / 2.0).rename("donchian_mid")
    return upper, mid, lower


def elder_ray_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> Tuple[pd.Series, pd.Series]:
    basis = ema(close, period)
    bull = (high - basis).rename("elder_bull_power")
    bear = (low - basis).rename("elder_bear_power")
    return bull, bear


def chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    adl = (mfm.fillna(0.0) * volume).cumsum()
    return (adl.ewm(span=fast, adjust=False).mean() - adl.ewm(span=slow, adjust=False).mean()).rename("chaikin_osc")


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    raw = close.diff() * volume
    return raw.ewm(span=period, adjust=False).mean().rename(f"force_index_{period}")


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Classic, Fibonacci, and Camarilla pivot-derived levels from the previous bar."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    prev_range = prev_high - prev_low

    classic_pivot = ((prev_high + prev_low + prev_close) / 3.0).rename("classic_pivot")
    fib_r1 = (classic_pivot + 0.382 * prev_range).rename("fib_r1")
    camarilla_r1 = (prev_close + (prev_range * 1.1 / 12.0)).rename("camarilla_r1")
    return pd.DataFrame({
        "classic_pivot": classic_pivot,
        "fib_r1": fib_r1,
        "camarilla_r1": camarilla_r1,
    })


def fibonacci_retracement_levels(
    high: pd.Series,
    low: pd.Series,
    period: int = 252,
) -> pd.DataFrame:
    rolling_high = high.rolling(period, min_periods=60).max()
    rolling_low = low.rolling(period, min_periods=60).min()
    span = (rolling_high - rolling_low).replace(0, np.nan)
    fib_382 = (rolling_high - 0.382 * span).rename("fib_382")
    fib_618 = (rolling_high - 0.618 * span).rename("fib_618")
    return pd.DataFrame({
        "fib_382": fib_382,
        "fib_618": fib_618,
        "rolling_high_252": rolling_high.rename("rolling_high_252"),
        "rolling_low_252": rolling_low.rename("rolling_low_252"),
    })


def rolling_slope(series: pd.Series, period: int = 10) -> pd.Series:
    def _slope(window: np.ndarray) -> float:
        x = np.arange(len(window), dtype=float)
        mask = np.isfinite(window)
        if mask.sum() < 2:
            return np.nan
        coef = np.polyfit(x[mask], window[mask], 1)[0]
        scale = np.nanmean(np.abs(window[mask])) + 1e-9
        return coef / scale

    return series.rolling(period).apply(_slope, raw=True).rename(f"{series.name}_slope_{period}")


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — raw volatility measure in price units."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename("atr")


def historical_volatility(close: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling annualized historical volatility from log returns."""
    log_ret = np.log(close / close.shift())
    vol = log_ret.rolling(period).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.rename(f"hvol_{period}")


# ── Volume ────────────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative buying/selling pressure."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("obv")


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price — fair value benchmark."""
    typical = (high + low + close) / 3
    return ((typical * volume).cumsum() / volume.cumsum()).rename("vwap")


def money_flow_index(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """MFI — volume-weighted RSI (0–100)."""
    typical = (high + low + close) / 3
    raw_mf = typical * volume
    pos_mf = raw_mf.where(typical > typical.shift(), 0.0)
    neg_mf = raw_mf.where(typical < typical.shift(), 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - 100 / (1 + mfr)).rename("mfi")


def chaikin_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """CMF — buying/selling pressure relative to price range."""
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    return (mfm * volume).rolling(period).sum() / volume.rolling(period).sum()


def cci_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    if talib is not None:
        values = talib.CCI(high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float), timeperiod=period)
        return pd.Series(values, index=close.index, name=f"cci_{period}")
    typical = (high + low + close) / 3
    sma_tp = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return ((typical - sma_tp) / (0.015 * mad.replace(0, np.nan))).rename(f"cci_{period}")


def cmo_indicator(close: pd.Series, period: int = 14) -> pd.Series:
    if talib is not None:
        values = talib.CMO(close.to_numpy(dtype=float), timeperiod=period)
        return pd.Series(values, index=close.index, name=f"cmo_{period}")
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).sum()
    down = (-delta.clip(upper=0)).rolling(period).sum()
    return (100 * (up - down) / (up + down).replace(0, np.nan)).rename(f"cmo_{period}")


def trix_indicator(close: pd.Series, period: int = 30) -> pd.Series:
    if talib is not None:
        values = talib.TRIX(close.to_numpy(dtype=float), timeperiod=period)
        return pd.Series(values, index=close.index, name=f"trix_{period}")
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return (ema3.pct_change() * 100).rename(f"trix_{period}")


def ppo_indicator(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    if talib is not None:
        values = talib.PPO(close.to_numpy(dtype=float), fastperiod=fast, slowperiod=slow, matype=0)
        return pd.Series(values, index=close.index, name="ppo")
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    return (((fast_ema - slow_ema) / slow_ema.replace(0, np.nan)) * 100).rename("ppo")


def bop_indicator(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if talib is not None:
        values = talib.BOP(open_.to_numpy(dtype=float), high.to_numpy(dtype=float), low.to_numpy(dtype=float), close.to_numpy(dtype=float))
        return pd.Series(values, index=close.index, name="bop")
    return ((close - open_) / (high - low).replace(0, np.nan)).rename("bop")


def aroon_osc_indicator(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    if talib is not None:
        values = talib.AROONOSC(high.to_numpy(dtype=float), low.to_numpy(dtype=float), timeperiod=period)
        return pd.Series(values, index=high.index, name=f"aroon_osc_{period}")
    aroon_up = high.rolling(period + 1).apply(lambda x: 100.0 * np.argmax(x) / period, raw=True)
    aroon_down = low.rolling(period + 1).apply(lambda x: 100.0 * np.argmin(x) / period, raw=True)
    return (aroon_up - aroon_down).rename(f"aroon_osc_{period}")


# ── Additional momentum ───────────────────────────────────────────────────────

def stochastic_rsi(
    close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """StochRSI — applies the stochastic formula to RSI values (0–100)."""
    rsi_vals = rsi(close, period)
    rsi_min = rsi_vals.rolling(period).min()
    rsi_max = rsi_vals.rolling(period).max()
    k = (100 * (rsi_vals - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
         ).rolling(smooth_k).mean().rename("stochrsi_k")
    d = k.rolling(smooth_d).mean().rename("stochrsi_d")
    return k, d


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Ultimate Oscillator — blends 7, 14, 28-period buying pressure (0–100)."""
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = (pd.concat([high, prev_close], axis=1).max(axis=1)
          - pd.concat([low, prev_close], axis=1).min(axis=1)).replace(0, np.nan)
    avg7  = bp.rolling(7).sum()  / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    return (100 * (4 * avg7 + 2 * avg14 + avg28) / 7.0).rename("uo")


def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Awesome Oscillator — 5-period vs 34-period midpoint SMA difference."""
    mid = (high + low) / 2.0
    return (mid.rolling(5).mean() - mid.rolling(34).mean()).rename("ao")


def detrended_price_oscillator(close: pd.Series, period: int = 20) -> pd.Series:
    """DPO — removes trend to expose cycles."""
    shift = period // 2 + 1
    return (close - close.rolling(period).mean().shift(shift)).rename(f"dpo_{period}")


def kst_oscillator(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """KST (Know Sure Thing) — sum of smoothed rate-of-change values."""
    rcma1 = close.pct_change(10).rolling(10).mean()
    rcma2 = close.pct_change(13).rolling(13).mean()
    rcma3 = close.pct_change(14).rolling(14).mean()
    rcma4 = close.pct_change(15).rolling(15).mean()
    kst_line = (rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4).rename("kst")
    signal = kst_line.rolling(9).mean().rename("kst_signal")
    return kst_line, signal


def mass_index_indicator(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    """Mass Index — detects trend reversals via high-low range expansion."""
    hl = high - low
    ema1 = hl.ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    return (ema1 / ema2.replace(0, np.nan)).rolling(period).sum().rename("mass_index")


# ── Additional trend / MA variants ───────────────────────────────────────────

def dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA — reduces lag vs standard EMA."""
    e1 = series.ewm(span=period, adjust=False).mean()
    return (2 * e1 - e1.ewm(span=period, adjust=False).mean()).rename(f"dema_{period}")


def tema(series: pd.Series, period: int) -> pd.Series:
    """Triple EMA — further lag reduction."""
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return (3 * e1 - 3 * e2 + e3).rename(f"tema_{period}")


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average — smooth and responsive."""
    half = max(period // 2, 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    raw = 2 * series.rolling(half).mean() - series.rolling(period).mean()
    return raw.rolling(sqrt_p).mean().rename(f"hma_{period}")


def kama(close: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive MA — adjusts speed based on efficiency ratio."""
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    result = close.copy().astype(float)
    result.iloc[:period] = np.nan
    for i in range(period, len(close)):
        direction  = abs(close.iloc[i] - close.iloc[i - period])
        volatility = close.iloc[i - period : i + 1].diff().abs().sum()
        er  = direction / volatility if volatility > 0 else 0.0
        sc  = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        prev = result.iloc[i - 1] if not np.isnan(result.iloc[i - 1]) else close.iloc[i]
        result.iloc[i] = prev + sc * (close.iloc[i] - prev)
    return result.rename("kama")


def vortex_indicator(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series]:
    """Vortex Indicator — VI+ and VI- trend direction signals."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    tr_sum = tr.rolling(period).sum().replace(0, np.nan)
    vi_p = ((high - low.shift(1)).abs().rolling(period).sum() / tr_sum).rename("vi_plus")
    vi_m = ((low  - high.shift(1)).abs().rolling(period).sum() / tr_sum).rename("vi_minus")
    return vi_p, vi_m


# ── Additional volatility ─────────────────────────────────────────────────────

def ulcer_index(close: pd.Series, period: int = 14) -> pd.Series:
    """Ulcer Index — RMS of rolling drawdown from peak."""
    rolling_max = close.rolling(period, min_periods=1).max()
    pct_dd = ((close - rolling_max) / rolling_max.replace(0, np.nan)) * 100
    return np.sqrt((pct_dd ** 2).rolling(period).mean()).rename(f"ulcer_{period}")


def chaikin_volatility_indicator(high: pd.Series, low: pd.Series, period: int = 10) -> pd.Series:
    """Chaikin Volatility — rate of change of H-L range EMA."""
    hl_ema = (high - low).ewm(span=period, adjust=False).mean()
    return (hl_ema.pct_change(period) * 100).rename(f"chaikin_vol_{period}")


# ── Additional volume ─────────────────────────────────────────────────────────

def ease_of_movement(
    high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """EMV — relates price change to volume (high EMV = easy upward move)."""
    midpoint_move = ((high + low) / 2).diff()
    box_ratio = volume / (high - low).replace(0, np.nan)
    return (midpoint_move / box_ratio.replace(0, np.nan)).rolling(period).mean().rename("emv")


def klinger_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    fast: int = 34, slow: int = 55,
) -> pd.Series:
    """Klinger Volume Oscillator — long-term money flow vs short-term."""
    trend = np.sign(((high + low + close) / 3).diff().fillna(0))
    sv = volume * trend
    return (sv.ewm(span=fast, adjust=False).mean()
            - sv.ewm(span=slow, adjust=False).mean()).rename("klinger")


def negative_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """NVI — cumulative price performance on low-volume days (smart money proxy)."""
    pct = close.pct_change().fillna(0)
    on_low_vol = pct.where(volume < volume.shift(1), 0.0)
    return (1000.0 * (1 + on_low_vol).cumprod()).rename("nvi")


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """VPT — cumulative volume × pct-change (similar to OBV but magnitude-weighted)."""
    return (volume * close.pct_change().fillna(0)).cumsum().rename("vpt")


# ── Candlestick pattern features ──────────────────────────────────────────────

def candlestick_features(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """Normalized candlestick shape features useful for pattern detection."""
    body = (close - open_) / open_.replace(0, np.nan)
    upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close.replace(0, np.nan)
    lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close.replace(0, np.nan)
    daily_range = (high - low) / close.replace(0, np.nan)
    return pd.DataFrame({
        "body": body,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "daily_range": daily_range,
    })


def candlestick_patterns(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """Binary candlestick pattern detectors (1.0 = pattern present, 0.0 = absent)."""
    body_size  = (close - open_).abs()
    total_range = (high - low).replace(0, np.nan)
    body_ratio  = body_size / total_range
    upper_wick  = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick  = pd.concat([open_, close], axis=1).min(axis=1) - low

    doji = (body_ratio < 0.1).astype(float)

    hammer = (
        (body_ratio < 0.35) &
        (lower_wick >= 2.0 * body_size) &
        (upper_wick <= 0.15 * total_range)
    ).astype(float)

    inv_hammer = (
        (body_ratio < 0.35) &
        (upper_wick >= 2.0 * body_size) &
        (lower_wick <= 0.15 * total_range)
    ).astype(float)

    engulf_bull = (
        (close.shift(1) < open_.shift(1)) &   # prev bearish
        (close > open_) &                       # curr bullish
        (open_ <= close.shift(1)) &
        (close >= open_.shift(1))
    ).astype(float)

    engulf_bear = (
        (close.shift(1) > open_.shift(1)) &   # prev bullish
        (close < open_) &                       # curr bearish
        (open_ >= close.shift(1)) &
        (close <= open_.shift(1))
    ).astype(float)

    shooting_star = (
        (body_ratio < 0.35) &
        (upper_wick >= 2.0 * body_size) &
        (lower_wick <= 0.15 * total_range) &
        (close < open_)
    ).astype(float)

    # Morning star (3-bar simplified): big down, small indecision, big up
    morning_star = (
        (close.shift(2) < open_.shift(2)) &
        (body_ratio.shift(1) < 0.25) &
        (close > open_) &
        (close > (open_.shift(2) + close.shift(2)) / 2)
    ).astype(float)

    spinning_top = (
        (body_ratio < 0.25) &
        (upper_wick > body_size) &
        (lower_wick > body_size)
    ).astype(float)

    return pd.DataFrame({
        "candle_doji":          doji,
        "candle_hammer":        hammer,
        "candle_inv_hammer":    inv_hammer,
        "candle_engulf_bull":   engulf_bull,
        "candle_engulf_bear":   engulf_bear,
        "candle_shooting_star": shooting_star,
        "candle_morning_star":  morning_star,
        "candle_spinning_top":  spinning_top,
    })


# ── Master feature assembler ──────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator and return a flat DataFrame.

    Input DataFrame must have columns: open, high, low, close, volume.
    All column names are lowercase.
    NaN rows at the start (due to warmup periods) are expected — callers
    should drop or forward-fill after calling this function.
    """
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    out = df.copy()

    # RSI family
    out["rsi_14"] = rsi(c, 14)
    out["rsi_28"] = rsi(c, 28)

    # MACD
    macd_line, sig_line, hist_line = macd(c)
    out["macd"] = macd_line
    out["macd_signal"] = sig_line
    out["macd_hist"] = hist_line
    out["macd_hist_norm"] = hist_line / c  # normalize to price scale

    # Stochastic
    sk, sd = stochastic(h, l, c)
    out["stoch_k"] = sk
    out["stoch_d"] = sd

    # Williams %R
    out["williams_r"] = williams_r(h, l, c)

    # ROC
    out["roc_10"] = rate_of_change(c, 10)
    out["roc_20"] = rate_of_change(c, 20)

    # Bollinger Bands
    bb_up, bb_mid, bb_lo, bb_bw, bb_pct = bollinger_bands(c)
    out["bb_upper"] = bb_up
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lo
    out["bb_width"] = bb_bw
    out["bb_pct"] = bb_pct

    # ADX / DI
    adx_s, plus_di, minus_di = adx(h, l, c)
    out["adx"] = adx_s
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["di_diff"] = plus_di - minus_di

    # ATR
    atr_s = atr(h, l, c)
    out["atr"] = atr_s
    out["atr_pct"] = atr_s / c  # normalized
    out["psar_gap"] = c / parabolic_sar(h, l).replace(0, np.nan) - 1.0

    # Ichimoku / trend overlays
    tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_cloud(h, l, c)
    out["ichimoku_tenkan_gap"] = c / tenkan.replace(0, np.nan) - 1.0
    out["ichimoku_kijun_gap"] = c / kijun.replace(0, np.nan) - 1.0
    out["ichimoku_senkou_a_gap"] = c / senkou_a.replace(0, np.nan) - 1.0
    out["ichimoku_senkou_b_gap"] = c / senkou_b.replace(0, np.nan) - 1.0
    out["ichimoku_chikou_gap"] = chikou / c.replace(0, np.nan) - 1.0

    supertrend_line, supertrend_dir = supertrend(h, l, c)
    out["supertrend_gap"] = c / supertrend_line.replace(0, np.nan) - 1.0
    out["supertrend_dir"] = supertrend_dir

    kc_upper, kc_mid, kc_lower = keltner_channel(h, l, c)
    out["keltner_width"] = (kc_upper - kc_lower) / kc_mid.replace(0, np.nan)
    out["keltner_pct"] = (c - kc_lower) / (kc_upper - kc_lower).replace(0, np.nan)

    dc_upper, dc_mid, dc_lower = donchian_channel(h, l)
    out["donchian_width"] = (dc_upper - dc_lower) / dc_mid.replace(0, np.nan)
    out["donchian_pct"] = (c - dc_lower) / (dc_upper - dc_lower).replace(0, np.nan)

    # Moving averages (as ratio to close — avoids raw price leaking into model)
    for p in [5, 10, 20, 50, 200]:
        out[f"close_vs_sma{p}"] = c / sma(c, p) - 1
        out[f"close_vs_ema{p}"] = c / ema(c, p) - 1

    # Historical volatility
    out["hvol_10"] = historical_volatility(c, 10)
    out["hvol_20"] = historical_volatility(c, 20)
    out["hvol_60"] = historical_volatility(c, 60)

    # Additional TA-Lib style oscillators / trend strength features
    out["cci_20"] = cci_indicator(h, l, c, 20)
    out["cmo_14"] = cmo_indicator(c, 14)
    out["trix_30"] = trix_indicator(c, 30)
    out["ppo"] = ppo_indicator(c)
    out["bop"] = bop_indicator(o, h, l, c)
    out["aroon_osc_14"] = aroon_osc_indicator(h, l, 14)
    bull_power, bear_power = elder_ray_index(h, l, c, 13)
    out["elder_bull_power"] = bull_power / c.replace(0, np.nan)
    out["elder_bear_power"] = bear_power / c.replace(0, np.nan)
    out["chaikin_osc"] = chaikin_oscillator(h, l, c, v) / (v.rolling(20).mean().replace(0, np.nan) * c.replace(0, np.nan))
    out["force_index_13"] = force_index(c, v, 13) / (v.rolling(13).mean().replace(0, np.nan) * c.replace(0, np.nan))

    # Returns (multi-horizon)
    for p in [1, 5, 10, 20, 60]:
        out[f"ret_{p}d"] = c.pct_change(p)

    # Volume features
    vol_sma20 = sma(v, 20).rename("volume_sma20")
    out["volume_sma20"] = vol_sma20
    out["rel_volume"] = v / vol_sma20.replace(0, np.nan)
    out["obv"] = obv(c, v)
    out["obv_slope_10"] = rolling_slope(out["obv"], 10)
    out["mfi"] = money_flow_index(h, l, c, v)

    # VWAP deviation
    vwap_s = vwap(h, l, c, v)
    out["close_vs_vwap"] = c / vwap_s.replace(0, np.nan) - 1

    # Pivot / Fibonacci / 52-week context
    pivots = pivot_points(h, l, c)
    out["classic_pivot_gap"] = c / pivots["classic_pivot"].replace(0, np.nan) - 1.0
    out["fib_r1_gap"] = c / pivots["fib_r1"].replace(0, np.nan) - 1.0
    out["camarilla_r1_gap"] = c / pivots["camarilla_r1"].replace(0, np.nan) - 1.0

    fib = fibonacci_retracement_levels(h, l, 252)
    span_252 = (fib["rolling_high_252"] - fib["rolling_low_252"]).replace(0, np.nan)
    out["fib_382_gap"] = c / fib["fib_382"].replace(0, np.nan) - 1.0
    out["fib_618_gap"] = c / fib["fib_618"].replace(0, np.nan) - 1.0
    out["fib_range_pos"] = (c - fib["rolling_low_252"]) / span_252
    out["rank_52w"] = (c - fib["rolling_low_252"]) / span_252
    out["dist_52w_high"] = c / fib["rolling_high_252"].replace(0, np.nan) - 1.0
    out["dist_52w_low"] = c / fib["rolling_low_252"].replace(0, np.nan) - 1.0

    # Optional benchmark / market-microstructure context
    out["relative_strength_20d"] = 0.0
    if "benchmark_close" in df.columns:
        benchmark = df["benchmark_close"].astype(float)
        out["relative_strength_20d"] = c.pct_change(20) - benchmark.pct_change(20)

    out["sector_relative_strength_20d"] = 0.0
    if "sector_benchmark_close" in df.columns:
        sector_benchmark = df["sector_benchmark_close"].astype(float)
        out["sector_relative_strength_20d"] = c.pct_change(20) - sector_benchmark.pct_change(20)

    out["delivery_pct_feature"] = 0.0
    if "delivery_pct" in df.columns:
        out["delivery_pct_feature"] = df["delivery_pct"].astype(float) / 100.0 - 0.5

    out["put_call_ratio_feature"] = 0.0
    if "put_call_ratio" in df.columns:
        out["put_call_ratio_feature"] = df["put_call_ratio"].astype(float) - 1.0

    # Candlestick shape features
    cs = candlestick_features(o, h, l, c)
    for col in cs.columns:
        out[col] = cs[col]

    # ── 62 new indicators (Goal 4: reach 140) ────────────────────────────────
    # Collect into a dict first, then concat once to avoid DataFrame fragmentation.
    new_cols: dict = {}

    srsi_k, srsi_d = stochastic_rsi(c)
    kst_line, kst_sig = kst_oscillator(c)
    ao_s  = awesome_oscillator(h, l)
    dpo_s = detrended_price_oscillator(c, 20)
    kama_s = kama(c)
    vi_p, vi_m = vortex_indicator(h, l, c)
    atr_7  = atr(h, l, c,  7)
    atr_21 = atr(h, l, c, 21)
    emv_s     = ease_of_movement(h, l, v)
    klinger_s = klinger_oscillator(h, l, c, v)
    nvi_s = negative_volume_index(c, v)
    vpt_s = volume_price_trend(c, v)
    cp    = candlestick_patterns(o, h, l, c)
    hvol_10_s = out["hvol_10"]
    hvol_60_s = out["hvol_60"]
    natr_7_s  = atr_7  / c.replace(0, np.nan)
    natr_21_s = atr_21 / c.replace(0, np.nan)
    vol_scale = v.rolling(34).mean().replace(0, np.nan) * c.replace(0, np.nan)
    returns_abs = c.pct_change().abs()

    new_cols.update({
        # Momentum
        "rsi_7":           rsi(c, 7),
        "rsi_21":          rsi(c, 21),
        "stochrsi_k":      srsi_k,
        "stochrsi_d":      srsi_d,
        "uo":              ultimate_oscillator(h, l, c),
        "ao_norm":         ao_s / c.replace(0, np.nan),
        "dpo_20_norm":     dpo_s / c.replace(0, np.nan),
        "kst_norm":        kst_line / 100.0,
        "kst_signal_norm": kst_sig  / 100.0,
        "mass_index_norm": mass_index_indicator(h, l) / 100.0,
        "roc_1":           rate_of_change(c, 1),
        "roc_5":           rate_of_change(c, 5),
        # Trend / MA
        "close_vs_sma3":   c / sma(c,   3).replace(0, np.nan) - 1.0,
        "close_vs_ema3":   c / ema(c,   3).replace(0, np.nan) - 1.0,
        "close_vs_sma100": c / sma(c, 100).replace(0, np.nan) - 1.0,
        "close_vs_dema20": c / dema(c, 20).replace(0, np.nan) - 1.0,
        "close_vs_tema20": c / tema(c, 20).replace(0, np.nan) - 1.0,
        "close_vs_hma20":  c / hma(c,  20).replace(0, np.nan) - 1.0,
        "close_vs_kama":   c / kama_s.replace(0, np.nan) - 1.0,
        "vi_plus":         vi_p,
        "vi_minus":        vi_m,
        "vi_diff":         vi_p - vi_m,
        "lin_reg_slope_20": rolling_slope(c, 20),
        # Volatility
        "ulcer_14":        ulcer_index(c, 14),
        "chaikin_vol_10":  chaikin_volatility_indicator(h, l, 10),
        "hvol_5":          historical_volatility(c,   5),
        "hvol_120":        historical_volatility(c, 120),
        "hvol_ratio":      hvol_10_s / hvol_60_s.replace(0, np.nan),
        "natr_7":          natr_7_s,
        "natr_21":         natr_21_s,
        "atr_ratio":       natr_7_s / natr_21_s.replace(0, np.nan),
        # Volume
        "cmf_20":          chaikin_money_flow(h, l, c, v, 20),
        "vol_roc_10":      v.pct_change(10) * 100,
        "emv_norm":        emv_s / c.replace(0, np.nan),
        "klinger_norm":    klinger_s / vol_scale,
        "nvi_slope":       rolling_slope(nvi_s, 10),
        "vpt_slope":       rolling_slope(vpt_s, 10),
        "vol_osc":         v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan) - 1.0,
        "vol_trend":       v / v.ewm(span=20, adjust=False).mean().replace(0, np.nan) - 1.0,
        "volume_surge":    (v > 2.0 * vol_sma20).astype(float),
        # Returns / price
        "ret_2d":          c.pct_change(2),
        "ret_3d":          c.pct_change(3),
        "ret_30d":         c.pct_change(30),
        "ret_120d":        c.pct_change(120),
        "gap_norm":        o / c.shift(1).replace(0, np.nan) - 1.0,
        "intraday_move":   (c - o) / o.replace(0, np.nan),
        "high_low_ratio":  h / l.replace(0, np.nan) - 1.0,
        "price_accel":     c.diff().diff() / c.replace(0, np.nan),
        # Structure / signals
        "amihud":          (returns_abs / v.replace(0, np.nan)).rolling(20).mean() * 1e6,
        "bb_squeeze":      ((bb_up <= kc_upper) & (bb_lo >= kc_lower)).astype(float),
        "macd_crossover":  np.sign(hist_line.diff()).fillna(0.0),
        "stoch_crossover": np.sign((sk - sd).diff()).fillna(0.0),
        "rsi_slope_5":     rolling_slope(out["rsi_14"], 5),
        "adx_slope_5":     rolling_slope(adx_s, 5),
        # Candlestick patterns
        **{col: cp[col] for col in cp.columns},
    })

    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


# Features selected for model input (no raw prices, only normalized/relative values)
MODEL_FEATURE_COLS = [
    # ── Original 78 ───────────────────────────────────────────────────────────
    "rsi_14", "rsi_28",
    "macd", "macd_signal", "macd_hist_norm",
    "stoch_k", "stoch_d",
    "williams_r",
    "roc_10", "roc_20",
    "bb_width", "bb_pct",
    "adx", "plus_di", "minus_di", "di_diff",
    "atr_pct",
    "psar_gap",
    "ichimoku_tenkan_gap", "ichimoku_kijun_gap", "ichimoku_senkou_a_gap",
    "ichimoku_senkou_b_gap", "ichimoku_chikou_gap",
    "supertrend_gap", "supertrend_dir",
    "keltner_width", "keltner_pct",
    "donchian_width", "donchian_pct",
    "close_vs_sma5", "close_vs_sma10", "close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
    "close_vs_ema5", "close_vs_ema10", "close_vs_ema20", "close_vs_ema50", "close_vs_ema200",
    "hvol_10", "hvol_20", "hvol_60",
    "cci_20", "cmo_14", "trix_30", "ppo", "bop", "aroon_osc_14",
    "elder_bull_power", "elder_bear_power", "chaikin_osc", "force_index_13",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "rel_volume",
    "mfi", "obv_slope_10",
    "close_vs_vwap",
    "classic_pivot_gap", "fib_r1_gap", "camarilla_r1_gap",
    "fib_382_gap", "fib_618_gap", "fib_range_pos",
    "rank_52w", "dist_52w_high", "dist_52w_low",
    "relative_strength_20d", "sector_relative_strength_20d",
    "delivery_pct_feature", "put_call_ratio_feature",
    "body", "upper_shadow", "lower_shadow", "daily_range",

    # ── New 62: momentum ──────────────────────────────────────────────────────
    "rsi_7", "rsi_21",
    "stochrsi_k", "stochrsi_d",
    "uo",
    "ao_norm",
    "dpo_20_norm",
    "kst_norm", "kst_signal_norm",
    "mass_index_norm",
    "roc_1", "roc_5",

    # ── New 62: trend / MA ────────────────────────────────────────────────────
    "close_vs_sma3", "close_vs_ema3", "close_vs_sma100",
    "close_vs_dema20", "close_vs_tema20", "close_vs_hma20", "close_vs_kama",
    "vi_plus", "vi_minus", "vi_diff",
    "lin_reg_slope_20",

    # ── New 62: volatility ────────────────────────────────────────────────────
    "ulcer_14",
    "chaikin_vol_10",
    "hvol_5", "hvol_120",
    "hvol_ratio",
    "natr_7", "natr_21",
    "atr_ratio",

    # ── New 62: volume ────────────────────────────────────────────────────────
    "cmf_20",
    "vol_roc_10",
    "emv_norm",
    "klinger_norm",
    "nvi_slope",
    "vpt_slope",
    "vol_osc",
    "vol_trend",
    "volume_surge",

    # ── New 62: returns / price ───────────────────────────────────────────────
    "ret_2d", "ret_3d", "ret_30d", "ret_120d",
    "gap_norm",
    "intraday_move",
    "high_low_ratio",
    "price_accel",

    # ── New 62: structure / signals ───────────────────────────────────────────
    "amihud",
    "bb_squeeze",
    "macd_crossover",
    "stoch_crossover",
    "rsi_slope_5",
    "adx_slope_5",

    # ── New 62: candlestick patterns ──────────────────────────────────────────
    "candle_doji",
    "candle_hammer",
    "candle_inv_hammer",
    "candle_engulf_bull",
    "candle_engulf_bear",
    "candle_shooting_star",
    "candle_morning_star",
    "candle_spinning_top",
]

N_MODEL_FEATURES = len(MODEL_FEATURE_COLS)   # must equal 140
