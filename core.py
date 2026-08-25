from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourcePolicy:
    min_tier: int = 2
    min_freshness: float = 0.70
    max_age_days_price: int = 3
    max_age_days_disclosure: int = 120


@dataclass(frozen=True)
class ModelConfig:
    horizon_days: int = 20
    min_train_rows: int = 504
    test_rows: int = 126
    buy_probability: float = 0.62
    sell_probability: float = 0.38
    transaction_cost_bps: float = 12.0
    annualization: int = 252


SOURCE_TIERS = {
    "JPX/J-Quants": 1,
    "TDnet": 1,
    "EDINET/FSA": 1,
    "BOJ": 1,
    "e-Stat": 1,
    "FRED/ALFRED": 1,
    "Issuer IR": 1,
    "Reuters": 2,
    "Nikkei": 2,
}


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast = s.ewm(span=12, adjust=False).mean()
    slow = s.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def max_drawdown_from_returns(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_index()
    close = x["Close"].astype(float)
    volume = x["Volume"].astype(float)
    for n in [5, 10, 20, 50, 100, 200]:
        x[f"ret_{n}"] = close.pct_change(n)
    for n in [20, 50, 100, 200]:
        x[f"sma_{n}"] = close.rolling(n).mean()
        x[f"dist_sma_{n}"] = close / x[f"sma_{n}"] - 1
    x["rsi_14"] = rsi(close)
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(close)
    x["atr_14"] = atr(x)
    x["atr_pct"] = x["atr_14"] / close
    x["vol_20"] = close.pct_change().rolling(20).std() * math.sqrt(252)
    x["vol_60"] = close.pct_change().rolling(60).std() * math.sqrt(252)
    x["volume_ratio_20"] = volume / volume.rolling(20).mean()
    x["turnover_proxy"] = close * volume
    x["turnover_ratio_20"] = x["turnover_proxy"] / x["turnover_proxy"].rolling(20).mean()
    x["high_252"] = close.rolling(252).max()
    x["low_252"] = close.rolling(252).min()
    x["pct_from_52w_high"] = close / x["high_252"] - 1
    x["pct_from_52w_low"] = close / x["low_252"] - 1
    x["trend_alignment"] = (
        (close > x["sma_200"]).astype(float)
        + (x["sma_20"] > x["sma_50"]).astype(float)
        + (x["sma_50"] > x["sma_100"]).astype(float)
        + (x["sma_100"] > x["sma_200"]).astype(float)
    ) / 4.0
    return x


def merge_asof_feature(base: pd.DataFrame, feature: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if feature is None or feature.empty:
        return base
    b = base.reset_index().rename(columns={base.index.name or "index": "Date"})
    f = feature.copy().reset_index().rename(columns={feature.index.name or "index": "Date"})
    b["Date"] = pd.to_datetime(b["Date"])
    f["Date"] = pd.to_datetime(f["Date"])
    keep = [c for c in f.columns if c != "Date"]
    f = f.rename(columns={c: f"{prefix}_{c}" for c in keep})
    out = pd.merge_asof(b.sort_values("Date"), f.sort_values("Date"), on="Date", direction="backward")
    return out.set_index("Date")


def add_fundamental_features(price_features: pd.DataFrame, fundamentals: Optional[pd.DataFrame]) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return price_features
    f = fundamentals.copy().sort_index()
    numeric = f.select_dtypes(include=[np.number]).columns
    for c in numeric:
        f[f"{c}_yoy"] = f[c].pct_change(4).replace([np.inf, -np.inf], np.nan)
    return merge_asof_feature(price_features, f, "fund")


def add_macro_features(df: pd.DataFrame, macro: Optional[pd.DataFrame]) -> pd.DataFrame:
    if macro is None or macro.empty:
        return df
    m = macro.copy().sort_index()
    for c in m.columns:
        if pd.api.types.is_numeric_dtype(m[c]):
            m[f"{c}_chg20"] = m[c].pct_change(20).replace([np.inf, -np.inf], np.nan)
            m[f"{c}_z252"] = (m[c] - m[c].rolling(252).mean()) / m[c].rolling(252).std()
    return merge_asof_feature(df, m, "macro")


def make_supervised(features: pd.DataFrame, horizon_days: int = 20) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = features.copy()
    future_return = x["Close"].shift(-horizon_days) / x["Close"] - 1
    y = (future_return > 0).astype(int)
    excluded = {"Open", "High", "Low", "Close", "Volume", "future_return", "target"}
    cols = [c for c in x.columns if c not in excluded and pd.api.types.is_numeric_dtype(x[c])]
    X = x[cols].replace([np.inf, -np.inf], np.nan)
    valid = future_return.notna()
    return X.loc[valid], y.loc[valid], future_return.loc[valid]


def deterministic_score(row: pd.Series) -> Dict[str, float]:
    trend = clamp((float(row.get("trend_alignment", 0.5)) - 0.5) * 2)
    mom = np.nanmean([
        clamp(float(row.get("ret_20", 0) or 0) * 5),
        clamp(float(row.get("ret_60", 0) or 0) * 2.5),
        clamp(float(row.get("macd_hist", 0) or 0) / max(abs(float(row.get("Close", 1))) * 0.01, 1e-9)),
    ])
    r = float(row.get("rsi_14", 50) or 50)
    rsi_component = 0.5 if 50 <= r <= 68 else (-0.4 if r > 75 or r < 30 else 0.0)
    momentum = clamp(0.75 * float(mom) + 0.25 * rsi_component)
    volume = clamp((float(row.get("turnover_ratio_20", 1) or 1) - 1) / 1.5)
    vol = float(row.get("atr_pct", 0.03) or 0.03)
    risk = clamp((0.045 - vol) / 0.045)
    total = 100 * (0.38 * trend + 0.32 * momentum + 0.12 * volume + 0.18 * risk)
    return {"trend": trend * 100, "momentum": momentum * 100, "volume": volume * 100, "risk": risk * 100, "technical_total": total}


def signal_from_probability(p_up: float, confidence: float, score: float, cfg: ModelConfig) -> str:
    # Require model probability and deterministic trend to broadly agree.
    if confidence < 0.55:
        return "様子見（信頼度不足）"
    if p_up >= cfg.buy_probability and score >= 15:
        return "買い候補"
    if p_up <= cfg.sell_probability and score <= -15:
        return "売り・縮小候補"
    return "様子見"


def backtest_from_probabilities(
    close: pd.Series,
    probabilities: pd.Series,
    cfg: ModelConfig,
) -> pd.DataFrame:
    d = pd.DataFrame({"Close": close, "p_up": probabilities}).dropna().copy()
    position = 0
    pos = []
    for p in d["p_up"]:
        if position == 0 and p >= cfg.buy_probability:
            position = 1
        elif position == 1 and p <= cfg.sell_probability:
            position = 0
        pos.append(position)
    d["Position"] = pd.Series(pos, index=d.index).shift(1).fillna(0)
    d["AssetRet"] = d["Close"].pct_change().fillna(0)
    turnover = d["Position"].diff().abs().fillna(d["Position"].abs())
    d["StrategyRet"] = d["Position"] * d["AssetRet"] - turnover * cfg.transaction_cost_bps / 10000
    d["StrategyEquity"] = (1 + d["StrategyRet"]).cumprod()
    d["BuyHoldEquity"] = (1 + d["AssetRet"]).cumprod()
    return d


def performance_metrics(bt: pd.DataFrame, annualization: int = 252) -> Dict[str, float]:
    if bt.empty:
        return {}
    r = bt["StrategyRet"].fillna(0)
    total = float(bt["StrategyEquity"].iloc[-1] - 1)
    years = max(len(bt) / annualization, 1 / annualization)
    cagr = float(bt["StrategyEquity"].iloc[-1] ** (1 / years) - 1)
    sharpe = float(np.sqrt(annualization) * r.mean() / r.std()) if r.std() > 0 else float("nan")
    maxdd = max_drawdown_from_returns(r)
    bh = float(bt["BuyHoldEquity"].iloc[-1] - 1)
    return {"total": total, "cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "buy_hold": bh}


def _score_linear(value: float, low: float, high: float) -> float:
    """Map value to 0..100, clipping outside the interval."""
    if value is None or not np.isfinite(value):
        return 50.0
    if high == low:
        return 50.0
    return float(np.clip((value - low) / (high - low) * 100.0, 0.0, 100.0))


def buy_score_components(
    row: pd.Series,
    p_up: float,
    model_conf: float,
    source_conf: float,
    data_coverage: float,
    event_risk: bool = False,
    supply_summary: Optional[Dict[str, float]] = None,
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return a transparent 0..100 buy-candidate score and weighted components.

    The score is intentionally not a direct probability. It combines price action,
    relative strength, participation, available fundamentals, model evidence and
    source quality. Missing optional inputs are neutral rather than silently bullish.
    """
    # 1) Trend / structure (max 22)
    align = float(row.get("trend_alignment", 0.5) or 0.5)
    d20 = float(row.get("dist_sma_20", 0) or 0)
    d50 = float(row.get("dist_sma_50", 0) or 0)
    d200 = float(row.get("dist_sma_200", 0) or 0)
    trend_quality = 0.55 * (align * 100.0)
    trend_quality += 0.15 * _score_linear(d20, -0.08, 0.08)
    trend_quality += 0.15 * _score_linear(d50, -0.12, 0.15)
    trend_quality += 0.15 * _score_linear(d200, -0.20, 0.30)
    trend_quality = float(np.clip(trend_quality, 0, 100))

    # 2) Momentum / early-stage quality (max 13). Penalize excessive RSI.
    r20 = float(row.get("ret_20", 0) or 0)
    r60 = float(row.get("ret_60", 0) or 0)
    rsi_v = float(row.get("rsi_14", 50) or 50)
    macd_hist = float(row.get("macd_hist", 0) or 0)
    close_v = max(abs(float(row.get("Close", 1) or 1)), 1e-9)
    momentum_quality = (
        0.35 * _score_linear(r20, -0.12, 0.18)
        + 0.30 * _score_linear(r60, -0.20, 0.35)
        + 0.20 * _score_linear(macd_hist / close_v, -0.02, 0.02)
    )
    if 52 <= rsi_v <= 68:
        rsi_q = 90.0
    elif 45 <= rsi_v < 52:
        rsi_q = 68.0
    elif 68 < rsi_v <= 74:
        rsi_q = 62.0
    elif rsi_v > 80:
        rsi_q = 18.0
    elif rsi_v > 74:
        rsi_q = 35.0
    elif 30 <= rsi_v < 45:
        rsi_q = 38.0
    else:
        rsi_q = 22.0
    momentum_quality = float(np.clip(momentum_quality + 0.15 * rsi_q, 0, 100))

    # 3) Volume / money participation (max 12)
    volume_ratio = float(row.get("volume_ratio_20", 1) or 1)
    turnover_ratio = float(row.get("turnover_ratio_20", 1) or 1)
    participation = 0.45 * _score_linear(volume_ratio, 0.65, 2.0) + 0.55 * _score_linear(turnover_ratio, 0.65, 2.0)

    # 4) Relative strength / market context (max 8)
    rs20 = float(row.get("relative_strength_20", 0) or 0)
    rs60 = float(row.get("relative_strength_60", 0) or 0)
    market_ret20 = float(row.get("market_ret_20", 0) or 0)
    relative_strength = 0.55 * _score_linear(rs20, -0.12, 0.18) + 0.45 * _score_linear(rs60, -0.20, 0.30)
    market_context = _score_linear(market_ret20, -0.12, 0.12)
    relative_market = 0.8 * relative_strength + 0.2 * market_context

    # 5) Fundamental evidence (max 15). Only use fields that actually exist.
    candidate_fields = [
        "fund_NetSales_yoy", "fund_OperatingProfit_yoy", "fund_OrdinaryProfit_yoy",
        "fund_Profit_yoy", "fund_EarningsPerShare_yoy", "fund_ForecastNetSales_yoy",
        "fund_ForecastOperatingProfit_yoy", "fund_ForecastProfit_yoy",
        "fund_ForecastEarningsPerShare_yoy",
    ]
    fundamental_items = []
    for c in candidate_fields:
        v = row.get(c, np.nan)
        if pd.notna(v) and np.isfinite(float(v)):
            fundamental_items.append(_score_linear(float(v), -0.30, 0.40))
    fundamental = float(np.mean(fundamental_items)) if fundamental_items else 50.0

    # 6) Risk / entry quality (max 10). Favor tradable volatility, not extreme calm or panic.
    atr_pct = float(row.get("atr_pct", 0.03) or 0.03)
    from_high = float(row.get("pct_from_52w_high", -0.10) or -0.10)
    vol20 = float(row.get("vol_20", 0.30) or 0.30)
    atr_quality = 100.0 - abs(_score_linear(atr_pct, 0.005, 0.09) - 42.0) * 1.25
    vol_quality = 100.0 - abs(_score_linear(vol20, 0.08, 0.80) - 40.0) * 1.1
    high_quality = 85.0 if -0.12 <= from_high <= -0.01 else (55.0 if -0.25 <= from_high < -0.12 else 35.0)
    entry_quality = float(np.clip(0.35 * atr_quality + 0.25 * vol_quality + 0.40 * high_quality, 0, 100))

    # 7) ML forecast quality (max 12): probability is discounted by model confidence.
    p_quality = _score_linear(float(p_up), 0.42, 0.78)
    ml_quality = float(np.clip(0.65 * p_quality + 0.35 * (model_conf * 100.0), 0, 100))

    # 8) Source / coverage quality (max 8)
    evidence_quality = float(np.clip(0.55 * source_conf * 100.0 + 0.45 * data_coverage * 100.0, 0, 100))

    # 9) Official supply/demand evidence (max 12). Missing evidence stays neutral, never bullish.
    supply_summary = supply_summary or {}
    supply_quality_raw = float(supply_summary.get("supply_demand_quality", 50.0) or 50.0)
    supply_coverage = float(supply_summary.get("supply_demand_coverage", 0.0) or 0.0)
    supply_coverage = float(np.clip(supply_coverage, 0.0, 1.0))
    supply_quality = 50.0 + (float(np.clip(supply_quality_raw, 0, 100)) - 50.0) * supply_coverage

    # Rebalanced to keep the headline score exactly 100 points while adding supply/demand.
    baseline_weights = {
        "trend": 20.0,
        "momentum": 12.0,
        "participation": 10.0,
        "relative_market": 7.0,
        "fundamental": 13.0,
        "entry_quality": 8.0,
        "ml": 10.0,
        "evidence": 8.0,
        "supply_demand": 12.0,
    }
    weights = baseline_weights.copy()
    if custom_weights:
        clean = {k: max(0.0, float(custom_weights.get(k, baseline_weights[k]))) for k in baseline_weights}
        total_w = sum(clean.values())
        if total_w > 0:
            weights = {k: 100.0 * v / total_w for k, v in clean.items()}
    quality = {
        "trend": trend_quality,
        "momentum": momentum_quality,
        "participation": participation,
        "relative_market": relative_market,
        "fundamental": fundamental,
        "entry_quality": entry_quality,
        "ml": ml_quality,
        "evidence": evidence_quality,
        "supply_demand": supply_quality,
    }
    points = {k: quality[k] / 100.0 * weights[k] for k in weights}
    total = float(sum(points.values()))

    # Guardrails: uncertainty or event risk caps the headline score.
    if model_conf < 0.45:
        total = min(total, 64.0)
    elif model_conf < 0.55:
        total = min(total, 72.0)
    if source_conf < 0.55 or data_coverage < 0.60:
        total = min(total, 69.0)
    if event_risk:
        total = min(total, 74.0)

    out: Dict[str, float] = {"buy_score": round(total, 1)}
    for k, v in points.items():
        out[f"points_{k}"] = round(float(v), 2)
        out[f"quality_{k}"] = round(float(quality[k]), 1)
    return out


def classify_market_phase(row: pd.Series, buy_score: float, score_change: float = 0.0) -> str:
    """Classify the current setup into a practical action-oriented phase."""
    align = float(row.get("trend_alignment", 0.5) or 0.5)
    rsi_v = float(row.get("rsi_14", 50) or 50)
    ret20 = float(row.get("ret_20", 0) or 0)
    ret60 = float(row.get("ret_60", 0) or 0)
    d20 = float(row.get("dist_sma_20", 0) or 0)
    d50 = float(row.get("dist_sma_50", 0) or 0)
    d200 = float(row.get("dist_sma_200", 0) or 0)
    from_high = float(row.get("pct_from_52w_high", -0.1) or -0.1)
    turnover = float(row.get("turnover_ratio_20", 1) or 1)

    if rsi_v >= 78 or (from_high > -0.015 and ret20 > 0.15):
        return "過熱警戒"
    if align <= 0.25 and d200 < -0.05 and ret60 < -0.10:
        return "下落基調"
    if align <= 0.50 and ret20 > 0.03 and score_change >= 8 and turnover >= 1.15:
        return "底打ち候補"
    if buy_score >= 76 and score_change >= 8 and d200 > 0 and rsi_v < 72 and turnover >= 1.05:
        return "初動候補"
    if align >= 0.75 and d50 > -0.02 and -0.06 <= d20 <= 0.01 and 45 <= rsi_v <= 62:
        return "押し目候補"
    if buy_score >= 78 and align >= 0.75 and ret20 > 0 and ret60 > 0:
        return "上昇継続"
    if buy_score < 55 and d20 < 0 and ret20 < 0:
        return "売り警戒"
    return "中立・監視"


def buy_score_label(score: float) -> str:
    if score >= 90:
        return "S｜最有力買い候補"
    if score >= 80:
        return "A｜買い候補"
    if score >= 70:
        return "B｜監視強化"
    if score >= 60:
        return "C｜様子見"
    if score >= 45:
        return "D｜弱い"
    return "E｜買い対象外"
