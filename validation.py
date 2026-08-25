from __future__ import annotations

from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

from supply_demand import summarize_supply_demand


def _linear(s, low, high):
    if isinstance(s, pd.Series):
        return ((s - low) / (high - low) * 100).clip(0, 100)
    return float(np.clip((s - low) / (high - low) * 100, 0, 100))


def _rsi14(s: pd.Series) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def historical_price_radar(px: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Point-in-time price radar history. Every row uses only information available on that date."""
    d = px.copy().sort_index()
    d.index = pd.to_datetime(d.index)
    close = pd.to_numeric(d["Close"], errors="coerce")
    volume = pd.to_numeric(d.get("Volume", 0), errors="coerce").fillna(0)
    turnover = close * volume
    for n in [20, 50, 200]:
        d[f"sma{n}"] = close.rolling(n).mean()
    d["ret20"] = close.pct_change(20)
    d["ret60"] = close.pct_change(60)
    d["rsi14"] = _rsi14(close)
    d["volume_ratio20"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    d["turnover_ratio20"] = turnover / turnover.rolling(20).mean().replace(0, np.nan)
    d["high252"] = close.rolling(252, min_periods=180).max()
    d["from_high"] = close / d["high252"] - 1

    if benchmark is not None and not benchmark.empty:
        b = pd.to_numeric(benchmark["Close"], errors="coerce").sort_index()
        b.index = pd.to_datetime(b.index)
        d["bm_ret20"] = b.pct_change(20).reindex(d.index, method="ffill")
        d["bm_ret60"] = b.pct_change(60).reindex(d.index, method="ffill")
    else:
        d["bm_ret20"] = 0.0
        d["bm_ret60"] = 0.0
    d["rs20"] = d["ret20"] - d["bm_ret20"].fillna(0)
    d["rs60"] = d["ret60"] - d["bm_ret60"].fillna(0)

    trend = (
        0.30 * _linear((close / d["sma20"] - 1).fillna(-0.2), -0.10, 0.12)
        + 0.25 * _linear((d["sma20"] / d["sma50"] - 1).fillna(-0.2), -0.08, 0.10)
        + 0.25 * _linear((d["sma50"] / d["sma200"] - 1).fillna(-0.2), -0.12, 0.18)
        + 0.20 * _linear((close / d["sma200"] - 1).fillna(-0.3), -0.18, 0.28)
    )
    momentum = 0.45 * _linear(d["ret20"].fillna(-0.3), -0.18, 0.25) + 0.35 * _linear(d["ret60"].fillna(-0.4), -0.30, 0.50)
    rsi = d["rsi14"].fillna(50)
    rsi_q = np.where(rsi < 35, 20, np.where(rsi < 50, 45 + (rsi - 35) * 2.2, np.where(rsi <= 68, 78 + (rsi - 50) * 1.2, np.where(rsi <= 78, 100 - (rsi - 68) * 4.0, 35))))
    momentum = 0.80 * momentum + 0.20 * rsi_q
    participation = 0.45 * _linear(d["volume_ratio20"].fillna(0.5), 0.65, 2.2) + 0.55 * _linear(d["turnover_ratio20"].fillna(0.5), 0.65, 2.2)
    relative = 0.55 * _linear(d["rs20"].fillna(-0.2), -0.12, 0.18) + 0.45 * _linear(d["rs60"].fillna(-0.3), -0.20, 0.30)
    fh = d["from_high"].fillna(-0.5)
    entry = np.where((fh >= -0.12) & (fh <= -0.01), 90, np.where((fh >= -0.25) & (fh < -0.12), 65, np.where(fh > -0.01, 42, 35)))
    overheat = np.where((rsi >= 78) | ((fh > -0.01) & (d["ret20"] > 0.18)), 18.0, 0.0)
    d["radar_score"] = np.clip(0.34 * trend + 0.23 * momentum + 0.18 * participation + 0.15 * relative + 0.10 * entry - overheat, 0, 100)
    d["radar_change"] = d["radar_score"].diff()
    return d[["Close", "ret20", "ret60", "rsi14", "turnover_ratio20", "from_high", "radar_score", "radar_change"]]


def _dated_subset(df: Optional[pd.DataFrame], dt: pd.Timestamp, candidates: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    col = next((c for c in candidates if c in x.columns), None)
    if col is None:
        return x
    dates = pd.to_datetime(x[col], errors="coerce")
    return x.loc[dates <= dt]


def historical_supply_series(bundle: Dict[str, pd.DataFrame], dates: Iterable[pd.Timestamp], min_spacing_days: int = 1) -> pd.DataFrame:
    """Create point-in-time supply scores by truncating every official dataset at each historical date."""
    rows = []
    last_dt = None
    for raw_dt in pd.DatetimeIndex(dates):
        dt = pd.Timestamp(raw_dt)
        if last_dt is not None and (dt - last_dt).days < min_spacing_days:
            continue
        s = summarize_supply_demand(
            breakdown=_dated_subset(bundle.get("breakdown"), dt, ["Date"]),
            daily_margin=_dated_subset(bundle.get("daily_margin"), dt, ["ApplicationDate", "PublishedDate"]),
            weekly_margin=_dated_subset(bundle.get("weekly_margin"), dt, ["Date"]),
            shorts=_dated_subset(bundle.get("shorts"), dt, ["CalculatedDate", "DisclosedDate"]),
            buyback_tdnet=_dated_subset(bundle.get("buyback_tdnet"), dt, ["DisclosedDate", "PurchaseDate"]),
            buyback_edinet=_dated_subset(bundle.get("buyback_edinet"), dt, ["SubmittedDate"]),
            offauction_buyback=_dated_subset(bundle.get("offauction_buyback"), dt, ["ImplementationDate", "PublicationDate"]),
        )
        rows.append({"Date": dt, **s})
        last_dt = dt
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Date").sort_index()


def make_signal_history(
    px: pd.DataFrame,
    benchmark: Optional[pd.DataFrame],
    bundle: Optional[Dict[str, pd.DataFrame]] = None,
    radar_min: float = 65,
    supply_min: float = 70,
    supply_coverage_min: float = 0.35,
    max_ret20: float = 0.12,
    max_rsi: float = 76,
) -> pd.DataFrame:
    h = historical_price_radar(px, benchmark)
    if bundle:
        # Supply endpoints are usually sparse; recent trading dates are enough for point-in-time validation.
        s = historical_supply_series(bundle, h.index[-320:])
        h = h.join(s[[c for c in ["supply_demand_quality", "supply_demand_coverage", "supply_demand_signal"] if c in s.columns]], how="left")
    if "supply_demand_quality" not in h:
        h["supply_demand_quality"] = 50.0
        h["supply_demand_coverage"] = 0.0
    h["supply_demand_quality"] = pd.to_numeric(h["supply_demand_quality"], errors="coerce").fillna(50)
    h["supply_demand_coverage"] = pd.to_numeric(h["supply_demand_coverage"], errors="coerce").fillna(0)
    # If supply is unavailable, the validator can still test the price radar. If available, require it.
    has_supply = h["supply_demand_coverage"].max() >= supply_coverage_min
    condition = (h["radar_score"] >= radar_min) & (h["ret20"].fillna(9) <= max_ret20) & (h["rsi14"].fillna(100) <= max_rsi)
    if has_supply:
        condition &= (h["supply_demand_quality"] >= supply_min) & (h["supply_demand_coverage"] >= supply_coverage_min)
    h["signal"] = condition
    h["validation_mode"] = "価格＋需給" if has_supply else "価格のみ（需給履歴不足）"
    return h


def event_study(signals: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None, horizons=(5, 20, 60), cooldown: int = 5):
    """Evaluate distinct signal events without using future information in signal construction."""
    if signals is None or signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = signals.copy().sort_index()
    candidates = d.index[d["signal"].fillna(False)]
    selected = []
    last_i = -10**9
    positions = {dt: i for i, dt in enumerate(d.index)}
    for dt in candidates:
        i = positions[dt]
        if i - last_i >= cooldown:
            selected.append(dt)
            last_i = i
    bm_close = None
    if benchmark is not None and not benchmark.empty:
        bm_close = pd.to_numeric(benchmark["Close"], errors="coerce").sort_index()
        bm_close.index = pd.to_datetime(bm_close.index)

    events = []
    for dt in selected:
        i = positions[dt]
        row = {"Date": dt, "radar_score": d.loc[dt, "radar_score"], "supply_score": d.loc[dt, "supply_demand_quality"], "ret20_at_signal": d.loc[dt, "ret20"]}
        for h in horizons:
            if i + h >= len(d):
                row[f"ret_{h}"] = np.nan
                row[f"excess_{h}"] = np.nan
                row[f"mae_{h}"] = np.nan
                continue
            entry = float(d["Close"].iloc[i])
            exit_ = float(d["Close"].iloc[i + h])
            fwd = exit_ / entry - 1
            window = d["Close"].iloc[i + 1:i + h + 1] / entry - 1
            row[f"ret_{h}"] = fwd
            row[f"mae_{h}"] = float(window.min()) if len(window) else np.nan
            if bm_close is not None:
                b = bm_close.reindex(d.index, method="ffill")
                if pd.notna(b.iloc[i]) and pd.notna(b.iloc[i + h]):
                    row[f"excess_{h}"] = fwd - (float(b.iloc[i + h]) / float(b.iloc[i]) - 1)
                else:
                    row[f"excess_{h}"] = np.nan
            else:
                row[f"excess_{h}"] = np.nan
        events.append(row)
    ev = pd.DataFrame(events)
    if ev.empty:
        return ev, pd.DataFrame()
    metrics = []
    for h in horizons:
        r = pd.to_numeric(ev[f"ret_{h}"], errors="coerce").dropna()
        ex = pd.to_numeric(ev[f"excess_{h}"], errors="coerce").dropna()
        mae = pd.to_numeric(ev[f"mae_{h}"], errors="coerce").dropna()
        metrics.append({
            "期間": f"{h}営業日後",
            "有効件数": int(len(r)),
            "勝率": float((r > 0).mean()) if len(r) else np.nan,
            "平均騰落率": float(r.mean()) if len(r) else np.nan,
            "中央値": float(r.median()) if len(r) else np.nan,
            "平均超過リターン": float(ex.mean()) if len(ex) else np.nan,
            "平均最大不利変動": float(mae.mean()) if len(mae) else np.nan,
        })
    return ev, pd.DataFrame(metrics)


def validation_grade(metrics: pd.DataFrame) -> Dict[str, object]:
    if metrics is None or metrics.empty:
        return {"grade": "未判定", "score": 0, "reason": "検証イベントがありません"}
    row20 = metrics[metrics["期間"] == "20営業日後"]
    row = row20.iloc[0] if not row20.empty else metrics.iloc[0]
    n = int(row["有効件数"])
    win = float(row["勝率"]) if pd.notna(row["勝率"]) else 0
    avg = float(row["平均騰落率"]) if pd.notna(row["平均騰落率"]) else 0
    excess = float(row["平均超過リターン"]) if pd.notna(row["平均超過リターン"]) else 0
    sample_q = min(1.0, n / 30)
    score = 100 * (0.30 * sample_q + 0.30 * np.clip((win - 0.45) / 0.25, 0, 1) + 0.25 * np.clip((avg + 0.01) / 0.10, 0, 1) + 0.15 * np.clip((excess + 0.01) / 0.08, 0, 1))
    if score >= 75 and n >= 20:
        grade = "A 有効性高め"
    elif score >= 60 and n >= 12:
        grade = "B 有効性あり"
    elif score >= 45:
        grade = "C 参考レベル"
    else:
        grade = "D 根拠不足"
    return {"grade": grade, "score": round(float(score), 1), "reason": f"20日基準: n={n}, 勝率={win:.1%}, 平均={avg:.1%}, 超過={excess:.1%}"}
