from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScanConfig:
    lookback_calendar_days: int = 390
    min_price_rows: int = 205
    min_turnover_yen: float = 50_000_000.0
    cache_dir: str = ".scanner_cache"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _linear(x, lo, hi):
    if hi == lo:
        return np.zeros_like(x, dtype=float)
    return 100.0 * _clip01((x - lo) / (hi - lo))


def filter_common_equities(listed: pd.DataFrame) -> pd.DataFrame:
    """Keep TSE Prime/Standard/Growth common-equity universe as robustly as fields allow."""
    if listed is None or listed.empty:
        return pd.DataFrame()
    d = listed.copy()
    for c in ["Code", "CompanyName", "MarketCodeName", "Sector33CodeName", "ScaleCategory"]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str)

    market = d["MarketCodeName"]
    ok_market = market.str.contains("プライム|スタンダード|グロース|Prime|Standard|Growth", case=False, regex=True)
    # Exclude obvious non-common-stock instruments when names expose them.
    name = d["CompanyName"]
    excluded = name.str.contains("ETF|ＥＴＦ|REIT|リート|投資法人|ETN|上場投信|インフラファンド", case=False, regex=True)
    d = d[ok_market & ~excluded].copy()
    return d.drop_duplicates("Code")


def fast_market_features(panel: pd.DataFrame, benchmark_code: str = "1306") -> pd.DataFrame:
    """Vectorized first-pass features for thousands of stocks.

    Input columns: Date, Code, Close, Volume, optional TurnoverValue.
    Returns one latest row per Code with a 0-100 radar score and day-over-day radar change.
    This is a discovery score, not the final 100-point score.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    d = panel.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Code"] = d["Code"].astype(str)
    d["Close"] = _num(d["Close"])
    d["Volume"] = _num(d.get("Volume", pd.Series(index=d.index, dtype=float)))
    if "TurnoverValue" in d.columns:
        d["TurnoverValue"] = _num(d["TurnoverValue"])
    else:
        d["TurnoverValue"] = d["Close"] * d["Volume"]
    d = d.dropna(subset=["Date", "Code", "Close"]).sort_values(["Code", "Date"])

    g = d.groupby("Code", group_keys=False)
    for n in [20, 50, 200]:
        d[f"sma{n}"] = g["Close"].transform(lambda s, n=n: s.rolling(n).mean())
    d["ret20"] = g["Close"].pct_change(20)
    d["ret60"] = g["Close"].pct_change(60)
    d["rsi14"] = g["Close"].transform(_rsi14)
    d["vol_avg20"] = g["Volume"].transform(lambda s: s.rolling(20).mean())
    d["turn_avg20"] = g["TurnoverValue"].transform(lambda s: s.rolling(20).mean())
    d["volume_ratio20"] = d["Volume"] / d["vol_avg20"].replace(0, np.nan)
    d["turnover_ratio20"] = d["TurnoverValue"] / d["turn_avg20"].replace(0, np.nan)
    d["high252"] = g["Close"].transform(lambda s: s.rolling(252, min_periods=180).max())
    d["from_high"] = d["Close"] / d["high252"] - 1.0

    # Benchmark 20d/60d returns are mapped by date.
    bm = d[d["Code"].str.startswith(str(benchmark_code)[:4])][["Date", "Close"]].drop_duplicates("Date").sort_values("Date")
    if not bm.empty:
        bm["bm_ret20"] = bm["Close"].pct_change(20)
        bm["bm_ret60"] = bm["Close"].pct_change(60)
        d = d.merge(bm[["Date", "bm_ret20", "bm_ret60"]], on="Date", how="left")
    else:
        d["bm_ret20"] = 0.0
        d["bm_ret60"] = 0.0
    d["rs20"] = d["ret20"] - d["bm_ret20"].fillna(0)
    d["rs60"] = d["ret60"] - d["bm_ret60"].fillna(0)

    trend = (
        0.30 * _linear((d["Close"] / d["sma20"] - 1.0).fillna(-0.2), -0.10, 0.12)
        + 0.25 * _linear((d["sma20"] / d["sma50"] - 1.0).fillna(-0.2), -0.08, 0.10)
        + 0.25 * _linear((d["sma50"] / d["sma200"] - 1.0).fillna(-0.2), -0.12, 0.18)
        + 0.20 * _linear((d["Close"] / d["sma200"] - 1.0).fillna(-0.3), -0.18, 0.28)
    )
    momentum = 0.45 * _linear(d["ret20"].fillna(-0.3), -0.18, 0.25) + 0.35 * _linear(d["ret60"].fillna(-0.4), -0.30, 0.50)
    # RSI sweet spot: strongest around 55-68, penalize extreme overbought.
    rsi = d["rsi14"].fillna(50)
    rsi_q = np.where(rsi < 35, 20, np.where(rsi < 50, 45 + (rsi - 35) * 2.2, np.where(rsi <= 68, 78 + (rsi - 50) * 1.2, np.where(rsi <= 78, 100 - (rsi - 68) * 4.0, 35))))
    momentum = 0.80 * momentum + 0.20 * rsi_q
    participation = 0.45 * _linear(d["volume_ratio20"].fillna(0.5), 0.65, 2.2) + 0.55 * _linear(d["turnover_ratio20"].fillna(0.5), 0.65, 2.2)
    relative = 0.55 * _linear(d["rs20"].fillna(-0.2), -0.12, 0.18) + 0.45 * _linear(d["rs60"].fillna(-0.3), -0.20, 0.30)
    # Entry quality: avoid vertical extensions at the 52w high; reward controlled proximity.
    fh = d["from_high"].fillna(-0.5)
    entry = np.where((fh >= -0.12) & (fh <= -0.01), 90, np.where((fh >= -0.25) & (fh < -0.12), 65, np.where(fh > -0.01, 42, 35)))
    overheat_penalty = np.where((rsi >= 78) | ((fh > -0.01) & (d["ret20"] > 0.18)), 18.0, 0.0)

    # Discovery radar score: intentionally excludes fundamentals/ML. Final score is computed only in deep analysis.
    d["radar_score"] = np.clip(
        0.34 * trend + 0.23 * momentum + 0.18 * participation + 0.15 * relative + 0.10 * entry - overheat_penalty,
        0, 100,
    )

    d["radar_change"] = d.groupby("Code")["radar_score"].diff(1)
    d["radar_change5"] = d.groupby("Code")["radar_score"].diff(5)
    d["rows"] = d.groupby("Code").cumcount() + 1

    latest = d.groupby("Code", as_index=False).tail(1).copy()
    latest["phase_hint"] = np.select(
        [
            (latest["radar_score"] >= 72) & (latest["radar_change"] >= 6) & (latest["turnover_ratio20"] >= 1.05) & (latest["rsi14"] < 75),
            (latest["sma20"] >= latest["sma50"]) & (latest["Close"] >= latest["sma200"]) & (latest["Close"] <= latest["sma20"] * 1.015) & (latest["rsi14"].between(43, 63)),
            (latest["radar_score"] >= 76) & (latest["ret20"] > 0) & (latest["ret60"] > 0),
            (latest["rsi14"] >= 78) | ((latest["from_high"] > -0.01) & (latest["ret20"] > 0.18)),
        ],
        ["初動レーダー", "押し目レーダー", "上昇継続レーダー", "過熱警戒"],
        default="監視",
    )
    return latest.sort_values(["radar_score", "radar_change"], ascending=[False, False])


class MarketSnapshotStore:
    """Disk cache for official all-market daily snapshots.

    CSV is used instead of parquet to avoid extra optional dependencies.
    """

    def __init__(self, root: str = ".scanner_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dt: date) -> Path:
        return self.root / f"quotes_{dt.isoformat()}.csv.gz"

    def load(self, dt: date) -> Optional[pd.DataFrame]:
        p = self.path_for(dt)
        if not p.exists():
            return None
        try:
            return pd.read_csv(p, dtype={"Code": str})
        except Exception:
            return None

    def save(self, dt: date, df: pd.DataFrame) -> None:
        if df is not None and not df.empty:
            df.to_csv(self.path_for(dt), index=False, compression="gzip")


def collect_market_panel(
    fetch_date: Callable[[str], pd.DataFrame],
    end_date: date,
    config: ScanConfig,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Collect/cache all-market daily snapshots over a calendar lookback.

    Fetches weekdays only. Holidays return empty and are ignored. Cached dates are never re-fetched.
    """
    store = MarketSnapshotStore(config.cache_dir)
    start_date = end_date - timedelta(days=config.lookback_calendar_days)
    dates = pd.date_range(start_date, end_date, freq="B").date.tolist()
    frames: List[pd.DataFrame] = []
    fetched = cached = empty = failed = 0
    total = max(len(dates), 1)

    for i, dt in enumerate(dates):
        if progress_cb:
            progress_cb((i + 1) / total, f"全市場データ {dt.isoformat()} ({i+1}/{total})")
        d = store.load(dt)
        if d is not None:
            cached += 1
        else:
            try:
                d = fetch_date(dt.isoformat())
                if d is None or d.empty:
                    empty += 1
                    continue
                store.save(dt, d)
                fetched += 1
                time.sleep(0.03)
            except Exception:
                failed += 1
                continue
        if not d.empty:
            frames.append(d)

    if not frames:
        return pd.DataFrame(), {"fetched": fetched, "cached": cached, "empty": empty, "failed": failed}
    panel = pd.concat(frames, ignore_index=True)
    return panel, {"fetched": fetched, "cached": cached, "empty": empty, "failed": failed}


def enrich_with_listed(radar: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    if radar is None or radar.empty:
        return pd.DataFrame()
    if listed is None or listed.empty:
        return radar
    cols = [c for c in ["Code", "CompanyName", "MarketCodeName", "Sector33CodeName", "ScaleCategory"] if c in listed.columns]
    meta = listed[cols].drop_duplicates("Code").copy()
    meta["Code"] = meta["Code"].astype(str)
    out = radar.copy()
    out["Code"] = out["Code"].astype(str)
    return out.merge(meta, on="Code", how="left")


class SupplyMarketCache:
    def __init__(self, root: str = ".supply_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, endpoint: str, dt: date) -> Path:
        safe = endpoint.replace("/", "_")
        return self.root / f"{safe}_{dt.isoformat()}.csv.gz"

    def get_or_fetch(self, endpoint: str, dt: date, fetch: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
        p = self.path_for(endpoint, dt)
        if p.exists():
            try:
                return pd.read_csv(p, dtype={"Code": str})
            except Exception:
                pass
        try:
            d = fetch(dt.isoformat())
        except Exception:
            return pd.DataFrame()
        if d is not None and not d.empty:
            d.to_csv(p, index=False, compression="gzip")
            return d
        return pd.DataFrame()


def collect_recent_supply_market(client, end_date: date, cache_dir: str = ".supply_cache", progress_cb=None) -> Dict[str, pd.DataFrame]:
    """Collect recent market-wide official supply/demand snapshots with disk caching.

    This is intentionally a compact discovery window. Full histories are pulled only for deep analysis.
    """
    store = SupplyMarketCache(cache_dir)
    bdays = pd.bdate_range(end=end_date, periods=45).date.tolist()
    jobs = []
    # Breakdown and daily margin: recent 10 business days.
    for dt in bdays[-10:]:
        jobs.append(("breakdown", dt, client.breakdown_for_date))
        jobs.append(("daily_margin", dt, client.daily_margin_interest_for_date))
    # Short reports: recent 15 business days; absence is normal.
    for dt in bdays[-15:]:
        jobs.append(("shorts", dt, client.short_selling_positions_for_date))
    # Buyback TDnet: recent 40 business days because announcements are sparse.
    for dt in bdays[-40:]:
        jobs.append(("buyback_tdnet", dt, client.share_buyback_tdnet_for_date))
    # Weekly margin record date is usually Friday. Query recent Fridays only.
    fridays = pd.date_range(end=end_date, periods=10, freq="W-FRI").date.tolist()
    for dt in fridays:
        jobs.append(("weekly_margin", dt, client.weekly_margin_interest_for_date))

    out: Dict[str, List[pd.DataFrame]] = {k: [] for k in ["breakdown", "daily_margin", "weekly_margin", "shorts", "buyback_tdnet"]}
    total = max(len(jobs), 1)
    for i, (kind, dt, fetch) in enumerate(jobs):
        if progress_cb:
            progress_cb((i + 1) / total, f"需給データ {kind} {dt.isoformat()} ({i+1}/{total})")
        d = store.get_or_fetch(kind, dt, fetch)
        if d is not None and not d.empty:
            out[kind].append(d)
        time.sleep(0.02)
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}
