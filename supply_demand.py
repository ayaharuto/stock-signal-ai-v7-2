from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


def _num(v, default=np.nan) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _clip100(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def _linear(x: float, lo: float, hi: float, neutral: float = 50.0) -> float:
    if not np.isfinite(_num(x)) or hi == lo:
        return neutral
    return _clip100((float(x) - lo) / (hi - lo) * 100.0)


def _latest_by_date(df: pd.DataFrame, date_candidates: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    c = next((x for x in date_candidates if x in d.columns), None)
    if c:
        d[c] = pd.to_datetime(d[c], errors="coerce")
        return d.sort_values(c)
    return d


def summarize_supply_demand(
    breakdown: Optional[pd.DataFrame] = None,
    daily_margin: Optional[pd.DataFrame] = None,
    weekly_margin: Optional[pd.DataFrame] = None,
    shorts: Optional[pd.DataFrame] = None,
    buyback_tdnet: Optional[pd.DataFrame] = None,
    buyback_edinet: Optional[pd.DataFrame] = None,
    offauction_buyback: Optional[pd.DataFrame] = None,
) -> Dict[str, float | str | bool]:
    """Build a conservative 0-100 supply/demand quality score from official JPX data.

    Missing sources are neutral and reduce coverage instead of being interpreted bullishly.
    Positive inputs: cash buying share, short-covering, improving margin balance, buybacks.
    Negative inputs: aggressive new margin longs, rising reportable shorts, heavy fresh shorting.
    """
    components: list[tuple[str, float, float]] = []  # name, quality, weight
    out: Dict[str, float | str | bool] = {}

    # 1) Daily trading breakdown (40% of supply/demand score when available)
    b = _latest_by_date(breakdown, ["Date"])
    if not b.empty:
        tail = b.tail(min(10, len(b))).copy()
        cash_buy = pd.to_numeric(tail.get("va_3_0_0"), errors="coerce")
        margin_new_buy = pd.to_numeric(tail.get("va_3_2_0"), errors="coerce")
        short_cover = pd.to_numeric(tail.get("va_3_4_0"), errors="coerce")
        cash_sell = pd.to_numeric(tail.get("va_1_0_0"), errors="coerce")
        nonmargin_short = pd.to_numeric(tail.get("va_1_0_5"), errors="coerce").fillna(0) + pd.to_numeric(tail.get("va_1_0_7"), errors="coerce").fillna(0)
        margin_new_short = pd.to_numeric(tail.get("va_1_2_5"), errors="coerce").fillna(0) + pd.to_numeric(tail.get("va_1_2_7"), errors="coerce").fillna(0)
        buy_total = cash_buy.fillna(0) + margin_new_buy.fillna(0) + short_cover.fillna(0)
        sell_total = cash_sell.fillna(0) + nonmargin_short.fillna(0) + margin_new_short.fillna(0)
        total = (buy_total + sell_total).replace(0, np.nan)
        cash_buy_share = float((cash_buy / total).mean()) if total.notna().any() else np.nan
        leverage_buy_share = float((margin_new_buy / buy_total.replace(0, np.nan)).mean()) if buy_total.notna().any() else np.nan
        shorting_share = float(((nonmargin_short + margin_new_short) / total).mean()) if total.notna().any() else np.nan
        cover_share = float((short_cover / buy_total.replace(0, np.nan)).mean()) if buy_total.notna().any() else np.nan
        q = (
            0.40 * _linear(cash_buy_share, 0.20, 0.55)
            + 0.25 * (100 - _linear(leverage_buy_share, 0.08, 0.40))
            + 0.20 * (100 - _linear(shorting_share, 0.08, 0.35))
            + 0.15 * _linear(cover_share, 0.01, 0.20)
        )
        components.append(("breakdown", q, 0.40))
        out.update({
            "cash_buy_share": cash_buy_share,
            "margin_new_buy_share": leverage_buy_share,
            "shorting_share": shorting_share,
            "short_cover_share": cover_share,
        })

    # 2) Weekly margin balance (25%) - falling long/short ratio burden is constructive.
    w = _latest_by_date(weekly_margin, ["Date"])
    if not w.empty and {"LongMarginOutstanding", "ShortMarginOutstanding"}.issubset(w.columns):
        w = w.dropna(subset=["LongMarginOutstanding", "ShortMarginOutstanding"]).tail(8)
        if not w.empty:
            long = pd.to_numeric(w["LongMarginOutstanding"], errors="coerce")
            short = pd.to_numeric(w["ShortMarginOutstanding"], errors="coerce")
            ratio = long / short.replace(0, np.nan)
            latest_ratio = float(ratio.iloc[-1]) if ratio.notna().any() else np.nan
            ratio_change4 = float(ratio.iloc[-1] / ratio.iloc[-min(5, len(ratio))] - 1) if len(ratio) >= 2 and np.isfinite(ratio.iloc[-1]) and np.isfinite(ratio.iloc[-min(5, len(ratio))]) else np.nan
            long_change4 = float(long.iloc[-1] / long.iloc[-min(5, len(long))] - 1) if len(long) >= 2 and long.iloc[-min(5, len(long))] != 0 else np.nan
            q = 0.55 * (100 - _linear(latest_ratio, 0.7, 8.0)) + 0.45 * (100 - _linear(ratio_change4, -0.25, 0.35))
            components.append(("weekly_margin", q, 0.25))
            out.update({"margin_ratio": latest_ratio, "margin_ratio_change4w": ratio_change4, "margin_long_change4w": long_change4})

    # 3) Daily-published margin names (supplement, 10%). Absence is neutral because only selected names are published.
    dm = _latest_by_date(daily_margin, ["ApplicationDate", "PublishedDate"])
    if not dm.empty:
        last = dm.iloc[-1]
        short_o = _num(last.get("ShortMarginOutstanding"))
        long_std = _num(last.get("LongStandardizedMarginOutstanding"), 0.0)
        long_neg = _num(last.get("LongNegotiableMarginOutstanding"), 0.0)
        long_o = long_std + long_neg if np.isfinite(long_std) and np.isfinite(long_neg) else np.nan
        ratio = long_o / short_o if np.isfinite(long_o) and np.isfinite(short_o) and short_o > 0 else np.nan
        dlong = _num(last.get("DailyChangeLongStandardizedMarginOutstanding"), 0.0) + _num(last.get("DailyChangeLongNegotiableMarginOutstanding"), 0.0)
        dshort = _num(last.get("DailyChangeShortMarginOutstanding"))
        change_balance = (dshort - dlong) / max(abs(long_o) + abs(short_o), 1.0) if np.isfinite(dshort) and np.isfinite(long_o) and np.isfinite(short_o) else np.nan
        q = 0.65 * (100 - _linear(ratio, 0.5, 8.0)) + 0.35 * _linear(change_balance, -0.08, 0.08)
        components.append(("daily_margin", q, 0.10))
        out.update({"daily_margin_ratio": ratio, "daily_margin_balance_change": change_balance})

    # 4) Reportable short positions (20%) - use aggregate ratio across reporters and trend.
    sh = _latest_by_date(shorts, ["CalculatedDate", "DisclosedDate"])
    if not sh.empty and "ShortPositionsToSharesOutstandingRatio" in sh.columns:
        date_col = "CalculatedDate" if "CalculatedDate" in sh.columns else "DisclosedDate"
        sh[date_col] = pd.to_datetime(sh[date_col], errors="coerce")
        sh["ratio"] = pd.to_numeric(sh["ShortPositionsToSharesOutstandingRatio"], errors="coerce")
        agg = sh.groupby(date_col)["ratio"].sum().dropna().sort_index()
        if not agg.empty:
            latest_short = float(agg.iloc[-1])
            short_change = float(agg.iloc[-1] - agg.iloc[-min(6, len(agg))]) if len(agg) >= 2 else 0.0
            # Ratios are percentage-point values in J-Quants (e.g. 0.5 means 0.5%).
            q = 0.55 * (100 - _linear(latest_short, 0.0, 6.0)) + 0.45 * (100 - _linear(short_change, -1.0, 1.5))
            components.append(("short_positions", q, 0.20))
            out.update({"reportable_short_ratio_sum": latest_short, "reportable_short_ratio_change": short_change})

    # 5) Buyback support (15%): recent authorisation/execution is constructive, completion alone is smaller.
    buyback_qs = []
    td = _latest_by_date(buyback_tdnet, ["DisclosedDate", "PurchaseDate"])
    if not td.empty:
        latest = td.iloc[-1]
        typ = str(latest.get("DisclosureType", ""))
        max_cost = _num(latest.get("MaximumTotalAcquisitionCost"), 0.0)
        cumul = _num(latest.get("CumulativeTotalPurchasePrice"), 0.0)
        execution = _num(latest.get("TotalPurchasePrice"), 0.0)
        progress = cumul / max_cost if max_cost > 0 else np.nan
        type_q = {"start": 90, "status": 78, "complete": 60, "alteration": 58, "correction": 50, "cancellation": 20}.get(typ, 55)
        exec_q = _linear(execution, 0, max(max_cost * 0.15, 1e9)) if execution > 0 else 50
        buyback_qs.append(0.75 * type_q + 0.25 * exec_q)
        out.update({"buyback_tdnet_type": typ, "buyback_progress": progress})

    ed = _latest_by_date(buyback_edinet, ["SubmittedDate"])
    if not ed.empty:
        latest = ed.iloc[-1]
        ps = _num(latest.get("BoardResolutionAcquisitionsProgressPercentageShares"))
        pa = _num(latest.get("BoardResolutionAcquisitionsProgressPercentageAmountYen"))
        progress = np.nanmean([x for x in [ps, pa] if np.isfinite(x)]) if np.isfinite(ps) or np.isfinite(pa) else np.nan
        buyback_qs.append(_linear(progress, 0, 100) if np.isfinite(progress) else 55)
        out["buyback_edinet_progress_pct"] = progress

    oa = _latest_by_date(offauction_buyback, ["ImplementationDate", "PublicationDate"])
    if not oa.empty:
        latest = oa.iloc[-1]
        planned = _num(latest.get("NumberOfSharesToBePurchased"), _num(latest.get("PlannedNumberOfShares")))
        traded = _num(latest.get("NumberOfTradedShares"), _num(latest.get("NumberOfSharesPurchased")))
        fill = traded / planned if np.isfinite(traded) and np.isfinite(planned) and planned > 0 else np.nan
        buyback_qs.append(_linear(fill, 0.0, 1.0) if np.isfinite(fill) else 65)
        out["tostnet_buyback_fill"] = fill

    if buyback_qs:
        components.append(("buyback", float(np.mean(buyback_qs)), 0.15))
        out["buyback_active"] = True
    else:
        out["buyback_active"] = False

    if not components:
        out.update({"supply_demand_quality": 50.0, "supply_demand_coverage": 0.0, "supply_demand_signal": "需給データ未取得"})
        return out

    total_weight = sum(w for _, _, w in components)
    quality = sum(q * w for _, q, w in components) / total_weight
    # Coverage is capped at 100; possible weights overlap intentionally because daily margin is supplemental.
    coverage = min(1.0, total_weight / 0.90)
    signal = "需給改善" if quality >= 68 else ("需給悪化" if quality <= 38 else "需給中立")
    if quality >= 75 and _num(out.get("cash_buy_share"), 0) >= 0.35:
        signal = "先回り需給候補"
    out.update({"supply_demand_quality": round(_clip100(quality), 1), "supply_demand_coverage": round(coverage, 3), "supply_demand_signal": signal})
    return out


def merge_supply_snapshot(radar: pd.DataFrame, supply: pd.DataFrame) -> pd.DataFrame:
    """Merge a per-Code supply snapshot into the fast radar and create an early-warning score."""
    if radar is None or radar.empty:
        return pd.DataFrame()
    out = radar.copy()
    if supply is None or supply.empty or "Code" not in supply.columns:
        out["supply_demand_quality"] = 50.0
        out["supply_demand_coverage"] = 0.0
    else:
        s = supply.copy()
        s["Code"] = s["Code"].astype(str)
        out["Code"] = out["Code"].astype(str)
        out = out.merge(s, on="Code", how="left")
        out["supply_demand_quality"] = pd.to_numeric(out.get("supply_demand_quality"), errors="coerce").fillna(50.0)
        out["supply_demand_coverage"] = pd.to_numeric(out.get("supply_demand_coverage"), errors="coerce").fillna(0.0)
    # Keep price radar dominant but allow strong official supply-demand evidence to surface early.
    effective_supply = 50 + (out["supply_demand_quality"] - 50) * out["supply_demand_coverage"].clip(0, 1)
    out["early_radar_score"] = np.clip(0.72 * out["radar_score"] + 0.28 * effective_supply, 0, 100)
    out["early_signal"] = np.select(
        [
            (out["supply_demand_coverage"] >= 0.45) & (out["supply_demand_quality"] >= 72) & (out["ret20"].fillna(0) < 0.12),
            (out["supply_demand_coverage"] >= 0.45) & (out["supply_demand_quality"] <= 35),
        ],
        ["需給先回り候補", "需給悪化警戒"],
        default=out.get("phase_hint", "監視"),
    )
    return out.sort_values(["early_radar_score", "radar_change"], ascending=[False, False])


def build_market_supply_snapshot(
    breakdown: pd.DataFrame,
    daily_margin: pd.DataFrame,
    weekly_margin: pd.DataFrame,
    shorts: pd.DataFrame,
    buyback_tdnet: pd.DataFrame,
    offauction_buyback: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create one official supply/demand row per Code for market-wide discovery."""
    frames = [x for x in [breakdown, daily_margin, weekly_margin, shorts, buyback_tdnet, offauction_buyback] if x is not None and not x.empty and "Code" in x.columns]
    if not frames:
        return pd.DataFrame()
    codes = sorted(set().union(*[set(x["Code"].astype(str)) for x in frames]))
    rows = []
    for code in codes:
        def part(df):
            if df is None or df.empty or "Code" not in df.columns:
                return pd.DataFrame()
            return df[df["Code"].astype(str) == code]
        s = summarize_supply_demand(
            breakdown=part(breakdown),
            daily_margin=part(daily_margin),
            weekly_margin=part(weekly_margin),
            shorts=part(shorts),
            buyback_tdnet=part(buyback_tdnet),
            offauction_buyback=part(offauction_buyback),
        )
        rows.append({"Code": code, **s})
    return pd.DataFrame(rows)
