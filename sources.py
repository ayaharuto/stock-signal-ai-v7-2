from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import io
import requests
import numpy as np
import pandas as pd


@dataclass
class SourceStatus:
    name: str
    tier: int
    ok: bool
    message: str
    last_observation: Optional[pd.Timestamp] = None
    rows: int = 0


class JQuantsAPIClient:
    """Official JPX J-Quants API V2 client for individual investors.

    Authentication uses the API key issued in the J-Quants dashboard (x-api-key).
    The methods intentionally expose the old internal column names expected by the
    existing Stock Signal AI analysis engine, so upgrading the data source does not
    remove analytical features.
    """

    BASE = "https://api.jquants.com/v2"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("JQUANTS_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def headers(self):
        return {"x-api-key": self.api_key, "User-Agent": "StockSignalAI/7.3"}

    def _get(self, path: str, params: Optional[Dict] = None, data_key: str = "data") -> dict:
        if not self.api_key:
            raise RuntimeError("J-Quants APIキーが設定されていません")
        url = self.BASE + path
        rows: list = []
        q = {k: v for k, v in dict(params or {}).items() if v not in (None, "")}
        pagination_key = ""
        retry = 0
        while True:
            if pagination_key:
                q["pagination_key"] = pagination_key
            r = self.session.get(url, params=q, headers=self.headers, timeout=self.timeout)
            if r.status_code == 429 and retry < 4:
                # Be conservative with the official service. Honour Retry-After when present.
                import time
                wait = float(r.headers.get("Retry-After", 2 ** retry))
                time.sleep(max(1.0, min(wait, 30.0)))
                retry += 1
                continue
            if not r.ok:
                try:
                    detail = r.json().get("message", r.text)
                except Exception:
                    detail = r.text
                if r.status_code in (401, 403):
                    raise RuntimeError(f"J-Quants API認証/プラン権限エラー ({r.status_code}): {detail}")
                raise RuntimeError(f"J-Quants APIエラー ({r.status_code}): {detail}")
            retry = 0
            payload = r.json()
            batch = payload.get(data_key, [])
            if isinstance(batch, list):
                rows.extend(batch)
            pagination_key = payload.get("pagination_key", "")
            if not pagination_key:
                return {"rows": rows, "raw": payload}

    @staticmethod
    def _normalize_code(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)

    def latest_data_date(self, sample_code: str = "7203", lookback_days: int = 160) -> Optional[pd.Timestamp]:
        end = date.today()
        start = end - timedelta(days=lookback_days)
        d = self.daily_quotes(sample_code, start.isoformat(), end.isoformat())
        if d.empty:
            return None
        return pd.Timestamp(d.index.max()).normalize()

    def listed_info(self, target_date: Optional[str] = None) -> pd.DataFrame:
        rows = self._get("/equities/master", {"date": target_date})["rows"]
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows)
        d = d.rename(columns={
            "CoName": "CompanyName", "CoNameEn": "CompanyNameEnglish",
            "S17": "Sector17Code", "S17Nm": "Sector17CodeName",
            "S33": "Sector33Code", "S33Nm": "Sector33CodeName",
            "ScaleCat": "ScaleCategory", "Mkt": "MarketCode", "MktNm": "MarketCodeName",
            "Mrgn": "MarginCode", "MrgnNm": "MarginCodeName",
        })
        if "Code" in d.columns:
            d["Code"] = self._normalize_code(d["Code"])
        return d

    def _map_daily_bars(self, rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows)
        if "Code" in d.columns:
            d["Code"] = self._normalize_code(d["Code"])
        # Adjusted prices/volume are the stable inputs for historical analysis.
        mapping = {
            "AdjO": "Open", "AdjH": "High", "AdjL": "Low", "AdjC": "Close", "AdjVo": "Volume",
            "Va": "TurnoverValue", "O": "RawOpen", "H": "RawHigh", "L": "RawLow", "C": "RawClose", "Vo": "RawVolume",
        }
        d = d.rename(columns=mapping)
        # Some records can have adjusted fields absent/blank; fall back to raw values.
        for adj, raw in [("Open", "RawOpen"), ("High", "RawHigh"), ("Low", "RawLow"), ("Close", "RawClose"), ("Volume", "RawVolume")]:
            if adj not in d.columns and raw in d.columns:
                d[adj] = d[raw]
            elif adj in d.columns and raw in d.columns:
                a = pd.to_numeric(d[adj], errors="coerce")
                d[adj] = a.where(a.notna(), pd.to_numeric(d[raw], errors="coerce"))
        for c in ["Open", "High", "Low", "Close", "Volume", "TurnoverValue"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        return d

    def daily_quotes_for_date(self, target_date: str) -> pd.DataFrame:
        rows = self._get("/equities/bars/daily", {"date": target_date})["rows"]
        d = self._map_daily_bars(rows)
        if d.empty:
            return d
        keep = [c for c in ["Date", "Code", "Open", "High", "Low", "Close", "Volume", "TurnoverValue"] if c in d.columns]
        return d[keep].copy()

    def daily_quotes(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/equities/bars/daily", {"code": code, "from": start, "to": end})["rows"]
        d = self._map_daily_bars(rows)
        if d.empty:
            return d
        return d.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index().dropna(subset=["Close"])

    def statements(self, code: str) -> pd.DataFrame:
        rows = self._get("/fins/summary", {"code": code})["rows"]
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows).rename(columns={
            "DiscDate": "DisclosedDate", "Sales": "NetSales", "OP": "OperatingProfit",
            "OdP": "OrdinaryProfit", "NP": "Profit", "EPS": "EarningsPerShare",
            "Eq": "Equity", "TA": "TotalAssets", "CFO": "CashFlowsFromOperatingActivities",
            "FSales": "ForecastNetSales", "FOP": "ForecastOperatingProfit",
            "FNP": "ForecastProfit", "FEPS": "ForecastEarningsPerShare",
        })
        dt_col = "DisclosedDate" if "DisclosedDate" in d.columns else "Date"
        d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
        numeric_candidates = [
            "NetSales", "OperatingProfit", "OrdinaryProfit", "Profit", "EarningsPerShare",
            "Equity", "TotalAssets", "CashFlowsFromOperatingActivities", "ForecastNetSales",
            "ForecastOperatingProfit", "ForecastProfit", "ForecastEarningsPerShare",
        ]
        out = pd.DataFrame(index=d[dt_col])
        for c in numeric_candidates:
            if c in d.columns:
                out[c] = pd.to_numeric(d[c].values, errors="coerce")
        return out[~out.index.isna()].sort_index()

    def breakdown(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/breakdown", {"code": code, "from": start, "to": end})["rows"]
        return self._map_breakdown(rows)

    def breakdown_for_date(self, target_date: str) -> pd.DataFrame:
        return self._map_breakdown(self._get("/markets/breakdown", {"date": target_date})["rows"])

    def _map_breakdown(self, rows: list) -> pd.DataFrame:
        d = self._market_frame(rows, ["Date"])
        if d.empty:
            return d
        # Compatibility aliases used by supply_demand.py.
        aliases = {
            "LongBuyVa": "va_3_0_0",
            "MrgnBuyNewVa": "va_3_2_0",
            "MrgnBuyCloseVa": "va_3_4_0",
            "LongSellVa": "va_1_0_0",
            "ShrtNoMrgnVa": "va_1_0_5",
            "MrgnSellNewVa": "va_1_2_5",
        }
        for src, dst in aliases.items():
            if src in d.columns:
                d[dst] = pd.to_numeric(d[src], errors="coerce")
        # V2 aggregates these categories compared with the Pro schema. Keep absent sub-buckets at zero
        # rather than duplicating values, which would overstate sell/buy pressure.
        for c in ["va_1_0_7", "va_1_2_7"]:
            if c not in d.columns:
                d[c] = 0.0
        return d

    def daily_margin_interest(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/margin-alert", {"code": code, "from": start, "to": end})["rows"]
        return self._map_daily_margin(rows)

    def daily_margin_interest_for_date(self, target_date: str) -> pd.DataFrame:
        return self._map_daily_margin(self._get("/markets/margin-alert", {"date": target_date})["rows"])

    def _map_daily_margin(self, rows: list) -> pd.DataFrame:
        d = self._market_frame(rows, ["AppDate", "PubDate"])
        if d.empty:
            return d
        d = d.rename(columns={
            "AppDate": "ApplicationDate", "PubDate": "PublishedDate", "PubReason": "PublishReason",
            "ShrtOut": "ShortMarginOutstanding", "ShrtOutChg": "DailyChangeShortMarginOutstanding",
            "LongStdOut": "LongStandardizedMarginOutstanding", "LongNegOut": "LongNegotiableMarginOutstanding",
            "LongStdOutChg": "DailyChangeLongStandardizedMarginOutstanding",
            "LongNegOutChg": "DailyChangeLongNegotiableMarginOutstanding",
        })
        return d

    def weekly_margin_interest(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/margin-interest", {"code": code, "from": start, "to": end})["rows"]
        return self._map_weekly_margin(rows)

    def weekly_margin_interest_for_date(self, target_date: str) -> pd.DataFrame:
        return self._map_weekly_margin(self._get("/markets/margin-interest", {"date": target_date})["rows"])

    def _map_weekly_margin(self, rows: list) -> pd.DataFrame:
        d = self._market_frame(rows, ["Date"])
        if d.empty:
            return d
        return d.rename(columns={"LongVol": "LongMarginOutstanding", "ShrtVol": "ShortMarginOutstanding"})

    def short_selling_positions(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/short-sale-report", {
            "code": code, "disc_date_from": start, "disc_date_to": end
        })["rows"]
        return self._map_short_positions(rows)

    def short_selling_positions_for_date(self, target_date: str) -> pd.DataFrame:
        rows = self._get("/markets/short-sale-report", {"disc_date": target_date})["rows"]
        return self._map_short_positions(rows)

    def _map_short_positions(self, rows: list) -> pd.DataFrame:
        d = self._market_frame(rows, ["CalcDate", "DiscDate"])
        if d.empty:
            return d
        return d.rename(columns={
            "CalcDate": "CalculatedDate", "DiscDate": "DisclosedDate",
            "ShrtPosToSO": "ShortPositionsToSharesOutstandingRatio",
        })

    def _td_list(self, *, code: str = "", date_: str = "", start: str = "", end: str = "") -> pd.DataFrame:
        params: Dict[str, str] = {}
        if date_:
            params["date"] = date_
        else:
            params.update({"code": code, "from": start, "to": end})
        rows = self._get("/td/list", params)["rows"]
        return self._market_frame(rows, ["DiscDate"])

    @staticmethod
    def _buyback_from_td(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty or "Title" not in d.columns:
            return pd.DataFrame()
        title = d["Title"].fillna("").astype(str)
        mask = title.str.contains(r"自己株|自己株式", regex=True) & title.str.contains(r"取得|消却|買付|買い付け", regex=True)
        out = d[mask].copy()
        if out.empty:
            return out
        out["DisclosedDate"] = pd.to_datetime(out.get("DiscDate"), errors="coerce")
        out["DisclosureType"] = "start"
        out.loc[title[mask].str.contains(r"取得状況|進捗"), "DisclosureType"] = "status"
        out.loc[title[mask].str.contains(r"終了|完了"), "DisclosureType"] = "complete"
        out.loc[title[mask].str.contains(r"訂正"), "DisclosureType"] = "correction"
        out.loc[title[mask].str.contains(r"中止|取消"), "DisclosureType"] = "cancellation"
        return out

    def share_buyback_tdnet(self, code: str, start: str, end: str) -> pd.DataFrame:
        return self._buyback_from_td(self._td_list(code=code, start=start, end=end))

    def share_buyback_tdnet_for_date(self, target_date: str) -> pd.DataFrame:
        return self._buyback_from_td(self._td_list(date_=target_date))

    def share_buyback_edinet(self, code: str) -> pd.DataFrame:
        # Dedicated Pro buyback endpoint is not part of individual V2. The separate EDINETClient
        # remains available for official filing presence; missing buyback amount is neutral in scoring.
        return pd.DataFrame()

    def off_auction_share_buyback(self, code: str, start: str, end: str) -> pd.DataFrame:
        # ToSTNeT-3 dedicated dataset is Pro-only. Never fabricate a substitute.
        return pd.DataFrame()

    def _market_frame(self, rows: list, date_cols: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows)
        if "Code" in d.columns:
            d["Code"] = self._normalize_code(d["Code"])
        for c in date_cols:
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        text_cols = {
            "Code", "CompanyName", "CompanyNameEnglish", "StockName", "StockNameEnglish",
            "DisclosureType", "PurchasingMethod", "Notes", "PublishReason", "Title", "Name",
            "DiscStatus", "DiscItems", "Docs", "SSName", "SSAddr", "DICName", "DICAddr", "FundName",
        }
        for c in d.columns:
            if c in text_cols:
                continue
            if d[c].dtype == object:
                converted = pd.to_numeric(d[c].replace({"-": np.nan, "*": np.nan, "": np.nan}), errors="coerce")
                if converted.notna().sum() >= max(1, int(len(d) * 0.4)):
                    d[c] = converted
        return d

class EDINETClient:
    BASE = "https://api.edinet-fsa.go.jp/api/v2"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("EDINET_API_KEY", "")
        self.timeout = timeout

    def list_documents(self, target_date: date) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("EDINET API key is not configured")
        params = {"date": target_date.isoformat(), "type": 2, "Subscription-Key": self.api_key}
        r = requests.get(f"{self.BASE}/documents.json", params=params, timeout=self.timeout)
        r.raise_for_status()
        return pd.DataFrame(r.json().get("results", []))

    def recent_documents_for_sec_code(self, sec_code: str, days: int = 30) -> pd.DataFrame:
        frames = []
        for i in range(days):
            dt = date.today() - timedelta(days=i)
            try:
                d = self.list_documents(dt)
            except requests.HTTPError:
                continue
            if d.empty or "secCode" not in d.columns:
                continue
            m = d[d["secCode"].fillna("").astype(str).str.startswith(str(sec_code)[:4])]
            if not m.empty:
                frames.append(m)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class FREDClient:
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.timeout = timeout

    def series(self, series_id: str, start: str, end: Optional[str] = None) -> pd.Series:
        if not self.api_key:
            raise RuntimeError("FRED API key is not configured")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
        }
        if end:
            params["observation_end"] = end
        r = requests.get(self.BASE, params=params, timeout=self.timeout)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        s = pd.Series(
            {pd.to_datetime(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs},
            name=series_id,
            dtype=float,
        )
        return s.sort_index()


class BOJClient:
    BASE = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def get_series(self, db: str, code: str, start_yyyymm: str, end_yyyymm: str) -> pd.Series:
        params = {
            "format": "csv",
            "lang": "en",
            "db": db,
            "startDate": start_yyyymm,
            "endDate": end_yyyymm,
            "code": code,
        }
        r = requests.get(self.BASE, params=params, timeout=self.timeout)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        # API CSV column names may differ by database. Find first date-like and numeric value columns robustly.
        date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), df.columns[0])
        value_cols = [c for c in df.columns if c != date_col]
        if not value_cols:
            return pd.Series(dtype=float, name=code)
        val_col = value_cols[-1]
        idx = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        vals = pd.to_numeric(df[val_col], errors="coerce")
        s = pd.Series(vals.values, index=idx, name=code).dropna()
        return s[~s.index.isna()].sort_index()


def build_macro_frame(fred: FREDClient, start: str, end: Optional[str] = None) -> Tuple[pd.DataFrame, List[SourceStatus]]:
    # High-signal, official macro series. FRED serves data from primary statistical agencies/Federal Reserve sources.
    series_map = {
        "us10y": "DGS10",
        "us2y": "DGS2",
        "vix": "VIXCLS",
        "usd_jpy": "DEXJPUS",
        "fed_funds": "DFF",
    }
    statuses: List[SourceStatus] = []
    cols = []
    for name, sid in series_map.items():
        try:
            s = fred.series(sid, start, end).rename(name)
            cols.append(s)
            statuses.append(SourceStatus("FRED/ALFRED", 1, True, f"{sid} loaded", s.index.max() if len(s) else None, len(s)))
        except Exception as e:
            statuses.append(SourceStatus("FRED/ALFRED", 1, False, f"{sid}: {e}"))
    if not cols:
        return pd.DataFrame(), statuses
    out = pd.concat(cols, axis=1).sort_index().ffill()
    out["yield_curve_10y2y"] = out.get("us10y") - out.get("us2y")
    return out, statuses


def source_confidence(statuses: List[SourceStatus]) -> float:
    if not statuses:
        return 0.0
    weights = []
    scores = []
    now = pd.Timestamp.now(tz=None).normalize()
    for s in statuses:
        w = 1.0 if s.tier == 1 else 0.65
        weights.append(w)
        if not s.ok:
            scores.append(0.0)
            continue
        freshness = 1.0
        if s.last_observation is not None:
            ts = pd.Timestamp(s.last_observation).tz_localize(None) if pd.Timestamp(s.last_observation).tzinfo else pd.Timestamp(s.last_observation)
            age = max((now - ts.normalize()).days, 0)
            freshness = max(0.35, np.exp(-age / 45))
        scores.append(float(freshness))
    return float(np.average(scores, weights=weights)) if sum(weights) else 0.0
