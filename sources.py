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


class JQuantsProClient:
    """Official JPX J-Quants Pro v2 client. Requires a valid bearer token."""

    BASE = "https://api.jquants-pro.com/v2"

    def __init__(self, token: Optional[str] = None, timeout: int = 20):
        self.token = token or os.getenv("JQUANTS_TOKEN", "")
        self.timeout = timeout

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    @classmethod
    def login(cls, mailaddress: str, password: str, timeout: int = 20):
        """Obtain fresh ID/refresh tokens from official J-Quants Pro auth endpoint."""
        if not mailaddress or not password:
            raise RuntimeError("J-Quants Pro mail address/password is not configured")
        r = requests.post(
            cls.BASE + "/token/auth_user",
            json={"mailaddress": mailaddress, "password": password},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("idToken", "")
        if not token:
            raise RuntimeError("J-Quants Pro did not return idToken")
        return cls(token, timeout=timeout), data.get("refreshToken", "")

    @classmethod
    def from_refresh_token(cls, refresh_token: str, timeout: int = 20):
        if not refresh_token:
            raise RuntimeError("J-Quants Pro refresh token is not configured")
        r = requests.post(
            cls.BASE + "/token/auth_refresh",
            json={"refreshtoken": refresh_token},
            timeout=timeout,
        )
        r.raise_for_status()
        token = r.json().get("idToken", "")
        if not token:
            raise RuntimeError("J-Quants Pro did not return idToken")
        return cls(token, timeout=timeout)

    def _get(self, path: str, params: Dict) -> dict:
        if not self.token:
            raise RuntimeError("J-Quants token is not configured")
        url = self.BASE + path
        payload = []
        pagination_key = None
        while True:
            q = dict(params)
            if pagination_key:
                q["pagination_key"] = pagination_key
            r = requests.get(url, params=q, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            list_key = next((k for k, v in data.items() if isinstance(v, list)), None)
            if list_key:
                payload.extend(data[list_key])
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                break
        return {"rows": payload}

    def listed_info(self, target_date: Optional[str] = None) -> pd.DataFrame:
        params: Dict[str, str] = {}
        if target_date:
            params["date"] = target_date
        data = self._get("/listed/info", params)["rows"]
        if not data:
            return pd.DataFrame()
        d = pd.DataFrame(data)
        if "Code" in d.columns:
            d["Code"] = d["Code"].astype(str)
        return d

    def daily_quotes_for_date(self, target_date: str) -> pd.DataFrame:
        data = self._get("/prices/daily_quotes", {"date": target_date})["rows"]
        if not data:
            return pd.DataFrame()
        d = pd.DataFrame(data)
        keep = [c for c in ["Date", "Code", "Open", "High", "Low", "Close", "Volume", "TurnoverValue",
                              "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose", "AdjustmentVolume",
                              "CompanyName", "Sector17Code", "Sector33Code", "ScaleCategory", "MarketCode", "MarginCode"] if c in d.columns]
        d = d[keep].copy()
        if "Code" in d.columns:
            d["Code"] = d["Code"].astype(str)
        for c in ["Open", "High", "Low", "Close", "Volume", "TurnoverValue"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        # For market-wide screening, use adjusted OHLC/volume when supplied by the API.
        for src, dst in [("AdjustmentOpen", "Open"), ("AdjustmentHigh", "High"), ("AdjustmentLow", "Low"),
                         ("AdjustmentClose", "Close"), ("AdjustmentVolume", "Volume")]:
            if src in d.columns:
                vals = pd.to_numeric(d[src], errors="coerce")
                d[dst] = vals.where(vals.notna(), d.get(dst))
        return d

    def daily_quotes(self, code: str, start: str, end: str) -> pd.DataFrame:
        data = self._get("/prices/daily_quotes", {"code": code, "from": start, "to": end})["rows"]
        if not data:
            return pd.DataFrame()
        d = pd.DataFrame(data)
        d["Date"] = pd.to_datetime(d["Date"])
        mapping = {
            "AdjustmentOpen": "Open",
            "AdjustmentHigh": "High",
            "AdjustmentLow": "Low",
            "AdjustmentClose": "Close",
            "AdjustmentVolume": "Volume",
        }
        for src, dst in mapping.items():
            if src in d.columns:
                d[dst] = pd.to_numeric(d[src], errors="coerce")
            elif dst in d.columns:
                d[dst] = pd.to_numeric(d[dst], errors="coerce")
        return d.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

    def statements(self, code: str) -> pd.DataFrame:
        data = self._get("/fins/statements", {"code": code})["rows"]
        if not data:
            return pd.DataFrame()
        d = pd.DataFrame(data)
        dt_col = "DisclosedDate" if "DisclosedDate" in d.columns else "Date"
        d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
        numeric_candidates = [
            "NetSales", "OperatingProfit", "OrdinaryProfit", "Profit",
            "EarningsPerShare", "Equity", "TotalAssets", "CashFlowsFromOperatingActivities",
            "ForecastNetSales", "ForecastOperatingProfit", "ForecastProfit", "ForecastEarningsPerShare",
        ]
        out = pd.DataFrame(index=d[dt_col])
        for c in numeric_candidates:
            if c in d.columns:
                out[c] = pd.to_numeric(d[c].values, errors="coerce")
        return out[~out.index.isna()].sort_index()


    def breakdown(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/breakdown", {"code": code, "from": start, "to": end})["rows"]
        return self._market_frame(rows, ["Date"])

    def breakdown_for_date(self, target_date: str) -> pd.DataFrame:
        return self._market_frame(self._get("/markets/breakdown", {"date": target_date})["rows"], ["Date"])

    def daily_margin_interest(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/daily_margin_interest", {"code": code, "from": start, "to": end})["rows"]
        return self._market_frame(rows, ["ApplicationDate", "PublishedDate"])

    def daily_margin_interest_for_date(self, target_date: str) -> pd.DataFrame:
        return self._market_frame(self._get("/markets/daily_margin_interest", {"date": target_date})["rows"], ["ApplicationDate", "PublishedDate"])

    def weekly_margin_interest(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/weekly_margin_interest", {"code": code, "from": start, "to": end})["rows"]
        return self._market_frame(rows, ["Date"])

    def weekly_margin_interest_for_date(self, target_date: str) -> pd.DataFrame:
        return self._market_frame(self._get("/markets/weekly_margin_interest", {"date": target_date})["rows"], ["Date"])

    def short_selling_positions(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/short_selling_positions", {
            "code": code, "disclosed_date_from": start, "disclosed_date_to": end
        })["rows"]
        return self._market_frame(rows, ["CalculatedDate", "DisclosedDate"])

    def short_selling_positions_for_date(self, target_date: str) -> pd.DataFrame:
        rows = self._get("/markets/short_selling_positions", {"disclosed_date": target_date})["rows"]
        return self._market_frame(rows, ["CalculatedDate", "DisclosedDate"])

    def share_buyback_tdnet(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/share_buyback_tdnet", {"code": code, "from": start, "to": end})["rows"]
        return self._market_frame(rows, ["DisclosedDate", "PurchaseDate"])

    def share_buyback_tdnet_for_date(self, target_date: str) -> pd.DataFrame:
        rows = self._get("/markets/share_buyback_tdnet", {"date": target_date})["rows"]
        return self._market_frame(rows, ["DisclosedDate", "PurchaseDate"])

    def share_buyback_edinet(self, code: str) -> pd.DataFrame:
        rows = self._get("/markets/share_buyback_edinet", {"code": code})["rows"]
        return self._market_frame(rows, ["SubmittedDate", "TreasurySharesHoldingReportDate"])

    def off_auction_share_buyback(self, code: str, start: str, end: str) -> pd.DataFrame:
        rows = self._get("/markets/off_auction_share_buyback", {
            "code": code, "publication_date_from": start, "publication_date_to": end
        })["rows"]
        return self._market_frame(rows, ["PublicationDate", "ImplementationDate"])

    def _market_frame(self, rows: list, date_cols: list[str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        d = pd.DataFrame(rows)
        if "Code" in d.columns:
            d["Code"] = d["Code"].astype(str)
        for c in date_cols:
            if c in d.columns:
                d[c] = pd.to_datetime(d[c], errors="coerce")
        for c in d.columns:
            if c in {"Code", "CompanyName", "CompanyNameEnglish", "StockName", "StockNameEnglish", "DisclosureType", "PurchasingMethod", "Notes", "PublishReason"}:
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
