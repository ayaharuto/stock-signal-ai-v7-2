from __future__ import annotations

"""
Stock Signal AI v8.1 - Prediction journal / answer-check engine.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd
import requests


DEFAULT_GIST_FILENAME = "prediction_history.json"
SCHEMA_VERSION = "v8.1"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if not isinstance(value, (str, bytes, bool)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


@dataclass
class ForecastSnapshot:
    snapshot_id: str
    schema_version: str
    created_at_utc: str
    base_date: str
    code: str
    company_name: str
    decision: str
    action: str
    timing: str
    score: float
    ai_probability: float
    confidence: float
    phase: str
    market_regime: str
    readiness: str
    base_close: float
    entry_low: float
    entry_high: float
    stop: float
    target1: float
    target2: float
    benchmark_code: str
    model_horizon_days: int
    next_condition: str
    reason: str
    app_version: str = "v8.1"

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


def make_snapshot(
    *,
    code: str,
    company_name: str,
    result: Dict[str, Any],
    plan: Dict[str, Any],
    benchmark_code: str,
    model_horizon_days: int,
    app_version: str = "v8.1",
) -> Dict[str, Any]:
    base_date = str(plan.get("基準日") or "")
    if not base_date:
        raise ValueError("基準日がないため予測履歴を保存できません。")
    code_s = str(code).strip()
    snapshot_id = f"{base_date}:{code_s}"
    obj = ForecastSnapshot(
        snapshot_id=snapshot_id,
        schema_version=SCHEMA_VERSION,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        base_date=base_date,
        code=code_s,
        company_name=str(company_name or ""),
        decision=str(plan.get("判定", "")),
        action=str(plan.get("行動", "")),
        timing=str(plan.get("買いタイミング", "")),
        score=_finite(result.get("score"), 0.0),
        ai_probability=_finite(result.get("live_p"), 0.5),
        confidence=_finite(result.get("confidence"), 0.0),
        phase=str(result.get("phase", "")),
        market_regime=str(plan.get("市場環境", "")),
        readiness=str(plan.get("実戦準備度", "")),
        base_close=_finite(plan.get("基準終値"), 0.0),
        entry_low=_finite(plan.get("買い下限"), np.nan),
        entry_high=_finite(plan.get("買い上限"), np.nan),
        stop=_finite(plan.get("損切り"), np.nan),
        target1=_finite(plan.get("利確1"), np.nan),
        target2=_finite(plan.get("利確2"), np.nan),
        benchmark_code=str(benchmark_code),
        model_horizon_days=int(model_horizon_days),
        next_condition=str(plan.get("買い昇格条件", "")),
        reason=str(plan.get("判断理由", "")),
        app_version=str(app_version),
    )
    return obj.to_dict()


def _gist_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "stock-signal-ai-v8.1",
    }


def load_gist_records(
    token: str,
    gist_id: str,
    filename: str = DEFAULT_GIST_FILENAME,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    if not token or not gist_id:
        return []
    url = f"https://api.github.com/gists/{gist_id}"
    r = requests.get(url, headers=_gist_headers(token), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", {})
    f = files.get(filename)
    if not f:
        return []
    content = f.get("content", "")
    if f.get("truncated") and f.get("raw_url"):
        rr = requests.get(f["raw_url"], headers=_gist_headers(token), timeout=timeout)
        rr.raise_for_status()
        content = rr.text
    if not str(content).strip():
        return []
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
        return parsed["records"]
    raise ValueError("Gist履歴ファイルの形式が不正です。")


def save_gist_records(
    token: str,
    gist_id: str,
    records: Sequence[Dict[str, Any]],
    filename: str = DEFAULT_GIST_FILENAME,
    timeout: int = 20,
) -> None:
    if not token or not gist_id:
        raise ValueError("GITHUB_GIST_TOKEN または PREDICTION_GIST_ID が未設定です。")
    safe_records = [_json_safe(dict(x)) for x in records]
    payload = {
        "files": {
            filename: {
                "content": json.dumps(
                    safe_records,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                )
            }
        }
    }
    url = f"https://api.github.com/gists/{gist_id}"
    r = requests.patch(url, headers=_gist_headers(token), json=payload, timeout=timeout)
    r.raise_for_status()


def upsert_gist_records(
    token: str,
    gist_id: str,
    new_records: Sequence[Dict[str, Any]],
    filename: str = DEFAULT_GIST_FILENAME,
    timeout: int = 20,
) -> Tuple[List[Dict[str, Any]], int, int]:
    current = load_gist_records(token, gist_id, filename=filename, timeout=timeout)
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for rec in current:
        sid = str(rec.get("snapshot_id") or "")
        if not sid:
            continue
        if sid not in by_id:
            order.append(sid)
        by_id[sid] = dict(rec)

    inserted = 0
    updated = 0
    for rec in new_records:
        rec = _json_safe(dict(rec))
        sid = str(rec.get("snapshot_id") or "")
        if not sid:
            continue
        if sid in by_id:
            updated += 1
        else:
            inserted += 1
            order.append(sid)
        by_id[sid] = rec

    merged = [by_id[sid] for sid in order]
    merged.sort(key=lambda x: (str(x.get("base_date", "")), str(x.get("code", ""))))
    save_gist_records(token, gist_id, merged, filename=filename, timeout=timeout)
    return merged, inserted, updated


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    d = df.copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.set_index("Date")
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    for c in ["Open", "High", "Low", "Close"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["Open", "High", "Low", "Close"]].dropna(subset=["Close"])


def _first_touch_entry(
    future: pd.DataFrame,
    entry_low: float,
    entry_high: float,
) -> Tuple[Optional[pd.Timestamp], float]:
    if not (np.isfinite(entry_low) and np.isfinite(entry_high)):
        return None, np.nan
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low
    for dt, row in future.iterrows():
        lo = _finite(row.get("Low"), np.nan)
        hi = _finite(row.get("High"), np.nan)
        op = _finite(row.get("Open"), np.nan)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if np.isfinite(op) and op > entry_high and lo > entry_high:
            continue
        if lo <= entry_high and hi >= entry_low:
            if np.isfinite(op) and entry_low <= op <= entry_high:
                fill = op
            elif np.isfinite(op) and op < entry_low:
                fill = float(np.clip(op, lo, hi))
            else:
                fill = entry_high
            return pd.Timestamp(dt), float(fill)
    return None, np.nan


def _trade_path_after_entry(
    future: pd.DataFrame,
    entry_date: Optional[pd.Timestamp],
    entry_price: float,
    stop: float,
    target1: float,
    target2: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "entry_date": None,
        "entry_price": np.nan,
        "stop_hit": False,
        "stop_date": None,
        "target1_hit": False,
        "target1_date": None,
        "target2_hit": False,
        "target2_date": None,
        "first_resolution": "no_entry",
        "mfe": np.nan,
        "mae": np.nan,
    }
    if entry_date is None or not np.isfinite(entry_price) or entry_price <= 0:
        return out

    d = future.loc[future.index >= entry_date].copy()
    if d.empty:
        return out
    out["entry_date"] = pd.Timestamp(entry_date).date().isoformat()
    out["entry_price"] = float(entry_price)

    max_high = _finite(d["High"].max(), np.nan)
    min_low = _finite(d["Low"].min(), np.nan)
    if np.isfinite(max_high):
        out["mfe"] = float(max_high / entry_price - 1.0)
    if np.isfinite(min_low):
        out["mae"] = float(min_low / entry_price - 1.0)

    first_resolution = None
    for dt, row in d.iterrows():
        lo = _finite(row.get("Low"), np.nan)
        hi = _finite(row.get("High"), np.nan)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue

        hit_stop = np.isfinite(stop) and lo <= stop
        hit_t1 = np.isfinite(target1) and hi >= target1
        hit_t2 = np.isfinite(target2) and hi >= target2

        if hit_stop and not out["stop_hit"]:
            out["stop_hit"] = True
            out["stop_date"] = pd.Timestamp(dt).date().isoformat()
            if first_resolution is None:
                first_resolution = "stop"
            break

        if hit_t1 and not out["target1_hit"]:
            out["target1_hit"] = True
            out["target1_date"] = pd.Timestamp(dt).date().isoformat()
            if first_resolution is None:
                first_resolution = "target1"
        if hit_t2 and not out["target2_hit"]:
            out["target2_hit"] = True
            out["target2_date"] = pd.Timestamp(dt).date().isoformat()
            first_resolution = "target2"
            break

    out["first_resolution"] = first_resolution or "open"
    return out


def _horizon_stats(
    base_close: float,
    future: pd.DataFrame,
    benchmark_future: pd.DataFrame,
    benchmark_base_close: float,
    horizon: int,
) -> Dict[str, Any]:
    key = str(horizon)
    if len(future) < horizon or base_close <= 0:
        return {
            f"ret_{key}": np.nan,
            f"benchmark_ret_{key}": np.nan,
            f"excess_{key}": np.nan,
            f"mfe_closebase_{key}": np.nan,
            f"mae_closebase_{key}": np.nan,
            f"matured_{key}": False,
        }
    d = future.iloc[:horizon]
    last_close = _finite(d["Close"].iloc[-1], np.nan)
    ret = last_close / base_close - 1.0 if np.isfinite(last_close) else np.nan
    max_high = _finite(d["High"].max(), np.nan)
    min_low = _finite(d["Low"].min(), np.nan)
    mfe = max_high / base_close - 1.0 if np.isfinite(max_high) else np.nan
    mae = min_low / base_close - 1.0 if np.isfinite(min_low) else np.nan

    bret = np.nan
    if benchmark_base_close > 0 and benchmark_future is not None and len(benchmark_future) >= horizon:
        bclose = _finite(benchmark_future["Close"].iloc[horizon - 1], np.nan)
        if np.isfinite(bclose):
            bret = bclose / benchmark_base_close - 1.0

    return {
        f"ret_{key}": float(ret) if np.isfinite(ret) else np.nan,
        f"benchmark_ret_{key}": float(bret) if np.isfinite(bret) else np.nan,
        f"excess_{key}": float(ret - bret) if np.isfinite(ret) and np.isfinite(bret) else np.nan,
        f"mfe_closebase_{key}": float(mfe) if np.isfinite(mfe) else np.nan,
        f"mae_closebase_{key}": float(mae) if np.isfinite(mae) else np.nan,
        f"matured_{key}": True,
    }


def _primary_answer_label(
    snapshot: Dict[str, Any],
    stats: Dict[str, Any],
    trade_path: Dict[str, Any],
) -> Tuple[str, float, str]:
    matured = bool(stats.get("matured_20", False))
    if not matured:
        return "判定待ち", np.nan, "20営業日未経過"

    decision = str(snapshot.get("decision", ""))
    action = str(snapshot.get("action", ""))
    actionable = decision in {"買い候補", "条件付き買い"} or action.startswith("買い候補")
    ret20 = _finite(stats.get("ret_20"), np.nan)
    excess20 = _finite(stats.get("excess_20"), np.nan)
    mae20 = _finite(stats.get("mae_closebase_20"), np.nan)
    mfe20 = _finite(stats.get("mfe_closebase_20"), np.nan)

    if actionable:
        if trade_path.get("entry_date") is None:
            if np.isfinite(mfe20) and mfe20 >= 0.12:
                return "機会損失", 45.0, "買いゾーンに入らず、その後大きく上昇"
            return "買い場なし", 72.0, "買いゾーンに入らず、無理に追わなかった"

        if trade_path.get("target2_hit"):
            score = 96.0
            if np.isfinite(excess20) and excess20 > 0:
                score = min(100.0, score + 3.0)
            return "大成功", score, "利確②まで到達"
        if trade_path.get("target1_hit"):
            score = 86.0
            if np.isfinite(excess20) and excess20 > 0:
                score = min(95.0, score + 3.0)
            return "成功", score, "利確①に先に到達"
        if trade_path.get("stop_hit"):
            return "失敗", 18.0, "損切りに先に到達"
        if np.isfinite(ret20) and ret20 > 0:
            return "概ね成功", 68.0, "20日後はプラスだが利確目標未達"
        return "未達", 40.0, "20日後まで目標未達"

    if np.isfinite(ret20):
        if ret20 <= -0.05:
            return "見送り正解", 94.0, f"買わずに約{abs(ret20):.1%}の下落を回避"
        if np.isfinite(mae20) and mae20 <= -0.08 and ret20 <= 0.03:
            return "見送り正解", 88.0, "期間中の大幅下落を回避"
        if ret20 <= 0:
            return "見送り妥当", 82.0, "20日後リターンは非プラス"
        if ret20 < 0.05:
            return "見送り妥当", 70.0, "上昇は小さく、見送りコストは限定的"
        if ret20 < 0.10:
            return "やや機会損失", 50.0, "5〜10%上昇を取り逃した"
        return "機会損失", 25.0, "見送り後に10%以上上昇"
    return "判定不能", np.nan, "20日後価格を取得できません"


def evaluate_snapshot(
    snapshot: Dict[str, Any],
    quotes: pd.DataFrame,
    benchmark_quotes: Optional[pd.DataFrame] = None,
    horizons: Sequence[int] = (5, 20, 60),
) -> Dict[str, Any]:
    px = _normalize_ohlc(quotes)
    bm = _normalize_ohlc(benchmark_quotes if benchmark_quotes is not None else pd.DataFrame())
    base_date = pd.Timestamp(snapshot.get("base_date"))
    base_close = _finite(snapshot.get("base_close"), np.nan)

    future = px.loc[px.index > base_date].copy()

    benchmark_base_close = np.nan
    benchmark_future = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    if not bm.empty:
        bm_base = bm.loc[bm.index <= base_date]
        if not bm_base.empty:
            benchmark_base_close = _finite(bm_base["Close"].iloc[-1], np.nan)
            benchmark_future = bm.loc[bm.index > base_date].copy()

    stats: Dict[str, Any] = {}
    for h in horizons:
        stats.update(
            _horizon_stats(
                base_close=base_close,
                future=future,
                benchmark_future=benchmark_future,
                benchmark_base_close=benchmark_base_close,
                horizon=int(h),
            )
        )

    max_h = max([int(h) for h in horizons], default=20)
    path_window = future.iloc[:max_h] if not future.empty else future
    entry_date, entry_price = _first_touch_entry(
        path_window,
        _finite(snapshot.get("entry_low"), np.nan),
        _finite(snapshot.get("entry_high"), np.nan),
    )
    trade_path = _trade_path_after_entry(
        path_window,
        entry_date,
        entry_price,
        _finite(snapshot.get("stop"), np.nan),
        _finite(snapshot.get("target1"), np.nan),
        _finite(snapshot.get("target2"), np.nan),
    )

    label, match_score, reason = _primary_answer_label(snapshot, stats, trade_path)

    result = dict(snapshot)
    result.update(stats)
    result.update(trade_path)
    result.update({
        "answer_label": label,
        "match_score": match_score,
        "answer_reason": reason,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_future_rows": int(len(future)),
    })
    return _json_safe(result)


def answer_table(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "基準日": r.get("base_date", ""),
            "Code": r.get("code", ""),
            "会社名": r.get("company_name", ""),
            "当時判定": r.get("decision", ""),
            "当時行動": r.get("action", ""),
            "100点評価": r.get("score", np.nan),
            "AI確率%": _finite(r.get("ai_probability"), np.nan) * 100,
            "答え": r.get("answer_label", "未評価"),
            "一致度": r.get("match_score", np.nan),
            "5日%": _finite(r.get("ret_5"), np.nan) * 100,
            "20日%": _finite(r.get("ret_20"), np.nan) * 100,
            "60日%": _finite(r.get("ret_60"), np.nan) * 100,
            "20日超過%": _finite(r.get("excess_20"), np.nan) * 100,
            "MFE20%": _finite(r.get("mfe_closebase_20"), np.nan) * 100,
            "MAE20%": _finite(r.get("mae_closebase_20"), np.nan) * 100,
            "買い日": r.get("entry_date", ""),
            "最初の決着": r.get("first_resolution", ""),
            "理由": r.get("answer_reason", ""),
        })
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    if "基準日" in d.columns:
        d = d.sort_values(["基準日", "Code"], ascending=[False, True])
    return d


def summarize_answers(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    d = answer_table(records)
    if d.empty:
        return {
            "matured_20": 0,
            "avg_match_score": np.nan,
            "buy_success_rate": np.nan,
            "skip_success_rate": np.nan,
            "avg_ret20": np.nan,
            "avg_excess20": np.nan,
        }

    matured = d[pd.to_numeric(d["20日%"], errors="coerce").notna()].copy()
    if matured.empty:
        return {
            "matured_20": 0,
            "avg_match_score": np.nan,
            "buy_success_rate": np.nan,
            "skip_success_rate": np.nan,
            "avg_ret20": np.nan,
            "avg_excess20": np.nan,
        }

    buy_mask = matured["当時判定"].isin(["買い候補", "条件付き買い"])
    buy = matured[buy_mask]
    skip = matured[~buy_mask]

    buy_success = buy["答え"].isin(["大成功", "成功", "概ね成功"]).mean() if len(buy) else np.nan
    skip_success = skip["答え"].isin(["見送り正解", "見送り妥当"]).mean() if len(skip) else np.nan

    return {
        "matured_20": int(len(matured)),
        "avg_match_score": float(pd.to_numeric(matured["一致度"], errors="coerce").mean()),
        "buy_success_rate": float(buy_success) if np.isfinite(buy_success) else np.nan,
        "skip_success_rate": float(skip_success) if np.isfinite(skip_success) else np.nan,
        "avg_ret20": float(pd.to_numeric(matured["20日%"], errors="coerce").mean() / 100.0),
        "avg_excess20": float(pd.to_numeric(matured["20日超過%"], errors="coerce").mean() / 100.0),
    }
