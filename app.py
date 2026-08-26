from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core import (
    ModelConfig,
    add_price_features,
    add_fundamental_features,
    add_macro_features,
    merge_asof_feature,
    make_supervised,
    deterministic_score,
    backtest_from_probabilities,
    performance_metrics,
    buy_score_components,
    classify_market_phase,
    buy_score_label,
)
from models import walk_forward_ensemble, fit_final_probability, model_confidence
from market_scanner import (
    ScanConfig, collect_market_panel, fast_market_features, filter_common_equities,
    enrich_with_listed, collect_recent_supply_market,
)
from supply_demand import summarize_supply_demand, build_market_supply_snapshot, merge_supply_snapshot
from validation import make_signal_history, event_study, validation_grade, execution_trade_backtest, execution_metrics, execution_grade
from optimizer import optimize_adaptive_weights, BASELINE_WEIGHTS
from sources import (
    JQuantsAPIClient,
    EDINETClient,
    FREDClient,
    build_macro_frame,
    SourceStatus,
    source_confidence,
)

st.set_page_config(page_title="Stock Signal AI v8.0.1", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 12px 14px;}
    .big-score {font-size: 4.1rem; font-weight: 800; line-height: 1; margin: .2rem 0;}
    .phase {font-size: 1.5rem; font-weight: 700; margin-bottom: .4rem;}
    .muted {opacity: .72; font-size: .92rem;}
    button[kind="primary"], button[kind="secondary"] {min-height: 48px; border-radius: 12px;}
    @media (max-width: 768px) {
      .block-container {padding: .65rem .7rem 5rem .7rem; max-width: 100%;}
      h1 {font-size: 1.75rem !important;}
      h2 {font-size: 1.35rem !important;}
      .big-score {font-size: 3rem;}
      div[data-testid="stMetric"] {padding: 9px 10px;}
      div[data-baseweb="tab-list"] {overflow-x: auto; white-space: nowrap; scrollbar-width: thin;}
      div[data-testid="stDataFrame"] {font-size: .80rem;}
      .stButton>button {width:100%; min-height:50px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock Signal AI v8.0.1")
st.caption("個人向けJ-Quants API V2対応。V8は『当てる』だけでなく、**市場環境・アウトオブサンプル優位性・資金管理・現実的な約定検証**を通過した銘柄だけを実戦候補にします。取得できないデータは推測で埋めません。")

def _safe_secret(name: str, default: str = "") -> str:
    """Read hosting secret first, then environment variable. Never expose secrets in UI."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def _sanitize_api_error(exc: Exception) -> str:
    text = str(exc)
    # Do not echo secrets if an upstream service ever includes request headers in an error.
    key = _safe_secret("JQUANTS_API_KEY", "")
    return text.replace(key, "***") if key else text


@st.cache_data(ttl=900, show_spinner=False)
def check_jquants_connection(api_key: str) -> dict:
    if not api_key:
        return {"ok": False, "latest": None, "age_days": None, "error": "APIキー未設定"}
    try:
        latest = JQuantsAPIClient(api_key).latest_data_date()
        if latest is None:
            return {"ok": False, "latest": None, "age_days": None, "error": "株価データを確認できませんでした"}
        age_days = max(0, (pd.Timestamp(date.today()) - pd.Timestamp(latest)).days)
        return {"ok": True, "latest": pd.Timestamp(latest), "age_days": age_days, "error": ""}
    except Exception as e:
        return {"ok": False, "latest": None, "age_days": None, "error": _sanitize_api_error(e)}


jq_api_key = _safe_secret("JQUANTS_API_KEY", "")
edinet_key = _safe_secret("EDINET_API_KEY", "")
fred_key = _safe_secret("FRED_API_KEY", "")
jq_status = check_jquants_connection(jq_api_key) if jq_api_key else {"ok": False, "latest": None, "age_days": None, "error": "APIキー未設定"}
jq_connected = bool(jq_status.get("ok"))
jq_latest_date = jq_status.get("latest")
jq_data_age_days = jq_status.get("age_days")
# The Free plan is 12 weeks delayed. A generous 14-day threshold avoids mistaking weekends/holidays
# for delayed data while preventing stale data from being labelled as "today".
jq_live_eligible = bool(jq_connected and jq_data_age_days is not None and jq_data_age_days <= 14)

status_col, action_col = st.columns([1.55, 1])
with status_col:
    if jq_live_eligible:
        st.success(f"✅ J-Quants API V2 接続済み ｜ 最新データ {jq_latest_date.date()}")
    elif jq_connected:
        st.warning(f"🕒 J-Quants接続済みですが最新データは {jq_latest_date.date()}（{jq_data_age_days}日前）。遅延データとして検証には使えますが、『今日の買い候補』には使いません。")
    elif jq_api_key:
        st.error(f"J-Quants APIへ接続できません: {jq_status.get('error','不明なエラー')}")
    else:
        st.warning("🔑 初回だけ J-Quants APIキーを設定してください")
with action_col:
    if jq_api_key and st.button("接続状態を再確認", use_container_width=True):
        check_jquants_connection.clear()
        st.rerun()

if not jq_api_key:
    with st.expander("🔑 初回設定（APIキーを1回だけ登録）", expanded=True):
        st.markdown("""
J-Quants **個人向けAPI V2**を使います。J-Quants Proの契約は不要です。  
APIキーはJ-Quants公式ダッシュボードで発行し、Streamlitの **Settings → Secrets** に次の1行だけ保存します。

`JQUANTS_API_KEY = "あなたのAPIキー"`

APIキーはこのチャットやGitHubへ貼らないでください。保存後はこの入力画面を普段開く必要はありません。
""")
elif jq_connected and not jq_live_eligible:
    st.info("現在のデータ鮮度では、個別の過去分析・AI検証は利用できます。『今日』と『100点ランキング』は誤判断防止のためライブ売買候補としては停止します。")

with st.expander("👋 使い方（30秒）", expanded=False):
    st.markdown("""
**普段はこの3操作だけです。**

1. **🌐 今日**を開く
2. **今日の結論を更新**を押す（全市場→100点→売買プランまで自動）
3. 「今日の結論」で **買い候補・買い目安・損切り・利確** を確認し、気になる銘柄だけ **🔎 個別** で詳細確認

公式・一次情報を優先します。SNS・匿名掲示板・まとめサイトは標準スコアには使いません。  
契約プランで取得できない公式データは**推測で補完せず「未取得」扱い**にし、情報充足度・信頼度を下げます。
""")

with st.sidebar:
    st.header("詳細設定（普段は触らなくてOK）")
    years = st.slider("価格履歴", 3, 12, 8, help="長いほど検証は安定しやすい一方、相場構造の変化も混ざります。")
    horizon = st.selectbox("AI予測期間", [5, 10, 20, 60], index=2, format_func=lambda x: f"{x}営業日")
    tx_cost = st.slider("片道売買コスト", 0, 50, 12, format="%d bps")
    benchmark_code = st.text_input("市場ベンチマーク", "1306")
    st.divider()
    st.subheader("資金管理")
    account_capital = st.number_input("運用資金（円）", min_value=100_000, max_value=500_000_000, value=1_000_000, step=100_000, help="推奨株数の計算にだけ使います。証券口座とは接続しません。")
    risk_per_trade_pct = st.slider("1回の最大許容損失", 0.2, 2.0, 0.6, 0.1, format="%.1f%%")
    max_position_pct = st.slider("1銘柄の最大投資比率", 5, 40, 20, 1, format="%d%%")
    max_positions = st.slider("同時保有上限", 1, 10, 5)
    slippage_bps = st.slider("想定スリッページ（片道）", 0, 40, 8, 1, format="%d bps")
    st.divider()
    st.caption("EDINET/FREDは公開側Secretsに設定可能。未設定でも主要機能は動きますが、情報信頼度・マクロ分析の一部が減ります。")

cfg = ModelConfig(horizon_days=horizon, transaction_cost_bps=tx_cost)


def parse_official_csv(uploaded) -> pd.DataFrame:
    d = pd.read_csv(uploaded)
    aliases = {
        "Date": ["Date", "date", "日付"],
        "Open": ["Open", "AdjustmentOpen", "始値"],
        "High": ["High", "AdjustmentHigh", "高値"],
        "Low": ["Low", "AdjustmentLow", "安値"],
        "Close": ["Close", "AdjustmentClose", "終値"],
        "Volume": ["Volume", "AdjustmentVolume", "取引高", "出来高"],
    }
    rename = {}
    for target, opts in aliases.items():
        src = next((c for c in opts if c in d.columns), None)
        if src:
            rename[src] = target
    d = d.rename(columns=rename)
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}")
    d["Date"] = pd.to_datetime(d["Date"])
    for c in needed[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("Date")[needed[1:]].sort_index().dropna(subset=["Close"])


@st.cache_data(ttl=1800, show_spinner=False)
def load_jq_quotes(api_key: str, code_: str, start: str, end: str) -> pd.DataFrame:
    return JQuantsAPIClient(api_key).daily_quotes(code_, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_jq_fundamentals(api_key: str, code_: str) -> pd.DataFrame:
    return JQuantsAPIClient(api_key).statements(code_)


@st.cache_data(ttl=3600, show_spinner=False)
def load_jq_supply_bundle(api_key: str, code_: str, end_date: str):
    client = JQuantsAPIClient(api_key)
    end_ts = pd.Timestamp(end_date)
    start90 = (end_ts - pd.Timedelta(days=120)).date().isoformat()
    start180 = (end_ts - pd.Timedelta(days=220)).date().isoformat()
    start365 = (end_ts - pd.Timedelta(days=420)).date().isoformat()
    bundle = {"_errors": {}}
    getters = {
        "breakdown": lambda: client.breakdown(code_, start90, end_date),
        "daily_margin": lambda: client.daily_margin_interest(code_, start180, end_date),
        "weekly_margin": lambda: client.weekly_margin_interest(code_, start365, end_date),
        "shorts": lambda: client.short_selling_positions(code_, start180, end_date),
        "buyback_tdnet": lambda: client.share_buyback_tdnet(code_, start365, end_date),
        "buyback_edinet": lambda: client.share_buyback_edinet(code_),
        "offauction_buyback": lambda: client.off_auction_share_buyback(code_, start365, end_date),
    }
    for key, fn in getters.items():
        try:
            bundle[key] = fn()
        except Exception as e:
            bundle[key] = pd.DataFrame()
            # Keep a sanitized reason so the UI can distinguish "no event" from "plan not available".
            bundle["_errors"][key] = _sanitize_api_error(e)[:160]
    return bundle


def supply_summary_from_bundle(bundle: dict) -> dict:
    return summarize_supply_demand(
        breakdown=bundle.get("breakdown"),
        daily_margin=bundle.get("daily_margin"),
        weekly_margin=bundle.get("weekly_margin"),
        shorts=bundle.get("shorts"),
        buyback_tdnet=bundle.get("buyback_tdnet"),
        buyback_edinet=bundle.get("buyback_edinet"),
        offauction_buyback=bundle.get("offauction_buyback"),
    )


def supply_statuses(bundle: dict) -> list[SourceStatus]:
    # These datasets are structurally sparse (e.g. daily margin only selected issues,
    # short reports only reportable positions, buybacks only when disclosed). Missing
    # rows are therefore not treated as source failures. The supply score itself carries
    # an explicit coverage ratio.
    rows = sum(len(d) for d in bundle.values() if isinstance(d, pd.DataFrame))
    any_data = any(isinstance(d, pd.DataFrame) and not d.empty for d in bundle.values())
    errors = bundle.get("_errors", {}) if isinstance(bundle, dict) else {}
    if any_data and errors:
        msg = f"公式需給データ {rows}件 / プラン等で未取得 {len(errors)}項目"
    elif any_data:
        msg = f"公式需給データ {rows}件"
    elif errors:
        msg = f"需給データ未取得（プラン/権限 {len(errors)}項目・中立扱い）"
    else:
        msg = "該当データなし（中立扱い）"
    return [SourceStatus("JPX/J-Quants", 1, bool(any_data or not errors), msg, None, rows)]


@st.cache_data(ttl=21600, show_spinner=False)
def load_jq_listed(api_key: str, target_date: str | None = None) -> pd.DataFrame:
    return JQuantsAPIClient(api_key).listed_info(target_date)


@st.cache_data(ttl=21600, show_spinner=False)
def load_jq_market_date(api_key: str, target_date: str) -> pd.DataFrame:
    return JQuantsAPIClient(api_key).daily_quotes_for_date(target_date)


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro(key: str, start: str, end: str):
    return build_macro_frame(FREDClient(key), start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_edinet(key: str, code_: str):
    return EDINETClient(key).recent_documents_for_sec_code(code_, days=30)


def prepare_features(px: pd.DataFrame, bm: pd.DataFrame, fundamentals: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    features = add_price_features(px)
    if bm is not None and not bm.empty:
        bmf = add_price_features(bm)[["ret_20", "ret_60", "dist_sma_50", "dist_sma_200", "rsi_14", "vol_20"]]
        features = merge_asof_feature(features, bmf, "market")
        features["relative_strength_20"] = features["ret_20"] - features.get("market_ret_20", 0)
        features["relative_strength_60"] = features["ret_60"] - features.get("market_ret_60", 0)
    features = add_fundamental_features(features, fundamentals)
    if macro is not None and not macro.empty:
        features = add_macro_features(features, macro)
    return features


def analyze_one(
    code: str,
    px: pd.DataFrame,
    bm: pd.DataFrame,
    fundamentals: pd.DataFrame,
    macro: pd.DataFrame,
    statuses: list[SourceStatus],
    edinet_docs: pd.DataFrame,
    supply_summary: dict | None = None,
) -> Dict:
    features = prepare_features(px, bm, fundamentals, macro)
    X_train, y, future_return = make_supervised(features, cfg.horizon_days)
    coverage_by_col = X_train.notna().mean()
    usable_cols = coverage_by_col[coverage_by_col >= 0.60].index.tolist()
    if len(usable_cols) < 5:
        raise ValueError("有効な特徴量が不足しています。")
    X_train = X_train[usable_cols]
    X_live = features[usable_cols].tail(1)
    data_coverage = float(X_live.notna().mean(axis=1).iloc[0])

    wf = walk_forward_ensemble(
        X_train,
        y,
        min_train_rows=cfg.min_train_rows,
        test_rows=cfg.test_rows,
        embargo_rows=cfg.horizon_days,
    )
    live_p = fit_final_probability(X_train, y, X_live)
    src_conf = source_confidence(statuses)
    conf = model_confidence(wf.metrics, src_conf, data_coverage)

    event_risk = False
    event_notes = []
    if edinet_docs is not None and not edinet_docs.empty:
        desc = edinet_docs.get("docDescription", pd.Series(dtype=str)).fillna("").astype(str)
        risky = desc.str.contains("訂正|臨時報告|Correction|Extraordinary", case=False, regex=True)
        if risky.any():
            event_risk = True
            event_notes = desc[risky].head(5).tolist()

    latest = features.iloc[-1]
    comp = buy_score_components(latest, live_p, conf, src_conf, data_coverage, event_risk, supply_summary=supply_summary)
    current_score = comp["buy_score"]

    # Previous comparable score: use the latest available OOS probability for a date before today.
    prev_score = np.nan
    prev_date = None
    if len(wf.probabilities.dropna()) >= 2:
        for dt, p in wf.probabilities.dropna().sort_index(ascending=False).items():
            if dt in features.index and dt < features.index[-1]:
                prev_comp = buy_score_components(features.loc[dt], float(p), conf, src_conf, data_coverage, False)
                prev_score = prev_comp["buy_score"]
                prev_date = dt
                break
    if np.isnan(prev_score):
        # Conservative fallback: compare structural score only, marked as an estimate.
        prior = features.iloc[-2]
        prior_p = 0.5
        prev_comp = buy_score_components(prior, prior_p, conf, src_conf, data_coverage, False)
        prev_score = prev_comp["buy_score"]
        prev_date = features.index[-2]
    score_change = round(current_score - float(prev_score), 1)
    phase = classify_market_phase(latest, current_score, score_change)

    ds = deterministic_score(latest)
    bt = backtest_from_probabilities(px["Close"].reindex(wf.probabilities.index), wf.probabilities, cfg)
    pm = performance_metrics(bt)

    return {
        "code": code,
        "features": features,
        "wf": wf,
        "live_p": live_p,
        "source_conf": src_conf,
        "confidence": conf,
        "coverage": data_coverage,
        "components": comp,
        "score": current_score,
        "prev_score": float(prev_score),
        "prev_date": prev_date,
        "score_change": score_change,
        "phase": phase,
        "label": buy_score_label(current_score),
        "event_risk": event_risk,
        "event_notes": event_notes,
        "edinet_docs": edinet_docs,
        "statuses": statuses,
        "supply_summary": supply_summary or {},
        "deterministic": ds,
        "backtest": bt,
        "performance": pm,
        "future_return": future_return,
        "px": px,
        "bm": bm,
    }


def score_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 42}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.28},
            "steps": [
                {"range": [0, 45]},
                {"range": [45, 60]},
                {"range": [60, 70]},
                {"range": [70, 80]},
                {"range": [80, 90]},
                {"range": [90, 100]},
            ],
            "threshold": {"line": {"width": 4}, "thickness": 0.75, "value": score},
        },
        title={"text": "買い候補スコア"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig



def _finite_float(value, fallback=np.nan) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float(fallback)
    except Exception:
        return float(fallback)


def market_regime_from_benchmark(bm: pd.DataFrame) -> Dict[str, object]:
    """Simple transparent market gate using only confirmed benchmark daily bars."""
    if bm is None or bm.empty or len(bm) < 210:
        return {"state": "不明", "score": 50.0, "reason": "ベンチマーク履歴不足"}
    f = add_price_features(bm).dropna(subset=["Close"])
    if f.empty:
        return {"state": "不明", "score": 50.0, "reason": "ベンチマーク計算不可"}
    r = f.iloc[-1]
    close = _finite_float(r.get("Close"), 0)
    sma20 = _finite_float(r.get("sma_20"), close)
    sma50 = _finite_float(r.get("sma_50"), close)
    sma200 = _finite_float(r.get("sma_200"), close)
    ret20 = _finite_float(r.get("ret_20"), 0)
    ret60 = _finite_float(r.get("ret_60"), 0)
    vol20 = _finite_float(r.get("vol_20"), 0.25)
    score = 0.0
    score += 24 if close > sma200 else 4
    score += 18 if sma50 > sma200 else 5
    score += 16 if sma20 > sma50 else 5
    score += float(np.clip((ret20 + 0.08) / 0.16, 0, 1)) * 18
    score += float(np.clip((ret60 + 0.12) / 0.24, 0, 1)) * 16
    score += float(np.clip((0.38 - vol20) / 0.28, 0, 1)) * 8
    score = float(np.clip(score, 0, 100))
    if score >= 68:
        state = "強気"
    elif score >= 48:
        state = "中立"
    else:
        state = "弱気"
    reason = f"20日 {ret20:+.1%} / 60日 {ret60:+.1%} / 年率ボラ {vol20:.1%}"
    return {"state": state, "score": round(score, 1), "reason": reason}


def oos_edge_profile(result: Dict) -> Dict[str, object]:
    """Choose a probability threshold on earlier OOS rows, evaluate it on later OOS rows."""
    wf = result.get("wf")
    future_return = result.get("future_return", pd.Series(dtype=float))
    if wf is None or future_return is None or len(future_return) == 0:
        return {"threshold": 0.60, "n": 0, "win_rate": np.nan, "avg_return": np.nan, "status": "insufficient"}
    p = pd.to_numeric(wf.probabilities, errors="coerce").dropna()
    r = pd.to_numeric(future_return.reindex(p.index), errors="coerce")
    d = pd.DataFrame({"p": p, "r": r}).dropna().sort_index()
    if len(d) < 80:
        return {"threshold": 0.60, "n": 0, "win_rate": np.nan, "avg_return": np.nan, "status": "insufficient"}
    split = max(int(len(d) * 0.60), 40)
    cal, ev = d.iloc[:split], d.iloc[split:]
    cost = 2.0 * (float(cfg.transaction_cost_bps) + float(slippage_bps)) / 10000.0
    best_t, best_u = 0.60, -1e9
    for t in np.arange(0.52, 0.73, 0.02):
        rr = cal.loc[cal["p"] >= t, "r"] - cost
        if len(rr) < 10:
            continue
        utility = float(rr.mean()) * np.sqrt(len(rr)) + 0.015 * float((rr > 0).mean() - 0.5)
        if utility > best_u:
            best_t, best_u = float(t), utility
    rr = ev.loc[ev["p"] >= best_t, "r"] - cost
    if len(rr) < 6:
        return {"threshold": round(best_t, 2), "n": int(len(rr)), "win_rate": np.nan, "avg_return": np.nan, "status": "small_sample"}
    return {
        "threshold": round(best_t, 2),
        "n": int(len(rr)),
        "win_rate": float((rr > 0).mean()),
        "avg_return": float(rr.mean()),
        "median_return": float(rr.median()),
        "status": "ok" if len(rr) >= 12 else "small_sample",
    }


def practical_readiness(result: Dict) -> Dict[str, object]:
    m = result.get("wf").metrics if result.get("wf") is not None else {}
    pm = result.get("performance", {})
    edge = oos_edge_profile(result)
    regime = market_regime_from_benchmark(result.get("bm", pd.DataFrame()))
    auc = _finite_float(m.get("auc"), np.nan)
    bal = _finite_float(m.get("balanced_accuracy"), 0.5)
    brier = _finite_float(m.get("brier"), 0.25)
    checks = {
        "OOS標本": int(m.get("oos_rows", 0) or 0) >= 252,
        "モデル": ((np.isfinite(auc) and auc >= 0.52) or bal >= 0.52) and brier <= 0.255,
        "確率別優位性": int(edge.get("n", 0) or 0) >= 8 and _finite_float(edge.get("avg_return"), -1) > 0,
        "戦略損益": _finite_float(pm.get("cagr"), -1) > 0 and _finite_float(pm.get("sharpe"), -9) > 0,
        "情報品質": _finite_float(result.get("source_conf"), 0) >= 0.65 and _finite_float(result.get("coverage"), 0) >= 0.65,
        "市場環境": regime.get("state") != "弱気",
    }
    passed = int(sum(bool(v) for v in checks.values()))
    grade = "A" if passed >= 6 else ("B" if passed >= 5 else ("C" if passed >= 4 else "D"))
    return {"grade": grade, "passed": passed, "total": len(checks), "checks": checks, "edge": edge, "regime": regime}


def build_trade_plan(result: Dict) -> Dict[str, object]:
    """Translate the analytical result into a conservative, daily-bar trade plan.

    Prices are reference levels derived from confirmed daily OHLC data. They are not
    real-time quotes and intentionally avoid telling the user to chase a gap-up open.
    """
    f = result.get("features", pd.DataFrame())
    if f is None or f.empty:
        return {}
    f = f.dropna(subset=["Close"]).copy()
    if f.empty:
        return {}
    row = f.iloc[-1]
    close = _finite_float(row.get("Close"), 0.0)
    if close <= 0:
        return {}
    atr_v = _finite_float(row.get("atr_14"), close * 0.03)
    atr_v = max(atr_v, close * 0.005)
    sma20 = _finite_float(row.get("sma_20"), np.nan)
    sma50 = _finite_float(row.get("sma_50"), np.nan)
    rsi_v = _finite_float(row.get("rsi_14"), 50.0)
    score = _finite_float(result.get("score"), 0.0)
    p_up = _finite_float(result.get("live_p"), 0.5)
    conf = _finite_float(result.get("confidence"), 0.0)
    phase = str(result.get("phase", "中立・監視"))
    event_risk = bool(result.get("event_risk", False))

    # Entry zone: phase-specific and deliberately conservative.
    if phase == "初動候補":
        entry_low, entry_high = close - 0.55 * atr_v, close + 0.15 * atr_v
    elif phase == "押し目候補" and np.isfinite(sma20):
        anchor = sma20
        entry_low, entry_high = anchor - 0.35 * atr_v, anchor + 0.35 * atr_v
    elif phase == "上昇継続":
        entry_low, entry_high = close - 0.85 * atr_v, close - 0.20 * atr_v
    elif phase == "底打ち候補":
        entry_low, entry_high = close - 0.55 * atr_v, close + 0.10 * atr_v
    elif phase == "過熱警戒":
        entry_low, entry_high = close - 1.55 * atr_v, close - 0.75 * atr_v
    else:
        entry_low, entry_high = close - 0.75 * atr_v, close - 0.15 * atr_v

    entry_low = max(entry_low, close * 0.70)
    entry_high = max(entry_high, entry_low + max(close * 0.002, 0.01))
    entry_mid = (entry_low + entry_high) / 2.0

    # Structural stop: nearest reasonable support below the entry zone, bounded to
    # roughly 3-10% risk so a single odd candle cannot create an absurd stop.
    recent10_low = _finite_float(f["Low"].tail(10).min() if "Low" in f else np.nan, np.nan)
    candidates = [entry_low - 1.15 * atr_v]
    if np.isfinite(recent10_low) and recent10_low < entry_low:
        candidates.append(recent10_low - 0.20 * atr_v)
    if np.isfinite(sma20) and sma20 < entry_low:
        candidates.append(sma20 - 0.65 * atr_v)
    if np.isfinite(sma50) and sma50 < entry_low:
        candidates.append(sma50 - 0.35 * atr_v)
    structural = max([x for x in candidates if np.isfinite(x) and x < entry_low], default=entry_low - 1.5 * atr_v)
    stop = float(np.clip(structural, entry_mid * 0.90, entry_mid * 0.97))
    if stop >= entry_low:
        stop = min(entry_low - 0.35 * atr_v, entry_mid * 0.97)
    stop = max(stop, 0.01)

    risk = max(entry_mid - stop, close * 0.01)
    recent20_high = _finite_float(f["High"].tail(20).max() if "High" in f else np.nan, np.nan)
    target1 = entry_mid + 1.6 * risk
    if np.isfinite(recent20_high) and recent20_high > entry_mid:
        target1 = max(target1, recent20_high)
    target2 = entry_mid + 2.6 * risk
    target2 = max(target2, target1 + 0.6 * risk)

    ready = practical_readiness(result)
    edge = ready["edge"]
    regime = ready["regime"]
    dynamic_p = max(0.54, float(edge.get("threshold", 0.60)))
    if event_risk or phase in {"売り警戒", "下落基調", "過熱警戒"}:
        decision = "見送り"
    elif regime.get("state") == "弱気":
        decision = "監視"
    elif score >= 86 and p_up >= dynamic_p and conf >= 0.60 and ready["grade"] in {"A", "B"}:
        decision = "買い候補"
    elif score >= 80 and p_up >= max(0.54, dynamic_p - 0.03) and conf >= 0.55 and ready["grade"] in {"A", "B", "C"}:
        decision = "条件付き買い"
    elif score >= 72:
        decision = "監視"
    else:
        decision = "見送り"

    if decision in {"買い候補", "条件付き買い"}:
        if close > entry_high:
            timing = "押し目待ち（上値を追わない）"
        elif close < entry_low:
            timing = "反発確認待ち"
        else:
            timing = "買いゾーン内"
    elif decision == "監視":
        timing = "まだ買わない"
    else:
        timing = "新規買い見送り"

    # If momentum is stretched, prohibit chasing even if the aggregate score is high.
    if rsi_v >= 72 and decision in {"買い候補", "条件付き買い"}:
        timing = "押し目待ち（RSI高め）"

    rr1 = (target1 - entry_mid) / max(entry_mid - stop, 1e-9)
    rr2 = (target2 - entry_mid) / max(entry_mid - stop, 1e-9)
    risk_budget = float(account_capital) * float(risk_per_trade_pct) / 100.0
    risk_per_share = max(entry_mid - stop, 0.01)
    shares_risk = int(risk_budget // (risk_per_share * 100)) * 100
    shares_alloc = int((float(account_capital) * float(max_position_pct) / 100.0) // (entry_mid * 100)) * 100
    suggested_shares = max(0, min(shares_risk, shares_alloc))
    position_value = suggested_shares * entry_mid
    estimated_loss = suggested_shares * risk_per_share
    base_date = pd.Timestamp(f.index[-1]).date().isoformat()
    return {
        "判定": decision,
        "買いタイミング": timing,
        "基準日": base_date,
        "基準終値": round(close, 1),
        "買い下限": round(entry_low, 1),
        "買い上限": round(entry_high, 1),
        "損切り": round(stop, 1),
        "利確1": round(target1, 1),
        "利確2": round(target2, 1),
        "RR1": round(rr1, 2),
        "RR2": round(rr2, 2),
        "市場環境": regime.get("state", "不明"),
        "市場環境点": regime.get("score", 50.0),
        "実戦準備度": ready.get("grade", "D"),
        "準備通過": f"{ready.get('passed',0)}/{ready.get('total',6)}",
        "AI必要確率": round(dynamic_p * 100, 1),
        "OOS期待値": edge.get("avg_return", np.nan),
        "OOS件数": edge.get("n", 0),
        "推奨株数": suggested_shares,
        "想定投資額": round(position_value, 0),
        "想定最大損失": round(estimated_loss, 0),
        "売りルール": f"{stop:,.1f}円割れで撤退。{target1:,.1f}円で半分利確後、残りの損切りを建値近辺へ引き上げ、{target2:,.1f}円または終値が20日線を明確に割れたら利確/縮小。",
    }


def render_trade_plan(result: Dict, title: str = "売買プラン") -> None:
    plan = build_trade_plan(result)
    if not plan:
        st.info("売買プランを計算できる価格データが不足しています。")
        return
    st.markdown(f"### {title}")
    decision = str(plan["判定"])
    timing = str(plan["買いタイミング"])
    if decision == "買い候補":
        st.success(f"**判定：{decision}** ｜ {timing}")
    elif decision == "条件付き買い":
        st.warning(f"**判定：{decision}** ｜ {timing}")
    elif decision == "監視":
        st.info(f"**判定：{decision}** ｜ {timing}")
    else:
        st.error(f"**判定：{decision}** ｜ {timing}")
    st.markdown(
        f"**買い目安：{plan['買い下限']:,.1f}〜{plan['買い上限']:,.1f}円**  ｜  "
        f"損切り：**{plan['損切り']:,.1f}円**  ｜  利確①：**{plan['利確1']:,.1f}円**  ｜  利確②：**{plan['利確2']:,.1f}円**"
    )
    st.markdown(f"**資金管理例**：推奨 **{int(plan['推奨株数']):,}株** ｜ 投資額 約**{plan['想定投資額']:,.0f}円** ｜ 損切り時の想定損失 約**{plan['想定最大損失']:,.0f}円**")
    oos = plan.get("OOS期待値", np.nan)
    oos_txt = f"{oos:+.1%}" if np.isfinite(_finite_float(oos, np.nan)) else "未判定"
    st.caption(f"基準日 {plan['基準日']}・終値 {plan['基準終値']:,.1f}円。市場環境 {plan['市場環境']}({plan['市場環境点']:.0f}点) / 実戦準備度 {plan['実戦準備度']}({plan['準備通過']}) / OOS期待値 {oos_txt}・n={plan['OOS件数']} / AI必要確率 {plan['AI必要確率']:.1f}% / RR ①{plan['RR1']:.2f} ②{plan['RR2']:.2f}。")
    st.caption("J-Quants Standard単体の本システムは確定日足を基準にします。J-Quantsには分足・Tickの有料アドオンもありますが日次配信でリアルタイム配信ではないため、この価格はリアルタイム注文価格ではありません。寄付きが買い上限を超える場合は追いかけず再判定します。")
    st.caption(str(plan["売りルール"]))


def component_table(result: Dict) -> pd.DataFrame:
    comp = result["components"]
    labels = {
        "trend": ("トレンド", 20),
        "momentum": ("モメンタム・初動性", 12),
        "participation": ("出来高・資金流入", 10),
        "relative_market": ("市場比・相対強度", 7),
        "fundamental": ("業績・ファンダ", 13),
        "entry_quality": ("リスク・買い位置", 8),
        "ml": ("AI予測", 10),
        "evidence": ("情報信頼度", 8),
        "supply_demand": ("信用・空売り・自己株買い需給", 12),
    }
    rows = []
    for key, (name, maxp) in labels.items():
        rows.append({
            "評価項目": name,
            "獲得点": comp[f"points_{key}"],
            "満点": maxp,
            "内部評価": comp[f"quality_{key}"],
        })
    return pd.DataFrame(rows)


market_tab, main_tab, scanner_tab, validation_tab, adaptive_tab, guide_tab = st.tabs(["🌐 今日", "🔎 個別", "🏆 100点", "🧪 検証", "🧠 配点", "📘 仕組み"])

with main_tab:
    left, right = st.columns([1.15, 0.85])
    with left:
        code = st.text_input("銘柄コード", "7203", key="single_code")
        mode = st.radio("価格データ", ["J-Quants公式API", "公式CSV"], horizontal=True)
        price_upload = benchmark_upload = None
        if mode == "公式CSV":
            c1, c2 = st.columns(2)
            price_upload = c1.file_uploader("対象株の公式CSV", type=["csv"], key="px_csv")
            benchmark_upload = c2.file_uploader("ベンチマーク公式CSV（任意）", type=["csv"], key="bm_csv")
    with right:
        st.info("**目安**\n\n90点以上: 最有力候補\n\n80–89点: 買い候補\n\n70–79点: 監視強化\n\nただし『過熱警戒』なら高得点でも追い買いを避ける設計です。")

    if st.button("この銘柄を分析", type="primary", use_container_width=True):
        try:
            end = date.today()
            start = end - timedelta(days=int(years * 365.25 + 300))
            statuses: list[SourceStatus] = []

            if mode == "J-Quants公式API":
                if not jq_connected:
                    st.error("J-Quants APIへ正常接続できていません。画面上部の接続状態を確認してください。非公式価格サイトには自動切替しません。")
                    st.stop()
                px = load_jq_quotes(jq_api_key, code, start.isoformat(), end.isoformat())
                bm = load_jq_quotes(jq_api_key, benchmark_code, start.isoformat(), end.isoformat())
                fundamentals = load_jq_fundamentals(jq_api_key, code)
                statuses += [
                    SourceStatus("JPX/J-Quants", 1, not px.empty, "対象株価", px.index.max() if not px.empty else None, len(px)),
                    SourceStatus("JPX/J-Quants", 1, not bm.empty, "市場ベンチマーク", bm.index.max() if not bm.empty else None, len(bm)),
                    SourceStatus("JPX/J-Quants", 1, not fundamentals.empty, "四半期財務", fundamentals.index.max() if not fundamentals.empty else None, len(fundamentals)),
                ]
            else:
                if price_upload is None:
                    st.error("対象株の公式CSVをアップロードしてください。")
                    st.stop()
                px = parse_official_csv(price_upload)
                bm = parse_official_csv(benchmark_upload) if benchmark_upload else pd.DataFrame()
                fundamentals = pd.DataFrame()
                statuses.append(SourceStatus("JPX/J-Quants", 1, True, "公式CSV", px.index.max(), len(px)))
                if not bm.empty:
                    statuses.append(SourceStatus("JPX/J-Quants", 1, True, "ベンチマーク公式CSV", bm.index.max(), len(bm)))

            if len(px) < cfg.min_train_rows + cfg.test_rows + cfg.horizon_days:
                raise ValueError("学習に必要な価格履歴が不足しています。分析期間を長くしてください。")

            macro = pd.DataFrame()
            if fred_key:
                macro, macro_status = load_macro(fred_key, start.isoformat(), end.isoformat())
                statuses += macro_status
            else:
                statuses.append(SourceStatus("FRED/ALFRED", 1, False, "API key未設定"))

            edinet_docs = pd.DataFrame()
            if edinet_key:
                try:
                    edinet_docs = load_edinet(edinet_key, code)
                    statuses.append(SourceStatus("EDINET/FSA", 1, True, f"直近30日 {len(edinet_docs)}件", pd.Timestamp.today(), len(edinet_docs)))
                except Exception as e:
                    statuses.append(SourceStatus("EDINET/FSA", 1, False, f"取得失敗: {e}"))
            else:
                statuses.append(SourceStatus("EDINET/FSA", 1, False, "API key未設定"))

            supply_summary = {}
            supply_bundle = {}
            if mode == "J-Quants公式API":
                supply_bundle = load_jq_supply_bundle(jq_api_key, code, end.isoformat())
                supply_summary = supply_summary_from_bundle(supply_bundle)
                statuses += supply_statuses(supply_bundle)

            with st.spinner("公式データを検証し、AI・需給・100点評価を計算しています…"):
                result = analyze_one(code, px, bm, fundamentals, macro, statuses, edinet_docs, supply_summary)
                result["supply_bundle"] = supply_bundle
                result["px"] = px
                result["bm"] = bm

            st.session_state["latest_result"] = result
        except Exception as e:
            st.exception(e)

    if "latest_result" in st.session_state:
        r = st.session_state["latest_result"]
        score = r["score"]
        c1, c2 = st.columns([0.9, 1.1])
        with c1:
            st.plotly_chart(score_gauge(score), use_container_width=True)
        with c2:
            st.markdown(f"<div class='phase'>{r['phase']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-score'>{score:.1f}<span style='font-size:1.4rem'> / 100</span></div>", unsafe_allow_html=True)
            st.markdown(f"**{r['label']}**")
            delta_prefix = "+" if r["score_change"] >= 0 else ""
            st.markdown(f"前回比較: **{delta_prefix}{r['score_change']:.1f}点**　｜　AI {horizon}営業日上昇確率: **{r['live_p']*100:.1f}%**")
            st.markdown(f"総合信頼度: **{r['confidence']*100:.1f}%**　｜　情報源信頼度: **{r['source_conf']*100:.1f}%**")
            if r["event_risk"]:
                st.warning("EDINETに訂正・臨時報告等の重要開示候補があります。買い候補スコアに上限をかけています。")

        headline = st.columns(6)
        headline[0].metric("終値", f"{r['features']['Close'].iloc[-1]:,.1f}")
        headline[1].metric("20日騰落", f"{r['features']['ret_20'].iloc[-1]*100:+.1f}%")
        headline[2].metric("RSI14", f"{r['features']['rsi_14'].iloc[-1]:.1f}")
        headline[3].metric("売買代金比", f"{r['features']['turnover_ratio_20'].iloc[-1]:.2f}x")
        headline[4].metric("52週高値比", f"{r['features']['pct_from_52w_high'].iloc[-1]*100:+.1f}%")
        sq = float(r.get("supply_summary", {}).get("supply_demand_quality", 50.0))
        headline[5].metric("需給スコア", f"{sq:.1f}/100")

        render_trade_plan(r, "この銘柄の売買プラン")

        detail_tabs = st.tabs(["100点内訳", "需給詳細", "チャート", "AI検証", "情報源監査", "重要開示"])
        with detail_tabs[0]:
            ct = component_table(r)
            st.dataframe(ct, use_container_width=True, hide_index=True, column_config={
                "獲得点": st.column_config.ProgressColumn("獲得点", min_value=0, max_value=22, format="%.1f"),
                "内部評価": st.column_config.ProgressColumn("内部評価", min_value=0, max_value=100, format="%.0f%%"),
            })
            best = ct.sort_values("獲得点", ascending=False).head(3)["評価項目"].tolist()
            weak = ct.sort_values("内部評価", ascending=True).head(2)["評価項目"].tolist()
            st.success("強み: " + " / ".join(best))
            st.warning("弱み・確認項目: " + " / ".join(weak))

        with detail_tabs[1]:
            ss = r.get("supply_summary", {})
            st.markdown(f"### {ss.get('supply_demand_signal', '需給データ未取得')}")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("需給内部評価", f"{float(ss.get('supply_demand_quality', 50)):.1f}/100")
            sc2.metric("需給データ充足", f"{float(ss.get('supply_demand_coverage', 0))*100:.0f}%")
            sc3.metric("自己株買い", "あり" if ss.get('buyback_active') else "確認なし")
            rows_sd = []
            label_map = {
                "cash_buy_share":"現物買い比率", "margin_new_buy_share":"信用新規買い比率",
                "shorting_share":"空売り・信用新規売り比率", "short_cover_share":"信用返済買い比率",
                "margin_ratio":"信用倍率(週次)", "margin_ratio_change4w":"信用倍率4週変化",
                "reportable_short_ratio_sum":"報告対象空売り残高比率合計",
                "reportable_short_ratio_change":"空売り残高比率変化",
                "buyback_progress":"TDnet自己株買い進捗", "buyback_edinet_progress_pct":"EDINET自己株買い進捗%",
            }
            for k, name in label_map.items():
                if k in ss and pd.notna(ss[k]):
                    rows_sd.append({"需給指標": name, "値": ss[k]})
            if rows_sd:
                st.dataframe(pd.DataFrame(rows_sd), use_container_width=True, hide_index=True)
            st.caption("日々公表信用残高は対象銘柄のみ収録されるため、未掲載を弱材料とは扱いません。空売り残高報告も0.5%以上の報告対象が中心です。")

        with detail_tabs[2]:
            f = r["features"].tail(300)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=f.index, open=f["Open"], high=f["High"], low=f["Low"], close=f["Close"], name="株価"))
            for n in [20, 50, 200]:
                if f"sma_{n}" in f.columns:
                    fig.add_trace(go.Scatter(x=f.index, y=f[f"sma_{n}"], mode="lines", name=f"SMA{n}"))
            fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with detail_tabs[3]:
            m = r["wf"].metrics
            cols = st.columns(4)
            cols[0].metric("ROC-AUC", f"{m.get('auc', float('nan')):.3f}")
            cols[1].metric("Balanced Accuracy", f"{m.get('balanced_accuracy', 0):.3f}")
            cols[2].metric("Brier score", f"{m.get('brier', 0):.3f}")
            cols[3].metric("OOS予測数", f"{int(m.get('oos_rows', 0)):,}")
            pm = r["performance"]
            pcols = st.columns(5)
            pcols[0].metric("戦略累積", f"{pm.get('total', 0)*100:.1f}%")
            pcols[1].metric("CAGR", f"{pm.get('cagr', 0)*100:.1f}%")
            pcols[2].metric("Sharpe", f"{pm.get('sharpe', float('nan')):.2f}")
            pcols[3].metric("最大DD", f"{pm.get('maxdd', 0)*100:.1f}%")
            pcols[4].metric("Buy & Hold", f"{pm.get('buy_hold', 0)*100:.1f}%")
            if not r["backtest"].empty:
                st.line_chart(r["backtest"][["StrategyEquity", "BuyHoldEquity"]])
            st.caption("時系列をシャッフルしないWalk-forward検証。予測期間と同じembargoを置き、未来情報の混入を抑えています。")

        with detail_tabs[4]:
            audit = pd.DataFrame([{
                "情報源": s.name,
                "Tier": s.tier,
                "状態": "OK" if s.ok else "未取得",
                "用途/メッセージ": s.message,
                "最終観測": s.last_observation,
                "件数": s.rows,
            } for s in r["statuses"]])
            st.dataframe(audit, use_container_width=True, hide_index=True)

        with detail_tabs[5]:
            if r["edinet_docs"] is None or r["edinet_docs"].empty:
                st.info("直近EDINET情報はありません、またはAPIキー未設定です。")
            else:
                cols = [c for c in ["submitDateTime", "docDescription", "docTypeCode", "edinetCode", "secCode"] if c in r["edinet_docs"].columns]
                st.dataframe(r["edinet_docs"][cols], use_container_width=True, hide_index=True)


with market_tab:
    st.subheader("本日の日本株レーダー（前営業日確定データ基準）")
    st.caption("東証プライム・スタンダード・グロースの普通株を公式J-Quantsデータで横断スキャンします。V8は高得点だけで買わず、市場環境・OOS優位性・リスク量を通過した候補だけを『買い候補』にします。")

    mc1, mc2, mc3, mc4 = st.columns(4)
    radar_days = mc1.selectbox("全市場履歴", [330, 390, 450], index=1, format_func=lambda x: f"約{x}暦日")
    deep_n = mc2.selectbox("最終精査する上位数", [10, 20, 30, 50], index=1)
    min_turn_m = mc3.number_input("最低売買代金/日（百万円）", min_value=0, max_value=5000, value=50, step=50, help="百万円。流動性が極端に低い銘柄を除外します。")
    preferred = mc4.multiselect("優先レーダー", ["需給先回り候補", "初動レーダー", "押し目レーダー", "上昇継続レーダー", "過熱警戒", "需給悪化警戒", "監視"], default=["需給先回り候補", "初動レーダー", "押し目レーダー", "上昇継続レーダー"])

    st.info("初回だけ過去の全市場日次データを公式APIから収集します。取得済み日付はサーバー側にキャッシュするため、2回目以降は差分中心になります。")
    st.caption("データ範囲はJ-Quants契約プランに連動します。Light以上で現在株価、Standard以上で信用・空売り、Premiumで売買内訳が追加されます。取得できない項目は推測せず、需給充足度を下げて評価します。")

    if st.button("本日の結論を更新（全市場→100点→実戦ゲート→売買プラン）", type="primary", use_container_width=True, key="all_market_scan"):
        st.session_state["auto_deep_scan_requested"] = False
        if not jq_connected:
            st.error("J-Quants APIへ正常接続できていません。画面上部の接続状態を確認してください。")
        elif not jq_live_eligible:
            st.error("現在のJ-Quantsデータは遅延しています。遅延データを『今日の買い候補』として表示すると危険なので停止しました。ライブ候補にはLight以上が必要です。")
        else:
            progress = st.progress(0, text="全市場データを準備中")
            try:
                listed = load_jq_listed(jq_api_key, None)
                universe = filter_common_equities(listed)
                universe_codes = set(universe["Code"].astype(str)) if not universe.empty else set()
                scan_cfg = ScanConfig(
                    lookback_calendar_days=int(radar_days),
                    min_price_rows=205,
                    min_turnover_yen=float(min_turn_m) * 1_000_000,
                    cache_dir=".scanner_cache",
                )

                def _pcb(frac, txt):
                    progress.progress(float(frac), text=txt)

                panel, stats = collect_market_panel(
                    lambda dt: load_jq_market_date(jq_api_key, dt),
                    date.today(),
                    scan_cfg,
                    progress_cb=_pcb,
                )
                if panel.empty:
                    raise ValueError("全市場の日次株価を取得できませんでした。APIキー、契約プラン、データ提供期間、通信状態を確認してください。")
                # Calculate relative strength while the benchmark ETF is still in the panel,
                # then restrict the resulting universe to common equities.
                radar = fast_market_features(panel, benchmark_code=benchmark_code)
                if universe_codes and not radar.empty:
                    radar = radar[radar["Code"].astype(str).isin(universe_codes)].copy()
                radar = enrich_with_listed(radar, universe)
                radar = radar[radar["rows"] >= scan_cfg.min_price_rows]
                radar = radar[radar["TurnoverValue"].fillna(0) >= scan_cfg.min_turnover_yen]
                progress.progress(0.05, text="公式需給データを横断集計中")
                client = JQuantsAPIClient(jq_api_key)
                supply_frames = collect_recent_supply_market(client, date.today(), cache_dir=".supply_cache", progress_cb=_pcb)
                supply_snap = build_market_supply_snapshot(
                    supply_frames.get("breakdown", pd.DataFrame()),
                    supply_frames.get("daily_margin", pd.DataFrame()),
                    supply_frames.get("weekly_margin", pd.DataFrame()),
                    supply_frames.get("shorts", pd.DataFrame()),
                    supply_frames.get("buyback_tdnet", pd.DataFrame()),
                )
                radar = merge_supply_snapshot(radar, supply_snap)
                st.session_state["market_supply_snapshot"] = supply_snap
                st.session_state["market_radar"] = radar
                st.session_state["market_listed"] = universe
                st.session_state["market_scan_stats"] = stats
                st.session_state["auto_deep_scan_requested"] = True
                progress.empty()
            except Exception as e:
                progress.empty()
                st.exception(e)

    if "market_radar" in st.session_state:
        radar = st.session_state["market_radar"].copy()
        stats = st.session_state.get("market_scan_stats", {})
        st.caption(f"対象 {len(radar):,}銘柄 ｜ 新規取得 {stats.get('fetched',0)}日 ｜ キャッシュ {stats.get('cached',0)}日 ｜ 休場等 {stats.get('empty',0)}日 ｜ 失敗 {stats.get('failed',0)}日")

        if preferred:
            visible = radar[radar["early_signal"].isin(preferred)].copy()
        else:
            visible = radar.copy()

        show_cols = [c for c in ["Code", "CompanyName", "MarketCodeName", "Sector33CodeName", "early_radar_score", "radar_score", "supply_demand_quality", "supply_demand_coverage", "radar_change", "radar_change5", "early_signal", "Close", "ret20", "ret60", "rsi14", "turnover_ratio20", "TurnoverValue"] if c in visible.columns]
        display = visible[show_cols].head(100).copy()
        ren = {
            "CompanyName": "会社名", "MarketCodeName": "市場", "Sector33CodeName": "業種",
            "early_radar_score": "先回りレーダー", "radar_score": "価格レーダー", "supply_demand_quality": "需給スコア",
            "supply_demand_coverage": "需給充足", "radar_change": "1日変化", "radar_change5": "5日変化",
            "early_signal": "一次判定", "Close": "終値", "ret20": "20日騰落", "ret60": "60日騰落",
            "rsi14": "RSI14", "turnover_ratio20": "売買代金比", "TurnoverValue": "売買代金",
        }
        display = display.rename(columns=ren)
        for c in ["20日騰落", "60日騰落"]:
            if c in display.columns:
                display[c] = display[c] * 100
        with st.expander("一次レーダー詳細（上級者向け・上位100件）", expanded=False):
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "先回りレーダー": st.column_config.ProgressColumn("先回りレーダー", min_value=0, max_value=100, format="%.1f"),
                    "価格レーダー": st.column_config.ProgressColumn("価格レーダー", min_value=0, max_value=100, format="%.1f"),
                    "需給スコア": st.column_config.ProgressColumn("需給スコア", min_value=0, max_value=100, format="%.1f"),
                    "需給充足": st.column_config.ProgressColumn("需給充足", min_value=0, max_value=1, format="%.0f%%"),
                    "1日変化": st.column_config.NumberColumn("1日変化", format="%+.1f"),
                    "5日変化": st.column_config.NumberColumn("5日変化", format="%+.1f"),
                    "20日騰落": st.column_config.NumberColumn("20日騰落", format="%+.1f%%"),
                    "60日騰落": st.column_config.NumberColumn("60日騰落", format="%+.1f%%"),
                    "売買代金比": st.column_config.NumberColumn("売買代金比", format="%.2fx"),
                    "売買代金": st.column_config.NumberColumn("売買代金", format="%,.0f"),
                },
            )

        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.metric("需給先回り候補数", int((radar["early_signal"] == "需給先回り候補").sum()))
        rr2.metric("初動候補数", int((radar["early_signal"] == "初動レーダー").sum()))
        rr3.metric("押し目候補数", int((radar["early_signal"] == "押し目レーダー").sum()))
        rr4.metric("先回り80点以上", int((radar["early_radar_score"] >= 80).sum()))

        st.markdown("### 上位候補を最終100点評価")
        st.caption("ここからは価格だけでなく、財務・AIのWalk-forward検証・情報信頼度を含めて再評価します。最終100点ランキングはこちらの結果を採用してください。")
        manual_deep = st.button(f"上位 {deep_n} 銘柄を再精査", type="secondary", use_container_width=True, key="deep_scan_top")
        auto_deep = bool(st.session_state.pop("auto_deep_scan_requested", False))
        if manual_deep or auto_deep:
            candidates = visible.head(int(deep_n))["Code"].astype(str).tolist()
            if not candidates:
                st.warning("優先レーダー条件に該当する候補がありません。フィルターを広げてください。")
            else:
                end = date.today()
                start = end - timedelta(days=int(years * 365.25 + 300))
                dprog = st.progress(0, text="最終100点評価を開始")
                try:
                    bm = load_jq_quotes(jq_api_key, benchmark_code, start.isoformat(), end.isoformat())
                except Exception:
                    bm = pd.DataFrame()
                macro = pd.DataFrame()
                macro_status = []
                if fred_key:
                    try:
                        macro, macro_status = load_macro(fred_key, start.isoformat(), end.isoformat())
                    except Exception:
                        pass
                final_rows = []
                listed_meta = st.session_state.get("market_listed", pd.DataFrame())
                name_map = dict(zip(listed_meta.get("Code", pd.Series(dtype=str)).astype(str), listed_meta.get("CompanyName", pd.Series(dtype=str)))) if not listed_meta.empty else {}
                for i, scode in enumerate(candidates):
                    dprog.progress((i + 1) / max(len(candidates), 1), text=f"{scode} を最終精査中")
                    try:
                        px = load_jq_quotes(jq_api_key, scode, start.isoformat(), end.isoformat())
                        fundamentals = load_jq_fundamentals(jq_api_key, scode)
                        statuses = [
                            SourceStatus("JPX/J-Quants", 1, not px.empty, "対象株価", px.index.max() if not px.empty else None, len(px)),
                            SourceStatus("JPX/J-Quants", 1, not bm.empty, "市場ベンチマーク", bm.index.max() if not bm.empty else None, len(bm)),
                            SourceStatus("JPX/J-Quants", 1, not fundamentals.empty, "四半期財務", fundamentals.index.max() if not fundamentals.empty else None, len(fundamentals)),
                        ] + macro_status
                        supply_bundle = load_jq_supply_bundle(jq_api_key, scode, end.isoformat())
                        ss = supply_summary_from_bundle(supply_bundle)
                        statuses += supply_statuses(supply_bundle)
                        # Batch deep scan skips the separate EDINET document crawl for speed, but J-Quants official buyback/market data are included.
                        res = analyze_one(scode, px, bm, fundamentals, macro, statuses, pd.DataFrame(), ss)
                        rr = radar[radar["Code"].astype(str) == scode].head(1)
                        plan = build_trade_plan(res)
                        final_rows.append({
                            "Code": scode,
                            "会社名": name_map.get(scode, ""),
                            "判定": plan.get("判定", ""),
                            "買いタイミング": plan.get("買いタイミング", ""),
                            "買い候補100点": res["score"],
                            "前回比": res["score_change"],
                            "相場フェーズ": res["phase"],
                            f"AI {horizon}日上昇確率": round(res["live_p"] * 100, 1),
                            "総合信頼度": round(res["confidence"] * 100, 1),
                            "先回りレーダー": round(float(rr["early_radar_score"].iloc[0]), 1) if not rr.empty and "early_radar_score" in rr else np.nan,
                            "需給スコア": round(float(ss.get("supply_demand_quality", 50)), 1),
                            "需給判定": ss.get("supply_demand_signal", ""),
                            "20日騰落%": round(res["features"]["ret_20"].iloc[-1] * 100, 1),
                            "売買代金比": round(res["features"]["turnover_ratio_20"].iloc[-1], 2),
                            "基準日": plan.get("基準日", ""),
                            "基準終値": plan.get("基準終値", np.nan),
                            "買い下限": plan.get("買い下限", np.nan),
                            "買い上限": plan.get("買い上限", np.nan),
                            "損切り": plan.get("損切り", np.nan),
                            "利確1": plan.get("利確1", np.nan),
                            "利確2": plan.get("利確2", np.nan),
                            "RR1": plan.get("RR1", np.nan),
                            "市場環境": plan.get("市場環境", ""),
                            "実戦準備度": plan.get("実戦準備度", ""),
                            "OOS期待値%": round(_finite_float(plan.get("OOS期待値"), np.nan) * 100, 2) if np.isfinite(_finite_float(plan.get("OOS期待値"), np.nan)) else np.nan,
                            "OOS件数": plan.get("OOS件数", 0),
                            "推奨株数": plan.get("推奨株数", 0),
                            "想定投資額": plan.get("想定投資額", 0),
                            "想定最大損失": plan.get("想定最大損失", 0),
                            "評価": res["label"],
                        })
                    except Exception as e:
                        # A single candidate must never crash the whole market scan.
                        # Keep the row shape stable so DataFrame construction/sorting is safe even
                        # when every candidate failed (e.g. temporary API/plan/data issue).
                        err = _sanitize_api_error(e)[:180]
                        final_rows.append({
                            "Code": scode,
                            "会社名": name_map.get(scode, ""),
                            "判定": "見送り",
                            "買いタイミング": "分析エラーのため見送り",
                            "買い候補100点": np.nan,
                            "前回比": np.nan,
                            "相場フェーズ": "分析失敗",
                            f"AI {horizon}日上昇確率": np.nan,
                            "総合信頼度": np.nan,
                            "先回りレーダー": np.nan,
                            "需給スコア": np.nan,
                            "需給判定": "未判定",
                            "20日騰落%": np.nan,
                            "売買代金比": np.nan,
                            "基準日": "",
                            "基準終値": np.nan,
                            "買い下限": np.nan,
                            "買い上限": np.nan,
                            "損切り": np.nan,
                            "利確1": np.nan,
                            "利確2": np.nan,
                            "RR1": np.nan,
                            "市場環境": "未判定",
                            "実戦準備度": "D",
                            "OOS期待値%": np.nan,
                            "OOS件数": 0,
                            "推奨株数": 0,
                            "想定投資額": 0,
                            "想定最大損失": 0,
                            "評価": "分析失敗",
                            "分析エラー": err,
                        })
                dprog.empty()
                final_rank = pd.DataFrame(final_rows)
                # Defensive schema: a failed batch can otherwise have no sort columns at all.
                required_defaults = {
                    "買い候補100点": np.nan, "前回比": np.nan, "判定": "見送り",
                    "会社名": "", "相場フェーズ": "分析失敗", "分析エラー": "",
                }
                for col, default in required_defaults.items():
                    if col not in final_rank.columns:
                        final_rank[col] = default
                if not final_rank.empty:
                    final_rank = final_rank.sort_values(
                        ["買い候補100点", "前回比"],
                        ascending=[False, False],
                        na_position="last",
                    )
                ok_n = int(pd.to_numeric(final_rank["買い候補100点"], errors="coerce").notna().sum()) if not final_rank.empty else 0
                fail_n = int(len(final_rank) - ok_n)
                st.session_state["deep_market_rank"] = final_rank
                st.session_state["deep_market_scan_counts"] = {"ok": ok_n, "failed": fail_n}

        if "deep_market_rank" in st.session_state:
            final_rank = st.session_state["deep_market_rank"].copy()
            counts = st.session_state.get("deep_market_scan_counts", {})
            ok_n = int(counts.get("ok", pd.to_numeric(final_rank.get("買い候補100点", pd.Series(dtype=float)), errors="coerce").notna().sum()))
            fail_n = int(counts.get("failed", max(len(final_rank) - ok_n, 0)))
            if fail_n:
                if ok_n == 0:
                    st.error("最終精査した銘柄を正常評価できませんでした。買い判断は出さず、下のエラー概要を確認してください。")
                else:
                    st.warning(f"最終精査 {ok_n}銘柄成功 / {fail_n}銘柄失敗。失敗銘柄は自動的に見送り扱いです。")
                if "分析エラー" in final_rank.columns:
                    err_view = final_rank.loc[final_rank["分析エラー"].fillna("").astype(str).ne(""), [c for c in ["Code", "会社名", "分析エラー"] if c in final_rank.columns]].head(10)
                    if not err_view.empty:
                        with st.expander("失敗理由を確認（APIキー等の秘密情報は表示しません）"):
                            st.dataframe(err_view, use_container_width=True, hide_index=True)
            st.markdown("## 今日の結論")
            st.caption("最終100点評価とAI・需給を通過した上位候補です。価格はJ-Quantsの確定日足から計算した目安で、リアルタイム気配ではありません。")
            priority = {"買い候補": 0, "条件付き買い": 1, "監視": 2, "見送り": 3}
            if "判定" in final_rank.columns:
                final_rank["_priority"] = final_rank["判定"].map(priority).fillna(9)
                score_num = pd.to_numeric(final_rank.get("買い候補100点", pd.Series(index=final_rank.index, dtype=float)), errors="coerce")
                actionable = final_rank[final_rank["判定"].isin(["買い候補", "条件付き買い"]) & score_num.notna()].copy()
                conclusion = actionable.sort_values(["_priority", "買い候補100点", "前回比"], ascending=[True, False, False], na_position="last").head(5)
            else:
                conclusion = pd.DataFrame()
            if conclusion.empty:
                st.info("**本日は新規買いなし**。実戦ゲートを通過した銘柄がありません。条件を緩めて無理に買わないことをV8の正式な判断とします。")
            else:
                for rank_no, (_, row_) in enumerate(conclusion.iterrows(), 1):
                    code_ = str(row_.get("Code", ""))
                    name_ = str(row_.get("会社名", ""))
                    decision_ = str(row_.get("判定", "監視"))
                    timing_ = str(row_.get("買いタイミング", ""))
                    score_ = _finite_float(row_.get("買い候補100点"), np.nan)
                    ai_ = _finite_float(row_.get(f"AI {horizon}日上昇確率"), np.nan)
                    line1 = f"**{rank_no}位　{code_} {name_}｜{decision_}**　{score_:.1f}点 / AI上昇確率 {ai_:.1f}%"
                    details = (
                        f"{timing_}　｜　買い目安 **{_finite_float(row_.get('買い下限')):,.1f}〜{_finite_float(row_.get('買い上限')):,.1f}円**　｜　"
                        f"損切り **{_finite_float(row_.get('損切り')):,.1f}円**　｜　利確① **{_finite_float(row_.get('利確1')):,.1f}円**　｜　利確② **{_finite_float(row_.get('利確2')):,.1f}円**\n\n"
                        f"市場 **{row_.get('市場環境','')}** ｜ 実戦準備度 **{row_.get('実戦準備度','')}** ｜ OOS期待値 **{_finite_float(row_.get('OOS期待値%'), np.nan):+.2f}%** (n={int(_finite_float(row_.get('OOS件数'),0))}) ｜ 推奨 **{int(_finite_float(row_.get('推奨株数'),0)):,}株**"
                    )
                    if decision_ == "買い候補":
                        st.success(line1 + "\n\n" + details)
                    elif decision_ == "条件付き買い":
                        st.warning(line1 + "\n\n" + details)
                    elif decision_ == "監視":
                        st.info(line1 + "\n\n" + details)
                    else:
                        st.error(line1 + "\n\n" + details)
            st.caption("売りの基本：損切りに達したら撤退。利確①で半分を落とし、残りのストップを建値近辺へ引き上げます。利確②または20日線割れで残りを縮小。寄付きが買い上限より上なら追い買いしません。")

            with st.expander("最終100点ランキング詳細", expanded=False):
                show_rank = final_rank.drop(columns=["_priority"], errors="ignore")
                st.dataframe(
                    show_rank,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "買い候補100点": st.column_config.ProgressColumn("買い候補100点", min_value=0, max_value=100, format="%.1f"),
                        "前回比": st.column_config.NumberColumn("前回比", format="%+.1f"),
                        f"AI {horizon}日上昇確率": st.column_config.ProgressColumn(f"AI {horizon}日上昇確率", min_value=0, max_value=100, format="%.1f%%"),
                        "総合信頼度": st.column_config.ProgressColumn("総合信頼度", min_value=0, max_value=100, format="%.1f%%"),
                        "先回りレーダー": st.column_config.ProgressColumn("先回りレーダー", min_value=0, max_value=100, format="%.1f"),
                        "需給スコア": st.column_config.ProgressColumn("需給スコア", min_value=0, max_value=100, format="%.1f"),
                        "基準終値": st.column_config.NumberColumn("基準終値", format="%,.1f円"),
                        "買い下限": st.column_config.NumberColumn("買い下限", format="%,.1f円"),
                        "買い上限": st.column_config.NumberColumn("買い上限", format="%,.1f円"),
                        "損切り": st.column_config.NumberColumn("損切り", format="%,.1f円"),
                        "利確1": st.column_config.NumberColumn("利確1", format="%,.1f円"),
                        "利確2": st.column_config.NumberColumn("利確2", format="%,.1f円"),
                        "OOS期待値%": st.column_config.NumberColumn("OOS期待値", format="%+.2f%%"),
                        "推奨株数": st.column_config.NumberColumn("推奨株数", format="%,d株"),
                        "想定投資額": st.column_config.NumberColumn("想定投資額", format="%,.0f円"),
                        "想定最大損失": st.column_config.NumberColumn("想定最大損失", format="%,.0f円"),
                    },
                )
            st.download_button("最終100点ランキングCSVを保存", final_rank.drop(columns=["_priority"], errors="ignore").to_csv(index=False).encode("utf-8-sig"), file_name="japan_stock_final_100_ranking_v74.csv", mime="text/csv", key="download_deep")
            st.warning("最終売買前は、上位候補を『個別』タブで開き、重要開示・決算日も確認してください。売買プランは確定日足ベースの参考値であり、利益を保証するものではありません。")

with scanner_tab:
    st.subheader("最終100点ランキング")
    st.caption("候補銘柄をJ-Quants公式価格・財務＋AI検証で再審査し、同じ100点基準で比較します。全市場レーダーで候補を絞ってから使うのが推奨です。")
    default_codes = "7203\n6758\n8306\n9984\n9432"
    code_text = st.text_area("スキャンする銘柄コード（1行1銘柄）", default_codes, height=140)
    code_file = st.file_uploader("または Code 列を持つCSV", type=["csv"], key="code_list_csv")
    max_scan = st.slider("最大スキャン数", 5, 50, 20)
    phase_filter = st.multiselect("優先表示フェーズ", ["初動候補", "押し目候補", "上昇継続", "底打ち候補", "中立・監視", "過熱警戒", "売り警戒", "下落基調"])

    if st.button("ランキングを作成", type="primary", use_container_width=True):
        if not jq_connected:
            st.error("ランキング作成にはJ-Quants APIへの正常接続が必要です。")
        elif not jq_live_eligible:
            st.error("現在のデータは遅延しています。現在の売買候補ランキングとしては表示しません。個別タブの過去分析・検証は利用できます。")
        else:
            codes = [x.strip() for x in code_text.replace(",", "\n").splitlines() if x.strip()]
            if code_file is not None:
                d = pd.read_csv(code_file)
                if "Code" in d.columns:
                    codes += d["Code"].dropna().astype(str).str.strip().tolist()
            codes = list(dict.fromkeys(codes))[:max_scan]
            end = date.today()
            start = end - timedelta(days=int(years * 365.25 + 300))
            rows = []
            progress = st.progress(0, text="スキャン開始")
            try:
                bm = load_jq_quotes(jq_api_key, benchmark_code, start.isoformat(), end.isoformat())
            except Exception:
                bm = pd.DataFrame()
            macro = pd.DataFrame()
            macro_status = []
            if fred_key:
                try:
                    macro, macro_status = load_macro(fred_key, start.isoformat(), end.isoformat())
                except Exception:
                    pass

            for i, scode in enumerate(codes):
                progress.progress((i + 1) / max(len(codes), 1), text=f"{scode} を分析中")
                try:
                    px = load_jq_quotes(jq_api_key, scode, start.isoformat(), end.isoformat())
                    fundamentals = load_jq_fundamentals(jq_api_key, scode)
                    statuses = [
                        SourceStatus("JPX/J-Quants", 1, not px.empty, "対象株価", px.index.max() if not px.empty else None, len(px)),
                        SourceStatus("JPX/J-Quants", 1, not bm.empty, "市場ベンチマーク", bm.index.max() if not bm.empty else None, len(bm)),
                        SourceStatus("JPX/J-Quants", 1, not fundamentals.empty, "四半期財務", fundamentals.index.max() if not fundamentals.empty else None, len(fundamentals)),
                    ] + macro_status
                    edocs = pd.DataFrame()  # Batch mode avoids slow per-name EDINET document crawl.
                    supply_bundle = load_jq_supply_bundle(jq_api_key, scode, end.isoformat())
                    ss = supply_summary_from_bundle(supply_bundle)
                    statuses += supply_statuses(supply_bundle)
                    res = analyze_one(scode, px, bm, fundamentals, macro, statuses, edocs, ss)
                    rows.append({
                        "Code": scode,
                        "買い候補スコア": res["score"],
                        "前回比": res["score_change"],
                        "相場フェーズ": res["phase"],
                        f"AI {horizon}日上昇確率": round(res["live_p"] * 100, 1),
                        "総合信頼度": round(res["confidence"] * 100, 1),
                        "20日騰落%": round(res["features"]["ret_20"].iloc[-1] * 100, 1),
                        "RSI": round(res["features"]["rsi_14"].iloc[-1], 1),
                        "売買代金比": round(res["features"]["turnover_ratio_20"].iloc[-1], 2),
                        "需給スコア": round(float(ss.get("supply_demand_quality", 50)), 1),
                        "需給判定": ss.get("supply_demand_signal", ""),
                        "評価": res["label"],
                    })
                except Exception as e:
                    rows.append({"Code": scode, "買い候補スコア": np.nan, "相場フェーズ": f"分析失敗: {str(e)[:45]}"})
            progress.empty()
            rank = pd.DataFrame(rows)
            if not rank.empty:
                rank = rank.sort_values(["買い候補スコア", "前回比"], ascending=[False, False], na_position="last")
                if phase_filter:
                    rank = rank[rank["相場フェーズ"].isin(phase_filter)]
                st.dataframe(
                    rank,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "買い候補スコア": st.column_config.ProgressColumn("買い候補スコア", min_value=0, max_value=100, format="%.1f"),
                        "前回比": st.column_config.NumberColumn("前回比", format="%+.1f"),
                        f"AI {horizon}日上昇確率": st.column_config.ProgressColumn(f"AI {horizon}日上昇確率", min_value=0, max_value=100, format="%.1f%%"),
                        "総合信頼度": st.column_config.ProgressColumn("総合信頼度", min_value=0, max_value=100, format="%.1f%%"),
                        "需給スコア": st.column_config.ProgressColumn("需給スコア", min_value=0, max_value=100, format="%.1f"),
                    },
                )
                st.download_button("ランキングCSVを保存", rank.to_csv(index=False).encode("utf-8-sig"), file_name="buy_candidate_ranking.csv", mime="text/csv")
                st.caption("ランキング画面では速度優先のためEDINET全件確認を省略。最終候補は個別銘柄画面で重要開示まで再確認してください。")

with validation_tab:
    st.subheader("過去シグナル有効性検証")
    st.caption("現在の買い候補条件を、過去の時点で利用可能だった情報だけで再現し、5・20・60営業日後の成績を確認します。未来のデータをシグナル作成には使いません。")

    if "latest_result" not in st.session_state:
        st.info("先に『🔎 個別銘柄』で1銘柄を分析してください。その銘柄の価格・需給履歴を使って検証します。")
    else:
        vr = st.session_state["latest_result"]
        c1, c2, c3, c4 = st.columns(4)
        radar_min = c1.slider("レーダー最低点", 40, 90, 65, 1)
        supply_min = c2.slider("需給最低点", 40, 95, 70, 1)
        max_ret20_pct = c3.slider("20日上昇率の上限", 0, 30, 12, 1, format="%d%%")
        cooldown = c4.slider("同一シグナル間隔", 1, 30, 5, 1, help="連日の同じ局面を重複カウントしすぎないための間隔です。")
        cov_min = st.slider("需給データ最低充足率", 0, 100, 35, 5, format="%d%%")

        if st.button("この条件を過去検証", type="primary", use_container_width=True, key="run_validation"):
            with st.spinner("過去時点のシグナルを再構築しています…"):
                vh = make_signal_history(
                    vr["px"], vr.get("bm"), vr.get("supply_bundle", {}),
                    radar_min=float(radar_min), supply_min=float(supply_min),
                    supply_coverage_min=float(cov_min) / 100.0,
                    max_ret20=float(max_ret20_pct) / 100.0,
                )
                events, metrics = event_study(vh, vr.get("bm"), horizons=(5, 20, 60), cooldown=int(cooldown))
                grade = validation_grade(metrics)
                st.session_state["validation_result"] = (vh, events, metrics, grade)

        if "validation_result" in st.session_state:
            vh, events, metrics, grade = st.session_state["validation_result"]
            m1, m2, m3 = st.columns(3)
            m1.metric("検証信頼スコア", f"{grade['score']:.1f} / 100")
            m2.metric("判定", grade["grade"])
            m3.metric("検証方式", str(vh["validation_mode"].dropna().iloc[-1]) if "validation_mode" in vh and vh["validation_mode"].notna().any() else "不明")
            st.caption(grade["reason"])

            if metrics.empty:
                st.warning("条件に一致する過去シグナルが不足しています。条件を少し緩めて再検証してください。")
            else:
                show = metrics.copy()
                for c in ["勝率", "平均騰落率", "中央値", "平均超過リターン", "平均最大不利変動"]:
                    if c in show:
                        show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.markdown("**読み方**: 勝率だけでなく、平均騰落率・TOPIX等に対する超過リターン・シグナル後の平均最大不利変動を同時に見ます。A/B判定でも将来利益を保証するものではありません。")

                st.markdown("### 現実約定テスト（V8）")
                st.caption("シグナル日の終値では買えたことにせず、翌営業日の始値で約定。大幅ギャップアップは見送り、ストップと利確が同日に両方触れた場合は保守的にストップ先行として計算します。手数料＋スリッページも差し引きます。")
                trades = execution_trade_backtest(
                    vh, vr["px"], horizon=int(horizon), transaction_cost_bps=float(tx_cost),
                    slippage_bps=float(slippage_bps), cooldown=int(cooldown),
                )
                em = execution_metrics(trades)
                eg = execution_grade(em)
                e1,e2,e3,e4,e5 = st.columns(5)
                e1.metric("実戦検証", eg["grade"])
                e2.metric("約定件数", f"{int(em.get('trades',0))}件")
                e3.metric("勝率", f"{_finite_float(em.get('win_rate'),0):.1%}")
                e4.metric("1回平均", f"{_finite_float(em.get('expectancy'),0):+.2%}")
                pfv = _finite_float(em.get('profit_factor'), np.nan)
                e5.metric("Profit Factor", f"{pfv:.2f}" if np.isfinite(pfv) else "-")
                cil = _finite_float(em.get('mean_ci10'), np.nan); cih = _finite_float(em.get('mean_ci90'), np.nan)
                if np.isfinite(cil) and np.isfinite(cih):
                    st.caption(f"平均損益のブートストラップ80%範囲: {cil:+.2%} ～ {cih:+.2%}。下限がマイナスなら、利益が出ていても確証は弱いと扱います。")
                if trades is not None and not trades.empty:
                    with st.expander("現実約定テスト明細"):
                        st.dataframe(trades.sort_values("SignalDate", ascending=False), use_container_width=True, hide_index=True)

            if events is not None and not events.empty:
                ev_show = events.sort_values("Date", ascending=False).copy()
                for c in [x for x in ev_show.columns if x.startswith("ret_") or x.startswith("excess_") or x.startswith("mae_")]:
                    ev_show[c] = ev_show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                with st.expander(f"過去シグナル {len(events)}件の明細"):
                    st.dataframe(ev_show, use_container_width=True, hide_index=True)
                    st.download_button("検証結果CSVを保存", events.to_csv(index=False).encode("utf-8-sig"), file_name=f"validation_{vr['code']}.csv", mime="text/csv")

with adaptive_tab:
    st.subheader("V8 適応配点エンジン")
    st.caption("過去データのアウト・オブ・サンプル成績から、価格・需給系の配点を学習します。過学習を避けるため、学習結果は従来配点へ65%縮小して採用します。")
    if "latest_result" not in st.session_state:
        st.info("まず『🔎 個別』で銘柄を分析してください。その銘柄の履歴から適応配点を検証できます。")
    else:
        rr = st.session_state["latest_result"]
        if st.button("この銘柄で適応配点を検証", type="primary", use_container_width=True, key="run_opt_weights"):
            with st.spinner("未来情報を使わず配点を学習・検証しています…"):
                opt = optimize_adaptive_weights(rr["px"], rr.get("bm"), rr.get("supply_bundle", {}), horizon=horizon)
            st.session_state["v7_optimizer"] = opt
        if "v7_optimizer" in st.session_state:
            opt=st.session_state["v7_optimizer"]
            m=opt.get("metrics",{})
            c1,c2,c3=st.columns(3)
            c1.metric("状態", opt.get("status",""))
            c2.metric("検証AUC", f"{m.get('auc', float('nan')):.3f}" if m else "-")
            c3.metric("Balanced Acc", f"{m.get('balanced_accuracy', float('nan')):.3f}" if m else "-")
            df=pd.DataFrame({"項目":list(opt["weights"].keys()),"V8推奨配点":list(opt["weights"].values()),"従来配点":[BASELINE_WEIGHTS[k] for k in opt["weights"]]})
            df["差"] = df["V8推奨配点"]-df["従来配点"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            if opt.get("status") != "optimized":
                st.warning("検証力が不足しているため、V8は従来配点を維持します。無理に最適化しません。")

with guide_tab:
    st.subheader("100点の配点")
    allocation = pd.DataFrame([
        ["トレンド", 20, "長短移動平均、200日線、トレンド配列"],
        ["モメンタム・初動性", 12, "20/60日騰落、MACD、RSI。過熱は減点"],
        ["出来高・資金流入", 10, "出来高と売買代金の20日平均比"],
        ["市場比・相対強度", 7, "ベンチマークに対して強いか"],
        ["業績・ファンダ", 13, "売上・利益・EPS・会社予想の成長"],
        ["リスク・買い位置", 8, "ATR、実現ボラ、高値圏での過熱度"],
        ["AI予測", 10, "複数モデルの上昇確率を検証精度で割引"],
        ["情報信頼度", 8, "公式ソース取得状況、鮮度、特徴量充足率"],
        ["信用・空売り・自己株買い需給", 12, "売買内訳、信用残高、空売り残高、TDnet/EDINET自己株買い"],
    ], columns=["項目", "満点", "主な内容"])
    st.dataframe(allocation, use_container_width=True, hide_index=True)
    st.markdown("""
**評価帯**

- **90–100点：S｜最有力買い候補** — 条件がかなり揃っている。ただし過熱・重要開示があれば追い買いしない。
- **80–89点：A｜買い候補** — 複数要因が同方向。個別ニュース・決算日を確認してエントリーを検討。
- **70–79点：B｜監視強化** — 初動前後で最も変化を見る帯。前回比の上昇を重視。
- **60–69点：C｜様子見** — 優位性不足。
- **45–59点：D｜弱い** — 買い優位性が乏しい。
- **0–44点：E｜買い対象外** — 下落基調や情報不足を含む。

**重要:** 100点は「100%上がる」という意味ではありません。**現時点の買い条件がどれだけ整っているかを100点で表した指数**です。AI上昇確率とは別物です。
    """)

st.divider()
st.caption("V8の原則：高得点でも、弱い市場・弱いOOS検証・過大なリスクなら買いません。Standard単体では確定日足を基準にし、リアルタイム気配としては扱いません。本システムは売買判断支援ツールで、利益を保証するものではありません。")
