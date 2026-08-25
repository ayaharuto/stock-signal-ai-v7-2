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
from validation import make_signal_history, event_study, validation_grade
from optimizer import optimize_adaptive_weights, BASELINE_WEIGHTS
from sources import (
    JQuantsProClient,
    EDINETClient,
    FREDClient,
    build_macro_frame,
    SourceStatus,
    source_confidence,
)

st.set_page_config(page_title="Stock Signal AI v7.2", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

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

st.title("Stock Signal AI v7.2")
st.caption("スマホ最適化版。機能は維持したまま、ログインと日常操作を簡略化。公式・一次情報を優先し、日本株全体スキャン、100点評価、需給先回り、過去検証、適応配点を1つのURLで使えます。")

def _safe_secret(name: str, default: str = "") -> str:
    """Read hosting secret first, then environment variable. Never expose secrets in UI."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def _ensure_jquants_session() -> str:
    if st.session_state.get("jq_token"):
        return st.session_state["jq_token"]
    direct = _safe_secret("JQUANTS_TOKEN", "")
    if direct:
        st.session_state["jq_token"] = direct
        return direct
    # When credentials are stored securely on the hosting side, login is automatic.
    mail = _safe_secret("JQUANTS_MAIL", "")
    password = _safe_secret("JQUANTS_PASSWORD", "")
    if mail and password and not st.session_state.get("jq_auto_login_attempted"):
        st.session_state["jq_auto_login_attempted"] = True
        try:
            client, refresh = JQuantsProClient.login(mail, password)
            st.session_state["jq_token"] = client.token
            if refresh:
                st.session_state["jq_refresh_token"] = refresh
            return client.token
        except Exception as e:
            st.session_state["jq_auto_login_error"] = str(e)
    return ""


jq_token = _ensure_jquants_session()
edinet_key = _safe_secret("EDINET_API_KEY", "")
fred_key = _safe_secret("FRED_API_KEY", "")

status_col, action_col = st.columns([1.4, 1])
with status_col:
    if jq_token:
        st.success("✅ 公式データ接続済み — そのまま使えます")
    else:
        st.warning("🔐 初回だけJ-Quantsへログインしてください")
with action_col:
    if jq_token and st.button("接続をやり直す", use_container_width=True):
        for k in ["jq_token", "jq_refresh_token", "jq_auto_login_attempted", "jq_auto_login_error"]:
            st.session_state.pop(k, None)
        st.rerun()

if not jq_token:
    with st.expander("🔐 初回ログイン（普段は開きません）", expanded=True):
        st.caption("J-Quants Proの公式アカウントでログインします。入力したパスワードをこのアプリのファイルへ保存しません。公開担当者がSecrets設定を済ませれば、この画面自体を省略できます。")
        with st.form("jquants_login_form"):
            jq_mail_ui = st.text_input("J-Quantsのメールアドレス", autocomplete="email")
            jq_pw_ui = st.text_input("J-Quantsのパスワード", type="password")
            login_submit = st.form_submit_button("公式データに接続", type="primary", use_container_width=True)
        if login_submit:
            try:
                with st.spinner("J-Quants公式APIへ接続中…"):
                    client, refresh = JQuantsProClient.login(jq_mail_ui.strip(), jq_pw_ui)
                st.session_state["jq_token"] = client.token
                if refresh:
                    st.session_state["jq_refresh_token"] = refresh
                st.session_state.pop("jq_auto_login_error", None)
                st.rerun()
            except Exception as e:
                st.error(f"接続できませんでした: {e}")

with st.expander("👋 使い方（30秒）", expanded=False):
    st.markdown("""
**普段はこの3操作だけです。**

1. **🌐 全市場**を開く
2. **今日の買い候補を更新**を押す
3. 上位候補を見て、気になる銘柄は **🔎 個別** で100点評価・需給・過去検証を確認

公開担当者がJ-Quants/EDINET/FREDの認証をSecretsへ設定すると、スマホ側ではログイン入力も不要にできます。SNS・匿名掲示板・まとめサイトは標準スコアには使いません。
""")

with st.sidebar:
    st.header("詳細設定（普段は触らなくてOK）")
    years = st.slider("価格履歴", 3, 12, 8, help="長いほど検証は安定しやすい一方、相場構造の変化も混ざります。")
    horizon = st.selectbox("AI予測期間", [5, 10, 20, 60], index=2, format_func=lambda x: f"{x}営業日")
    tx_cost = st.slider("片道売買コスト", 0, 50, 12, format="%d bps")
    benchmark_code = st.text_input("市場ベンチマーク", "1306")
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
def load_jq_quotes(token: str, code_: str, start: str, end: str) -> pd.DataFrame:
    return JQuantsProClient(token).daily_quotes(code_, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_jq_fundamentals(token: str, code_: str) -> pd.DataFrame:
    return JQuantsProClient(token).statements(code_)


@st.cache_data(ttl=3600, show_spinner=False)
def load_jq_supply_bundle(token: str, code_: str, end_date: str):
    client = JQuantsProClient(token)
    end_ts = pd.Timestamp(end_date)
    start90 = (end_ts - pd.Timedelta(days=120)).date().isoformat()
    start180 = (end_ts - pd.Timedelta(days=220)).date().isoformat()
    start365 = (end_ts - pd.Timedelta(days=420)).date().isoformat()
    bundle = {}
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
        except Exception:
            bundle[key] = pd.DataFrame()
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
    msg = f"公式需給データ {rows}件" if any_data else "該当データなし（中立扱い）"
    return [SourceStatus("JPX/J-Quants", 1, True, msg, None, rows)]


@st.cache_data(ttl=21600, show_spinner=False)
def load_jq_listed(token: str, target_date: str | None = None) -> pd.DataFrame:
    return JQuantsProClient(token).listed_info(target_date)


@st.cache_data(ttl=21600, show_spinner=False)
def load_jq_market_date(token: str, target_date: str) -> pd.DataFrame:
    return JQuantsProClient(token).daily_quotes_for_date(target_date)


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
    X_train, y, _ = make_supervised(features, cfg.horizon_days)
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
                if not jq_token:
                    st.error("J-Quants Proへの接続が必要です。画面上部の「初回ログイン」から接続してください。非公式価格サイトには自動切替しません。")
                    st.stop()
                px = load_jq_quotes(jq_token, code, start.isoformat(), end.isoformat())
                bm = load_jq_quotes(jq_token, benchmark_code, start.isoformat(), end.isoformat())
                fundamentals = load_jq_fundamentals(jq_token, code)
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
                supply_bundle = load_jq_supply_bundle(jq_token, code, end.isoformat())
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
    st.subheader("今日の日本株レーダー")
    st.caption("東証プライム・スタンダード・グロースの普通株を公式J-Quantsデータで横断スキャンします。一次選別は高速レーダー、その後に上位銘柄だけを最終100点評価します。")

    mc1, mc2, mc3, mc4 = st.columns(4)
    radar_days = mc1.selectbox("全市場履歴", [330, 390, 450], index=1, format_func=lambda x: f"約{x}暦日")
    deep_n = mc2.selectbox("最終精査する上位数", [10, 20, 30, 50], index=1)
    min_turn_m = mc3.number_input("最低売買代金/日", min_value=0, max_value=5000, value=50, step=50, help="百万円。流動性が極端に低い銘柄を除外します。")
    preferred = mc4.multiselect("優先レーダー", ["需給先回り候補", "初動レーダー", "押し目レーダー", "上昇継続レーダー", "過熱警戒", "需給悪化警戒", "監視"], default=["需給先回り候補", "初動レーダー", "押し目レーダー", "上昇継続レーダー"])

    st.info("初回だけ過去の全市場日次データを公式APIから収集します。取得済み日付はPC内にキャッシュするため、2回目以降は差分中心になります。")

    if st.button("今日の買い候補を更新", type="primary", use_container_width=True, key="all_market_scan"):
        if not jq_token:
            st.error("J-Quants Proへの接続が必要です。画面上部の「初回ログイン」から接続してください。")
        else:
            progress = st.progress(0, text="全市場データを準備中")
            try:
                listed = load_jq_listed(jq_token, None)
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
                    lambda dt: load_jq_market_date(jq_token, dt),
                    date.today(),
                    scan_cfg,
                    progress_cb=_pcb,
                )
                if panel.empty:
                    raise ValueError("全市場の日次株価を取得できませんでした。ID token、契約データセット、通信状態を確認してください。")
                if universe_codes:
                    panel = panel[panel["Code"].astype(str).isin(universe_codes)]
                radar = fast_market_features(panel, benchmark_code=benchmark_code)
                radar = enrich_with_listed(radar, universe)
                radar = radar[radar["rows"] >= scan_cfg.min_price_rows]
                radar = radar[radar["TurnoverValue"].fillna(0) >= scan_cfg.min_turnover_yen]
                progress.progress(0.05, text="公式需給データを横断集計中")
                client = JQuantsProClient(jq_token)
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
        rr1.metric("需給先回り", int((radar["early_signal"] == "需給先回り候補").sum()))
        rr2.metric("初動候補", int((radar["early_signal"] == "初動レーダー").sum()))
        rr3.metric("押し目候補", int((radar["early_signal"] == "押し目レーダー").sum()))
        rr4.metric("先回り80点以上", int((radar["early_radar_score"] >= 80).sum()))

        st.markdown("### 上位候補を最終100点評価")
        st.caption("ここからは価格だけでなく、財務・AIのWalk-forward検証・情報信頼度を含めて再評価します。最終100点ランキングはこちらの結果を採用してください。")
        if st.button(f"上位 {deep_n} 銘柄を精査", type="secondary", use_container_width=True, key="deep_scan_top"):
            candidates = visible.head(int(deep_n))["Code"].astype(str).tolist()
            if not candidates:
                st.warning("優先レーダー条件に該当する候補がありません。フィルターを広げてください。")
            else:
                end = date.today()
                start = end - timedelta(days=int(years * 365.25 + 300))
                dprog = st.progress(0, text="最終100点評価を開始")
                try:
                    bm = load_jq_quotes(jq_token, benchmark_code, start.isoformat(), end.isoformat())
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
                        px = load_jq_quotes(jq_token, scode, start.isoformat(), end.isoformat())
                        fundamentals = load_jq_fundamentals(jq_token, scode)
                        statuses = [
                            SourceStatus("JPX/J-Quants", 1, not px.empty, "対象株価", px.index.max() if not px.empty else None, len(px)),
                            SourceStatus("JPX/J-Quants", 1, not bm.empty, "市場ベンチマーク", bm.index.max() if not bm.empty else None, len(bm)),
                            SourceStatus("JPX/J-Quants", 1, not fundamentals.empty, "四半期財務", fundamentals.index.max() if not fundamentals.empty else None, len(fundamentals)),
                        ] + macro_status
                        supply_bundle = load_jq_supply_bundle(jq_token, scode, end.isoformat())
                        ss = supply_summary_from_bundle(supply_bundle)
                        statuses += supply_statuses(supply_bundle)
                        # Batch deep scan skips the separate EDINET document crawl for speed, but J-Quants official buyback/market data are included.
                        res = analyze_one(scode, px, bm, fundamentals, macro, statuses, pd.DataFrame(), ss)
                        rr = radar[radar["Code"].astype(str) == scode].head(1)
                        final_rows.append({
                            "Code": scode,
                            "会社名": name_map.get(scode, ""),
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
                            "評価": res["label"],
                        })
                    except Exception as e:
                        final_rows.append({"Code": scode, "会社名": name_map.get(scode, ""), "買い候補100点": np.nan, "相場フェーズ": f"分析失敗: {str(e)[:45]}"})
                dprog.empty()
                final_rank = pd.DataFrame(final_rows).sort_values(["買い候補100点", "前回比"], ascending=[False, False], na_position="last")
                st.session_state["deep_market_rank"] = final_rank

        if "deep_market_rank" in st.session_state:
            final_rank = st.session_state["deep_market_rank"]
            st.dataframe(
                final_rank,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "買い候補100点": st.column_config.ProgressColumn("買い候補100点", min_value=0, max_value=100, format="%.1f"),
                    "前回比": st.column_config.NumberColumn("前回比", format="%+.1f"),
                    f"AI {horizon}日上昇確率": st.column_config.ProgressColumn(f"AI {horizon}日上昇確率", min_value=0, max_value=100, format="%.1f%%"),
                    "総合信頼度": st.column_config.ProgressColumn("総合信頼度", min_value=0, max_value=100, format="%.1f%%"),
                    "先回りレーダー": st.column_config.ProgressColumn("先回りレーダー", min_value=0, max_value=100, format="%.1f"),
                    "需給スコア": st.column_config.ProgressColumn("需給スコア", min_value=0, max_value=100, format="%.1f"),
                },
            )
            st.download_button("最終100点ランキングCSVを保存", final_rank.to_csv(index=False).encode("utf-8-sig"), file_name="japan_stock_final_100_ranking.csv", mime="text/csv", key="download_deep")
            st.warning("最終売買前は、上位候補を『個別銘柄』タブで開き、EDINET重要開示まで確認してください。")

with scanner_tab:
    st.subheader("最終100点ランキング")
    st.caption("候補銘柄をJ-Quants公式価格・財務＋AI検証で再審査し、同じ100点基準で比較します。全市場レーダーで候補を絞ってから使うのが推奨です。")
    default_codes = "7203\n6758\n8306\n9984\n9432"
    code_text = st.text_area("スキャンする銘柄コード（1行1銘柄）", default_codes, height=140)
    code_file = st.file_uploader("または Code 列を持つCSV", type=["csv"], key="code_list_csv")
    max_scan = st.slider("最大スキャン数", 5, 50, 20)
    phase_filter = st.multiselect("優先表示フェーズ", ["初動候補", "押し目候補", "上昇継続", "底打ち候補", "中立・監視", "過熱警戒", "売り警戒", "下落基調"])

    if st.button("ランキングを作成", type="primary", use_container_width=True):
        if not jq_token:
            st.error("ランキング作成にはJ-Quants Proへの接続が必要です。")
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
                bm = load_jq_quotes(jq_token, benchmark_code, start.isoformat(), end.isoformat())
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
                    px = load_jq_quotes(jq_token, scode, start.isoformat(), end.isoformat())
                    fundamentals = load_jq_fundamentals(jq_token, scode)
                    statuses = [
                        SourceStatus("JPX/J-Quants", 1, not px.empty, "対象株価", px.index.max() if not px.empty else None, len(px)),
                        SourceStatus("JPX/J-Quants", 1, not bm.empty, "市場ベンチマーク", bm.index.max() if not bm.empty else None, len(bm)),
                        SourceStatus("JPX/J-Quants", 1, not fundamentals.empty, "四半期財務", fundamentals.index.max() if not fundamentals.empty else None, len(fundamentals)),
                    ] + macro_status
                    edocs = pd.DataFrame()  # Batch mode avoids slow per-name EDINET document crawl.
                    supply_bundle = load_jq_supply_bundle(jq_token, scode, end.isoformat())
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

            if events is not None and not events.empty:
                ev_show = events.sort_values("Date", ascending=False).copy()
                for c in [x for x in ev_show.columns if x.startswith("ret_") or x.startswith("excess_") or x.startswith("mae_")]:
                    ev_show[c] = ev_show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                with st.expander(f"過去シグナル {len(events)}件の明細"):
                    st.dataframe(ev_show, use_container_width=True, hide_index=True)
                    st.download_button("検証結果CSVを保存", events.to_csv(index=False).encode("utf-8-sig"), file_name=f"validation_{vr['code']}.csv", mime="text/csv")

with adaptive_tab:
    st.subheader("V7 適応配点エンジン")
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
            df=pd.DataFrame({"項目":list(opt["weights"].keys()),"V7推奨配点":list(opt["weights"].values()),"従来配点":[BASELINE_WEIGHTS[k] for k in opt["weights"]]})
            df["差"] = df["V7推奨配点"]-df["従来配点"]
            st.dataframe(df, use_container_width=True, hide_index=True)
            if opt.get("status") != "optimized":
                st.warning("検証力が不足しているため、V7は従来配点を維持します。無理に最適化しません。")

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
st.caption("本システムは売買判断支援ツールです。将来の利益を保証するものではありません。最終判断では決算日、流動性、板、ギャップ、ストップ高/安、税・手数料も確認してください。")
