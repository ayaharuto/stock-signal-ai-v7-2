from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
from validation import historical_price_radar, historical_supply_series

BASELINE_WEIGHTS = {
    "trend": 20.0, "momentum": 12.0, "participation": 10.0,
    "relative_market": 7.0, "fundamental": 13.0, "entry_quality": 8.0,
    "ml": 10.0, "evidence": 8.0, "supply_demand": 12.0,
}

OPT_KEYS = ["trend", "momentum", "participation", "relative_market", "entry_quality", "supply_demand"]

def _scale(s, lo, hi):
    return ((s-lo)/(hi-lo)*100).clip(0,100)

def historical_component_proxies(px: pd.DataFrame, benchmark: Optional[pd.DataFrame]=None, bundle: Optional[Dict[str,pd.DataFrame]]=None) -> pd.DataFrame:
    h = historical_price_radar(px, benchmark).copy()
    close = pd.to_numeric(px["Close"], errors="coerce").reindex(h.index)
    vol = pd.to_numeric(px.get("Volume", 0), errors="coerce").reindex(h.index).fillna(0)
    sma20 = close.rolling(20).mean(); sma50 = close.rolling(50).mean(); sma200 = close.rolling(200).mean()
    h["trend"] = 0.35*_scale(close/sma20-1,-.10,.12)+0.30*_scale(sma20/sma50-1,-.08,.10)+0.35*_scale(sma50/sma200-1,-.12,.18)
    h["momentum"] = 0.60*_scale(h["ret20"],-.18,.25)+0.40*_scale(h["ret60"],-.30,.50)
    vr = vol/vol.rolling(20).mean().replace(0,np.nan)
    tv = close*vol; tr = tv/tv.rolling(20).mean().replace(0,np.nan)
    h["participation"] = 0.45*_scale(vr,.65,2.2)+0.55*_scale(tr,.65,2.2)
    if benchmark is not None and not benchmark.empty:
        b = pd.to_numeric(benchmark["Close"], errors="coerce").reindex(h.index, method="ffill")
        rs20 = h["ret20"] - b.pct_change(20)
        rs60 = h["ret60"] - b.pct_change(60)
        h["relative_market"] = .55*_scale(rs20,-.12,.18)+.45*_scale(rs60,-.20,.30)
    else:
        h["relative_market"] = 50.0
    fh=h["from_high"]
    h["entry_quality"] = np.where((fh>=-.12)&(fh<=-.01),90,np.where((fh>=-.25)&(fh<-.12),65,np.where(fh>-.01,42,35)))
    h["supply_demand"] = 50.0
    if bundle:
        s = historical_supply_series(bundle, h.index[-420:])
        if not s.empty and "supply_demand_quality" in s:
            h.loc[s.index.intersection(h.index), "supply_demand"] = s["supply_demand_quality"].reindex(h.index).ffill().fillna(50)
    return h

def optimize_adaptive_weights(px: pd.DataFrame, benchmark: Optional[pd.DataFrame]=None, bundle: Optional[Dict[str,pd.DataFrame]]=None, horizon: int=20, shrinkage: float=.65, min_rows: int=240) -> Dict[str, object]:
    h = historical_component_proxies(px, benchmark, bundle)
    bm = None
    if benchmark is not None and not benchmark.empty:
        bm = pd.to_numeric(benchmark["Close"], errors="coerce").reindex(h.index, method="ffill")
    fwd = h["Close"].shift(-horizon)/h["Close"]-1
    if bm is not None:
        excess = fwd - (bm.shift(-horizon)/bm-1)
    else:
        excess = fwd
    y = (excess > 0).astype(float)
    X = h[OPT_KEYS].replace([np.inf,-np.inf],np.nan)
    d = X.copy(); d["y"]=y; d=d.dropna()
    if len(d) < min_rows:
        return {"weights": BASELINE_WEIGHTS.copy(), "status":"insufficient", "rows":len(d), "metrics":{}}
    split = int(len(d)*.72)
    if split < 160 or len(d)-split < 50:
        return {"weights": BASELINE_WEIGHTS.copy(), "status":"insufficient", "rows":len(d), "metrics":{}}
    Xtr=d[OPT_KEYS].iloc[:split]/100.0; ytr=d["y"].iloc[:split].astype(int)
    Xte=d[OPT_KEYS].iloc[split:]/100.0; yte=d["y"].iloc[split:].astype(int)
    model=LogisticRegression(C=.35, max_iter=2000, class_weight="balanced")
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
    try: auc=float(roc_auc_score(yte,p))
    except Exception: auc=float('nan')
    bal=float(balanced_accuracy_score(yte,pred)); brier=float(brier_score_loss(yte,p))
    coeff=np.maximum(model.coef_[0],0)
    if coeff.sum() <= 1e-9 or (np.isfinite(auc) and auc < .52) or bal < .52:
        return {"weights": BASELINE_WEIGHTS.copy(), "status":"weak_evidence", "rows":len(d), "metrics":{"auc":auc,"balanced_accuracy":bal,"brier":brier}}
    baseline_opt=np.array([BASELINE_WEIGHTS[k] for k in OPT_KEYS],dtype=float)
    learned=coeff/coeff.sum()*baseline_opt.sum()
    blended=shrinkage*baseline_opt+(1-shrinkage)*learned
    out=BASELINE_WEIGHTS.copy()
    for k,v in zip(OPT_KEYS,blended): out[k]=float(v)
    fixed=sum(BASELINE_WEIGHTS[k] for k in BASELINE_WEIGHTS if k not in OPT_KEYS)
    scale=(100-fixed)/sum(out[k] for k in OPT_KEYS)
    for k in OPT_KEYS: out[k]*=scale
    return {"weights":out,"status":"optimized","rows":len(d),"metrics":{"auc":auc,"balanced_accuracy":bal,"brier":brier},"coefficients":dict(zip(OPT_KEYS,map(float,model.coef_[0])))}
