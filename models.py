from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


@dataclass
class WalkForwardResult:
    probabilities: pd.Series
    metrics: Dict[str, float]
    latest_probability: float
    feature_count: int
    train_rows: int


def _models() -> List[Pipeline]:
    # Deliberately diverse ensemble: linear + nonlinear tree models.
    return [
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.35, max_iter=2000, class_weight="balanced")),
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=140, l2_regularization=1.0, random_state=42)),
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=180, max_depth=5, min_samples_leaf=12, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=1)),
        ]),
    ]


def _ensemble_predict(fitted: List[Pipeline], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(X)[:, 1] for m in fitted]
    return np.mean(np.vstack(preds), axis=0)


def walk_forward_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_rows: int = 504,
    test_rows: int = 126,
    embargo_rows: int = 20,
    X_live: pd.DataFrame | None = None,
) -> WalkForwardResult:
    X = X.copy()
    y = y.loc[X.index].astype(int)
    if len(X) < min_train_rows + test_rows:
        raise ValueError(f"Need at least {min_train_rows + test_rows} usable rows; got {len(X)}")

    probs = pd.Series(index=X.index, dtype=float)
    fold_metrics = []
    start = min_train_rows
    while start < len(X):
        end = min(start + test_rows, len(X))
        train_end = max(start - embargo_rows, 1)
        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        Xte, yte = X.iloc[start:end], y.iloc[start:end]
        if ytr.nunique() < 2:
            start = end
            continue
        fitted = []
        for m in _models():
            m.fit(Xtr, ytr)
            fitted.append(m)
        p = _ensemble_predict(fitted, Xte)
        probs.iloc[start:end] = p
        pred = (p >= 0.5).astype(int)
        fm = {
            "accuracy": accuracy_score(yte, pred),
            "balanced_accuracy": balanced_accuracy_score(yte, pred),
            "brier": brier_score_loss(yte, p),
        }
        try:
            fm["auc"] = roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan
        except Exception:
            fm["auc"] = np.nan
        fold_metrics.append(fm)
        start = end

    valid = probs.notna()
    if valid.sum() == 0:
        raise ValueError("Walk-forward produced no out-of-sample predictions")
    aggregate = {
        "accuracy": float(accuracy_score(y[valid], (probs[valid] >= 0.5).astype(int))),
        "balanced_accuracy": float(balanced_accuracy_score(y[valid], (probs[valid] >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y[valid], probs[valid])),
        "auc": float(roc_auc_score(y[valid], probs[valid])) if y[valid].nunique() > 1 else float("nan"),
        "oos_rows": int(valid.sum()),
        "folds": len(fold_metrics),
    }

    # Final live model uses all completed labels. No future labels are used.
    fitted = []
    for m in _models():
        m.fit(X, y)
        fitted.append(m)
    live_x = X_live[X.columns].tail(1) if X_live is not None and not X_live.empty else X.tail(1)
    latest_probability = float(_ensemble_predict(fitted, live_x)[0])
    return WalkForwardResult(probs, aggregate, latest_probability, X.shape[1], len(X))


def model_confidence(metrics: Dict[str, float], source_conf: float, data_coverage: float) -> float:
    auc = metrics.get("auc", np.nan)
    bal = metrics.get("balanced_accuracy", 0.5)
    brier = metrics.get("brier", 0.25)
    auc_score = 0.5 if np.isnan(auc) else np.clip((auc - 0.5) / 0.20, 0, 1)
    bal_score = np.clip((bal - 0.5) / 0.15, 0, 1)
    brier_score = np.clip((0.25 - brier) / 0.10, 0, 1)
    predictive = 0.4 * auc_score + 0.35 * bal_score + 0.25 * brier_score
    return float(np.clip(0.55 * predictive + 0.30 * source_conf + 0.15 * data_coverage, 0, 1))


def fit_final_probability(X: pd.DataFrame, y: pd.Series, X_live: pd.DataFrame) -> float:
    y = y.loc[X.index].astype(int)
    fitted = []
    for m in _models():
        m.fit(X, y)
        fitted.append(m)
    return float(_ensemble_predict(fitted, X_live[X.columns].tail(1))[0])
