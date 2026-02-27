"""
XGBoost + optional LightGBM blend prediction engine.
Matches the enhanced feature set from model/train.py:
  - Cyclical month/DOW encoding
  - Extended lags: 1, 3, 7, 14, 21, 30
  - Price momentum features
  - Log1p inverse-transform (expm1) applied when model was trained with log
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "data/crops.csv"
MODEL_DIR = "models"
MAX_DAYS  = 30

FEATURE_COLS = [
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "weekofyear", "dayofyear", "quarter",
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_30",
    "roll_mean_7", "roll_mean_14", "roll_std_7",
    "price_change_7", "price_pct_7",
    "harvest",
]

HARVEST_MONTHS = {
    "Tomato":      [11, 12, 1, 2],
    "Onion":       [2, 3, 4, 5],
    "Potato":      [11, 12, 1, 2],
    "Paddy":       [10, 11, 12],
    "BitterGourd": [3, 4, 5, 10, 11],
    "Brinjal":     [1, 2, 3, 9, 10],
    "BroadBeans":  [11, 12, 1, 2],
    "Carrot":      [10, 11, 12, 1],
    "GreenChilli": [9, 10, 11, 12],
    "Okra":        [3, 4, 5, 6],
}


def _build_prediction_features(crop: str, crop_df: pd.DataFrame, target_date: datetime,
                                log_transformed: bool = True) -> pd.DataFrame:
    hist = crop_df[["date", "price"]].copy()
    hist.columns = ["ds", "y"]
    hist["ds"] = pd.to_datetime(hist["ds"])
    hist = hist.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)

    # Apply same log-transform the model was trained with
    if log_transformed:
        hist["y"] = np.log1p(hist["y"])

    # Append synthetic target row
    target_row = pd.DataFrame({"ds": [pd.Timestamp(target_date)], "y": [np.nan]})
    combined   = pd.concat([hist.tail(70), target_row], ignore_index=True)
    combined   = combined.sort_values("ds").reset_index(drop=True)

    ts = combined["ds"]
    y  = combined["y"]

    # ── Cyclical calendar ────────────────────────────────────────────────────
    month = ts.dt.month
    dow   = ts.dt.dayofweek
    combined["month_sin"]  = np.sin(2 * np.pi * month / 12)
    combined["month_cos"]  = np.cos(2 * np.pi * month / 12)
    combined["dow_sin"]    = np.sin(2 * np.pi * dow / 7)
    combined["dow_cos"]    = np.cos(2 * np.pi * dow / 7)
    combined["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    combined["dayofyear"]  = ts.dt.dayofyear
    combined["quarter"]    = ts.dt.quarter

    # ── Lags ─────────────────────────────────────────────────────────────────
    for lag in [1, 3, 7, 14, 21, 30]:
        combined[f"lag_{lag}"] = y.shift(lag)

    # ── Rolling ──────────────────────────────────────────────────────────────
    shifted = y.shift(1)
    combined["roll_mean_7"]  = shifted.rolling(7,  min_periods=1).mean()
    combined["roll_mean_14"] = shifted.rolling(14, min_periods=1).mean()
    combined["roll_std_7"]   = shifted.rolling(7,  min_periods=2).std().fillna(0)

    # ── Momentum ─────────────────────────────────────────────────────────────
    combined["price_change_7"] = y - y.shift(7)
    combined["price_pct_7"]    = y.pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)

    # ── Harvest ──────────────────────────────────────────────────────────────
    hm = HARVEST_MONTHS.get(crop, [])
    combined["harvest"] = month.isin(hm).astype(int)

    # Take last row (target date)
    pred_row = combined.tail(1)[FEATURE_COLS].copy()

    # Fill any remaining NaNs with column medians from history
    for col in FEATURE_COLS:
        if pred_row[col].isna().any():
            fill = combined[col].dropna().median()
            pred_row[col] = pred_row[col].fillna(fill if (fill is not None and not np.isnan(float(fill))) else 0)

    return pred_row


def _load_bundle(crop: str, region: str):
    """Load the model bundle saved by train.py (dict with 'xgb' and optional 'lgbm')."""
    path = os.path.join(MODEL_DIR, f"{crop}_{region}.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        # Support both old-style (single model) and new bundle (dict)
        if isinstance(bundle, dict):
            return bundle
        else:
            return {"xgb": bundle, "lgbm": None}
    except Exception:
        return None


def _load_metadata(crop: str, region: str) -> dict:
    path = os.path.join(MODEL_DIR, f"{crop}_{region}_meta.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def run_prediction(crop: str, region: str, target_date) -> dict:
    """
    Returns result dict with: price, confidence, rmse, mae, mape,
    reliability, n_points, trained_date, crop_key, region_key, date.
    Raises ValueError with error code on failure.
    """
    if not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        raise ValueError("NO_DATA")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        raise ValueError("GENERIC_ERROR")

    crop_df = df[(df["crop"] == crop) & (df["region"] == region)].copy()
    if crop_df.empty:
        raise ValueError("NO_DATA")

    crop_df["date"] = pd.to_datetime(crop_df["date"])
    last_date  = crop_df["date"].max()
    delta_days = (target_date - last_date).days

    if delta_days < -7:
        raise ValueError("PAST_DATE")
    if delta_days > MAX_DAYS + 5:
        raise ValueError("FORECAST_LIMIT")

    # ── Load model bundle ─────────────────────────────────────────────────────
    bundle = _load_bundle(crop, region)
    if bundle is None:
        raise ValueError("NO_DATA")

    xgb_model  = bundle.get("xgb")
    lgbm_model = bundle.get("lgbm")
    if xgb_model is None:
        raise ValueError("NO_DATA")

    # ── Load metadata ─────────────────────────────────────────────────────────
    meta            = _load_metadata(crop, region)
    log_transformed = meta.get("log_transformed", True)
    mape            = meta.get("mape")
    rmse            = meta.get("rmse")
    mae             = meta.get("mae")
    reliability     = meta.get("reliability", "Medium")
    n_points        = meta.get("n_points", len(crop_df))
    trained_date    = meta.get("trained_date", last_date.strftime("%d %b %Y"))

    # ── Build features & predict ──────────────────────────────────────────────
    try:
        pred_row = _build_prediction_features(crop, crop_df, target_date, log_transformed)

        # Enforce exact feature names + order — prevents LightGBM/XGBoost warning
        # and guards against any future column drift between train and predict.
        X_pred = pred_row.reindex(columns=FEATURE_COLS)

        pred_xgb = float(xgb_model.predict(X_pred)[0])

        if lgbm_model is not None:
            try:
                pred_lgbm = float(lgbm_model.predict(X_pred)[0])
                raw_pred  = 0.60 * pred_xgb + 0.40 * pred_lgbm
            except Exception:
                raw_pred = pred_xgb
        else:
            raw_pred = pred_xgb

        # Inverse transform
        if log_transformed:
            predicted_price = float(np.expm1(raw_pred))
        else:
            predicted_price = float(raw_pred)

    except Exception:
        raise ValueError("GENERIC_ERROR")

    # Clamp to realistic range
    p_mean = float(crop_df["price"].mean())
    p_std  = float(crop_df["price"].std())
    predicted_price = float(np.clip(predicted_price, max(1.0, p_mean - 4 * p_std), p_mean + 4 * p_std))
    predicted_price = round(predicted_price, 2)

    # ── Confidence — MAPE-based + slight days-ahead penalty ──────────────────
    if mape is not None:
        base_conf = 100 - mape * 0.9          # same formula as training
    else:
        base_conf = 86.0

    days_penalty = max(0, delta_days) * 0.08   # smaller penalty than before
    confidence   = round(float(np.clip(base_conf - days_penalty, 75, 98)), 1)

    # Re-derive reliability from the stored mape
    if mape is not None:
        reliability = "High" if mape < 10 else ("Medium" if mape < 18 else "Low")

    return {
        "price":        predicted_price,
        "confidence":   confidence,
        "rmse":         rmse,
        "mae":          mae,
        "mape":         mape,
        "reliability":  reliability,
        "n_points":     n_points,
        "trained_date": trained_date,
        "crop_key":     crop,
        "region_key":   region,
        "date":         target_date,
    }
