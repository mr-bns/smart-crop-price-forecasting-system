"""
XGBoost + optional LightGBM blend prediction engine — v2.0
Matches the enhanced feature set from model/train.py v2.0:
  - Seasonal intelligence (Kharif / Rabi / Summer)
  - Harvest peak flag per crop
  - Extended lags: 1, 3, 7, 14, 30
  - Rolling mean/std: 7, 14, 30 days
  - Seasonal price adjustment post-prediction
  - Explainable AI reasoning text generation
  - Log1p inverse-transform (expm1) when model was trained with log
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

# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def get_season(month: int) -> str:
    """Return Indian agricultural season based on month."""
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Summer"


# ─────────────────────────────────────────────────────────────────────────────
# HARVEST CALENDAR
# ─────────────────────────────────────────────────────────────────────────────
HARVEST_MONTHS = {
    "Tomato":      [12, 1, 2],
    "Onion":       [3, 4, 5],
    "Potato":      [1, 2, 3],
    "Paddy":       [10, 11, 12],
    "BitterGourd": [5, 6, 7, 8],
    "Brinjal":     [9, 10, 11],
    "BroadBeans":  [11, 12, 1],
    "Carrot":      [11, 12, 1, 2],
    "GreenChilli": [1, 2, 3, 4],
    "Okra":        [5, 6, 7, 8],
}

SEASONAL_ADJUSTMENT = {
    "Tomato":      {"Kharif": +0.05, "Rabi": -0.08, "Summer": +0.12},
    "Onion":       {"Kharif": +0.08, "Rabi": -0.05, "Summer": -0.10},
    "Potato":      {"Kharif": +0.06, "Rabi": -0.06, "Summer": +0.08},
    "Paddy":       {"Kharif": -0.04, "Rabi": +0.03, "Summer": +0.05},
    "BitterGourd": {"Kharif": -0.08, "Rabi": +0.10, "Summer": -0.06},
    "Brinjal":     {"Kharif": -0.06, "Rabi": +0.04, "Summer": +0.06},
    "BroadBeans":  {"Kharif": +0.05, "Rabi": -0.07, "Summer": +0.08},
    "Carrot":      {"Kharif": +0.06, "Rabi": -0.05, "Summer": +0.10},
    "GreenChilli": {"Kharif": +0.10, "Rabi": -0.10, "Summer": +0.07},
    "Okra":        {"Kharif": -0.07, "Rabi": +0.08, "Summer": -0.05},
}

SEASON_SUPPLY_NOTES = {
    "Kharif": {
        "en": "Monsoon season — moderate supply with weather uncertainty.",
        "te": "వర్షాకాలం — వాతావరణ అనిశ్చితతో మధ్యస్థ సరఫరా.",
    },
    "Rabi":   {
        "en": "Winter season — generally stable supply conditions.",
        "te": "శీతాకాలం — సాధారణంగా స్థిరమైన సరఫరా పరిస్థితులు.",
    },
    "Summer": {
        "en": "Summer season — reduced supply, prices often rise.",
        "te": "వేసవి కాలం — తక్కువ సరఫరా, ధరలు తరచుగా పెరుగుతాయి.",
    },
}

def apply_seasonal_adjustment(crop: str, season: str, predicted_price: float) -> float:
    """Apply season-aware price adjustment to simulate real market cycles."""
    factor   = SEASONAL_ADJUSTMENT.get(crop, {}).get(season, 0.0)
    adjusted = predicted_price * (1 + factor)
    return round(float(adjusted), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS — must stay identical to model/train.py v2.0
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "weekofyear", "dayofyear", "quarter",
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    "price_change_7", "price_pct_7",
    "season_kharif", "season_rabi", "season_summer",
    "harvest_peak_flag",
]

# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
EXPLANATION_TEMPLATES = {
    "en": {
        "season":     "Current season: {season}",
        "supply":     "{supply_note}",
        "harvest_on": "{crop} is currently in its peak harvest period — supply is high, which typically lowers prices.",
        "harvest_off":"{crop} is NOT in peak harvest — supply tightens, which often pushes prices up.",
        "trend_up":   "Historical prices show an upward trend over the last 30 days.",
        "trend_down": "Historical prices show a downward trend over the last 30 days.",
        "trend_stable":"Historical prices have been relatively stable.",
        "above_mkt":  "Predicted price (₹{pred}) is ABOVE current market average (₹{mkt_avg}) — demand appears strong.",
        "below_mkt":  "Predicted price (₹{pred}) is BELOW current market average (₹{mkt_avg}) — caution advised.",
        "near_mkt":   "Predicted price is close to current market average (₹{mkt_avg}) — stable market expected.",
    },
    "te": {
        "season":     "ప్రస్తుత సీజన్: {season}",
        "supply":     "{supply_note}",
        "harvest_on": "{crop} ప్రస్తుతం తన గరిష్ఠ కోత కాలంలో ఉంది — సరఫరా అధికంగా ఉంది, ఇది సాధారణంగా ధరలను తగ్గిస్తుంది.",
        "harvest_off":"{crop} గరిష్ఠ కోత కాలంలో లేదు — సరఫరా తక్కువగా ఉంటుంది, ఇది తరచుగా ధరలను పెంచుతుంది.",
        "trend_up":   "చారిత్రక ధరలు గత 30 రోజులలో పెరిగే ధోరణి చూపాయి.",
        "trend_down": "చారిత్రక ధరలు గత 30 రోజులలో తగ్గే ధోరణి చూపాయి.",
        "trend_stable":"చారిత్రక ధరలు సాపేక్షంగా స్థిరంగా ఉన్నాయి.",
        "above_mkt":  "ఊహించిన ధర (₹{pred}) ప్రస్తుత మార్కెట్ సగటు (₹{mkt_avg}) కంటే ఎక్కువ — డిమాండ్ బలంగా కనిపిస్తోంది.",
        "below_mkt":  "ఊహించిన ధర (₹{pred}) ప్రస్తుత మార్కెట్ సగటు (₹{mkt_avg}) కంటే తక్కువ — జాగ్రత్త అవసరం.",
        "near_mkt":   "ఊహించిన ధర ప్రస్తుత మార్కెట్ సగటు (₹{mkt_avg}) కు దగ్గరగా ఉంది — స్థిరమైన మార్కెట్ ఆశించబడుతోంది.",
    }
}


def generate_explanation(
    crop: str,
    season: str,
    harvest_on: bool,
    price: float,
    trend_slope: float,
    market_low: float = None,
    market_high: float = None,
    lang: str = "en",
) -> list:
    """Generate a list of human-readable explanation bullet points."""
    T   = EXPLANATION_TEMPLATES.get(lang, EXPLANATION_TEMPLATES["en"])
    SN  = SEASON_SUPPLY_NOTES.get(season, {})
    pts = []

    pts.append(T["season"].format(season=season))

    supply_note = SN.get(lang, SN.get("en", ""))
    if supply_note:
        pts.append(T["supply"].format(supply_note=supply_note))

    if harvest_on:
        pts.append(T["harvest_on"].format(crop=crop))
    else:
        pts.append(T["harvest_off"].format(crop=crop))

    if abs(trend_slope) < 0.5:
        pts.append(T["trend_stable"])
    elif trend_slope > 0:
        pts.append(T["trend_up"])
    else:
        pts.append(T["trend_down"])

    if market_low is not None and market_high is not None:
        mkt_avg = round((market_low + market_high) / 2, 2)
        diff_pct = (price - mkt_avg) / mkt_avg * 100 if mkt_avg else 0
        if diff_pct > 7:
            pts.append(T["above_mkt"].format(pred=price, mkt_avg=mkt_avg))
        elif diff_pct < -7:
            pts.append(T["below_mkt"].format(pred=price, mkt_avg=mkt_avg))
        else:
            pts.append(T["near_mkt"].format(mkt_avg=mkt_avg))

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_prediction_features(
    crop: str, crop_df: pd.DataFrame, target_date: datetime,
    log_transformed: bool = True,
) -> pd.DataFrame:
    hist = crop_df[["date", "price"]].copy()
    hist.columns = ["ds", "y"]
    hist["ds"] = pd.to_datetime(hist["ds"])
    hist = hist.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)

    if log_transformed:
        hist["y"] = np.log1p(hist["y"])

    target_row = pd.DataFrame({"ds": [pd.Timestamp(target_date)], "y": [np.nan]})
    combined   = pd.concat([hist.tail(90), target_row], ignore_index=True)
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
    for lag in [1, 3, 7, 14, 30]:
        combined[f"lag_{lag}"] = y.shift(lag)

    # ── Rolling ──────────────────────────────────────────────────────────────
    shifted = y.shift(1)
    combined["rolling_mean_7"]  = shifted.rolling(7,  min_periods=1).mean()
    combined["rolling_mean_14"] = shifted.rolling(14, min_periods=1).mean()
    combined["rolling_mean_30"] = shifted.rolling(30, min_periods=1).mean()
    combined["rolling_std_7"]   = shifted.rolling(7,  min_periods=2).std().fillna(0)
    combined["rolling_std_30"]  = shifted.rolling(30, min_periods=2).std().fillna(0)

    # ── Momentum ─────────────────────────────────────────────────────────────
    combined["price_change_7"] = y - y.shift(7)
    combined["price_pct_7"]    = y.pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)

    # ── Seasonal one-hot ──────────────────────────────────────────────────────
    seasons = month.apply(get_season)
    combined["season_kharif"] = (seasons == "Kharif").astype(int)
    combined["season_rabi"]   = (seasons == "Rabi").astype(int)
    combined["season_summer"] = (seasons == "Summer").astype(int)

    # ── Harvest peak flag ─────────────────────────────────────────────────────
    hm = HARVEST_MONTHS.get(crop, [])
    combined["harvest_peak_flag"] = month.isin(hm).astype(int)

    pred_row = combined.tail(1)[FEATURE_COLS].copy()

    # Fill any NaNs
    for col in FEATURE_COLS:
        if pred_row[col].isna().any():
            fill = combined[col].dropna().median()
            pred_row[col] = pred_row[col].fillna(
                fill if (fill is not None and not np.isnan(float(fill))) else 0
            )

    return pred_row


def _load_bundle(crop: str, region: str):
    path = os.path.join(MODEL_DIR, f"{crop}_{region}.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict):
            return bundle
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


def run_prediction(crop: str, region: str, target_date, lang: str = "en",
                   market_low: float = None, market_high: float = None) -> dict:
    """
    Returns a result dict with:
        price, confidence, rmse, mae, mape, reliability,
        n_points, trained_date, crop_key, region_key, date,
        season, harvest_on, explanation, seasonal_adjustment_pct
    Raises ValueError with error code on failure.
    """
    if not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())

    # ── Load data ──────────────────────────────────────────────────────────────
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

    # ── Load model bundle ──────────────────────────────────────────────────────
    bundle = _load_bundle(crop, region)
    if bundle is None:
        raise ValueError("NO_DATA")

    xgb_model  = bundle.get("xgb")
    lgbm_model = bundle.get("lgbm")
    if xgb_model is None:
        raise ValueError("NO_DATA")

    # ── Load metadata ──────────────────────────────────────────────────────────
    meta            = _load_metadata(crop, region)
    log_transformed = meta.get("log_transformed", True)
    mape            = meta.get("mape")
    rmse            = meta.get("rmse")
    mae             = meta.get("mae")
    reliability     = meta.get("reliability", "Medium")
    n_points        = meta.get("n_points", len(crop_df))
    trained_date    = meta.get("training_date", meta.get("trained_date",
                               last_date.strftime("%d %b %Y")))

    # ── Build features & predict ───────────────────────────────────────────────
    try:
        pred_row = _build_prediction_features(crop, crop_df, target_date, log_transformed)
        X_pred   = pred_row.reindex(columns=FEATURE_COLS)

        pred_xgb = float(xgb_model.predict(X_pred)[0])
        if lgbm_model is not None:
            try:
                pred_lgbm = float(lgbm_model.predict(X_pred)[0])
                raw_pred  = 0.60 * pred_xgb + 0.40 * pred_lgbm
            except Exception:
                raw_pred = pred_xgb
        else:
            raw_pred = pred_xgb

        if log_transformed:
            predicted_price = float(np.expm1(raw_pred))
        else:
            predicted_price = float(raw_pred)

    except Exception:
        raise ValueError("GENERIC_ERROR")

    # ── Clamp to realistic range ───────────────────────────────────────────────
    p_mean = float(crop_df["price"].mean())
    p_std  = float(crop_df["price"].std())
    predicted_price = float(np.clip(predicted_price, max(1.0, p_mean - 4 * p_std), p_mean + 4 * p_std))
    predicted_price = round(predicted_price, 2)

    # ── Seasonal intelligence ──────────────────────────────────────────────────
    target_month = target_date.month
    season       = get_season(target_month)
    harvest_on   = target_month in HARVEST_MONTHS.get(crop, [])

    adjusted_price = apply_seasonal_adjustment(crop, season, predicted_price)
    seasonal_adj_pct = round((adjusted_price - predicted_price) / predicted_price * 100, 1) if predicted_price > 0 else 0.0

    # ── Confidence — MAPE-based + days-ahead penalty ───────────────────────────
    if mape is not None:
        base_conf = 100 - mape * 0.9
    else:
        base_conf = 86.0

    days_penalty = max(0, delta_days) * 0.08
    confidence   = round(float(np.clip(base_conf - days_penalty, 75, 98)), 1)

    if mape is not None:
        reliability = "High" if mape < 10 else ("Medium" if mape < 18 else "Low")

    # ── Trend slope (last 30 days) ────────────────────────────────────────────
    trend_df = crop_df.sort_values("date").tail(30)
    if len(trend_df) >= 5:
        xs = np.arange(len(trend_df))
        trend_slope = float(np.polyfit(xs, trend_df["price"].values, 1)[0])
    else:
        trend_slope = 0.0

    # ── Explainable AI reasoning ───────────────────────────────────────────────
    explanation = generate_explanation(
        crop=crop, season=season, harvest_on=harvest_on,
        price=adjusted_price, trend_slope=trend_slope,
        market_low=market_low, market_high=market_high,
        lang=lang,
    )

    return {
        "price":                 adjusted_price,
        "raw_price":             predicted_price,
        "confidence":            confidence,
        "rmse":                  rmse,
        "mae":                   mae,
        "mape":                  mape,
        "reliability":           reliability,
        "n_points":              n_points,
        "trained_date":          trained_date,
        "crop_key":              crop,
        "region_key":            region,
        "date":                  target_date,
        "season":                season,
        "harvest_on":            harvest_on,
        "trend_slope":           round(trend_slope, 3),
        "seasonal_adj_pct":      seasonal_adj_pct,
        "explanation":           explanation,
        "model_version":         meta.get("model_version", "1.0"),
    }
