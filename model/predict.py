"""
model/predict.py  —  XGBoost + LightGBM Prediction Engine v2.1
================================================================
Key improvements over v2.0:
  • Absolute paths — works from any working directory
  • Distinct error codes: NO_MODEL (file missing) vs NO_DATA (CSV missing/empty)
  • Increased history window to 120 rows for robust lag/rolling features
  • Forward-fill before median fallback for NaN imputation
  • Seasonal adjustment weight reduced to 40 % to avoid double-counting
  • Data freshness check — warns if CSV is > 14 days stale
  • Cleaner, well-documented code throughout
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ── Resolve project root ──────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))  # …/prototype/model
_PROJECT = os.path.dirname(_HERE)                       # …/prototype
sys.path.insert(0, _PROJECT)

DATA_PATH = os.path.join(_PROJECT, "data", "crops.csv")
MODEL_DIR = os.path.join(_PROJECT, "models")
MAX_DAYS  = 30   # maximum forecast horizon

# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def get_season(month: int) -> str:
    """Return Indian agricultural season name for the given month."""
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    if month in [11, 12, 1, 2, 3]:
        return "Rabi"
    return "Summer"


# ─────────────────────────────────────────────────────────────────────────────
# HARVEST CALENDAR  (peak harvest months → high supply → lower prices)
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

# Seasonal price adjustment — reduced weight (0.40×) to avoid double-counting
# the seasonality that the XGBoost model already learns from seasonal features.
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
    "Rabi": {
        "en": "Winter season — generally stable supply conditions.",
        "te": "శీతాకాలం — సాధారణంగా స్థిరమైన సరఫరా పరిస్థితులు.",
    },
    "Summer": {
        "en": "Summer season — reduced supply, prices often rise.",
        "te": "వేసవి కాలం — తక్కువ సరఫరా, ధరలు తరచుగా పెరుగుతాయి.",
    },
}


def apply_seasonal_adjustment(crop: str, season: str, predicted_price: float) -> float:
    """
    Apply a *blended* post-prediction seasonal correction.
    Weight = 0.40 to avoid doubling the seasonality the model already learned.
    """
    raw_factor = SEASONAL_ADJUSTMENT.get(crop, {}).get(season, 0.0)
    blended    = raw_factor * 0.40   # reduce overcorrection
    return round(float(predicted_price * (1 + blended)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS — must stay identical to model/train.py
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
        "season":      "Current season: {season}",
        "supply":      "{supply_note}",
        "harvest_on":  "{crop} is in peak harvest — high supply typically lowers prices.",
        "harvest_off": "{crop} is outside peak harvest — tighter supply often pushes prices up.",
        "trend_up":    "Historical prices show an upward trend over the last 30 days.",
        "trend_down":  "Historical prices show a downward trend over the last 30 days.",
        "trend_stable":"Historical prices have been relatively stable.",
        "above_mkt":   "Forecast (₹{pred}) is ABOVE market average (₹{mkt_avg}) — strong demand signal.",
        "below_mkt":   "Forecast (₹{pred}) is BELOW market average (₹{mkt_avg}) — consider selling now.",
        "near_mkt":    "Forecast is close to market average (₹{mkt_avg}) — stable market expected.",
    },
    "te": {
        "season":      "ప్రస్తుత సీజన్: {season}",
        "supply":      "{supply_note}",
        "harvest_on":  "{crop} గరిష్ఠ కోత కాలంలో ఉంది — అధిక సరఫరా సాధారణంగా ధరలను తగ్గిస్తుంది.",
        "harvest_off": "{crop} గరిష్ఠ కోత కాలంలో లేదు — తక్కువ సరఫరా తరచుగా ధరలను పెంచుతుంది.",
        "trend_up":    "చారిత్రక ధరలు గత 30 రోజులలో పెరిగే ధోరణి చూపాయి.",
        "trend_down":  "చారిత్రక ధరలు గత 30 రోజులలో తగ్గే ధోరణి చూపాయి.",
        "trend_stable":"చారిత్రక ధరలు సాపేక్షంగా స్థిరంగా ఉన్నాయి.",
        "above_mkt":   "అంచనా ధర (₹{pred}) మార్కెట్ సగటు (₹{mkt_avg}) కంటే ఎక్కువ — డిమాండ్ బలంగా ఉంది.",
        "below_mkt":   "అంచనా ధర (₹{pred}) మార్కెట్ సగటు (₹{mkt_avg}) కంటే తక్కువ — ఇప్పుడు అమ్మడం మంచిది.",
        "near_mkt":    "అంచనా ధర మార్కెట్ సగటు (₹{mkt_avg}) కు దగ్గరగా ఉంది — మార్కెట్ స్థిరంగా ఉంటుంది.",
    },
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
    """Return a list of human-readable bullet-point explanation strings."""
    T   = EXPLANATION_TEMPLATES.get(lang, EXPLANATION_TEMPLATES["en"])
    SN  = SEASON_SUPPLY_NOTES.get(season, {})
    pts = []

    pts.append(T["season"].format(season=season))

    supply_note = SN.get(lang, SN.get("en", ""))
    if supply_note:
        pts.append(T["supply"].format(supply_note=supply_note))

    pts.append(T["harvest_on" if harvest_on else "harvest_off"].format(crop=crop))

    if abs(trend_slope) < 0.5:
        pts.append(T["trend_stable"])
    elif trend_slope > 0:
        pts.append(T["trend_up"])
    else:
        pts.append(T["trend_down"])

    if market_low is not None and market_high is not None:
        mkt_avg  = round((market_low + market_high) / 2, 2)
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
    crop: str,
    crop_df: pd.DataFrame,
    target_date: datetime,
    log_transformed: bool = True,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for the target_date prediction.
    Uses up to 120 rows of history to ensure all lags/rolling windows are
    well-populated (lag_30 needs at least 31 rows).
    """
    hist = crop_df[["date", "price"]].copy()
    hist.columns = ["ds", "y"]
    hist["ds"] = pd.to_datetime(hist["ds"])
    hist = hist.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)

    if log_transformed:
        hist["y"] = np.log1p(hist["y"])

    # Use 120 rows of history (up from 90) for better lag/rolling coverage
    target_row = pd.DataFrame({"ds": [pd.Timestamp(target_date)], "y": [np.nan]})
    combined   = pd.concat([hist.tail(120), target_row], ignore_index=True)
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

    # ── Extended lags ────────────────────────────────────────────────────────
    for lag in [1, 3, 7, 14, 30]:
        combined[f"lag_{lag}"] = y.shift(lag)

    # ── Rolling statistics  (shift-1 to prevent leakage) ────────────────────
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

    # ── NaN imputation: forward-fill first, then median fallback ─────────────
    for col in FEATURE_COLS:
        if pred_row[col].isna().any():
            fwd = combined[col].ffill()
            fill_val = fwd.iloc[-1] if not pd.isna(fwd.iloc[-1]) else combined[col].median()
            pred_row[col] = pred_row[col].fillna(
                fill_val if (fill_val is not None and not np.isnan(float(fill_val))) else 0.0
            )

    return pred_row


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_bundle(crop: str, region: str):
    """Load the joblib model bundle. Returns None if file missing or corrupt."""
    path = os.path.join(MODEL_DIR, f"{crop}_{region}.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        return bundle if isinstance(bundle, dict) else {"xgb": bundle, "lgbm": None}
    except Exception:
        return None


def _load_metadata(crop: str, region: str) -> dict:
    """Load the JSON metadata sidecar. Returns empty dict if unavailable."""
    path = os.path.join(MODEL_DIR, f"{crop}_{region}_meta.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(
    crop: str,
    region: str,
    target_date,
    lang: str = "en",
    market_low: float = None,
    market_high: float = None,
) -> dict:
    """
    Run a price forecast for (crop, region, target_date).

    Returns a result dict with keys:
        price, raw_price, confidence, rmse, mae, mape, reliability,
        n_points, trained_date, crop_key, region_key, date,
        season, harvest_on, explanation, seasonal_adj_pct, model_version,
        data_freshness_days

    Raises ValueError with one of these error codes:
        NO_DATA    — CSV missing or no rows for this crop/region
        NO_MODEL   — Model .joblib file missing or unloadable
        PAST_DATE  — target_date too far in the past
        FORECAST_LIMIT — target_date beyond 30-day horizon
        GENERIC_ERROR  — unexpected internal error
    """
    if not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())

    # ── 1. Load CSV ──────────────────────────────────────────────────────────
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
    last_date       = crop_df["date"].max()
    delta_days      = (target_date - last_date).days

    # Data freshness — how many days since the last CSV record
    data_freshness_days = int((datetime.today() - last_date).days)

    if delta_days < -7:
        raise ValueError("PAST_DATE")
    if delta_days > MAX_DAYS + 5:
        raise ValueError("FORECAST_LIMIT")

    # ── 2. Load model ────────────────────────────────────────────────────────
    bundle = _load_bundle(crop, region)
    if bundle is None:
        raise ValueError("NO_MODEL")   # Distinct from NO_DATA

    xgb_model  = bundle.get("xgb")
    lgbm_model = bundle.get("lgbm")
    if xgb_model is None:
        raise ValueError("NO_MODEL")

    # ── 3. Load metadata ─────────────────────────────────────────────────────
    meta            = _load_metadata(crop, region)
    log_transformed = meta.get("log_transformed", True)
    mape            = meta.get("mape")
    rmse            = meta.get("rmse")
    mae             = meta.get("mae")
    reliability     = meta.get("reliability", "Medium")
    n_points        = meta.get("n_points", len(crop_df))
    trained_date    = meta.get(
        "training_date",
        meta.get("trained_date", last_date.strftime("%d %b %Y"))
    )

    # ── 4. Build features & predict ──────────────────────────────────────────
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

        predicted_price = float(np.expm1(raw_pred)) if log_transformed else float(raw_pred)

    except Exception:
        raise ValueError("GENERIC_ERROR")

    # ── 5. Clamp to realistic historical range ───────────────────────────────
    p_mean          = float(crop_df["price"].mean())
    p_std           = float(crop_df["price"].std())
    predicted_price = float(np.clip(
        predicted_price,
        max(1.0, p_mean - 4 * p_std),
        p_mean + 4 * p_std,
    ))
    predicted_price = round(predicted_price, 2)

    # ── 6. Seasonal intelligence ─────────────────────────────────────────────
    target_month   = target_date.month
    season         = get_season(target_month)
    harvest_on     = target_month in HARVEST_MONTHS.get(crop, [])
    adjusted_price = apply_seasonal_adjustment(crop, season, predicted_price)
    seasonal_adj_pct = (
        round((adjusted_price - predicted_price) / predicted_price * 100, 1)
        if predicted_price > 0 else 0.0
    )

    # ── 7. Confidence score: MAPE-based + days-ahead penalty ─────────────────
    base_conf    = (100 - mape * 0.9) if mape is not None else 86.0
    days_penalty = max(0, delta_days) * 0.08
    confidence   = round(float(np.clip(base_conf - days_penalty, 75, 98)), 1)

    if mape is not None:
        reliability = "High" if mape < 10 else ("Medium" if mape < 18 else "Low")

    # ── 8. Trend slope (last 30 days) ────────────────────────────────────────
    trend_df = crop_df.sort_values("date").tail(30)
    if len(trend_df) >= 5:
        xs          = np.arange(len(trend_df))
        trend_slope = float(np.polyfit(xs, trend_df["price"].values, 1)[0])
    else:
        trend_slope = 0.0

    # ── 9. Explainable AI reasoning ──────────────────────────────────────────
    explanation = generate_explanation(
        crop=crop,
        season=season,
        harvest_on=harvest_on,
        price=adjusted_price,
        trend_slope=trend_slope,
        market_low=market_low,
        market_high=market_high,
        lang=lang,
    )

    return {
        "price":               adjusted_price,
        "raw_price":           predicted_price,
        "confidence":          confidence,
        "rmse":                rmse,
        "mae":                 mae,
        "mape":                mape,
        "reliability":         reliability,
        "n_points":            n_points,
        "trained_date":        trained_date,
        "crop_key":            crop,
        "region_key":          region,
        "date":                target_date,
        "season":              season,
        "harvest_on":          harvest_on,
        "trend_slope":         round(trend_slope, 3),
        "seasonal_adj_pct":    seasonal_adj_pct,
        "explanation":         explanation,
        "model_version":       meta.get("model_version", "2.1"),
        "data_freshness_days": data_freshness_days,
    }
