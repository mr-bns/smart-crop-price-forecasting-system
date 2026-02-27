import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

DATA_PATH  = "data/crops.csv"
MODEL_DIR  = "models"
MAX_DAYS   = 30

def _compute_accuracy_metrics(model, crop_df):
    """
    Compute RMSE and MAPE using a held-out set (last 30 data points).
    Returns (rmse, mape).
    """
    try:
        df = crop_df[["date", "price"]].copy()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").drop_duplicates("ds")

        if len(df) < 40:
            return None, None

        # Hold out last 30 rows for evaluation
        holdout = df.tail(30).copy()
        train   = df.iloc[:-30].copy()
        if len(train) < 30:
            return None, None

        # Use the already-fitted model to forecast into the holdout period
        periods = (holdout["ds"].max() - train["ds"].max()).days
        periods = max(30, periods)
        future   = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast["ds"] = pd.to_datetime(forecast["ds"])

        # Inverse log transform if needed (model may have log-transformed y)
        has_log = getattr(model, '_log_transformed', False)

        merged = holdout.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
        if merged.empty:
            return None, None

        actual = merged["y"].values
        predicted = merged["yhat"].values

        if has_log:
            actual    = np.expm1(actual)
            predicted = np.expm1(predicted)

        predicted = np.clip(predicted, 0.1, None)
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mape = float(np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1e-9, actual))) * 100)

        return round(rmse, 2), round(mape, 2)

    except Exception:
        return None, None


def _load_model_metadata(crop, region):
    """Load saved metadata JSON if it exists."""
    meta_path = os.path.join(MODEL_DIR, f"{crop}_{region}_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_reliability(mape, confidence):
    """Compute reliability tag from MAPE and confidence."""
    if mape is None:
        if confidence >= 90:
            return "High"
        elif confidence >= 75:
            return "Medium"
        return "Low"

    if mape < 10 and confidence >= 90:
        return "High"
    elif mape <= 20:
        return "Medium"
    return "Low"


def run_prediction(crop, region, target_date):
    """
    Load pre-trained Prophet model and forecast price for target_date.
    Returns dict with keys:
        price, confidence, rmse, mape, reliability, n_points, trained_date
    Raises ValueError with error code string on failure.
    """

    # ── 1. Load CSV ──────────────────────────────────
    if not os.path.exists(DATA_PATH):
        raise ValueError("NO_DATA")

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        raise ValueError("GENERIC_ERROR")

    # ── 2. Filter crop/region ────────────────────────
    crop_df = df[(df["crop"] == crop) & (df["region"] == region)].copy()

    if crop_df.empty:
        raise ValueError("NO_DATA")

    # ── 3. Date validation ───────────────────────────
    try:
        crop_df["date"] = pd.to_datetime(crop_df["date"])
        last_date = crop_df["date"].max()
    except Exception:
        raise ValueError("GENERIC_ERROR")

    if not isinstance(target_date, datetime):
        target_date = datetime.combine(target_date, datetime.min.time())

    delta_days = (target_date - last_date).days

    if delta_days < -7:
        raise ValueError("PAST_DATE")
    if delta_days > MAX_DAYS + 5:
        raise ValueError("FORECAST_LIMIT")

    # ── 4. Load model ────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"{crop}_{region}.joblib")
    if not os.path.exists(model_path):
        raise ValueError("NO_DATA")

    try:
        model = joblib.load(model_path)
    except Exception:
        raise ValueError("GENERIC_ERROR")

    # ── 5. Generate forecast ─────────────────────────
    periods = max(1, delta_days)

    try:
        future   = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        forecast["ds"] = pd.to_datetime(forecast["ds"])
        target_ts = pd.Timestamp(target_date)
        closest_idx = (forecast["ds"] - target_ts).abs().idxmin()
        predicted_price = float(forecast.loc[closest_idx, "yhat"])

        # Confidence interval width for this prediction
        yhat_lower = float(forecast.loc[closest_idx, "yhat_lower"])
        yhat_upper = float(forecast.loc[closest_idx, "yhat_upper"])

        # Clamp to realistic range
        predicted_price = max(1.0, predicted_price)
        p_mean = crop_df["price"].mean()
        p_std  = crop_df["price"].std()
        lower_bound = max(1.0, p_mean - 5 * p_std)
        upper_bound = p_mean + 5 * p_std
        predicted_price = float(max(lower_bound, min(predicted_price, upper_bound)))

        # Store predicted confidence interval
        interval_width = max(0.1, yhat_upper - yhat_lower)
        relative_width = interval_width / max(predicted_price, 1)

    except Exception:
        raise ValueError("GENERIC_ERROR")

    # ── 6. Compute accuracy metrics ──────────────────
    rmse, mape = _compute_accuracy_metrics(model, crop_df)

    # ── 7. Confidence score ───────────────────────────
    # Primary: based on MAPE; Secondary: relative CI width; Tertiary: days ahead
    if mape is not None:
        # MAPE-based confidence:  MAPE 0% → 99%, MAPE 20% → 75%, MAPE 30% → 60%
        mape_conf = max(50.0, 99.0 - mape * 1.5)
    else:
        mape_conf = 90.0

    # Penalise for wide confidence interval
    ci_penalty = min(10.0, relative_width * 20)
    # Penalise for days ahead (slight)
    days_penalty = max(0, delta_days) * 0.15

    confidence = max(60.0, mape_conf - ci_penalty - days_penalty)

    # ── 8. Load metadata ─────────────────────────────
    meta = _load_model_metadata(crop, region)
    n_points    = meta.get("n_points", len(crop_df))
    trained_date = meta.get("trained_date",
                            last_date.strftime("%d %b %Y") if hasattr(last_date, "strftime") else "N/A")

    # ── 9. Reliability tag ────────────────────────────
    reliability = _get_reliability(mape, confidence)

    return {
        "price":        round(predicted_price, 2),
        "confidence":   round(confidence, 1),
        "rmse":         rmse,
        "mape":         mape,
        "reliability":  reliability,
        "n_points":     n_points,
        "trained_date": trained_date,
        "yhat_lower":   round(max(1.0, yhat_lower), 2),
        "yhat_upper":   round(yhat_upper, 2),
    }
