"""
Enhanced XGBoost crop price model training — v2.0
Improvements:
  1. Seasonal intelligence (Kharif / Rabi / Summer)
  2. Harvest-peak flag per crop (supply shock indicator)
  3. Extended lag features: 1,3,7,14,30 days
  4. Rolling mean/std: 7, 14, 30 days
  5. Seasonal price adjustment after prediction
  6. Broader hyperparameter search with early stopping
  7. Rich metadata storage (feature_list, model_version, season)
  8. log1p price transform to reduce RMSE
  9. TimeSeriesSplit CV (n_splits=8)
  10. Optional LightGBM blend (60% XGB + 40% LGBM)
"""

import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError as e:
    raise ImportError(f"Missing: {e}. Run: pip install xgboost scikit-learn")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logging.basicConfig(level=logging.WARNING)

DATA_PATH   = "data/crops.csv"
MODEL_DIR   = "models"
MODEL_VERSION = "2.0"

# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────
def get_season(month: int) -> str:
    """Return Indian agricultural season based on month."""
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:  # 4, 5
        return "Summer"


# ─────────────────────────────────────────────────────────────────────────────
# CROP HARVEST CALENDAR
# Peak harvest months → harvest_peak_flag = 1
# ─────────────────────────────────────────────────────────────────────────────
HARVEST_MONTHS = {
    "Tomato":      [12, 1, 2],       # Dec–Feb
    "Onion":       [3, 4, 5],        # Mar–May
    "Potato":      [1, 2, 3],        # Jan–Mar
    "Paddy":       [10, 11, 12],     # Oct–Dec
    "BitterGourd": [5, 6, 7, 8],     # May–Aug
    "Brinjal":     [9, 10, 11],      # Sep–Nov
    "BroadBeans":  [11, 12, 1],      # Nov–Jan
    "Carrot":      [11, 12, 1, 2],   # Nov–Feb
    "GreenChilli": [1, 2, 3, 4],     # Jan–Apr
    "Okra":        [5, 6, 7, 8],     # May–Aug
}

# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL PRICE ADJUSTMENT RULES
# ─────────────────────────────────────────────────────────────────────────────
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


def apply_seasonal_adjustment(crop: str, season: str, predicted_price: float) -> float:
    """Apply season-aware price adjustment to simulate real market cycles."""
    factor = SEASONAL_ADJUSTMENT.get(crop, {}).get(season, 0.0)
    adjusted = predicted_price * (1 + factor)
    return round(float(adjusted), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS (strict order — must match predict.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Calendar — cyclical encoded
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "weekofyear", "dayofyear", "quarter",
    # Required lags
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_30",
    # Rolling statistics
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_std_30",
    # Momentum
    "price_change_7", "price_pct_7",
    # Seasonal
    "season_kharif", "season_rabi", "season_summer",
    "harvest_peak_flag",
]


def engineer_features(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"])

    month = df["ds"].dt.month
    dow   = df["ds"].dt.dayofweek

    # ── Calendar (cyclical) ──────────────────────────────────────────────────
    df["month_sin"]  = np.sin(2 * np.pi * month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * month / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * dow / 7)
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    df["dayofyear"]  = df["ds"].dt.dayofyear
    df["quarter"]    = df["ds"].dt.quarter

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 3, 7, 14, 30]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # ── Rolling statistics (lag-1 shifted to prevent leakage) ────────────────
    shifted = df["y"].shift(1)
    df["rolling_mean_7"]  = shifted.rolling(7,  min_periods=1).mean()
    df["rolling_mean_14"] = shifted.rolling(14, min_periods=1).mean()
    df["rolling_mean_30"] = shifted.rolling(30, min_periods=1).mean()
    df["rolling_std_7"]   = shifted.rolling(7,  min_periods=2).std().fillna(0)
    df["rolling_std_30"]  = shifted.rolling(30, min_periods=2).std().fillna(0)

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["price_change_7"] = df["y"] - df["y"].shift(7)
    df["price_pct_7"]    = df["y"].pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)

    # ── Seasonal one-hot ──────────────────────────────────────────────────────
    seasons = month.apply(get_season)
    df["season_kharif"] = (seasons == "Kharif").astype(int)
    df["season_rabi"]   = (seasons == "Rabi").astype(int)
    df["season_summer"] = (seasons == "Summer").astype(int)

    # ── Harvest peak flag ─────────────────────────────────────────────────────
    hm = HARVEST_MONTHS.get(crop, [])
    df["harvest_peak_flag"] = month.isin(hm).astype(int)

    return df.dropna(subset=FEATURE_COLS)


def compute_metrics(actual, predicted):
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    predicted = np.clip(predicted, 0.01, None)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae  = float(mean_absolute_error(actual, predicted))
    mape = float(np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1e-9, actual))) * 100)
    return round(rmse, 3), round(mae, 3), round(mape, 3)


def train_single(crop: str, region: str, df: pd.DataFrame):
    raw = df[["date", "price"]].copy()
    raw.columns = ["ds", "y"]
    raw["ds"] = pd.to_datetime(raw["ds"])
    raw = raw.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)

    if len(raw) < 60:
        return None

    # ── Outlier removal (IQR method + z-score) ───────────────────────────────
    q1, q3 = raw["y"].quantile(0.25), raw["y"].quantile(0.75)
    iqr     = q3 - q1
    mu, sigma = raw["y"].mean(), raw["y"].std()
    raw = raw[
        (raw["y"] >= q1 - 1.5 * iqr) & (raw["y"] <= q3 + 1.5 * iqr) &
        (raw["y"] >= mu - 2.5 * sigma) & (raw["y"] <= mu + 2.5 * sigma)
    ].copy()

    if len(raw) < 50:
        return None

    price_mean_orig = float(raw["y"].mean())
    price_std_orig  = float(raw["y"].std())

    # ── Log1p transform ───────────────────────────────────────────────────────
    raw["y"] = np.log1p(raw["y"])

    feat_df = engineer_features(raw, crop)
    if len(feat_df) < 50:
        return None

    X = feat_df[FEATURE_COLS]
    y = feat_df["y"].values

    # ── 80/20 time-based holdout ──────────────────────────────────────────────
    split  = int(len(X) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    # ── Hyperparameter grid search with TimeSeriesSplit ───────────────────────
    tscv = TimeSeriesSplit(n_splits=8)
    best_params  = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.04,
                    "subsample": 0.85, "colsample_bytree": 0.80,
                    "min_child_weight": 3, "gamma": 0.1, "reg_alpha": 0.1}
    best_cv_rmse = float("inf")

    param_grid = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 2, "gamma": 0.0, "reg_alpha": 0.0},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.04, "min_child_weight": 3, "gamma": 0.1, "reg_alpha": 0.1},
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 4, "gamma": 0.0, "reg_alpha": 0.1},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 2, "gamma": 0.1, "reg_alpha": 0.0},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.04, "min_child_weight": 3, "gamma": 0.0, "reg_alpha": 0.0},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "min_child_weight": 5, "gamma": 0.1, "reg_alpha": 0.1},
    ]

    for params in param_grid:
        cv_rmses = []
        for tr_i, val_i in tscv.split(X_tr):
            if len(val_i) < 4:
                continue
            m = XGBRegressor(
                **params,
                reg_lambda=1.0, subsample=0.85, colsample_bytree=0.80,
                random_state=42, verbosity=0, n_jobs=-1,
                early_stopping_rounds=30,
                eval_metric="rmse",
            )
            m.fit(
                X_tr.iloc[tr_i], y_tr[tr_i],
                eval_set=[(X_tr.iloc[val_i], y_tr[val_i])],
                verbose=False,
            )
            preds = m.predict(X_tr.iloc[val_i])
            cv_rmses.append(np.sqrt(mean_squared_error(y_tr[val_i], preds)))

        if cv_rmses and np.mean(cv_rmses) < best_cv_rmse:
            best_cv_rmse = float(np.mean(cv_rmses))
            best_params  = {**params}

    # ── Train final XGBoost ───────────────────────────────────────────────────
    xgb_model = XGBRegressor(
        **best_params,
        reg_lambda=1.0, subsample=0.85, colsample_bytree=0.80,
        random_state=42, verbosity=0, n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr)

    # ── Optional LightGBM blend ───────────────────────────────────────────────
    lgbm_model = None
    if HAS_LGBM and len(X_tr) >= 60:
        try:
            lgbm_model = LGBMRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                learning_rate=best_params["learning_rate"],
                num_leaves=31, min_child_samples=10,
                reg_alpha=best_params["reg_alpha"],
                subsample=0.85, colsample_bytree=0.80,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            lgbm_model.fit(X_tr, y_tr)
        except Exception:
            lgbm_model = None

    # ── Evaluate on holdout ───────────────────────────────────────────────────
    rmse = mae = mape = None
    if len(X_te) >= 5:
        preds_xgb = xgb_model.predict(X_te)
        if lgbm_model is not None:
            preds_lgbm = lgbm_model.predict(X_te)
            preds_te   = 0.60 * preds_xgb + 0.40 * preds_lgbm
        else:
            preds_te = preds_xgb
        y_te_real  = np.expm1(y_te)
        preds_real = np.expm1(preds_te)
        rmse, mae, mape = compute_metrics(y_te_real, preds_real)

    # ── Re-fit on ALL data for deployment ────────────────────────────────────
    xgb_model.fit(X, y)
    if lgbm_model is not None:
        lgbm_model.fit(X, y)

    # ── Confidence & reliability ──────────────────────────────────────────────
    if mape is not None:
        confidence  = float(np.clip(100 - mape * 0.9, 75, 98))
        reliability = "High" if mape < 10 else ("Medium" if mape < 18 else "Low")
    else:
        confidence  = 86.0
        reliability = "Medium"

    metadata = {
        "crop":             crop,
        "region":           region,
        "n_points":         len(feat_df),
        "training_date":    datetime.now().strftime("%d %b %Y"),
        "training_records": len(raw),
        "best_params":      best_params,
        "use_lgbm_blend":   lgbm_model is not None,
        "log_transformed":  True,
        "rmse":             rmse,
        "mae":              mae,
        "mape":             mape,
        "confidence":       round(confidence, 1),
        "reliability":      reliability,
        "model_version":    MODEL_VERSION,
        "feature_list":     FEATURE_COLS,
        "price_mean":       round(price_mean_orig, 2),
        "price_std":        round(price_std_orig, 2),
        "last_date":        str(raw["ds"].max().date()),
    }

    model_bundle = {"xgb": xgb_model, "lgbm": lgbm_model}
    return model_bundle, metadata


def train_all_models():
    print("🚀 Starting Enhanced XGBoost v2 Training (seasonal + harvest calendar + IQR)…")
    if HAS_LGBM:
        print("   ✅ LightGBM detected — blend models enabled")
    else:
        print("   ℹ️  LightGBM not found — XGBoost only (pip install lightgbm)")

    if not os.path.exists(DATA_PATH):
        print(f"❌ {DATA_PATH} not found. Run generate_data.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ {e}")
        return

    # Remove duplicates globally
    df = df.drop_duplicates(subset=["date", "crop", "region"])
    df = df.dropna(subset=["price"])

    combinations = df[["crop", "region"]].drop_duplicates().values
    count, skipped = 0, 0

    for crop, region in combinations:
        print(f"   🌱 {crop}/{region} … ", end="", flush=True)
        crop_df = df[(df["crop"] == crop) & (df["region"] == region)].copy()

        try:
            result = train_single(crop, region, crop_df)
            if result is None:
                print("⚠️  skipped (not enough data)")
                skipped += 1
                continue

            bundle, meta = result
            joblib.dump(bundle, os.path.join(MODEL_DIR, f"{crop}_{region}.joblib"))
            with open(os.path.join(MODEL_DIR, f"{crop}_{region}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            blend_tag = " [XGB+LGBM]" if meta["use_lgbm_blend"] else " [XGB]"
            mape_str  = (
                f"RMSE={meta['rmse']}  MAPE={meta['mape']}%  "
                f"conf={meta['confidence']}%  [{meta['reliability']}]{blend_tag}"
            )
            print(f"✅  {mape_str}")
            count += 1

        except Exception as e:
            print(f"❌ {e}")
            skipped += 1

    print(f"\n✅ Done — {count} models saved, {skipped} skipped.")


if __name__ == "__main__":
    train_all_models()
