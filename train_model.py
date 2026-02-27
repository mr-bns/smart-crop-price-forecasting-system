import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

DATA_PATH = "data/crops.csv"
MODEL_DIR = "models"


def train_all_models():
    print("🚀 Starting Batch Model Training...")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Run generate_data.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    try:
        combinations = df[["crop", "region"]].drop_duplicates().values
    except KeyError:
        print("❌ Error: CSV must contain 'crop' and 'region' columns.")
        return

    count   = 0
    skipped = 0

    for crop, region in combinations:
        print(f"   🔹 Training {crop} — {region}...")

        crop_df = df[(df["crop"] == crop) & (df["region"] == region)].copy()

        if len(crop_df) < 30:
            print(f"      ⚠️  Skipping: Not enough data ({len(crop_df)} rows)")
            skipped += 1
            continue

        try:
            train_df = crop_df[["date", "price"]].copy()
            train_df.columns = ["ds", "y"]
            train_df["ds"] = pd.to_datetime(train_df["ds"])
            train_df = train_df.sort_values("ds").drop_duplicates("ds")

            # Remove obvious outliers (beyond 3 std dev)
            mean_y = train_df["y"].mean()
            std_y  = train_df["y"].std()
            train_df = train_df[
                (train_df["y"] >= mean_y - 3 * std_y) &
                (train_df["y"] <= mean_y + 3 * std_y)
            ].copy()

            n_points = len(train_df)

            # Hyperparameter candidates — pick best by cross-val RMSE on last 20 rows
            cps_candidates = [0.05, 0.10, 0.15, 0.25, 0.40]
            best_model = None
            best_rmse  = float("inf")
            best_cps   = 0.15

            holdout_n = min(20, n_points // 5)
            train_part  = train_df.iloc[:-holdout_n]
            holdout_part = train_df.iloc[-holdout_n:]

            if len(train_part) >= 30:
                for cps in cps_candidates:
                    try:
                        m_try = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            seasonality_mode="multiplicative",
                            changepoint_prior_scale=cps,
                            seasonality_prior_scale=12.0,
                            interval_width=0.80,
                        )
                        m_try.add_seasonality(name="monthly", period=30.5, fourier_order=8)
                        m_try.fit(train_part)

                        periods = (holdout_part["ds"].max() - train_part["ds"].max()).days
                        future_try = m_try.make_future_dataframe(periods=max(1, periods))
                        fc_try = m_try.predict(future_try)
                        fc_try["ds"] = pd.to_datetime(fc_try["ds"])

                        merged = holdout_part.merge(fc_try[["ds", "yhat"]], on="ds", how="inner")
                        if len(merged) >= 5:
                            rmse_try = float(np.sqrt(np.mean((merged["y"] - merged["yhat"]) ** 2)))
                            if rmse_try < best_rmse:
                                best_rmse = rmse_try
                                best_cps  = cps
                    except Exception:
                        continue

            # Fit final model on full data with best hyperparameter
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=best_cps,
                seasonality_prior_scale=12.0,
                interval_width=0.80,
            )
            model.add_seasonality(name="monthly", period=30.5, fourier_order=8)
            model.fit(train_df)

            # Save model
            filename  = f"{crop}_{region}.joblib"
            save_path = os.path.join(MODEL_DIR, filename)
            joblib.dump(model, save_path)

            # Save metadata JSON
            meta = {
                "crop":         crop,
                "region":       region,
                "n_points":     n_points,
                "best_cps":     best_cps,
                "trained_date": datetime.now().strftime("%d %b %Y"),
                "train_rmse":   round(best_rmse, 2) if best_rmse < float("inf") else None,
            }
            meta_path = os.path.join(MODEL_DIR, f"{crop}_{region}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            count += 1
            print(f"      ✅ Saved: {filename} | best_cps={best_cps} | train_rmse={meta['train_rmse']}")

        except Exception as e:
            print(f"      ❌ Training failed for {crop}/{region}: {e}")
            skipped += 1

    print(f"\n✅ Training Complete — {count} models saved, {skipped} skipped.")


if __name__ == "__main__":
    train_all_models()
