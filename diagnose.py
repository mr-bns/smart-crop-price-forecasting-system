"""
diagnose.py  —  Quick system health check
==========================================
Checks that all required files exist, models are present,
and runs a sample prediction to verify the pipeline end-to-end.

Usage:  python diagnose.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

DATA_PATH = os.path.join(_HERE, "data", "crops.csv")
MODEL_DIR = os.path.join(_HERE, "models")

SAMPLE_CROP   = "Tomato"
SAMPLE_REGION = "Guntur"


def check(label: str, ok: bool, detail: str = ""):
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}" + (f"  — {detail}" if detail else ""))
    return ok


def main():
    print("\n🔍 Smart Crop Price Forecasting System — Diagnostics\n")
    all_ok = True

    # 1. Data file
    data_ok = os.path.exists(DATA_PATH)
    all_ok &= check("crops.csv exists", data_ok, DATA_PATH)

    if data_ok:
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        rows = len(df)
        last = df["date"].max()
        check("CSV has data", rows > 0, f"{rows:,} rows, last date = {last}")

        import datetime
        staleness = (datetime.datetime.today() - pd.to_datetime(last)).days
        check(
            "Data freshness",
            staleness <= 7,
            f"{staleness} day(s) old" + (" — consider refreshing!" if staleness > 7 else ""),
        )

    # 2. Models directory + file count
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")] if os.path.exists(MODEL_DIR) else []
    all_ok &= check("models/ directory exists", os.path.exists(MODEL_DIR))
    all_ok &= check("Trained models present", len(models) == 60, f"{len(models)}/60 models found")

    sample_model = os.path.join(MODEL_DIR, f"{SAMPLE_CROP}_{SAMPLE_REGION}.joblib")
    all_ok &= check(f"{SAMPLE_CROP}_{SAMPLE_REGION} model", os.path.exists(sample_model))

    # 3. End-to-end prediction test
    print(f"\n  Running sample prediction: {SAMPLE_CROP} / {SAMPLE_REGION} …")
    try:
        from model.predict import run_prediction
        import datetime
        target = datetime.datetime.today() + datetime.timedelta(days=3)
        result = run_prediction(SAMPLE_CROP, SAMPLE_REGION, target, lang="en")
        price  = result.get("price")
        conf   = result.get("confidence")
        check(
            "Prediction pipeline",
            price is not None and price > 0,
            f"₹{price}/kg  |  confidence {conf}%  |  reliability {result.get('reliability')}",
        )
    except Exception as exc:
        all_ok = False
        check("Prediction pipeline", False, str(exc))

    print(f"\n{'✅ All checks passed!' if all_ok else '❌ Some checks failed — see above.'}\n")


if __name__ == "__main__":
    main()
