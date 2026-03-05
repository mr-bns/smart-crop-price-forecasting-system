"""
update_data.py  —  Data refresh utility
========================================
Regenerates crops.csv with fresh dates through today, then optionally
retriggers model training.

Usage:
    python update_data.py          # regenerate data only
    python update_data.py --train  # regenerate data + retrain all models
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from generate_data import main as generate_main


def main():
    print("=== Smart Crop Price Forecasting System — Data Refresh ===\n")

    # Step 1: Regenerate CSV
    generate_main()

    # Step 2: Optionally retrain
    if "--train" in sys.argv:
        print("\n=== Retraining all models ===\n")
        from model.train import train_all_models
        train_all_models()
    else:
        print(
            "\n✅ Data refreshed. To retrain models, run:\n"
            "   python update_data.py --train\n"
            "   — or —\n"
            "   python model/train.py"
        )


if __name__ == "__main__":
    main()
