"""
generate_data.py  —  Synthetic AP Mandi Price Generator
Generates realistic 2-year historical price data for 10 crops × 6 districts.

Run:  python generate_data.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Resolve paths ─────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(_HERE, "data", "crops.csv")

# ── Configuration ─────────────────────────────────────────────────────────────
CROPS = [
    "Tomato", "Onion", "Potato", "Paddy",
    "BitterGourd", "Brinjal", "BroadBeans", "Carrot", "GreenChilli", "Okra"
]

REGIONS = [
    "Krishna", "Guntur", "Visakhapatnam",
    "EastWestGodavari", "Chittoor", "Kurnool"
]

DAYS     = 730    # 2 years of history → better lag/rolling features
END_DATE = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=DAYS - 1)

# ── Crop statistics ───────────────────────────────────────────────────────────
# (base_price_₹/kg, volatility, harvest_dip_month, harvest_month_list)
CROP_STATS = {
    "Tomato":      (20.0, 5.0,  11, [10, 11, 12, 1]),
    "Onion":       (24.0, 4.5,   1, [12, 1, 2]),
    "Potato":      (22.0, 4.0,   2, [1, 2, 3]),
    "Paddy":       (23.5, 1.5,  11, [10, 11]),
    "BitterGourd": (30.0, 6.0,   7, [6, 7, 8, 9]),
    "Brinjal":     (15.0, 4.0,  10, [9, 10, 11]),
    "BroadBeans":  (38.0, 7.0,  12, [11, 12, 1]),
    "Carrot":      (25.0, 5.0,  12, [11, 12, 1, 2]),
    "GreenChilli": (40.0, 12.0,  2, [1, 2, 3]),
    "Okra":        (20.0, 5.0,   6, [5, 6, 7, 8]),
}

# Regional price multipliers relative to base
REGIONAL_FACTORS = {
    "Krishna": {
        "Tomato": 1.02, "Onion": 1.01, "Potato": 1.01, "Paddy": 1.00,
        "BitterGourd": 1.05, "Brinjal": 1.02, "BroadBeans": 1.03,
        "Carrot": 1.04, "GreenChilli": 1.10, "Okra": 1.03,
    },
    "Guntur": {
        "Tomato": 0.97, "Onion": 0.98, "Potato": 0.96, "Paddy": 0.99,
        "BitterGourd": 0.96, "Brinjal": 0.95, "BroadBeans": 0.98,
        "Carrot": 0.97, "GreenChilli": 0.85, "Okra": 0.96,  # Guntur chilli discount
    },
    "Visakhapatnam": {
        "Tomato": 1.10, "Onion": 1.08, "Potato": 1.09, "Paddy": 1.02,
        "BitterGourd": 1.12, "Brinjal": 1.08, "BroadBeans": 1.10,
        "Carrot": 1.10, "GreenChilli": 1.05, "Okra": 1.10,
    },
    "EastWestGodavari": {
        "Tomato": 1.00, "Onion": 0.99, "Potato": 0.99, "Paddy": 1.03,
        "BitterGourd": 1.00, "Brinjal": 0.98, "BroadBeans": 1.00,
        "Carrot": 1.00, "GreenChilli": 1.00, "Okra": 0.99,
    },
    "Chittoor": {
        "Tomato": 0.82, "Onion": 0.90, "Potato": 0.87, "Paddy": 0.97,
        "BitterGourd": 0.92, "Brinjal": 0.90, "BroadBeans": 0.94,
        "Carrot": 0.91, "GreenChilli": 0.95, "Okra": 0.93,  # Major tomato belt
    },
    "Kurnool": {
        "Tomato": 0.98, "Onion": 0.97, "Potato": 0.97, "Paddy": 0.98,
        "BitterGourd": 0.97, "Brinjal": 0.95, "BroadBeans": 0.97,
        "Carrot": 0.97, "GreenChilli": 0.98, "Okra": 0.97,
    },
}


def generate() -> pd.DataFrame:
    """Generate two years of synthetic daily mandi prices."""
    random.seed(42)
    np.random.seed(42)

    data = []
    print(f"Generating data from {START_DATE.date()} → {END_DATE.date()} …")

    for region in REGIONS:
        for crop in CROPS:
            base_price, volatility, _, harvest_months = CROP_STATS[crop]
            region_factor  = REGIONAL_FACTORS[region][crop]
            adjusted_base  = base_price * region_factor
            price          = adjusted_base

            for i in range(DAYS):
                date  = START_DATE + timedelta(days=i)
                month = date.month

                # 1. Seasonal effect  —  harvest = supply glut → lower prices
                if month in harvest_months:
                    seasonal_factor = np.random.uniform(0.75, 0.88)
                elif month in [(harvest_months[-1] % 12) + 1,
                               (harvest_months[-1] % 12) + 2]:
                    seasonal_factor = np.random.uniform(0.90, 1.00)   # recovering
                else:
                    seasonal_factor = np.random.uniform(1.05, 1.20)   # lean season

                # 2. Post-harvest 1-week recovery ramp
                weekday_effect = -0.5 if date.weekday() == 0 else 0.0

                # 3. Mean reversion toward seasonal target
                target       = adjusted_base * seasonal_factor
                mean_rev     = 0.15 * (target - price)

                # 4. Random daily noise
                noise = np.random.normal(0, volatility * 0.4)

                # 5. Occasional supply shock (~2% chance)
                shock = 0.0
                if np.random.random() > 0.98:
                    shock = (np.random.uniform(volatility, volatility * 2.5)
                             * np.random.choice([-1, 1]))

                price = price + mean_rev + noise + weekday_effect + shock
                # Hard clamp: 40%–250% of adjusted base
                price = float(np.clip(price, adjusted_base * 0.40, adjusted_base * 2.50))

                data.append({
                    "date":   date.strftime("%Y-%m-%d"),
                    "crop":   crop,
                    "region": region,
                    "price":  round(price, 2),
                })

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["date", "crop", "region"])
    df = df.sort_values(["crop", "region", "date"]).reset_index(drop=True)
    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ {len(df):,} rows → {OUTPUT_CSV}")
    print(f"   Date range : {df['date'].min()} → {df['date'].max()}")
    print(f"   Regions    : {sorted(df['region'].unique())}")
    print(f"   Crops      : {sorted(df['crop'].unique())}")


if __name__ == "__main__":
    main()
