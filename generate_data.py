import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CROPS = [
    "Tomato", "Onion", "Potato", "Paddy",
    "BitterGourd", "Brinjal", "BroadBeans", "Carrot", "GreenChilli", "Okra"
]

REGIONS = [
    "Krishna", "Guntur", "Visakhapatnam",
    "EastWestGodavari", "Chittoor", "Kurnool"
]

DAYS = 730   # 2 years of history for better model training
END_DATE = datetime(2026, 2, 24)
START_DATE = END_DATE - timedelta(days=DAYS)

# ─────────────────────────────────────────────
# REALISTIC CROP BASE PRICES (₹/kg)
# Based on actual AP mandi rates
# (mean_price, std_dev, peak_month, harvest_months)
# peak_month: month when price is LOWEST (post-harvest)
# ─────────────────────────────────────────────
CROP_STATS = {
    # crop: (base_price, volatility, harvest_dip_month, harvest_months_range)
    "Tomato":      (20.0, 5.0,  11, [10, 11, 12, 1]),    # Nov harvest → lowest price
    "Onion":       (24.0, 4.5,  1,  [12, 1, 2]),          # Jan harvest
    "Potato":      (22.0, 4.0,  2,  [1, 2, 3]),           # Feb-Mar harvest
    "Paddy":       (23.5, 1.5,  11, [10, 11]),             # Kharif season
    "BitterGourd": (30.0, 6.0,  7,  [6, 7, 8, 9]),        # Summer crop
    "Brinjal":     (15.0, 4.0,  10, [9, 10, 11]),         # Oct-Nov
    "BroadBeans":  (38.0, 7.0,  12, [11, 12, 1]),         # Winter crop
    "Carrot":      (25.0, 5.0,  12, [11, 12, 1, 2]),      # Winter harvest
    "GreenChilli": (40.0, 12.0, 2,  [1, 2, 3]),           # Guntur season - high volatility
    "Okra":        (20.0, 5.0,  6,  [5, 6, 7, 8])         # Summer crop
}

# Regional price multipliers (relative to base)
# Chittoor tomatoes are cheapest (Madanapalle belt = major producer)
# Guntur chillies are lowest (local production)
# Visakhapatnam has higher prices (demand center)
REGIONAL_FACTORS = {
    "Krishna":          {"Tomato": 1.02, "Onion": 1.01, "Potato": 1.01, "Paddy": 1.00,
                         "BitterGourd": 1.05, "Brinjal": 1.02, "BroadBeans": 1.03,
                         "Carrot": 1.04, "GreenChilli": 1.10, "Okra": 1.03},
    "Guntur":           {"Tomato": 0.97, "Onion": 0.98, "Potato": 0.96, "Paddy": 0.99,
                         "BitterGourd": 0.96, "Brinjal": 0.95, "BroadBeans": 0.98,
                         "Carrot": 0.97, "GreenChilli": 0.85, "Okra": 0.96},  # Chilli discount
    "Visakhapatnam":    {"Tomato": 1.10, "Onion": 1.08, "Potato": 1.09, "Paddy": 1.02,
                         "BitterGourd": 1.12, "Brinjal": 1.08, "BroadBeans": 1.10,
                         "Carrot": 1.10, "GreenChilli": 1.05, "Okra": 1.10},
    "EastWestGodavari": {"Tomato": 1.00, "Onion": 0.99, "Potato": 0.99, "Paddy": 1.03,
                         "BitterGourd": 1.00, "Brinjal": 0.98, "BroadBeans": 1.00,
                         "Carrot": 1.00, "GreenChilli": 1.00, "Okra": 0.99},
    "Chittoor":         {"Tomato": 0.82, "Onion": 0.90, "Potato": 0.87, "Paddy": 0.97,
                         "BitterGourd": 0.92, "Brinjal": 0.90, "BroadBeans": 0.94,
                         "Carrot": 0.91, "GreenChilli": 0.95, "Okra": 0.93},  # Major tomato belt
    "Kurnool":          {"Tomato": 0.98, "Onion": 0.97, "Potato": 0.97, "Paddy": 0.98,
                         "BitterGourd": 0.97, "Brinjal": 0.95, "BroadBeans": 0.97,
                         "Carrot": 0.97, "GreenChilli": 0.98, "Okra": 0.97}
}

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

data = []
print(f"Generating realistic data from {START_DATE.date()} to {END_DATE.date()}...")

for region in REGIONS:
    for crop in CROPS:
        base_price, volatility, harvest_month, harvest_months = CROP_STATS[crop]
        region_factor = REGIONAL_FACTORS[region][crop]
        adjusted_base = base_price * region_factor

        # Price is a mean-reverting series with seasonal component
        price = adjusted_base  # Start at base

        for i in range(DAYS):
            date = START_DATE + timedelta(days=i)
            month = date.month
            day_of_year = date.timetuple().tm_yday

            # 1. Seasonal effect: lower price during harvest, higher during lean season
            if month in harvest_months:
                # Harvest time: price dips 15-25%
                seasonal_factor = np.random.uniform(0.75, 0.88)
            elif month in [(m % 12) + 1 for m in [harvest_months[-1], harvest_months[-1] + 1]]:
                # Just after harvest: recovering
                seasonal_factor = np.random.uniform(0.90, 1.0)
            else:
                # Lean season: price rises 10-20%
                seasonal_factor = np.random.uniform(1.05, 1.20)

            # 2. Weekly cycle: Mondays have lower prices (market reset)
            weekday_effect = -0.5 if date.weekday() == 0 else 0.0

            # 3. Mean reversion: nudge back toward seasonal target
            target = adjusted_base * seasonal_factor
            mean_reversion = 0.15 * (target - price)

            # 4. Random daily noise
            noise = np.random.normal(0, volatility * 0.4)

            # 5. Occasional supply shock (2% chance)
            shock = 0
            if np.random.random() > 0.98:
                shock = np.random.uniform(volatility, volatility * 2.5) * np.random.choice([-1, 1])

            # Update price with all effects
            price = price + mean_reversion + noise + weekday_effect + shock

            # Hard clamp: don't go below 40% of base or above 250% of base
            price = float(np.clip(price, adjusted_base * 0.40, adjusted_base * 2.50))

            data.append({
                "date":   date.strftime("%Y-%m-%d"),
                "crop":   crop,
                "region": region,
                "price":  round(price, 2)
            })

df = pd.DataFrame(data)
df = df.drop_duplicates(subset=["date", "crop", "region"])
df = df.sort_values(["crop", "region", "date"]).reset_index(drop=True)
df.to_csv("data/crops.csv", index=False)

print(f"✅ Generated {len(df)} rows of realistic data in data/crops.csv")
print(f"   Regions: {REGIONS}")
print(f"   Crops:   {CROPS}")
