import random

# Base market prices (Low, High) per kg - Realistic 2026 AP Market Rates
# Prices based on actual mandi rates for Andhra Pradesh districts
BASE_DATA = {
    "Krishna": {
        "Tomato":      (18, 25),
        "Onion":       (22, 30),
        "Potato":      (20, 28),
        "Paddy":       (22, 26),   # per kg (MSP ~₹2300/quintal = ₹23/kg)
        "BitterGourd": (25, 38),
        "Brinjal":     (12, 20),
        "BroadBeans":  (30, 45),
        "Carrot":      (22, 35),
        "GreenChilli": (30, 60),
        "Okra":        (18, 28)
    },
    "Guntur": {
        "Tomato":      (16, 22),
        "Onion":       (20, 28),
        "Potato":      (19, 26),
        "Paddy":       (22, 26),
        "BitterGourd": (22, 35),
        "Brinjal":     (10, 18),
        "BroadBeans":  (28, 42),
        "Carrot":      (20, 32),
        "GreenChilli": (25, 55),  # Guntur is famous for chillies - wider range
        "Okra":        (16, 26)
    },
    "Visakhapatnam": {
        "Tomato":      (20, 30),
        "Onion":       (25, 35),
        "Potato":      (22, 32),
        "Paddy":       (23, 27),
        "BitterGourd": (28, 42),
        "Brinjal":     (14, 22),
        "BroadBeans":  (32, 48),
        "Carrot":      (25, 38),
        "GreenChilli": (28, 52),
        "Okra":        (20, 32)
    },
    "EastWestGodavari": {
        "Tomato":      (17, 24),
        "Onion":       (21, 29),
        "Potato":      (20, 27),
        "Paddy":       (23, 27),  # Godavari delta - major paddy belt
        "BitterGourd": (23, 36),
        "Brinjal":     (11, 19),
        "BroadBeans":  (29, 44),
        "Carrot":      (21, 33),
        "GreenChilli": (26, 50),
        "Okra":        (17, 27)
    },
    "Chittoor": {
        "Tomato":      (14, 20),  # Madanapalle is AP's tomato capital
        "Onion":       (18, 26),
        "Potato":      (17, 24),  # Also a major potato belt
        "Paddy":       (22, 25),
        "BitterGourd": (20, 32),
        "Brinjal":     (10, 17),
        "BroadBeans":  (26, 40),
        "Carrot":      (18, 30),
        "GreenChilli": (22, 45),
        "Okra":        (15, 24)
    },
    "Kurnool": {
        "Tomato":      (17, 23),
        "Onion":       (21, 29),
        "Potato":      (19, 27),
        "Paddy":       (22, 26),
        "BitterGourd": (22, 34),
        "Brinjal":     (11, 18),
        "BroadBeans":  (27, 42),
        "Carrot":      (20, 31),
        "GreenChilli": (24, 48),
        "Okra":        (16, 25)
    }
}

def simulate_live_variation(low, high):
    """
    Simulate real-time market fluctuation with small jitter.
    """
    if low is None or high is None:
        return None, None

    spread = max(high - low, low * 0.05)
    drift = random.uniform(-spread * 0.15, spread * 0.15)

    new_low = max(1.0, round(low + drift, 2))
    new_high = max(new_low + 1.0, round(high + drift, 2))

    return new_low, new_high

def get_market_price(region, crop):
    """
    Fetch the 'Live' market price for a given region and crop.
    """
    region_data = BASE_DATA.get(region, {})
    price_range = region_data.get(crop)

    if not price_range:
        return None, None

    return simulate_live_variation(*price_range)
