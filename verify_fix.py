from predictor import run_prediction
from datetime import datetime, timedelta

try:
    # Test 1: Future prediction
    target = datetime.now() + timedelta(days=5)
    price, confidence = run_prediction("Tomato", "Guntur", target)
    print(f"✅ Prediction Success: Tomato/Guntur on {target.date()} -> ₹{price} (Conf: {confidence}%)")

    # Test 2: Another crop
    target = datetime.now() + timedelta(days=10)
    price, confidence = run_prediction("Onion", "Hyderabad", target)
    print(f"✅ Prediction Success: Onion/Hyderabad on {target.date()} -> ₹{price} (Conf: {confidence}%)")

except Exception as e:
    print(f"❌ Verification Failed: {e}")
