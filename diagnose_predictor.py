import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from predictor import run_prediction

def test_prediction():
    crop = "Tomato"
    region = "Krishna"
    target_date = datetime.today() + timedelta(days=1)
    
    print(f"Testing prediction for {crop} in {region} on {target_date.date()}...")
    
    try:
        price, conf = run_prediction(crop, region, target_date)
        print(f"✅ Success! Price: ₹{price}, Confidence: {conf}%")
    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
