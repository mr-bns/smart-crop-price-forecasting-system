import pandas as pd
from datetime import datetime

# Load data
df = pd.read_csv("data/crops.csv")
df["date"] = pd.to_datetime(df["date"])

# Identify shift needed
max_date_in_data = df["date"].max()
current_date = datetime.now()

years_diff = current_date.year - max_date_in_data.year

if years_diff > 0:
    print(f"🔄 Data is from {max_date_in_data.year}. Shifting by {years_diff} years to {current_date.year}...")
    
    # Shift dates
    df["date"] = df["date"] + pd.DateOffset(years=years_diff)
    
    # Save back
    df.to_csv("data/crops.csv", index=False)
    print("✅ Dates updated successfully!")
else:
    print("✅ Data is already recent.")
