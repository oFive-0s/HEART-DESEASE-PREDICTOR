import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters for dataset generation
start_time = datetime(2024, 11, 18, 0, 0)
rows = 10000  # Total number of rows
data_labeled = []

# Generate dataset
for i in range(rows):
    timestamp = start_time + timedelta(minutes=i)
    heart_rate = random.randint(45, 125)  # Random heart rate in the specified range
    label = 1 if heart_rate <= 60 or heart_rate >= 100 else 0  # 1 for danger, 0 for safe
    data_labeled.append([timestamp.strftime("%Y-%m-%d %H:%M"), heart_rate, label])

# Create DataFrame with labels
df_labeled = pd.DataFrame(data_labeled, columns=["timestamp", "heart_rate", "label"])

# Save to CSV
file_path = "rate_labeled.csv"
df_labeled.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")