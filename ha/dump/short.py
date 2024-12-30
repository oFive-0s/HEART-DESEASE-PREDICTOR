import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters for dataset generation
start_time = datetime(2024, 11, 18, 0, 0)
total_rows = 100  # Total number of rows
abnormal_count = 5  # Number of abnormal heart rates

# Initialize list for dataset
data_labeled = []

# Generate normal heart rates
for i in range(total_rows - abnormal_count):
    timestamp = start_time + timedelta(minutes=i)
    heart_rate = random.randint(61, 99)  # Normal heart rates
    label = 0  # 0 for safe (normal)
    data_labeled.append([timestamp.strftime("%Y-%m-%d %H:%M"), heart_rate, label])

# Generate abnormal heart rates (5 instances)
for _ in range(abnormal_count):
    timestamp = start_time + timedelta(minutes=random.randint(0, total_rows - 1))
    heart_rate = random.choice([random.randint(45, 59), random.randint(100, 125)])  # Low or high heart rate
    label = 1  # 1 for danger (abnormal)
    data_labeled.append([timestamp.strftime("%Y-%m-%d %H:%M"), heart_rate, label])

# Shuffle the dataset to mix normal and abnormal entries
random.shuffle(data_labeled)

# Create DataFrame with labels
df_labeled = pd.DataFrame(data_labeled, columns=["timestamp", "heart_rate", "label"])

# Save to CSV
file_path = "rate_labeled_short.csv"
df_labeled.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")