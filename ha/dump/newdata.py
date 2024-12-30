import pandas as pd
import random

# Define ranges and categories for the attributes
def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        # Create the combined population for age
        age_population = list(range(18, 31)) + list(range(30, 76)) + list(range(76, 101))
        
        # Define the weights for the age ranges (ensuring that the number of weights matches the population)
        age_weights = [1] * len(range(18, 31)) + [4] * len(range(30, 76)) + [2] * len(range(76, 101))
        
        # Age: Good range and high-risk range
        age = random.choices(age_population, weights=age_weights, k=1)[0]

        # Sex: 0 = Female, 1 = Male
        sex = random.choice([0, 1])

        # Chest Pain Type (CP): 0-3
        cp = random.choice([0, 1, 2, 3])

        # Resting Blood Pressure (Trestbps): 90-200
        trestbps_population = list(range(90, 101)) + list(range(100, 131)) + list(range(131, 141)) + list(range(141, 201))
        trestbps_weights = [1] * len(range(90, 101)) + [3] * len(range(100, 131)) + [2] * len(range(131, 141)) + [2] * len(range(141, 201))
        trestbps = random.choices(trestbps_population, weights=trestbps_weights, k=1)[0]

        # Serum Cholesterol (Chol): 100-400
        chol_population = list(range(100, 201)) + list(range(201, 241)) + list(range(241, 401))
        chol_weights = [3] * len(range(100, 201)) + [2] * len(range(201, 241)) + [1] * len(range(241, 401))
        chol = random.choices(chol_population, weights=chol_weights, k=1)[0]

        # Fasting Blood Sugar (FBS): 0 or 1
        fbs = random.choice([0, 1])

        # Resting ECG Results (Restecg): 0-2
        restecg = random.choice([0, 1, 2])

        # Maximum Heart Rate (Thalach): 60-220
        thalach_population = list(range(60, 101)) + list(range(100, 171)) + list(range(171, 221))
        thalach_weights = [2] * len(range(60, 101)) + [3] * len(range(100, 171)) + [1] * len(range(171, 221))
        thalach = random.choices(thalach_population, weights=thalach_weights, k=1)[0]

        # Exercise Induced Angina (Exang): 0 or 1
        exang = random.choice([0, 1])

        # ST Depression (Oldpeak): 0-6.2
        oldpeak = round(random.uniform(0, 6.2), 1)

        # ST Segment Slope (Slope): 0-2
        slope = random.choice([0, 1, 2])

        # Number of Major Vessels (CA): 0-4
        ca = random.choice([0, 1, 2, 3, 4])

        # Thalassemia (Thal): 0-3
        thal = random.choice([0, 1, 2, 3])

        # Label (Normal or Danger)
        # Let's assume a "danger" label if cholesterol > 240 or heart rate < 100
        label = 1 if chol > 240 or thalach < 100 else 0

        # Append to the data list
        data.append([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, label])

    # Create a DataFrame
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate a dataset
dataset = generate_data(num_samples=1000)

# Save to a CSV file
output_file = "heart_disease_dataset_with_labels.csv"
dataset.to_csv(output_file, index=False)

print(f"Dataset generated and saved to {output_file}.")


