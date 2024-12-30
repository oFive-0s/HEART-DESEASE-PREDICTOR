import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Function to generate the dataset (this will be used if you don't have an existing CSV file)
def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        # Create the combined population for age
        age_population = list(range(18, 31)) + list(range(30, 76)) + list(range(76, 101))
        age_weights = [1] * len(range(18, 31)) + [4] * len(range(30, 76)) + [2] * len(range(76, 101))
        age = random.choices(age_population, weights=age_weights, k=1)[0]

        sex = random.choice([0, 1])
        cp = random.choice([0, 1, 2, 3])
        trestbps_population = list(range(90, 101)) + list(range(100, 131)) + list(range(131, 141)) + list(range(141, 201))
        trestbps_weights = [1] * len(range(90, 101)) + [3] * len(range(100, 131)) + [2] * len(range(131, 141)) + [2] * len(range(141, 201))
        trestbps = random.choices(trestbps_population, weights=trestbps_weights, k=1)[0]

        chol_population = list(range(100, 201)) + list(range(201, 241)) + list(range(241, 401))
        chol_weights = [3] * len(range(100, 201)) + [2] * len(range(201, 241)) + [1] * len(range(241, 401))
        chol = random.choices(chol_population, weights=chol_weights, k=1)[0]

        fbs = random.choice([0, 1])
        restecg = random.choice([0, 1, 2])
        thalach_population = list(range(60, 101)) + list(range(100, 171)) + list(range(171, 221))
        thalach_weights = [2] * len(range(60, 101)) + [3] * len(range(100, 171)) + [1] * len(range(171, 221))
        thalach = random.choices(thalach_population, weights=thalach_weights, k=1)[0]

        exang = random.choice([0, 1])
        oldpeak = round(random.uniform(0, 6.2), 1)
        slope = random.choice([0, 1, 2])
        ca = random.choice([0, 1, 2, 3, 4])
        thal = random.choice([0, 1, 2, 3])

        label = 1 if chol > 240 or thalach < 100 else 0

        data.append([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, label])

    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    df = pd.DataFrame(data, columns=columns)
    return df

# Load the dataset from a CSV file
# If you have your own dataset, replace 'heart_disease_input.csv' with the correct file path
input_file = "heart_disease_input.csv"
dataset = pd.read_csv(input_file)

# Check if the 'label' column is present
if 'label' not in dataset.columns:
    print("Label column not found in the input file. Generating labels based on the data...")
    # In case the label is not present, generate labels from the data
    dataset['label'] = dataset.apply(lambda row: 1 if row['chol'] > 240 or row['thalach'] < 100 else 0, axis=1)

# Split the data into features (X) and label (y)
X = dataset.drop('label', axis=1)  # Features
y = dataset['label']  # Target (heart disease label)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler using joblib
joblib.dump(clf, 'heart_disease_model_forest.joblib')  # Save the trained model
joblib.dump(scaler, 'scaler_forest.joblib')  # Save the scaler

print("Model and scaler saved.")

# Example: Predict the heart disease risk for a new data point
def predict_risk(data):
    # Scale the input data
    scaled_data = scaler.transform([data])
    # Predict the probability of heart disease
    prob = clf.predict_proba(scaled_data)[0][1]  # Get the probability of class 1 (heart disease)
    # Output the result as a percentage
    return f"Risk of heart disease: {prob * 100:.2f}%"

# Example of usage (passing a sample data point)
sample_data = [55, 1, 2, 120, 220, 1, 1, 150, 1, 2.5, 1, 2, 1]  # Sample data
print(predict_risk(sample_data))
