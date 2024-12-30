import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to generate the dataset (this will be used if you don't have an existing CSV file)
def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        # Age categorization
        age = random.randint(18, 100)
        if age <= 30:
            age_category = "Low risk"
        elif 30 < age <= 75:
            age_category = "Normal risk"
        else:
            age_category = "High risk"

        # Sex
        sex = random.choice([0, 1])

        # Chest Pain Type (CP)
        cp = random.choice([0, 1, 2, 3])

        # Resting Blood Pressure (Trestbps)
        trestbps = random.randint(90, 200)
        if trestbps < 100:
            trestbps_category = "Low"
        elif 100 <= trestbps <= 130:
            trestbps_category = "Normal"
        elif 130 < trestbps <= 140:
            trestbps_category = "High normal"
        else:
            trestbps_category = "High"

        # Serum Cholesterol (Chol)
        chol = random.randint(100, 400)
        if chol < 200:
            chol_category = "Optimal"
        elif 200 <= chol <= 240:
            chol_category = "Borderline high"
        else:
            chol_category = "High"

        # Fasting Blood Sugar (FBS)
        fbs = random.choice([0, 1])

        # Resting ECG Results (Restecg)
        restecg = random.choice([0, 1, 2])

        # Maximum Heart Rate (Thalach)
        thalach = random.randint(60, 220)
        if thalach < 100:
            thalach_category = "Low"
        elif 100 <= thalach <= 170:
            thalach_category = "Normal"
        else:
            thalach_category = "High"

        # Exercise Induced Angina (Exang)
        exang = random.choice([0, 1])

        # ST Depression (Oldpeak)
        oldpeak = round(random.uniform(0, 6.2), 1)
        if oldpeak <= 1.5:
            oldpeak_category = "Normal"
        elif 1.5 < oldpeak <= 2.5:
            oldpeak_category = "Borderline"
        else:
            oldpeak_category = "Abnormal"

        # ST Segment Slope (Slope)
        slope = random.choice([0, 1, 2])

        # Number of Major Vessels (CA)
        ca = random.choice([0, 1, 2, 3, 4])

        # Thalassemia (Thal)
        thal = random.choice([0, 1, 2, 3])

        # Define label based on a combination of features (you can adjust the conditions)
        # Heart disease risk label (1 for risk, 0 for no risk)
        label = 1 if chol > 240 or thalach < 100 or oldpeak > 2.5 else 0

        # Add the data row
        data.append([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, label])

    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label']
    df = pd.DataFrame(data, columns=columns)
    return df

# Load the dataset from a CSV file (use this if you already have an existing dataset)
input_file = "heart_disease_input.csv"
dataset = pd.read_csv(input_file)

# Check if the 'label' column is present
if 'label' not in dataset.columns:
    print("Label column not found in the input file. Generating labels based on the data...")
    # If the label is missing, generate labels based on conditions (cholesterol, heart rate, etc.)
    dataset['label'] = dataset.apply(lambda row: 1 if row['chol'] > 240 or row['thalach'] < 100 or row['oldpeak'] > 2.5 else 0, axis=1)

# Split the data into features (X) and label (y)
X = dataset.drop('label', axis=1)  # Features
y = dataset['label']  # Target (heart disease label)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler using joblib
joblib.dump(knn, 'heart_disease_knn_model.joblib')  # Save the trained KNN model
joblib.dump(scaler, 'scaler_knn.joblib')  # Save the scaler

print("Model and scaler saved.")

# Example: Predict the heart disease risk for a new data point
def predict_risk(data):
    # Scale the input data
    scaled_data = scaler.transform([data])
    # Predict the probability of heart disease
    prob = knn.predict_proba(scaled_data)[0][1]  # Get the probability of class 1 (heart disease)
    # Output the result as a percentage
    return f"Risk of heart disease: {prob * 100:.2f}%"

# Example of usage (passing a sample data point)
sample_data = [55, 1, 2, 120, 220, 1, 1, 150, 1, 2.5, 1, 2, 1]  # Sample data
print(predict_risk(sample_data))
