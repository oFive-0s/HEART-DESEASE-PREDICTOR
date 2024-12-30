import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import winsound

# Load the trained models and scalers
knn_model = joblib.load('heart_disease_knn_model.joblib')
forest_model = joblib.load('heart_disease_model_forest.joblib')
scaler_knn = joblib.load('scaler_knn.joblib')
scaler_forest = joblib.load('scaler_forest.joblib')

# Function to predict risk using both models
def predict_risk(data, selected_model):
    # Scale the input data for both models
    scaled_knn = scaler_knn.transform([data])
    scaled_forest = scaler_forest.transform([data])
    
    # Get the probabilities for both models
    knn_prob = knn_model.predict_proba(scaled_knn)[0][1]  # Probability of heart disease (class 1)
    forest_prob = forest_model.predict_proba(scaled_forest)[0][1]  # Probability of heart disease (class 1)
    
    # If the selected model is KNN
    if selected_model == 'KNN':
        return knn_prob * 100
    else:
        return forest_prob * 100

# Function to evaluate both models and print accuracy
def evaluate_models(X_test_knn, y_test_knn, X_test_forest, y_test_forest):
    # Predictions for KNN
    y_pred_knn = knn_model.predict(X_test_knn)
    accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
    
    # Predictions for Random Forest
    y_pred_forest = forest_model.predict(X_test_forest)
    accuracy_forest = accuracy_score(y_test_forest, y_pred_forest)
    
    # Print the accuracies of both models
    print(f"KNN Model Accuracy: {accuracy_knn * 100:.2f}%")
    print(f"Random Forest Model Accuracy: {accuracy_forest * 100:.2f}%")
    
    # Return the best model based on accuracy
    if accuracy_knn > accuracy_forest:
        print("KNN model is selected based on higher accuracy.\n")
        return 'KNN'
    else:
        print("Random Forest model is selected based on higher accuracy.\n")
        return 'Random Forest'

# Function to get user input for the prediction
def get_user_input():
    # Ask the user for the necessary parameters
    age = int(input("Enter age (18-100 years): "))
    sex = int(input("Enter sex (0 for Female, 1 for Male): "))
    cp = int(input("Enter chest pain type (0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic): "))
    trestbps = int(input("Enter resting blood pressure (90-200 mm Hg): "))
    chol = int(input("Enter serum cholesterol (100-400 mg/dl): "))
    fbs = int(input("Enter fasting blood sugar (0=Normal, 1=High): "))
    restecg = int(input("Enter resting ECG results (0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy): "))
    thalach = int(input("Enter maximum heart rate (60-220 bpm): "))
    exang = int(input("Enter exercise induced angina (0=No, 1=Yes): "))
    oldpeak = float(input("Enter ST depression (0-6.2 mm): "))
    slope = int(input("Enter ST segment slope (0=Upsloping, 1=Flat, 2=Downsloping): "))
    ca = int(input("Enter number of major vessels (0-4): "))
    thal = int(input("Enter thalassemia (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Not normal): "))
    
    # Return the data as a list
    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Function to check risk level and provide a response with a beep for high risk
def check_risk_and_notify(risk_percentage):
    print(f"Risk of heart disease: {risk_percentage:.2f}%\n")
    if risk_percentage >= 60.00:
        print("You are at high risk of heart disease. Please consult a doctor immediately.")
        # Beep sound (frequency: 1000 Hz, duration: 500 ms)
        winsound.Beep(1000, 500)
    else:
        print("Take rest and if feeling any serious symptoms, better call a doctor.")

# Example of usage for prediction:
# Get user input
user_data = get_user_input()

# For accuracy evaluation, load test data (replace with actual dataset for testing)
data = pd.read_csv('heart_disease_input.csv')

# Separate features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split data for test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data for both models
X_test_knn = scaler_knn.transform(X_test)
X_test_forest = scaler_forest.transform(X_test)

# Evaluate both models
best_model = evaluate_models(X_test_knn, y_test, X_test_forest, y_test)

# Display prediction using both models (based on selected model)
risk_percentage = predict_risk(user_data, best_model)

# Check the risk level and provide appropriate response
check_risk_and_notify(risk_percentage)
