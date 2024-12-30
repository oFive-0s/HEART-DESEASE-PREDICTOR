import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

def load_test_data():
    """
    Load the original dataset to calculate model accuracies
    """
    try:
        df = pd.read_csv(r"C:\poorna\New folder\ha\heart.csv")
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None, None

def calculate_model_accuracy(model, scaler, X, y):
    """
    Calculate accuracy for a given model
    """
    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        return accuracy_score(y, y_pred)
    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
        return 0

def load_models():
    """
    Load both Random Forest and KNN models with their respective scalers
    """
    try:
        # Load Random Forest model and scaler
        rf_model = joblib.load(r"C:\\poorna\\New folder\\ha\\heart_disease_model.joblib")
        rf_scaler = joblib.load(r"C:\\poorna\\New folder\\ha\\scaler.joblib")
        
        # Load KNN model and scaler
        knn_model = joblib.load(r"C:\\poorna\\New folder\\ha\\heart_disease_knn_model.joblib")
        knn_scaler = joblib.load(r"C:\\poorna\\New folder\\ha\\knn_scaler.joblib")
        
        return rf_model, rf_scaler, knn_model, knn_scaler
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None

def predict_heart_disease(patient_data):
    """
    Make predictions using both models and return results from the more accurate model
    """
    try:
        # Load models and scalers
        rf_model, rf_scaler, knn_model, knn_scaler = load_models()
        if None in (rf_model, rf_scaler, knn_model, knn_scaler):
            raise Exception("Failed to load models")
        
        # Load test data to calculate accuracies
        X_test, y_test = load_test_data()
        if X_test is None:
            raise Exception("Failed to load test data")
        
        # Calculate accuracies
        rf_accuracy = calculate_model_accuracy(rf_model, rf_scaler, X_test, y_test)
        knn_accuracy = calculate_model_accuracy(knn_model, knn_scaler, X_test, y_test)
        
        print(f"\nModel Accuracies:")
        print(f"Random Forest: {rf_accuracy:.3f}")
        print(f"KNN: {knn_accuracy:.3f}")
        
        # Use the model with higher accuracy
        if rf_accuracy >= knn_accuracy:
            print("Using Random Forest model (higher accuracy)")
            scaled_data = rf_scaler.transform(patient_data)
            prediction = rf_model.predict(scaled_data)
            probability = rf_model.predict_proba(scaled_data)
        else:
            print("Using KNN model (higher accuracy)")
            scaled_data = knn_scaler.transform(patient_data)
            prediction = knn_model.predict(scaled_data)
            probability = knn_model.predict_proba(scaled_data)
        
        return prediction, probability
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def validate_input(values):
    """
    Validate the input values
    """
    if len(values) != 13:
        raise ValueError("Expected 13 values for prediction")
    
    # Add any additional validation rules here
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError("All values must be numbers")

if __name__ == "__main__":
    # Example usage
    while True:
        try:
            print("\nEnter patient data (or 'quit' to exit):")
            print("Format: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal")
            print("Example: 52,1,0,125,212,0,1,168,0,1,2,2,3")
            
            user_input = input()
            if user_input.lower() == 'quit':
                break
                
            # Parse and validate the input
            values = [float(x) for x in user_input.split(',')]
            validate_input(values)
            patient_data = np.array([values])
            
            # Get prediction
            prediction, probability = predict_heart_disease(patient_data)
            
            if prediction is not None:
                print(f"\nPrediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
                print(f"Probability of heart disease: {probability[0][1]:.3f}")
            
        except ValueError as ve:
            print(f"Input error: {str(ve)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

            75,1,3,160,330,1,2,105,1,4,3,3,3
