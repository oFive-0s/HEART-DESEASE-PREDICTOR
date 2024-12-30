import joblib
import numpy as np

def predict_heart_disease(patient_data):
    try:
        # Load the saved model and scaler
        model = joblib.load(r"C:\poorna\New folder\ha\heart_disease_model.joblib")
        scaler = joblib.load(r"C:\poorna\New folder\ha\scaler.joblib")
        
        # Scale the input data
        scaled_data = scaler.transform(patient_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        return prediction, probability
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage
    while True:
        try:
            print("\nEnter patient data (or 'quit' to exit):")
            print("Format: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal")
            
            user_input = input()
            if user_input.lower() == 'quit':
                break
                
            # Parse the input
            values = [float(x) for x in user_input.split(',')]
            patient_data = np.array([values])
            
            # Get prediction
            prediction, probability = predict_heart_disease(patient_data)
            
            if prediction is not None:
                print(f"\nPrediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
                print(f"Probability of heart disease: {probability[0][1]:.3f}")
            
        except ValueError:
            print("Invalid input format. Please enter numbers separated by commas.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")