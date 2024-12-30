import pandas as pd
import joblib

# Load the saved model
def load_model(filename):
    model = joblib.load(filename)
    return model

# Load heart rate data from CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare data for prediction
def prepare_data(df):
    X = df[['heart_rate']]  # Features
    return X

# Make predictions
def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

# Main function for prediction
def main_predict():
    model_filename = 'heart_rate_model.pkl'  # Path to your saved model
    file_path = 'rate_labeled_short.csv'  # Path to your CSV file
    heart_rate_data = load_data(file_path)
    X = prepare_data(heart_rate_data)  # Prepare features
    model = load_model(model_filename)
    predictions = make_predictions(model, X)
    
    # Add predictions to the DataFrame
    heart_rate_data['predictions'] = predictions
    
    # Filter for high and low heart rates
    alert_data = heart_rate_data[(heart_rate_data['predictions'] == 'high') | (heart_rate_data['predictions'] == 'low')]
    
    if not alert_data.empty:
        print("Alert! The following heart rates are outside the normal range:")
        for index, row in alert_data.iterrows():
            print(f"Time: {row['timestamp']}, Heart Rate: {row['heart_rate']}, Status: {row['predictions']}")
            if row['predictions'] == 'high':
                print("High heart rate detected! Immediate attention is required.\n")
            elif row['predictions'] == 'low':
                print("Low heart rate detected! Immediate attention is required.\n")
    else:
        print("All heart rates are normal.")

# Start the prediction process
if __name__ == "__main__":
    main_predict()