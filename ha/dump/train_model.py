import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Define the file path
# Method 1: Using raw string
file_path = r"C:\\poorna\\New folder\\ha\\heart.csv"

# Method 2: Using os.path.join (more portable)
# file_path = os.path.join("C:", "poorna", "New folder", "ha", "heart.csv")

# Rest of your code remains the same
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the heart disease dataset
    """
    try:
        # Read the CSV file
        print(f"Attempting to read file from: {file_path}")
        df = pd.read_csv(file_path)
        print("File successfully loaded!")
        
        # Remove duplicates
        df_unique = df.drop_duplicates()
        print(f"Removed {len(df) - len(df_unique)} duplicate rows")
        
        # Handle missing values
        df_unique = df_unique.replace('?', np.nan)
        df_unique = df_unique.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with missing values
        df_clean = df_unique.dropna()
        print(f"Removed {len(df_unique) - len(df_clean)} rows with missing values")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        print("Please check if the file path is correct and the file exists.")
        raise
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def prepare_data(df):
    """
    Prepare data for training by splitting features and target
    """
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Train the Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train the final model
    rf_model.fit(X_train, y_train)
    
    return rf_model

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model and print results
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()

def main():
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data(file_path)
        
        # Prepare data for training
        print("\nPreparing data for training...")
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Train model
        print("\nTraining Random Forest model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, X_test, y_test, columns[:-1])
        
        # Save the model and scaler
        print("\nSaving model and scaler...")
        model_save_path = os.path.join(os.path.dirname(file_path), 'heart_disease_model.joblib')
        scaler_save_path = os.path.join(os.path.dirname(file_path), 'scaler.joblib')
        
        joblib.dump(model, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        
        print(f"Model saved as: {model_save_path}")
        print(f"Scaler saved as: {scaler_save_path}")
        
        return model, scaler
        
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Train and evaluate the model
    model, scaler = main()
    
    if model is not None and scaler is not None:
        # Example of making a prediction
        new_patient = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3]])
        prediction = model.predict(new_patient)
        probability = model.predict_proba(new_patient)
        print(f"\nPrediction for new patient: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        print(f"Probability: {probability[0][1]:.3f}")