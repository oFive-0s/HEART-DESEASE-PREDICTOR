import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Define the file path
file_path = r"C:\\poorna\\New folder\\ha\\heart.csv"

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the heart disease dataset
    """
    try:
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
    
    # Scale the features - very important for KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def find_optimal_k(X_train, y_train):
    """
    Find the optimal number of neighbors using cross-validation
    """
    k_range = range(1, 31, 2)  # Test odd numbers from 1 to 30
    cv_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, cv_scores, 'bo-')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Finding Optimal K for KNN')
    plt.grid(True)
    plt.show()
    
    # Return the optimal k
    optimal_k = k_range[np.argmax(cv_scores)]
    print(f"\nOptimal number of neighbors: {optimal_k}")
    return optimal_k

def train_model(X_train, y_train):
    """
    Train the KNN model with optimized parameters
    """
    # Find optimal k
    optimal_k = find_optimal_k(X_train, y_train)
    
    # Create and train the model with optimal k
    knn_model = KNeighborsClassifier(
        n_neighbors=optimal_k,
        weights='uniform',
        algorithm='auto',
        metric='minkowski',
        p=2  # Euclidean distance
    )
    
    # Perform cross-validation with optimal k
    cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train the final model
    knn_model.fit(X_train, y_train)
    
    return knn_model

def evaluate_model(model, X_test, y_test):
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
    
    # Plot decision boundary (for the first two features)
    plot_decision_boundary(model, X_test, y_test)

def plot_decision_boundary(model, X_test, y_test, feature_indices=[0, 1]):
    """
    Plot the decision boundary for the first two features
    """
    plt.figure(figsize=(10, 8))
    
    # Only use the first two features for visualization
    X_subset = X_test[:, feature_indices]
    
    # Create a mesh grid
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Create features for prediction
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), 
                          np.zeros((xx.ravel().shape[0], X_test.shape[1]-2))])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_test, alpha=0.8)
    plt.xlabel(f'Feature {feature_indices[0]}')
    plt.ylabel(f'Feature {feature_indices[1]}')
    plt.title('KNN Decision Boundary (First Two Features)')
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
        print("\nTraining KNN model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, X_test, y_test)
        
        # Save the model and scaler
        print("\nSaving model and scaler...")
        model_save_path = os.path.join(os.path.dirname(file_path), 'heart_disease_knn_model.joblib')
        scaler_save_path = os.path.join(os.path.dirname(file_path), 'knn_scaler.joblib')
        
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
        scaled_patient = scaler.transform(new_patient)
        prediction = model.predict(scaled_patient)
        probability = model.predict_proba(scaled_patient)
        print(f"\nPrediction for new patient: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        print(f"Probability: {probability[0][1]:.3f}")