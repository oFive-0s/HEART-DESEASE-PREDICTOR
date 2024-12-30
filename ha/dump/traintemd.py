import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load heart rate data from CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prepare data
def prepare_data(df):
    # Define categories based on heart rate
    conditions = [
        (df['heart_rate'] < 60),
        (df['heart_rate'] >= 60) & (df['heart_rate'] <= 100),
        (df['heart_rate'] > 100)
    ]
    categories = ['low', 'normal', 'high']
    df['label'] = pd.cut(df['heart_rate'], bins=[0, 60, 100, 200], labels=categories, right=False)

    X = df[['heart_rate']]  # Features
    y = df['label']  # Target variable (e.g., low, normal, high)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Save the model
def save_model(model, filename):
    joblib.dump(model, filename)

# Main function
def main():
    file_path = 'rate.csv'  # Path to your CSV file
    heart_rate_data = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(heart_rate_data)
    model = train_model(X_train, y_train)
    save_model(model, 'heart_rate_model.pkl')

# Start the process
if __name__ == "__main__":
    main()