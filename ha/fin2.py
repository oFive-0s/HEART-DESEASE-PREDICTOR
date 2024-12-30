import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Optionally, trim whitespace from column names
df.columns = df.columns.str.strip()

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Rename columns according to your requirements
df.rename(columns={
    'cp': 'ChestPainType',
    'target': 'HeartAttackRate'  # Uncomment if 'target' exists in your dataset
}, inplace=True)

# Convert HeartAttackRate to binary (0 and 1) if necessary
if 'HeartAttackRate' in df.columns:
    df['HeartAttackRate'] = df['HeartAttackRate'].apply(lambda x: 1 if x > 0 else 0)

# Update the target variable name to 'HeartAttackRate'
target_variable = 'HeartAttackRate'  # Update to the correct target variable name

if target_variable in df.columns:

    # Histograms for all numerical features
    num_features = df.select_dtypes(include=[np.number]).columns  # Get numerical features
    num_plots = len(num_features)

    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each numerical feature
    for i, feature in enumerate(num_features):
        plt.subplot((num_plots // 3) + 1, 3, i + 1)  # Adjust layout based on number of features
        sns.histplot(df[feature], bins=15, kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.suptitle('Histograms of Numerical Features', y=1.02)  # Title for the entire figure
    plt.show()
else:
    print(f"Cannot proceed with further analysis since the target variable '{target_variable}' is not found.")

# Calculate total heart attacks for each gender
if 'Gender' in df.columns and 'HeartAttackRate' in df.columns:
    total_attacks_by_gender = df.groupby('Gender')['HeartAttackRate'].sum().reset_index()

    # Print the total attacks for each gender
    print("Total Heart Attacks by Gender:")
    print(total_attacks_by_gender)

    # Create the bar graph for total heart attacks by gender
    plt.figure(figsize=(8, 5))
    
    # Define colors for each gender
    plt.bar(total_attacks_by_gender['Gender'], total_attacks_by_gender['HeartAttackRate'], color=['pink', 'blue'])

    plt.title('Gender wise heart attack')
    plt.xlabel('Gender')
    plt.ylabel('Total Heart Attacks')
    plt.xticks(rotation=0)  # Optional: Rotate x-axis labels if needed
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend
    plt.legend(handles=[plt.Rectangle((0,0),1,1, color='blue'), plt.Rectangle((0,0),1,1, color='pink')],
               labels=['Male', 'Female'], title='Gender')

    # Show the plot
    plt.show()