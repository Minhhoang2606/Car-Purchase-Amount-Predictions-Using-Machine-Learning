'''
Car Purchases Amount Prediction by Machine Learning
Author: Henry Ha
'''
# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv('Car_Purchasing_Data.csv')

#TODO EDA

# Display the first 5 rows of the dataset
print(data.head())

# Display descriptive statistics
print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms for numerical features
data[['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Car Purchase Amount']].hist(
    figsize=(12, 8), bins=20, edgecolor='black')
plt.tight_layout()
plt.show()

# Plot a correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Generate a pair plot
sns.pairplot(data[['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Car Purchase Amount']])
plt.show()

#TODO Data Preparation and Feature Engineering

# Check for missing values
print(data.isnull().sum())

# Drop irrelevant columns
data = data.drop(['Customer Name', 'Customer e-mail', 'Country'], axis=1)

from sklearn.preprocessing import StandardScaler

# Separate features and target variable
X = data.drop('Car Purchase Amount', axis=1)  # Features
y = data['Car Purchase Amount']  # Target variable

# Apply scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#TODO Model Building and Evaluation
from pycaret.regression import setup, compare_models, pull, save_model

# Initialize PyCaret regression setup
automl = setup(
    data=data,
    target='Car Purchase Amount',  # Target variable
    train_size=0.8,                # Train-test split ratio
    session_id=42,                 # For reproducibility
    normalize=True                 # Ensures feature scaling
)

# Compare models and select the best one
best_model = compare_models(sort='MAE')

# Display comparison results
results = pull()
print(results)

# Save the best model
import joblib

# Save the trained model
joblib.dump(best_model, 'car_purchase_model.pkl')

print("Model saved as 'car_purchase_model.pkl'")

