import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Load the dataset
data = pd.read_csv('data/daily_data.csv')

# Convert ALL_DATE to datetime format
data['ALL_DATE'] = pd.to_datetime(data['ALL_DATE'], format='%d-%m-%Y')

# Extract features from date
data['day'] = data['ALL_DATE'].dt.day
data['month'] = data['ALL_DATE'].dt.month
data['year'] = data['ALL_DATE'].dt.year
data['weekday'] = data['ALL_DATE'].dt.weekday

# Drop the original date column and non-numeric columns
data = data.drop(columns=['ALL_DATE', 'AREA_CODE', 'PSKU'])

# Handle missing values if any (for simplicity, let's fill with the median value)
data.fillna(data.median(), inplace=True)

# Split the data into features and target variable
X = data.drop(columns=['QTY'])
y = data['QTY']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training phase
def train_model(X_train, y_train, X_test, y_test, n_estimators, model_file_path):
    with mlflow.start_run():
        # Initialize the Random Forest Regressor
        rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        
        # Train the model
        rfr.fit(X_train, y_train)
        
        # Save the model to a file
        joblib.dump(rfr, model_file_path)
        
        # Log model parameters
        mlflow.log_param("n_estimators", n_estimators)
        
        # Make predictions on the test set
        y_pred = rfr.predict(X_test)
        
        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log the model file as an artifact
        mlflow.log_artifact(model_file_path)
        
        # Log the model with MLflow
        mlflow.sklearn.log_model(rfr, "model")
        
        # Print evaluation metrics
        print(f'n_estimators: {n_estimators}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'R-squared (R²): {r2}')
    
    return rfr

# Testing phase
def test_model(model_file_path, X_test, y_test):
    # Load the model from file
    rfr = joblib.load(model_file_path)
    
    # Make predictions on the test set
    y_pred = rfr.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R²): {r2}')
    
    return y_pred

# Train models with different parameters
n_estimators_list = [50, 100, 150]
model_file_path = 'random_forest_model.joblib'

for n_estimators in n_estimators_list:
    train_model(X_train, y_train, X_test, y_test, n_estimators, model_file_path)

# Test the final model
test_model(model_file_path, X_test, y_test)