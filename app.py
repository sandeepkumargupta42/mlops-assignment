from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('random_forest_model.joblib')

def test_model(model, X_test, run_name):
    with mlflow.start_run(run_name=run_name):
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics (if actual values are provided)
        if 'QTY' in X_test.columns:
            y_test = X_test['QTY']
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            # Return predictions and metrics
            return y_pred, {'mae': mae, 'mse': mse, 'r2': r2}
        
        # Return only predictions if actual values are not provided
        return y_pred, {}

@app.route('/api')
def home():
    return "Random Forest Regressor API"

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Use the test_model function to make predictions and log to MLflow
    run_name = "PredictionRun"
    predictions, metrics = test_model(model, df, run_name)
    
    # Prepare the response
    response = {
        'predictions': predictions.tolist(),
        'metrics': metrics
    }
    
    # Return response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)


#curl -X POST -H "Content-Type: application/json" -d '[{"day": 1, "month": 1, "year": 2020, "weekday": 2}]' http://127.0.0.1:5001/predict
