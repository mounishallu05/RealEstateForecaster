import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import json
import traceback

# Import local modules
from src.data_processing import preprocess_data, load_preprocessing_config
from src.feature_engineering import add_features_to_preprocessed
from src.model import load_model

app = Flask(__name__)

# Load model and preprocessing config
try:
    model, metadata = load_model()
    preprocessing_config = metadata['preprocessing_config']
    
    # Check if it's an ensemble
    if isinstance(model, dict) and 'models' in model:
        is_ensemble = True
        models = model['models']
        model_types = model['model_types']
    else:
        is_ensemble = False
    
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model_loaded = False
    preprocessing_config = None
    
    # Try to load just the preprocessing config
    try:
        preprocessing_config = load_preprocessing_config()
        print("Preprocessing config loaded successfully!")
    except Exception as e:
        print(f"Error loading preprocessing config: {e}")
        traceback.print_exc()

# Define state price factors for reference
STATE_PRICE_FACTORS = {
    'CA': 1.8, 'NY': 1.7, 'TX': 0.9, 'FL': 1.1, 'WA': 1.3, 
    'CO': 1.2, 'IL': 1.0, 'GA': 0.9, 'NC': 0.85, 'OH': 0.7,
    'PA': 0.8, 'MA': 1.5, 'AZ': 1.0, 'NJ': 1.4, 'MI': 0.6,
    'VA': 1.1, 'WI': 0.8, 'MN': 0.9, 'OR': 1.2, 'MD': 1.1
}

@app.route('/')
def home():
    """
    Render the home page
    """
    # Get list of states for the dropdown
    states = list(STATE_PRICE_FACTORS.keys())
    states.sort()
    
    return render_template('index.html', states=states, model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on form data
    """
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train a model first using the train.py script.'
        }), 400
    
    try:
        # Get form data
        data = {}
        data['state'] = request.form.get('state')
        data['zipcode'] = int(request.form.get('zipcode'))
        data['sqft'] = int(request.form.get('sqft'))
        data['bedrooms'] = int(request.form.get('bedrooms'))
        data['bathrooms'] = float(request.form.get('bathrooms'))
        data['year_built'] = int(request.form.get('year_built'))
        data['lot_size'] = int(request.form.get('lot_size'))
        data['has_garage'] = 1 if request.form.get('has_garage') == 'yes' else 0
        data['has_pool'] = 1 if request.form.get('has_pool') == 'yes' else 0
        data['house_type'] = request.form.get('house_type')
        data['neighborhood_quality'] = request.form.get('neighborhood_quality')
        
        # Create DataFrame from form data
        input_df = pd.DataFrame([data])
        
        # Preprocess the data
        X_processed, _, _ = preprocess_data(
            input_df,
            train_mode=False,
            preprocessing_config=preprocessing_config
        )
        
        # Apply feature engineering
        X_final, _ = add_features_to_preprocessed(X_processed, preprocessing_config)
        
        # Make prediction
        if is_ensemble:
            # Average predictions from all models
            predictions = []
            for model_type in model_types:
                model_instance = models[model_type]
                predictions.append(model_instance.predict(X_final)[0])
            
            prediction = np.mean(predictions)
            
            # Also get individual model predictions
            individual_predictions = {
                model_type: float(models[model_type].predict(X_final)[0])
                for model_type in model_types
            }
        else:
            prediction = model.predict(X_final)[0]
            individual_predictions = None
        
        # Format the prediction
        formatted_prediction = f"${prediction:,.2f}"
        
        # Return the prediction
        return jsonify({
            'prediction': formatted_prediction,
            'raw_prediction': float(prediction),
            'individual_predictions': individual_predictions
        })
    
    except Exception as e:
        # Print the exception for debugging
        traceback.print_exc()
        
        # Return an error response
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions on a batch of houses
    """
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train a model first using the train.py script.'
        }), 400
    
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400
        
        file = request.files['file']
        
        # Check if the file has a name
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        
        # Check if the file is a CSV
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV.'}), 400
        
        # Read the CSV
        input_df = pd.read_csv(file)
        
        # Validate the dataframe
        required_columns = ['state', 'zipcode', 'sqft', 'bedrooms', 'bathrooms', 'year_built']
        for col in required_columns:
            if col not in input_df.columns:
                return jsonify({'error': f'Missing required column: {col}'}), 400
        
        # Preprocess the data
        X_processed, _, _ = preprocess_data(
            input_df,
            train_mode=False,
            preprocessing_config=preprocessing_config
        )
        
        # Apply feature engineering
        X_final, _ = add_features_to_preprocessed(X_processed, preprocessing_config)
        
        # Make predictions
        if is_ensemble:
            # Average predictions from all models
            predictions = []
            for model_type in model_types:
                model_instance = models[model_type]
                predictions.append(model_instance.predict(X_final))
            
            batch_predictions = np.mean(predictions, axis=0)
        else:
            batch_predictions = model.predict(X_final)
        
        # Add predictions to the dataframe
        input_df['predicted_price'] = batch_predictions
        
        # Create a temporary file to save the results
        temp_file = os.path.join('temp', 'predictions.csv')
        os.makedirs('temp', exist_ok=True)
        input_df.to_csv(temp_file, index=False)
        
        # Return the path to the results
        return jsonify({
            'success': True,
            'file_path': temp_file,
            'num_predictions': len(batch_predictions)
        })
    
    except Exception as e:
        # Print the exception for debugging
        traceback.print_exc()
        
        # Return an error response
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 400

if __name__ == '__main__':
    # Create the templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple HTML template if it doesn't exist
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 40px;
        }
        h1 {
            color: #3b4b59;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .row {
            display: flex;
            flex-wrap: wrap;
            margin: 0 -10px;
        }
        .col {
            flex: 1;
            padding: 0 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            display: block;
            margin: 20px auto;
            min-width: 200px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background-color: #e9f7ef;
            border-radius: 5px;
            font-size: 24px;
            display: none;
        }
        .error {
            color: #d9534f;
            text-align: center;
            margin-top: 20px;
            display: none;
        }
        .model-warning {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            text-align: center;
        }
        .loader {
            border: 8px solid #f3f3f3;
            border-radius: 50%;
            border-top: 8px solid #3498db;
            width: 60px;
            height: 60px;
            margin: 20px auto;
            animation: spin 2s linear infinite;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>House Price Predictor</h1>
        
        {% if not model_loaded %}
            <div class="model-warning">
                <strong>Warning:</strong> Model not loaded. Please train a model first using the train.py script.
            </div>
        {% endif %}
        
        <form id="predictionForm">
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label for="state">State:</label>
                        <select id="state" name="state" required>
                            {% for state in states %}
                                <option value="{{ state }}">{{ state }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="zipcode">Zip Code:</label>
                        <input type="number" id="zipcode" name="zipcode" min="10000" max="99999" required>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label for="sqft">Square Feet:</label>
                        <input type="number" id="sqft" name="sqft" min="100" required>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="lot_size">Lot Size (sq ft):</label>
                        <input type="number" id="lot_size" name="lot_size" min="100" required>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label for="bedrooms">Bedrooms:</label>
                        <input type="number" id="bedrooms" name="bedrooms" min="0" required>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="bathrooms">Bathrooms:</label>
                        <select id="bathrooms" name="bathrooms">
                            <option value="1">1</option>
                            <option value="1.5">1.5</option>
                            <option value="2">2</option>
                            <option value="2.5">2.5</option>
                            <option value="3">3</option>
                            <option value="3.5">3.5</option>
                            <option value="4">4</option>
                            <option value="4.5">4.5</option>
                            <option value="5">5</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label for="year_built">Year Built:</label>
                        <input type="number" id="year_built" name="year_built" min="1800" max="2023" required>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="house_type">House Type:</label>
                        <select id="house_type" name="house_type">
                            <option value="Single Family">Single Family</option>
                            <option value="Townhouse">Townhouse</option>
                            <option value="Condo">Condo</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label for="has_garage">Has Garage:</label>
                        <select id="has_garage" name="has_garage">
                            <option value="yes">Yes</option>
                            <option value="no">No</option>
                        </select>
                    </div>
                </div>
                <div class="col">
                    <div class="form-group">
                        <label for="has_pool">Has Pool:</label>
                        <select id="has_pool" name="has_pool">
                            <option value="no">No</option>
                            <option value="yes">Yes</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <div class="form-group">
                <label for="neighborhood_quality">Neighborhood Quality:</label>
                <select id="neighborhood_quality" name="neighborhood_quality">
                    <option value="Low">Low</option>
                    <option value="Medium">Medium</option>
                    <option value="High">High</option>
                    <option value="Very High">Very High</option>
                </select>
            </div>
            
            <button type="button" id="predictBtn" {% if not model_loaded %}disabled{% endif %}>Predict Price</button>
            
            <div class="loader" id="loader"></div>
            <div class="result" id="result"></div>
            <div class="error" id="error"></div>
        </form>
    </div>
    
    <script>
        document.getElementById('predictBtn').addEventListener('click', function() {
            // Show loader
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            const formData = new FormData(document.getElementById('predictionForm'));
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('error').textContent = 'Error: ' + data.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    // Show prediction
                    document.getElementById('result').textContent = 'Estimated Price: ' + data.prediction;
                    document.getElementById('result').style.display = 'block';
                    
                    // If individual predictions are available
                    if (data.individual_predictions) {
                        const models = Object.keys(data.individual_predictions);
                        if (models.length > 0) {
                            let details = document.createElement('div');
                            details.style.fontSize = '16px';
                            details.style.marginTop = '15px';
                            
                            let detailsText = '<strong>Individual Model Predictions:</strong><br>';
                            for (const model of models) {
                                const value = data.individual_predictions[model];
                                detailsText += model + ': $' + value.toLocaleString('en-US', {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2
                                }) + '<br>';
                            }
                            
                            details.innerHTML = detailsText;
                            document.getElementById('result').appendChild(details);
                        }
                    }
                }
            })
            .catch(error => {
                // Hide loader
                document.getElementById('loader').style.display = 'none';
                
                // Show error
                document.getElementById('error').textContent = 'Error: ' + error.message;
                document.getElementById('error').style.display = 'block';
            });
        });
        
        // Set default values
        window.onload = function() {
            document.getElementById('sqft').value = '2000';
            document.getElementById('bedrooms').value = '3';
            document.getElementById('bathrooms').value = '2';
            document.getElementById('year_built').value = '2000';
            document.getElementById('lot_size').value = '10000';
            document.getElementById('zipcode').value = '90210';
            document.getElementById('neighborhood_quality').value = 'Medium';
        };
    </script>
</body>
</html>
            """)
    
    # Run the app
    app.run(debug=True) 