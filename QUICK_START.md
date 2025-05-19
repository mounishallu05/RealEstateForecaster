# Quick Start Guide: US House Price Prediction

This guide will help you get started with the house price prediction model.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The project can generate synthetic data for demonstration purposes:

```bash
python src/data_download.py
```

This will create training and test datasets in the `data/` directory.

## Training a Model

To train the default XGBoost model:

```bash
python src/train.py
```

### Training Options

You can customize the training with these arguments:

- `--model-type`: Choose model type (`xgboost`, `random_forest`, `lightgbm`, `elastic_net`, or `ensemble`)
- `--tune-hyperparams`: Enable hyperparameter tuning (takes longer)
- `--test-size`: Proportion of data to use for testing (default: 0.2)

Examples:

```bash
# Train an ensemble model
python src/train.py --model-type ensemble

# Train a LightGBM model with hyperparameter tuning
python src/train.py --model-type lightgbm --tune-hyperparams
```

## Evaluating a Model

To evaluate a trained model:

```bash
python src/evaluation.py
```

This will generate evaluation metrics and plots in the `data/plots/` and `data/evaluations/` directories.

## Using the Web App

To launch the web application for making predictions:

```bash
python app.py
```

Then open your browser to http://127.0.0.1:5000/

The web app provides:
- Individual house price predictions
- Batch prediction capability (upload CSV)

## Project Structure

- `data/`: Contains datasets and preprocessing configurations
- `models/`: Saved models and metadata
- `src/`: Source code
  - `data_download.py`: Downloads/generates dataset
  - `data_processing.py`: Data preprocessing functions
  - `feature_engineering.py`: Feature engineering functions
  - `model.py`: Model training and prediction
  - `evaluation.py`: Model evaluation
  - `train.py`: Main training script
- `app.py`: Web application for predictions
- `templates/`: HTML templates for the web app

## Making Predictions in Code

```python
from src.data_processing import preprocess_data, load_preprocessing_config
from src.feature_engineering import add_features_to_preprocessed
from src.model import load_model
import pandas as pd

# Load model and preprocessing config
model, metadata = load_model()
preprocessing_config = metadata['preprocessing_config']

# Prepare input data
data = {
    'state': 'CA', 
    'zipcode': 90210,
    'sqft': 2000, 
    'bedrooms': 3, 
    'bathrooms': 2,
    'year_built': 2000,
    'lot_size': 10000,
    'has_garage': 1,
    'has_pool': 0,
    'house_type': 'Single Family',
    'neighborhood_quality': 'High'
}
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
prediction = model.predict(X_final)[0]
print(f"Predicted price: ${prediction:,.2f}")
``` 