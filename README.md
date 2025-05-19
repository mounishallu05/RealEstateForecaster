# US House Price Prediction

A machine learning model to predict house prices in the United States based on various features like location, size, number of rooms, etc.

## Project Overview

This project provides a complete machine learning pipeline for predicting house prices in the US market. The system includes:

- Synthetic data generation with realistic US housing parameters
- Comprehensive data preprocessing and feature engineering
- Multiple model options (XGBoost, Random Forest, LightGBM, Elastic Net)
- Ensemble modeling capability
- Evaluation with metrics specific to real estate predictions
- Interactive web interface for making predictions
- Data exploration tools and visualizations

## Features

- **Regional Price Modeling**: Accounts for state-level price differences
- **Advanced Feature Engineering**: Creates derived features like price per sqft, lot utilization, etc.
- **Flexible Model Selection**: Choose from multiple algorithms or use ensemble approach
- **Comprehensive Evaluation**: RMSE, MAE, MAPE and percentage within price ranges
- **Interactive Web Interface**: Easy-to-use prediction UI
- **Batch Prediction**: Support for processing multiple properties at once

## Project Structure

```
project/
├── data/                  # Data files and preprocessing artifacts
│   ├── housing_data.csv   # Main dataset
│   ├── plots/             # Evaluation plots and visualizations
│   └── evaluations/       # Saved model evaluation results
├── models/                # Saved models and metadata
├── notebooks/             # Jupyter notebooks for exploration
│   ├── exploration.ipynb  # Data exploration notebook
│   └── data_exploration.py # Data exploration script
├── src/                   # Source code
│   ├── data_download.py   # Data generation script
│   ├── data_processing.py # Data preprocessing functions
│   ├── feature_engineering.py # Feature engineering functions
│   ├── model.py           # Model training and evaluation 
│   ├── evaluation.py      # Detailed model evaluation
│   └── train.py           # Training pipeline script
├── templates/             # Web app templates
│   └── index.html         # Main prediction interface
├── temp/                  # Temporary files for batch predictions
├── app.py                 # Flask web application
├── requirements.txt       # Project dependencies
├── QUICK_START.md         # Quick start guide
└── run.sh                 # Shell script for easy execution
```

## Getting Started

### Prerequisites

- Python 3.6+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Generation

Generate synthetic housing data:

```bash
python src/data_download.py
```

### Model Training

Train a model using the default settings (XGBoost):

```bash
python src/train.py
```

Or specify a model type:

```bash
python src/train.py --model-type ensemble
```

Available model types:
- `xgboost` (default)
- `random_forest`
- `lightgbm`
- `elastic_net`
- `ensemble` (combines multiple models)

For hyperparameter tuning:

```bash
python src/train.py --tune-hyperparams
```

### Evaluation

Evaluate the trained model:

```bash
python src/evaluation.py
```

This generates evaluation metrics and visualizations in `data/plots/` and `data/evaluations/`.

### Web Application

Run the web application for interactive predictions:

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:5000/`

### Using the Shell Script

For convenience, you can use the shell script:

```bash
chmod +x run.sh
./run.sh
```

This provides an interactive menu to:
1. Train a model
2. Evaluate a model
3. Start the web app
4. Run the full pipeline

## Data Exploration

Explore the housing data using the provided script:

```bash
python notebooks/data_exploration.py
```

This generates visualizations and a summary of the data in `notebooks/plots/`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The synthetic data is generated to mimic real US housing market trends
- Model architecture and feature engineering inspired by best practices in real estate prediction

## Dataset
This project uses the Zillow Housing Dataset (or similar) which includes housing data across various US regions.

## Model
The model uses gradient boosting (XGBoost) and ensemble techniques to predict house prices based on features like:
- Location (zip code, city, state)
- Size (square footage)
- Number of bedrooms/bathrooms
- Year built
- Lot size
- Neighborhood amenities
- And more... 