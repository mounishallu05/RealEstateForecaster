import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
import json
from datetime import datetime

# Import local modules
from data_processing import load_data, preprocess_data, load_preprocessing_config
from feature_engineering import add_features_to_preprocessed
from model import load_model

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using multiple metrics
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate median absolute percentage error (common in real estate)
    mape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate percentage of predictions within X% of true value
    within_5_percent = np.mean(np.abs((y_true - y_pred) / y_true) < 0.05) * 100
    within_10_percent = np.mean(np.abs((y_true - y_pred) / y_true) < 0.10) * 100
    within_20_percent = np.mean(np.abs((y_true - y_pred) / y_true) < 0.20) * 100
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'within_5_percent': within_5_percent,
        'within_10_percent': within_10_percent,
        'within_20_percent': within_20_percent
    }
    
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics in a readable format"""
    print(f"Evaluation metrics:")
    print(f"  RMSE:  ${metrics['rmse']:.2f}")
    print(f"  MAE:   ${metrics['mae']:.2f}")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  % Within 5% of true:  {metrics['within_5_percent']:.1f}%")
    print(f"  % Within 10% of true: {metrics['within_10_percent']:.1f}%")
    print(f"  % Within 20% of true: {metrics['within_20_percent']:.1f}%")

def plot_residuals(y_true, y_pred, max_samples=1000):
    """
    Plot residuals
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        max_samples (int): Maximum number of samples to plot
    """
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate residuals
    residuals = y_pred - y_true
    
    # Subsample if needed
    if len(y_true) > max_samples:
        idx = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        residuals = residuals[idx]
    
    # Create plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='-')
    axes[0].set_xlabel('Predicted Price ($)')
    axes[0].set_ylabel('Residuals ($)')
    axes[0].set_title('Residuals vs Predicted Values')
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='-')
    axes[1].set_xlabel('Residual Value ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/residuals.png')
    plt.close()
    
    print(f"Residual plots saved to data/plots/residuals.png")

def plot_error_by_feature(df, y_true, y_pred, feature, max_samples=1000):
    """
    Plot prediction errors by a specific feature
    
    Args:
        df (DataFrame): Input data
        y_true (array): True values
        y_pred (array): Predicted values
        feature (str): Feature name to analyze
        max_samples (int): Maximum number of samples to plot
    """
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate percentage error
    pct_error = np.abs((y_pred - y_true) / y_true) * 100
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        feature: df[feature],
        'Percent Error': pct_error
    })
    
    # Subsample if needed
    if len(plot_df) > max_samples:
        plot_df = plot_df.sample(max_samples, random_state=42)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Different plot types depending on feature type
    if plot_df[feature].dtype in [np.int64, np.float64]:
        # For numeric features, use a scatter plot with trend line
        sns.regplot(x=feature, y='Percent Error', data=plot_df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    else:
        # For categorical features, use a box plot
        sns.boxplot(x=feature, y='Percent Error', data=plot_df)
        plt.xticks(rotation=45)
    
    plt.title(f'Prediction Error by {feature}')
    plt.ylabel('Percent Error (%)')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig(f'data/plots/error_by_{feature}.png')
    plt.close()
    
    print(f"Error by {feature} plot saved to data/plots/error_by_{feature}.png")

def analyze_model_performance(model, X, y, df, features_to_analyze=None):
    """
    Comprehensive analysis of model performance
    
    Args:
        model: Trained model
        X (DataFrame): Feature data
        y (Series): Target data
        df (DataFrame): Original DataFrame with all features
        features_to_analyze (list): List of features to analyze specifically
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = evaluate_predictions(y, y_pred)
    print_metrics(metrics)
    
    # Plot residuals
    plot_residuals(y, y_pred)
    
    # Plot error by important features
    if features_to_analyze is None:
        # Default features to analyze
        if 'state' in df.columns:
            plot_error_by_feature(df, y, y_pred, 'state')
        if 'bedrooms' in df.columns:
            plot_error_by_feature(df, y, y_pred, 'bedrooms')
        if 'sqft' in df.columns:
            plot_error_by_feature(df, y, y_pred, 'sqft')
        if 'year_built' in df.columns:
            plot_error_by_feature(df, y, y_pred, 'year_built')
        if 'neighborhood_quality' in df.columns:
            plot_error_by_feature(df, y, y_pred, 'neighborhood_quality')
    else:
        # Analyze specified features
        for feature in features_to_analyze:
            if feature in df.columns:
                plot_error_by_feature(df, y, y_pred, feature)
    
    # Calculate price segment performance
    price_segments = [
        (0, 200000), 
        (200000, 400000), 
        (400000, 600000),
        (600000, 800000),
        (800000, 1000000),
        (1000000, float('inf'))
    ]
    
    price_segment_metrics = {}
    for low, high in price_segments:
        mask = (y >= low) & (y < high)
        if sum(mask) > 10:  # Only evaluate if we have enough samples
            segment_metrics = evaluate_predictions(y[mask], y_pred[mask])
            price_segment_metrics[f"${low/1000:.0f}k-${high/1000:.0f}k"] = {
                'count': int(sum(mask)),
                'rmse': segment_metrics['rmse'],
                'mape': segment_metrics['mape'],
                'r2': segment_metrics['r2'],
                'within_10_percent': segment_metrics['within_10_percent']
            }
    
    # Print price segment performance
    print("\nPerformance by price segment:")
    for segment, segment_metrics in price_segment_metrics.items():
        print(f"  {segment} ({segment_metrics['count']} houses):")
        print(f"    RMSE: ${segment_metrics['rmse']:.2f}")
        print(f"    MAPE: {segment_metrics['mape']:.2f}%")
        print(f"    % Within 10% of true: {segment_metrics['within_10_percent']:.1f}%")
    
    # Save metrics
    os.makedirs('data/evaluations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    evaluation_results = {
        'timestamp': timestamp,
        'overall_metrics': metrics,
        'price_segment_metrics': price_segment_metrics
    }
    
    with open(f'data/evaluations/evaluation_{timestamp}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to data/evaluations/evaluation_{timestamp}.json")
    
    return evaluation_results

def analyze_ensemble_performance(ensemble, X, y, df):
    """
    Analyze performance of an ensemble model
    
    Args:
        ensemble (dict): Ensemble model dictionary
        X (DataFrame): Feature data
        y (Series): Target data
        df (DataFrame): Original DataFrame with all features
    """
    print("Analyzing ensemble performance...")
    
    # Make predictions with individual models
    predictions = {}
    for model_type in ensemble['model_types']:
        model = ensemble['models'][model_type]
        predictions[model_type] = model.predict(X)
    
    # Ensemble prediction (average)
    y_pred_ensemble = np.mean([predictions[model_type] for model_type in ensemble['model_types']], axis=0)
    
    # Calculate metrics for each model and the ensemble
    metrics = {}
    for model_type in ensemble['model_types']:
        metrics[model_type] = evaluate_predictions(y, predictions[model_type])
    
    metrics['ensemble'] = evaluate_predictions(y, y_pred_ensemble)
    
    # Print metrics
    print("\nModel performance comparison:")
    for model_type in ensemble['model_types'] + ['ensemble']:
        print(f"\n{model_type.upper()}:")
        print_metrics(metrics[model_type])
    
    # Plot residuals for ensemble
    plot_residuals(y, y_pred_ensemble)
    
    # Plot comparison of model performances
    model_names = ensemble['model_types'] + ['ensemble']
    rmse_values = [metrics[model]['rmse'] for model in model_names]
    r2_values = [metrics[model]['r2'] for model in model_names]
    within_10_pct = [metrics[model]['within_10_percent'] for model in model_names]
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, rmse_values, color='skyblue')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'${rmse_values[i]:.0f}', ha='center')
    
    plt.title('RMSE Comparison Between Models')
    plt.ylabel('RMSE ($)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/ensemble_rmse_comparison.png')
    plt.close()
    
    # Plot accuracy within 10% comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, within_10_pct, color='lightgreen')
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{within_10_pct[i]:.1f}%', ha='center')
    
    plt.title('Predictions Within 10% of True Value')
    plt.ylabel('Percentage of Predictions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('data/plots/ensemble_accuracy_comparison.png')
    plt.close()
    
    # Analyze price segment performance for the ensemble
    price_segments = [
        (0, 200000), 
        (200000, 400000), 
        (400000, 600000),
        (600000, 800000),
        (800000, 1000000),
        (1000000, float('inf'))
    ]
    
    price_segment_metrics = {}
    for low, high in price_segments:
        mask = (y >= low) & (y < high)
        if sum(mask) > 10:  # Only evaluate if we have enough samples
            segment_metrics = evaluate_predictions(y[mask], y_pred_ensemble[mask])
            price_segment_metrics[f"${low/1000:.0f}k-${high/1000:.0f}k"] = {
                'count': int(sum(mask)),
                'rmse': segment_metrics['rmse'],
                'mape': segment_metrics['mape'],
                'r2': segment_metrics['r2'],
                'within_10_percent': segment_metrics['within_10_percent']
            }
    
    # Print price segment performance
    print("\nEnsemble performance by price segment:")
    for segment, segment_metrics in price_segment_metrics.items():
        print(f"  {segment} ({segment_metrics['count']} houses):")
        print(f"    RMSE: ${segment_metrics['rmse']:.2f}")
        print(f"    MAPE: {segment_metrics['mape']:.2f}%")
        print(f"    % Within 10% of true: {segment_metrics['within_10_percent']:.1f}%")
    
    # Save evaluation results
    os.makedirs('data/evaluations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    evaluation_results = {
        'timestamp': timestamp,
        'model_metrics': {model: metrics[model] for model in model_names},
        'price_segment_metrics': price_segment_metrics
    }
    
    with open(f'data/evaluations/ensemble_evaluation_{timestamp}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEnsemble evaluation results saved to data/evaluations/ensemble_evaluation_{timestamp}.json")
    
    return evaluation_results

if __name__ == "__main__":
    # If model exists, load it
    try:
        model, metadata = load_model()
        preprocessing_config = metadata['preprocessing_config']
        model_exists = True
        print("Loaded existing model for evaluation")
    except FileNotFoundError:
        model_exists = False
        print("No existing model found, loading preprocessing config only")
        try:
            preprocessing_config = load_preprocessing_config()
        except FileNotFoundError:
            preprocessing_config = None
            print("No preprocessing config found either, will create from scratch")
    
    # Load test data
    print("Loading test data...")
    try:
        test_df = load_data('data/housing_data_test.csv')
    except FileNotFoundError:
        print("No separate test file found, loading main dataset and splitting...")
        df = load_data()
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Preprocess test data
    print("Preprocessing test data...")
    X_test, y_test, preprocessing_config = preprocess_data(
        test_df,
        train_mode=False,
        preprocessing_config=preprocessing_config
    )
    
    # Apply feature engineering
    X_test, preprocessing_config = add_features_to_preprocessed(X_test, preprocessing_config)
    
    if model_exists:
        print("\nEvaluating model performance...")
        
        # Check if it's an ensemble
        if isinstance(model, dict) and 'models' in model:
            analyze_ensemble_performance(model, X_test, y_test, test_df)
        else:
            analyze_model_performance(
                model, 
                X_test, 
                y_test, 
                test_df, 
                features_to_analyze=['state', 'bedrooms', 'sqft', 'year_built', 'neighborhood_quality']
            )
    else:
        print("No model available for evaluation. Train a model first using model.py") 