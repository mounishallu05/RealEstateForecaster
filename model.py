import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import local modules
from data_processing import load_data, preprocess_data, load_preprocessing_config, save_preprocessing_config
from feature_engineering import add_features_to_preprocessed

def train_model(X, y, model_type='xgboost', tune_hyperparams=True, cv=5):
    """
    Train a model on the given data
    
    Args:
        X (DataFrame): Feature data
        y (Series): Target data
        model_type (str): Type of model to train ('xgboost', 'random_forest', 'lightgbm', 'elastic_net')
        tune_hyperparams (bool): Whether to tune hyperparameters using GridSearchCV
        cv (int): Number of cross-validation folds
        
    Returns:
        object: Trained model
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'xgboost':
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X, y)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            model.fit(X, y)
            return model
    
    elif model_type == 'random_forest':
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4],
            }
            model = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X, y)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            model.fit(X, y)
            return model
    
    elif model_type == 'lightgbm':
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
            model = lgb.LGBMRegressor(objective='regression', random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X, y)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='regression',
                random_state=42
            )
            model.fit(X, y)
            return model
    
    elif model_type == 'elastic_net':
        if tune_hyperparams:
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            }
            model = ElasticNet(random_state=42, max_iter=10000)
            grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
            model.fit(X, y)
            return model
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def evaluate_model(model, X, y, feature_names=None):
    """
    Evaluate a model on the given data
    
    Args:
        model: Trained model
        X (DataFrame): Feature data
        y (Series): Target data
        feature_names (list): Feature names (for feature importance)
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Calculate median absolute percentage error (common in real estate)
    mape = np.median(np.abs((y - y_pred) / y)) * 100
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    print(f"Evaluation metrics:")
    print(f"  RMSE:  ${rmse:.2f}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    
    # Plot feature importance if model supports it
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('data/plots', exist_ok=True)
        plt.savefig('data/plots/feature_importance.png')
        plt.close()
        
        print(f"Feature importance plot saved to data/plots/feature_importance.png")
        
        # Add to metrics
        metrics['top_features'] = feature_importance['feature'].tolist()[:10]
    
    return metrics

def plot_predictions(y_true, y_pred, max_samples=1000):
    """
    Plot predicted vs actual values
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        max_samples (int): Maximum number of samples to plot
    """
    # Subsample if needed
    if len(y_true) > max_samples:
        idx = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Predicted vs Actual House Prices')
    
    # Add metrics to plot
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    plt.annotate(f'RMSE: ${rmse:.2f}\nR²: {r2:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    plt.savefig('data/plots/prediction_scatter.png')
    plt.close()
    
    print(f"Prediction plot saved to data/plots/prediction_scatter.png")

def save_model(model, metrics, preprocessing_config, model_type='xgboost'):
    """
    Save model and metadata to disk
    
    Args:
        model: Trained model
        metrics (dict): Evaluation metrics
        preprocessing_config (dict): Preprocessing configuration
        model_type (str): Type of model
    """
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f"models/house_price_model_{model_type}_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'timestamp': timestamp,
        'metrics': metrics,
        'preprocessing_config': preprocessing_config
    }
    
    metadata_path = f"models/house_price_model_{model_type}_{timestamp}_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    
    # Also save as latest model for easy reference
    latest_model_path = f"models/house_price_model_{model_type}_latest.pkl"
    latest_metadata_path = f"models/house_price_model_{model_type}_latest_metadata.pkl"
    
    joblib.dump(model, latest_model_path)
    joblib.dump(metadata, latest_metadata_path)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Latest model saved to {latest_model_path}")

def load_model(model_path="models/house_price_model_xgboost_latest.pkl", 
               metadata_path="models/house_price_model_xgboost_latest_metadata.pkl"):
    """
    Load model and metadata from disk
    
    Args:
        model_path (str): Path to model file
        metadata_path (str): Path to metadata file
        
    Returns:
        tuple: (model, metadata)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    return model, metadata

def train_ensemble():
    """
    Train an ensemble of models
    
    Returns:
        dict: Ensemble model and metadata
    """
    print("Training ensemble of models...")
    
    # Load and preprocess data
    df = load_data()
    X_train_raw, y_train_raw = df.drop('price', axis=1), df['price']
    
    # Split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, random_state=42
    )
    
    # Preprocess data
    X_train_processed, y_train, preprocessing_config = preprocess_data(
        pd.concat([X_train, pd.DataFrame({'price': y_train})], axis=1)
    )
    
    # Apply same preprocessing to validation data
    X_val_processed, y_val, _ = preprocess_data(
        pd.concat([X_val, pd.DataFrame({'price': y_val})], axis=1),
        train_mode=False,
        preprocessing_config=preprocessing_config
    )
    
    # Apply feature engineering
    X_train_final, preprocessing_config = add_features_to_preprocessed(X_train_processed, preprocessing_config)
    X_val_final, _ = add_features_to_preprocessed(X_val_processed, preprocessing_config)
    
    # Train individual models
    models = {}
    predictions = {}
    
    model_types = ['xgboost', 'lightgbm', 'random_forest']
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        model = train_model(X_train_final, y_train, model_type=model_type, tune_hyperparams=False)
        models[model_type] = model
        
        # Make predictions
        predictions[model_type] = model.predict(X_val_final)
        
        # Evaluate
        metrics = evaluate_model(model, X_val_final, y_val, 
                                feature_names=preprocessing_config.get('feature_names'))
        print(f"{model_type} validation RMSE: ${metrics['rmse']:.2f}")
    
    # Create ensemble prediction (simple average)
    y_pred_ensemble = np.mean([predictions[model_type] for model_type in model_types], axis=0)
    
    # Calculate ensemble metrics
    ensemble_mse = mean_squared_error(y_val, y_pred_ensemble)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_mae = mean_absolute_error(y_val, y_pred_ensemble)
    ensemble_r2 = r2_score(y_val, y_pred_ensemble)
    ensemble_mape = np.median(np.abs((y_val - y_pred_ensemble) / y_val)) * 100
    
    print("\nEnsemble model performance:")
    print(f"  RMSE:  ${ensemble_rmse:.2f}")
    print(f"  MAE:   ${ensemble_mae:.2f}")
    print(f"  R²:    {ensemble_r2:.4f}")
    print(f"  MAPE:  {ensemble_mape:.2f}%")
    
    # Plot ensemble predictions
    plot_predictions(y_val.values, y_pred_ensemble)
    
    # Create ensemble metadata
    ensemble = {
        'models': models,
        'model_types': model_types,
        'preprocessing_config': preprocessing_config,
        'metrics': {
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'r2': ensemble_r2,
            'mape': ensemble_mape
        }
    }
    
    # Save ensemble
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_path = f"models/house_price_ensemble_{timestamp}.pkl"
    joblib.dump(ensemble, ensemble_path)
    
    # Also save as latest ensemble
    latest_ensemble_path = "models/house_price_ensemble_latest.pkl"
    joblib.dump(ensemble, latest_ensemble_path)
    
    print(f"Ensemble saved to {ensemble_path}")
    print(f"Latest ensemble saved to {latest_ensemble_path}")
    
    return ensemble

if __name__ == "__main__":
    # If preprocessing config exists, load it
    try:
        preprocessing_config = load_preprocessing_config()
        print("Loaded existing preprocessing configuration")
    except FileNotFoundError:
        preprocessing_config = None
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    
    # Split data
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Preprocess training data
    X_train, y_train, preprocessing_config = preprocess_data(train_df, preprocessing_config=preprocessing_config)
    
    # Apply feature engineering
    X_train, preprocessing_config = add_features_to_preprocessed(X_train, preprocessing_config)
    
    # Save preprocessing config for future use
    save_preprocessing_config(preprocessing_config)
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train, tune_hyperparams=False)
    
    # Preprocess test data using same transformations
    X_test, y_test, _ = preprocess_data(
        test_df, 
        train_mode=False,
        preprocessing_config=preprocessing_config
    )
    
    # Apply feature engineering to test data
    X_test, _ = add_features_to_preprocessed(X_test, preprocessing_config)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model, 
        X_test, 
        y_test, 
        feature_names=preprocessing_config.get('feature_names')
    )
    
    # Plot predictions
    y_pred = model.predict(X_test)
    plot_predictions(y_test.values, y_pred)
    
    # Save model
    save_model(model, metrics, preprocessing_config)
    
    print("\nTraining ensemble model...")
    # Train ensemble
    ensemble = train_ensemble() 