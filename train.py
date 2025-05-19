#!/usr/bin/env python
"""
Main training script for the house price prediction model
"""
import argparse
import time
import os

from data_download import download_dataset
from data_processing import load_data, preprocess_data, save_preprocessing_config
from feature_engineering import add_features_to_preprocessed
from model import train_model, evaluate_model, save_model, train_ensemble

def main():
    """
    Main training function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train house price prediction model')
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'random_forest', 'lightgbm', 'elastic_net', 'ensemble'],
                        help='Type of model to train')
    parser.add_argument('--tune-hyperparams', action='store_true',
                        help='Whether to tune hyperparameters')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    args = parser.parse_args()
    
    print(f"Training {args.model_type} model...")
    if args.tune_hyperparams:
        print("Hyperparameter tuning enabled")
    
    # Step 1: Download/Generate data if it doesn't exist
    print("\n=== Step 1: Downloading/Generating Data ===")
    if not os.path.exists('data/housing_data.csv'):
        print("Data file not found. Downloading/generating data...")
        download_dataset()
    else:
        print("Data file already exists. Skipping download/generation.")
    
    # Step 2: Load and preprocess data
    print("\n=== Step 2: Loading and Preprocessing Data ===")
    start_time = time.time()
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
    print(f"Split into {len(train_df)} training samples and {len(test_df)} test samples")
    
    # Preprocess
    X_train, y_train, preprocessing_config = preprocess_data(train_df)
    print(f"Preprocessed training data: {X_train.shape}")
    
    # Apply feature engineering
    X_train, preprocessing_config = add_features_to_preprocessed(X_train, preprocessing_config)
    print(f"Added engineered features: {X_train.shape}")
    
    # Save preprocessing config
    save_preprocessing_config(preprocessing_config)
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocessing_time:.1f} seconds")
    
    # Step 3: Train model
    print("\n=== Step 3: Training Model ===")
    start_time = time.time()
    
    if args.model_type == 'ensemble':
        # Train ensemble
        ensemble = train_ensemble()
        training_time = time.time() - start_time
        print(f"Ensemble training completed in {training_time:.1f} seconds")
        
        # Additional info from the ensemble training is already printed
        
    else:
        # Train single model
        model = train_model(
            X_train, 
            y_train, 
            model_type=args.model_type,
            tune_hyperparams=args.tune_hyperparams
        )
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.1f} seconds")
        
        # Step 4: Evaluate model
        print("\n=== Step 4: Evaluating Model ===")
        start_time = time.time()
        
        # Preprocess test data
        X_test, y_test, _ = preprocess_data(
            test_df, 
            train_mode=False,
            preprocessing_config=preprocessing_config
        )
        
        # Apply feature engineering to test data
        X_test, _ = add_features_to_preprocessed(X_test, preprocessing_config)
        
        # Evaluate
        print("\nTest Set Performance:")
        metrics = evaluate_model(
            model, 
            X_test, 
            y_test, 
            feature_names=preprocessing_config.get('feature_names')
        )
        
        # Step 5: Save model
        print("\n=== Step 5: Saving Model ===")
        save_model(model, metrics, preprocessing_config, model_type=args.model_type)
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main() 