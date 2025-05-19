import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import joblib
import category_encoders as ce

def load_data(data_path='data/housing_data.csv'):
    """
    Load housing data from CSV file
    
    Args:
        data_path (str): Path to the CSV data file
    
    Returns:
        DataFrame: Housing data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Run data_download.py first.")
    
    return pd.read_csv(data_path)

def analyze_data(df):
    """
    Perform basic data analysis
    
    Args:
        df (DataFrame): Housing data
        
    Returns:
        dict: Data analysis results
    """
    results = {}
    
    # Basic statistics
    results['shape'] = df.shape
    results['missing_values'] = df.isnull().sum().to_dict()
    results['numeric_stats'] = df.describe().to_dict()
    
    # Target variable analysis
    results['price_stats'] = {
        'mean': df['price'].mean(),
        'median': df['price'].median(),
        'min': df['price'].min(),
        'max': df['price'].max(),
        'std': df['price'].std()
    }
    
    # Correlation with price
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')
    results['price_correlations'] = {col: df[col].corr(df['price']) for col in numeric_cols}
    
    return results

def preprocess_data(df, target_col='price', train_mode=True, preprocessing_config=None):
    """
    Preprocess housing data for modeling
    
    Args:
        df (DataFrame): Housing data
        target_col (str): Name of the target column
        train_mode (bool): Whether in training mode (to fit transformers) or inference mode
        preprocessing_config (dict): Preprocessing configuration (if None, a new one will be created)
        
    Returns:
        tuple: (X, y, preprocessing_config)
            X: Features DataFrame
            y: Target Series
            preprocessing_config: Configuration with fitted transformers
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if train_mode and preprocessing_config is None:
        preprocessing_config = {}
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        binary_features = [col for col in X.columns if X[col].nunique() == 2]
        
        for bf in binary_features:
            if bf in numeric_features:
                numeric_features.remove(bf)
            if bf in categorical_features:
                categorical_features.remove(bf)
        
        preprocessing_config['numeric_features'] = numeric_features
        preprocessing_config['categorical_features'] = categorical_features
        preprocessing_config['binary_features'] = binary_features
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])
        
        # For states, we'll use target encoding instead of one-hot
        if 'state' in categorical_features:
            categorical_features.remove('state')
            preprocessing_config['state_encoder'] = ce.TargetEncoder()
            preprocessing_config['state_encoder'].fit(X['state'], y)
            X['state_encoded'] = preprocessing_config['state_encoder'].transform(X['state'])
            preprocessing_config['state_encoded'] = True
        
        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bin', binary_transformer, binary_features)
            ],
            remainder='passthrough'
        )
        
        # Fit preprocessor
        preprocessing_config['preprocessor'] = preprocessor
        preprocessing_config['preprocessor'].fit(X)
    
    # Transform the data
    if 'state_encoded' in preprocessing_config and preprocessing_config['state_encoded']:
        if train_mode:
            X['state_encoded'] = preprocessing_config['state_encoder'].transform(X['state'])
        else:
            X['state_encoded'] = preprocessing_config['state_encoder'].transform(X['state'])
    
    X_processed = preprocessing_config['preprocessor'].transform(X)
    
    # Get feature names
    if train_mode:
        # Get feature names from column transformer
        numeric_features = preprocessing_config['numeric_features']
        categorical_features = preprocessing_config['categorical_features']
        binary_features = preprocessing_config['binary_features']
        
        # Get transformed feature names
        feature_names = []
        
        # Numeric features keep their names
        feature_names.extend(numeric_features)
        
        # Categorical features are expanded by one-hot encoding
        cat_transformer = preprocessing_config['preprocessor'].named_transformers_['cat']
        if categorical_features:
            cat_encoder = cat_transformer.named_steps['encoder']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)
        
        # Binary features keep their names
        feature_names.extend(binary_features)
        
        # Add any passthrough columns
        if 'state_encoded' in preprocessing_config and preprocessing_config['state_encoded']:
            feature_names.append('state_encoded')
        
        preprocessing_config['feature_names'] = feature_names
    
    # Convert to DataFrame if feature names are available
    if 'feature_names' in preprocessing_config:
        X_processed = pd.DataFrame(X_processed, columns=preprocessing_config['feature_names'])
    
    return X_processed, y, preprocessing_config

def save_preprocessing_config(preprocessing_config, filepath='data/preprocessing_config.pkl'):
    """Save preprocessing configuration to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessing_config, filepath)
    print(f"Preprocessing configuration saved to {filepath}")

def load_preprocessing_config(filepath='data/preprocessing_config.pkl'):
    """Load preprocessing configuration from disk"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessing configuration file not found: {filepath}")
    
    return joblib.load(filepath)

if __name__ == "__main__":
    # Example usage
    df = load_data()
    analysis = analyze_data(df)
    print(f"Dataset shape: {analysis['shape']}")
    print(f"Price statistics: {analysis['price_stats']}")
    
    X, y, preprocessing_config = preprocess_data(df)
    save_preprocessing_config(preprocessing_config)
    
    print(f"Preprocessed features shape: {X.shape}")
    print(f"Top price correlations: {sorted(analysis['price_correlations'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]}") 