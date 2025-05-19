import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering specific to housing data
    """
    
    def __init__(self):
        self.features_added = []
    
    def fit(self, X, y=None):
        """
        Fit the transformer (just stores metadata in this case)
        """
        # Store column names for reference
        self.columns_ = X.columns if isinstance(X, pd.DataFrame) else None
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding engineered features
        """
        # Convert to DataFrame if not already
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Reset features_added to avoid duplicates when called multiple times
        self.features_added = []
        
        # Age of the house
        if 'year_built' in X_copy.columns:
            X_copy['house_age'] = 2023 - X_copy['year_built']
            self.features_added.append('house_age')
        
        # Price per square foot (for training data that includes price)
        if 'price' in X_copy.columns and 'sqft' in X_copy.columns:
            X_copy['price_per_sqft'] = X_copy['price'] / X_copy['sqft']
            self.features_added.append('price_per_sqft')
        
        # Room counts
        if 'bedrooms' in X_copy.columns and 'bathrooms' in X_copy.columns:
            X_copy['total_rooms'] = X_copy['bedrooms'] + X_copy['bathrooms']
            X_copy['bedroom_ratio'] = X_copy['bedrooms'] / (X_copy['total_rooms'] + 0.01)  # Avoid division by zero
            self.features_added.extend(['total_rooms', 'bedroom_ratio'])
        
        # Living area per bedroom
        if 'sqft' in X_copy.columns and 'bedrooms' in X_copy.columns:
            X_copy['sqft_per_bedroom'] = X_copy['sqft'] / (X_copy['bedrooms'] + 0.01)  # Avoid division by zero
            self.features_added.append('sqft_per_bedroom')
        
        # Lot utilization
        if 'sqft' in X_copy.columns and 'lot_size' in X_copy.columns:
            X_copy['lot_utilization'] = X_copy['sqft'] / (X_copy['lot_size'] + 1)  # Avoid division by zero
            self.features_added.append('lot_utilization')
        
        # Neighborhood quality numeric
        if 'neighborhood_quality' in X_copy.columns:
            quality_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            X_copy['neighborhood_quality_score'] = X_copy['neighborhood_quality'].map(quality_map)
            self.features_added.append('neighborhood_quality_score')
        
        # Zipcode first digit (region indicator in the US)
        if 'zipcode' in X_copy.columns:
            X_copy['zipcode_region'] = X_copy['zipcode'].astype(str).str[0].astype(int)
            self.features_added.append('zipcode_region')
        
        # House value indicators
        if 'has_pool' in X_copy.columns and 'has_garage' in X_copy.columns:
            X_copy['amenities_count'] = X_copy['has_pool'] + X_copy['has_garage']
            self.features_added.append('amenities_count')
        
        # House type encoding
        if 'house_type' in X_copy.columns:
            # Create value map based on typical pricing hierarchy
            type_value_map = {
                'Condo': 1,
                'Townhouse': 2,
                'Single Family': 3
            }
            X_copy['house_type_value'] = X_copy['house_type'].map(type_value_map)
            self.features_added.append('house_type_value')
        
        return X_copy
    
    def get_feature_names_out(self):
        """
        Get the feature names created by this transformer
        """
        return self.features_added

def create_polynomial_features(X, degree=2, interaction_only=True, include_bias=False):
    """
    Create polynomial features from the input data
    
    Args:
        X (DataFrame): Input features
        degree (int): Degree of polynomial features
        interaction_only (bool): Whether to include only interaction terms
        include_bias (bool): Whether to include a bias column of ones
        
    Returns:
        DataFrame: Input features with polynomial features added
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    # Select only numeric columns
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=['number']).columns
        X_numeric = X[numeric_cols]
    else:
        X_numeric = X
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    X_poly = poly.fit_transform(X_numeric)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Create DataFrame
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index if isinstance(X, pd.DataFrame) else None)
    
    # Remove the constant term if it exists and isn't wanted
    if not include_bias and '1' in X_poly_df.columns:
        X_poly_df = X_poly_df.drop('1', axis=1)
    
    # Add the original DataFrame columns that weren't included in polynomial features
    if isinstance(X, pd.DataFrame):
        non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            X_poly_df[col] = X[col].values
    
    return X_poly_df

def add_features_to_preprocessed(X_processed, preprocessing_config):
    """
    Add engineered features to already preprocessed data
    
    Args:
        X_processed (DataFrame): Preprocessed feature data
        preprocessing_config (dict): Preprocessing configuration
        
    Returns:
        DataFrame: Enhanced feature data
    """
    # Apply feature engineering
    feature_engineer = FeatureEngineer()
    X_enhanced = feature_engineer.fit_transform(X_processed)
    
    # Update feature names in preprocessing config
    if 'feature_names' in preprocessing_config:
        new_features = feature_engineer.get_feature_names_out()
        preprocessing_config['feature_names'].extend(new_features)
        preprocessing_config['engineered_features'] = new_features
    
    return X_enhanced, preprocessing_config

def save_feature_config(feature_config, filepath='data/feature_config.pkl'):
    """Save feature engineering configuration to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(feature_config, filepath)
    print(f"Feature engineering configuration saved to {filepath}")

def load_feature_config(filepath='data/feature_config.pkl'):
    """Load feature engineering configuration from disk"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature engineering configuration file not found: {filepath}")
    
    return joblib.load(filepath)

if __name__ == "__main__":
    # Example usage
    from data_processing import load_data, preprocess_data
    
    df = load_data()
    X_processed, y, preprocessing_config = preprocess_data(df)
    
    # Apply feature engineering
    X_enhanced, preprocessing_config = add_features_to_preprocessed(X_processed, preprocessing_config)
    
    # Print results
    print(f"Original features: {X_processed.shape[1]}")
    print(f"Enhanced features: {X_enhanced.shape[1]}")
    print(f"Added features: {preprocessing_config['engineered_features']}")
    
    # Save configuration
    save_feature_config({
        'added_features': preprocessing_config['engineered_features'],
        'feature_engineer': FeatureEngineer()
    }) 