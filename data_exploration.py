#!/usr/bin/env python
"""
US Housing Data Exploration Script

This script explores the housing dataset, generates visualizations, 
and provides insights about the data.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parent.parent))

# Configure visualizations
plt.style.use('seaborn')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directory for plots
output_dir = Path(__file__).parent / "plots"
output_dir.mkdir(exist_ok=True)

def load_data():
    """Load the housing dataset"""
    try:
        # Try to import from the local module
        from src.data_processing import load_data as load_data_func
        df = load_data_func('../data/housing_data.csv')
    except ImportError:
        # Fallback to pandas if module import fails
        df = pd.read_csv('../data/housing_data.csv')
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def explore_data_overview(df):
    """Basic data overview"""
    print("\n=== Data Overview ===")
    
    # Display basic info
    print("\nData Types:")
    print(df.dtypes)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe().round(2))
    
    # Check for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nMissing values per column:")
        print(missing)
    else:
        print("\nNo missing values found.")

def explore_categorical_features(df):
    """Explore categorical features"""
    print("\n=== Categorical Features ===")
    
    # Identify categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns: {cat_columns}")
    
    # Check unique values
    for col in cat_columns:
        unique_vals = df[col].unique()
        print(f"\n{col}: {len(unique_vals)} unique values")
        print(unique_vals)
    
    # Plot state distribution
    plt.figure(figsize=(15, 6))
    state_counts = df['state'].value_counts()
    sns.barplot(x=state_counts.index, y=state_counts.values)
    plt.title('Count of Houses by State')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'state_distribution.png')
    
    # Plot house type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='house_type', data=df)
    plt.title('Distribution of House Types')
    plt.xlabel('House Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'house_type_distribution.png')
    
    # Plot neighborhood quality distribution
    plt.figure(figsize=(10, 6))
    quality_order = ['Low', 'Medium', 'High', 'Very High']
    sns.countplot(x='neighborhood_quality', data=df, order=quality_order)
    plt.title('Distribution of Neighborhood Quality')
    plt.xlabel('Neighborhood Quality')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'neighborhood_quality_distribution.png')

def explore_price_distribution(df):
    """Explore the target variable (price)"""
    print("\n=== Price Distribution ===")
    
    # Basic stats
    print(f"Min price: ${df['price'].min():,.2f}")
    print(f"Max price: ${df['price'].max():,.2f}")
    print(f"Mean price: ${df['price'].mean():,.2f}")
    print(f"Median price: ${df['price'].median():,.2f}")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Distribution of House Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.axvline(df['price'].median(), color='red', linestyle='--', label=f"Median: ${df['price'].median():,.0f}")
    plt.axvline(df['price'].mean(), color='green', linestyle='--', label=f"Mean: ${df['price'].mean():,.0f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'price_distribution.png')
    
    # Boxplot by state
    plt.figure(figsize=(16, 8))
    state_order = df.groupby('state')['price'].median().sort_values().index
    sns.boxplot(x='state', y='price', data=df, order=state_order)
    plt.title('House Price Distribution by State')
    plt.xlabel('State')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'price_by_state.png')

def explore_feature_relationships(df):
    """Explore relationships between features and price"""
    print("\n=== Feature Relationships ===")
    
    # Calculate correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numeric_df.corr()
    
    # Print top correlations with price
    price_corr = correlation['price'].sort_values(ascending=False)
    print("Top correlations with price:")
    print(price_corr)
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 12))
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png')
    
    # Plot price vs sqft
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='sqft', y='price', data=df, alpha=0.5)
    plt.title('House Price vs. Square Footage')
    plt.xlabel('Square Footage')
    plt.ylabel('Price ($)')
    plt.tight_layout()
    plt.savefig(output_dir / 'price_vs_sqft.png')
    
    # Plot price vs bedrooms
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='bedrooms', y='price', data=df)
    plt.title('House Price vs. Number of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price ($)')
    plt.tight_layout()
    plt.savefig(output_dir / 'price_vs_bedrooms.png')

def feature_engineering_exploration(df):
    """Explore potential engineered features"""
    print("\n=== Feature Engineering Exploration ===")
    
    # Create engineered features
    df['house_age'] = 2023 - df['year_built']
    df['price_per_sqft'] = df['price'] / df['sqft']
    df['lot_utilization'] = df['sqft'] / df['lot_size']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['rooms_per_sqft'] = df['total_rooms'] / df['sqft']
    
    # Check correlation with price
    engineered_features = [
        'house_age', 'price_per_sqft', 'lot_utilization', 
        'total_rooms', 'rooms_per_sqft'
    ]
    
    for feature in engineered_features:
        corr = df[feature].corr(df['price'])
        print(f"{feature} correlation with price: {corr:.4f}")
    
    # Plot price per sqft by state
    plt.figure(figsize=(16, 8))
    state_order = df.groupby('state')['price_per_sqft'].median().sort_values().index
    sns.boxplot(x='state', y='price_per_sqft', data=df, order=state_order)
    plt.title('Price per Square Foot by State')
    plt.xlabel('State')
    plt.ylabel('Price per Square Foot ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'price_per_sqft_by_state.png')
    
    # Plot house age vs price
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='house_age', y='price', data=df, alpha=0.5)
    plt.title('House Price vs. House Age')
    plt.xlabel('House Age (years)')
    plt.ylabel('Price ($)')
    plt.tight_layout()
    plt.savefig(output_dir / 'price_vs_age.png')

def generate_summary(df):
    """Generate a summary of key insights"""
    print("\n=== Summary and Key Insights ===")
    
    # Calculate key insights
    total_homes = len(df)
    avg_price = df['price'].mean()
    median_price = df['price'].median()
    avg_sqft = df['sqft'].mean()
    avg_price_per_sqft = (df['price'] / df['sqft']).mean()
    most_expensive_state = df.groupby('state')['price'].median().sort_values(ascending=False).index[0]
    least_expensive_state = df.groupby('state')['price'].median().sort_values().index[0]
    
    # Print summary
    print(f"Total homes in dataset: {total_homes}")
    print(f"Average home price: ${avg_price:,.2f}")
    print(f"Median home price: ${median_price:,.2f}")
    print(f"Average square footage: {avg_sqft:.1f} sqft")
    print(f"Average price per square foot: ${avg_price_per_sqft:.2f}")
    print(f"Most expensive state (by median price): {most_expensive_state}")
    print(f"Least expensive state (by median price): {least_expensive_state}")
    
    # Save summary to a file
    summary = {
        'total_homes': total_homes,
        'avg_price': avg_price,
        'median_price': median_price,
        'avg_sqft': avg_sqft,
        'avg_price_per_sqft': avg_price_per_sqft,
        'most_expensive_state': most_expensive_state,
        'least_expensive_state': least_expensive_state
    }
    
    # Save as text file
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("=== US Housing Data Summary ===\n\n")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                if key.startswith('avg_') or key.endswith('_price'):
                    f.write(f"{key}: ${value:,.2f}\n")
                else:
                    f.write(f"{key}: {value:,}\n")
            else:
                f.write(f"{key}: {value}\n")
                
    print(f"\nSummary saved to {output_dir / 'summary.txt'}")
    print(f"All plots saved to {output_dir}")

def main():
    """Main function to run the exploration"""
    print("=== US Housing Data Exploration ===")
    
    # Load data
    df = load_data()
    
    # Run exploration functions
    explore_data_overview(df)
    explore_categorical_features(df)
    explore_price_distribution(df)
    explore_feature_relationships(df)
    feature_engineering_exploration(df)
    generate_summary(df)

if __name__ == "__main__":
    main() 