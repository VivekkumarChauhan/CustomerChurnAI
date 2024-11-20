import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as ce

def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    data = pd.read_csv(filepath)
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values by filling them based on the specified strategy.
    Default strategy: 'mean' for numerical and 'most_frequent' for categorical.
    """
    # Handle missing values in numerical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if strategy == 'mean':
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    elif strategy == 'median':
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    elif strategy == 'mode':
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mode().iloc[0])
    
    # Handle missing values in categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    
    return data

def encode_categorical_features(data, target_column=None):
    """
    Encode categorical features using the appropriate encoding method.
    Uses OneHotEncoder for nominal features and TargetEncoder for others.
    """
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # OneHotEncoding for nominal categories
    onehot_cols = [col for col in categorical_cols if len(data[col].value_counts()) > 2]
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)

    # Target Encoding for ordinal or other categorical columns
    target_cols = [col for col in categorical_cols if col not in onehot_cols]
    if target_column and target_cols:
        encoder = ce.TargetEncoder()
        data[target_cols] = encoder.fit_transform(data[target_cols], data[target_column])
    
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Performs feature scaling using StandardScaler on numerical features.
    """
    # Split into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale numerical features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test

def preprocess_data(data, target_column, missing_value_strategy='mean'):
    """
    Preprocess the data: 
    - Handle missing values
    - Encode categorical features
    - Split into train/test datasets.
    """
    # Handle missing values
    data = handle_missing_values(data, strategy=missing_value_strategy)
    
    # Encode categorical features
    data = encode_categorical_features(data, target_column)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    
    return X_train, X_test, y_train, y_test
