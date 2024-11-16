# utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Preprocess the data: handle missing values, encode categorical features.
    """
    # Fill missing values (example: with median for numerical columns)
    data.fillna(data.median(), inplace=True)
    
    # Example of encoding categorical variables using Target Encoding
    encoder = ce.TargetEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = encoder.fit_transform(data[categorical_columns], data['Churn'])
    
    return data

def split_data(data, target_column):
    """
    Split the data into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
