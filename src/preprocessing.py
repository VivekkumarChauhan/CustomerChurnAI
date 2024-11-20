import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def clean_data(filepath):
    """
    Clean the dataset by handling missing values and initial checks.
    """
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Handle missing values using SimpleImputer for categorical columns (most frequent)
    categorical_cols = ['Churn Category', 'Churn Reason']
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])
    
    # Handle missing values for numerical columns (mean strategy)
    numeric_cols = data.select_dtypes(include=np.number).columns
    imputer_num = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])
    
    return data

def preprocess_data(data):
    """
    Preprocess the data by encoding categorical features, scaling numerical features,
    and performing feature engineering if necessary.
    """
    # Define categorical and numerical columns
    categorical_cols = ['Gender', 'Senior Citizen', 'Married', 'Phone Service', 'Internet Service', 'Contract']
    numerical_cols = ['Tenure in Months', 'Monthly Charge', 'Total Charges']
    
    # Initialize LabelEncoder for binary or ordinal categories
    label_enc = LabelEncoder()
    for col in categorical_cols:
        if data[col].dtype == 'object':  # Label encoding only for object columns (binary/ordinal)
            data[col] = label_enc.fit_transform(data[col])

    # One-Hot Encoding for non-ordinal categorical features
    # Select columns that are nominal and apply one-hot encoding to them
    onehot_cols = ['Phone Service', 'Internet Service', 'Contract']
    data = pd.get_dummies(data, columns=onehot_cols, drop_first=True)
    
    # Handle scaling of numerical features using StandardScaler
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data

def feature_engineering(data):
    """
    Add additional features or transformations to improve model performance.
    """
    # Example of adding a new feature: Interaction between tenure and monthly charges
    data['Tenure_MonthlyCharge_Interaction'] = data['Tenure in Months'] * data['Monthly Charge']
    
    # Additional feature: Total Charges per Month (if Total Charges exist in the dataset)
    if 'Total Charges' in data.columns:
        data['Charges_Per_Month'] = data['Total Charges'] / (data['Tenure in Months'] + 1e-6)  # Adding small epsilon to avoid division by zero
    
    return data

def build_pipeline():
    """
    Create a pipeline that includes imputation, encoding, scaling, and model training.
    """
    # Define categorical and numerical columns
    categorical_cols = ['Gender', 'Senior Citizen', 'Married', 'Phone Service', 'Internet Service', 'Contract']
    numerical_cols = ['Tenure in Months', 'Monthly Charge', 'Total Charges']

    # Imputer for numerical and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ])
    
    # Combine transformers into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

# Example usage of the pipeline
# preprocessor = build_pipeline()
# X = preprocessor.fit_transform(data)  # Apply preprocessing pipeline to data
