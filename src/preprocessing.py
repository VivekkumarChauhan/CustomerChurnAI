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
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')  # Most frequent for categorical columns
    data['Churn Category'] = imputer.fit_transform(data[['Churn Category']])
    data['Churn Reason'] = imputer.fit_transform(data[['Churn Reason']])
    
    # Further handling for numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='mean')  # Mean for numerical columns
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    
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
        data[col] = label_enc.fit_transform(data[col])
    
    # One-Hot Encoding for non-ordinal categorical features (optional, depending on your dataset)
    # You can selectively apply this to other categorical columns if needed
    # data = pd.get_dummies(data, columns=['Gender', 'Married', 'Contract'], drop_first=True)
    
    # Handle scaling of numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data

def feature_engineering(data):
    """
    Add additional features or transformations to improve model performance.
    """
    # Example of adding a new feature: Interaction between tenure and monthly charges
    data['Tenure_MonthlyCharge_Interaction'] = data['Tenure in Months'] * data['Monthly Charge']
    
    return data
