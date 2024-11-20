import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset (update the path as needed)
data_path = 'D:/practice/Customer_Churn_Prediction/data/processed/cleaned_data.csv'

# Read the dataset
try:
    data = pd.read_csv(data_path)
    print(f"Dataset loaded successfully from {data_path}")
except FileNotFoundError:
    print(f"Error: The dataset file was not found at {data_path}. Please check the path.")
    exit()

# Ensure the 'Churn Label' column exists in the dataset
if 'Churn Label' not in data.columns:
    print("Error: 'Churn Label' column not found in the dataset.")
    print("Available columns:", data.columns)
    exit()

# Preprocessing (adjust as per your requirements)
# Let's assume 'Churn Label' is the target column and others are features
X = data.drop(columns=['Churn Label'])
y = data['Churn Label']

# Convert categorical columns to numeric using LabelEncoder or OneHotEncoder
# Identify categorical columns (example: 'Gender', 'Country', 'State', etc.)
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
label_encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and check the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model to disk
model_path = 'D:/practice/Customer_Churn_Prediction/models1/best_churn_model.pkl'
joblib.dump(model, model_path)
print(f"Model has been retrained and saved successfully at {model_path}")
