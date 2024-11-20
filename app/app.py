from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model only once
model = joblib.load('D:/practice/Customer_Churn_Prediction/models1/best_churn_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form submission
        gender = request.form.get('gender')
        senior_citizen = request.form.get('senior_citizen')
        married = request.form.get('married')
        phone_service = request.form.get('phone_service')
        internet_service = request.form.get('internet_service')
        contract = request.form.get('contract')
        tenure_months = request.form.get('tenure_months')
        monthly_charge = request.form.get('monthly_charge')
        total_charges = request.form.get('total_charges')

        # Convert input data to correct types
        try:
            tenure_months = float(tenure_months)
            monthly_charge = float(monthly_charge)
            total_charges = float(total_charges)
            senior_citizen = int(senior_citizen)
        except ValueError:
            return jsonify({'error': 'Invalid numerical value entered'}), 400

        # Prepare input data as a dictionary
        input_data = {
            'Gender': gender,
            'Senior Citizen': senior_citizen,
            'Married': married,
            'Phone Service': phone_service,
            'Internet Service': internet_service,
            'Contract': contract,
            'Tenure in Months': tenure_months,
            'Monthly Charge': monthly_charge,
            'Total Charges': total_charges
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure the features are in the correct order
        expected_columns = ['Gender', 'Senior Citizen', 'Married', 'Phone Service', 'Internet Service', 'Contract', 
                            'Tenure in Months', 'Monthly Charge', 'Total Charges']
        input_df = input_df[expected_columns]

        # Predict using the model
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[:, 1]  # Get the probability of class 1 (churn)

        # Return the prediction and probability to the template
        return render_template('index.html', prediction=prediction[0], prob=prob[0])

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
