from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('../models/best_churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = pd.DataFrame(request.json)
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[:, 1]
    
    return jsonify({'Churn Prediction': prediction.tolist(), 'Probability': prob.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
