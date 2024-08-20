import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
def load_model():
    model_path = 'logistic_regression_model.pkl'
    scaler_path = 'scaler.pkl'

    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Model or scaler file not found. Please ensure '{model_path}' and '{scaler_path}' are in the correct location.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_diabetes(input_data, model, scaler):
    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 
        'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'Sex', 
        'Age', 'Income'
    ]
    
    # Ensure all expected features are included in the input_data
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0
    
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect user input
            input_data = {
                'Age': int(request.form.get('age', 0)),
                'BMI': float(request.form.get('bmi', 0)),
                'HighBP': int(request.form.get('highbp', 0)),
                'HighChol': int(request.form.get('highchol', 0)),
                'CholCheck': int(request.form.get('cholcheck', 0)),
                'Smoker': int(request.form.get('smoker', 0)),
                'Stroke': int(request.form.get('stroke', 0)),
                'HeartDiseaseorAttack': int(request.form.get('heartdisease', 0)),
                'PhysActivity': int(request.form.get('physactivity', 0)),
                'HvyAlcoholConsump': int(request.form.get('hvyalcohol', 0)),
                'AnyHealthcare': int(request.form.get('anyhealthcare', 0)),
                'NoDocbcCost': int(request.form.get('nodocbccost', 0)),
                'GenHlth': int(request.form.get('genhlth', 1)),
                'MentHlth': int(request.form.get('menthlth', 1)),
                'Sex': int(request.form.get('sex', 0)),
                'Income': int(request.form.get('income', 1))
            }

            print(f"Received input_data: {input_data}")

            # Convert numerical values to human-readable strings
            input_data_readable = {
                'Age': input_data['Age'],
                'BMI': input_data['BMI'],
                'HighBP': 'Yes' if input_data['HighBP'] == 1 else 'No',
                'HighChol': 'Yes' if input_data['HighChol'] == 1 else 'No',
                'CholCheck': 'Yes' if input_data['CholCheck'] == 1 else 'No',
                'Smoker': 'Yes' if input_data['Smoker'] == 1 else 'No',
                'Stroke': 'Yes' if input_data['Stroke'] == 1 else 'No',
                'HeartDiseaseorAttack': 'Yes' if input_data['HeartDiseaseorAttack'] == 1 else 'No',
                'PhysActivity': 'Yes' if input_data['PhysActivity'] == 1 else 'No',
                'HvyAlcoholConsump': 'Yes' if input_data['HvyAlcoholConsump'] == 1 else 'No',
                'AnyHealthcare': 'Yes' if input_data['AnyHealthcare'] == 1 else 'No',
                'NoDocbcCost': 'Yes' if input_data['NoDocbcCost'] == 1 else 'No',
                'GenHlth': ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][input_data['GenHlth'] - 1],
                'MentHlth': ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][input_data['MentHlth'] - 1],
                'Sex': 'Female' if input_data['Sex'] == 1 else 'Male',
                'Income': [
                    'Less than 10000 rupees (per month)',
                    '10000 to 25000 rupees (per month)',
                    '25000 to 35000 rupees (per month)',
                    '35000 to 50000 rupees (per month)',
                    '50000 to 75000 rupees (per month)',
                    '75000 to 100000 rupees (per month)',
                    'More than 100000 rupees (per month)'
                ][input_data['Income'] - 1]
            }

            # Load model and scaler
            model, scaler = load_model()

            # Predict diabetes
            prediction = predict_diabetes(input_data, model, scaler)
            result_class = 'Diabetes' if prediction == 1 else 'No Diabetes'

            print(f"Prediction: {result_class}")

            # Generate recommendations
            recommendations = generate_recommendations(input_data, prediction)

            return render_template('result.html', 
                                   input_data=input_data_readable, 
                                   result_class=result_class, 
                                   recommendations=recommendations)
        except Exception as e:
            print(f"Error: {e}")
            return str(e)
    return render_template('index.html')

def generate_recommendations(input_data, prediction):
    advice = []
    if input_data['HighBP'] == 'Yes':
        advice.append("Managing blood pressure is crucial. Consider lifestyle changes and consult a healthcare provider.")
    if input_data['HighChol'] == 'Yes':
        advice.append("High cholesterol can lead to cardiovascular issues. A balanced diet and exercise are recommended.")
    if input_data['BMI'] >= 30:
        advice.append("Your BMI indicates obesity, a major risk factor for diabetes. Consider a weight management plan.")
    if input_data['PhysActivity'] == 'No':
        advice.append("Incorporate physical activity into your daily routine to reduce diabetes risk.")
    if input_data['Smoker'] == 'Yes':
        advice.append("Smoking is a risk factor for many chronic diseases. Quitting smoking can significantly improve your health.")
    if input_data['HeartDiseaseorAttack'] == 'Yes':
        advice.append("A history of heart disease increases your risk of further cardiovascular issues. Regular check-ups and a heart-healthy lifestyle are recommended.")
    
    if prediction == 1:
        advice.insert(0, "The model predicts a higher risk of diabetes. Regular monitoring and consultation with a healthcare provider are advised.")
    else:
        advice.insert(0, "The model predicts a lower risk of diabetes. Continue maintaining a healthy lifestyle.")
    
    return " ".join(advice)

if __name__ == '__main__':
    app.run(debug=True)
