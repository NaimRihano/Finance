from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

def predict_customer_default(input_data):
    # Load the trained model, column transformer, and scaler
    model = joblib.load('best_model.pkl')
    column_transformer = joblib.load('column_transformer.pkl')
    scaler = joblib.load('scaler.pkl')

    # Step 3: Apply column transformer to encode categorical features
    X_encoded = column_transformer.transform(input_data)

    # Step 4: Scale the features using the saved StandardScaler
    X_scaled = scaler.transform(X_encoded)

    # Step 5: Make the prediction using the trained model
    predicted_class = model.predict(X_scaled)

    # Convert binary result to categorical
    if predicted_class == 1:
        prediction = "Approved"
    else:
        prediction = "Denied"
    
    # Confidence score can be obtained from model's prediction probabilities
    confidence_score = np.max(model.predict_proba(X_scaled))

    # Format the confidence score as a percentage without decimals
    confidence_percentage = f"{int(confidence_score * 100)}%"

    result_message = f"The credit card application has been {prediction} with a confidence score of {confidence_percentage}."
    return prediction, confidence_percentage

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    gender = request.form.get('gender')
    own_car = request.form.get('own_car')
    own_realty = request.form.get('own_realty')
    cnt_children = int(request.form.get('cnt_children'))
    amt_income_total = float(request.form.get('amt_income_total'))
    income_type = request.form.get('income_type')
    education_type = request.form.get('education_type')
    family_status = request.form.get('family_status')
    housing_type = request.form.get('housing_type')
    days_employed = int(request.form.get('days_employed'))
    work_phone = int(request.form.get('work_phone'))
    phone = int(request.form.get('phone'))
    email = int(request.form.get('email'))
    occupation_type = request.form.get('occupation_type')
    cnt_fam_members = float(request.form.get('cnt_fam_members'))
    age_years = int(request.form.get('age_years'))

    # Create a DataFrame from the input data
    user_data = pd.DataFrame({
        'GENDER': [gender],
        'OWN_CAR': [own_car],
        'OWN_REALTY': [own_realty],
        'CNT_CHILDREN': [cnt_children],
        'AMT_INCOME_TOTAL': [amt_income_total],
        'INCOME_TYPE': [income_type],
        'EDUCATION_TYPE': [education_type],
        'FAMILY_STATUS': [family_status],
        'HOUSING_TYPE': [housing_type],
        'DAYS_EMPLOYED': [days_employed],
        'WORK_PHONE': [work_phone],
        'PHONE': [phone],
        'EMAIL': [email],
        'OCCUPATION_TYPE': [occupation_type],
        'CNT_FAM_MEMBERS': [cnt_fam_members],
        'AGE_YEARS': [age_years]
    })

    # Get the prediction and confidence score
    prediction, confidence_score = predict_customer_default(user_data)
    
    return jsonify({
        'prediction': prediction,
        'confidence_score': confidence_score
    })

if __name__ == '__main__':
    app.run(debug=True)
