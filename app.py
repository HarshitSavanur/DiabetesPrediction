from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import logging

app = Flask(__name__)
model = pickle.load(open('diabetes_model.pkl', 'rb'))                         

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/', methods= ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.form
            Pregnancies = int(data['Pregnancies'])
            Glucose = int(data['Glucose'])
            BloodPressure = int(data['BloodPressure'])
            SkinThickness = int(data['SkinThickness'])
            Insulin = int(data['Insulin'])
            BMI = float(data['BMI'])
            DiabetesPedigreeFunction = float(data['DiabetesPedigreeFunction'])
            Age = int(data['Age'])

            feature_values = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            prediction = model.predict(feature_values)

            return render_template('result.html', prediction=prediction)
        else:
            return jsonify({'error': 'Invalid request method'}), 405
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)