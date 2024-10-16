from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained models
dtr = pickle.load(open(r'C:\Users\Admin\Downloads\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\dtr.pkl', 'rb'))

preprocessor = pickle.load(open(r'C:\Users\Admin\Downloads\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\preprocessor.pkl', 'rb'))


# Flask app
app = Flask(__name__)

# Prediction function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = preprocessor.transform(features)

    # Make the prediction
    predicted_yield = dtr.predict(transformed_features)

    return predicted_yield[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form data
            Year = float(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']

            # Prediction
            result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)

            return render_template('index.html', prediction=result)
        
        except ValueError as ve:
            return render_template('index.html', error=f"Input error: {str(ve)}")

if __name__ == "__main__":
    app.run(debug=True)
