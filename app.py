from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained models
try:
    dtr = pickle.load(open(r'C:\Users\Admin\Downloads\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\dtr.pkl', 'rb'))
    preprocessor = pickle.load(open(r'C:\Users\Admin\Downloads\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\Crop-Yield-Prediction-Using-Machin-Learning-Python-main\preprocessor.pkl', 'rb'))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Flask app
app = Flask(__name__)

# Prediction function
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    
    # Transform the features using the preprocessor
    try:
        transformed_features = preprocessor.transform(features)
        # Make the prediction
        predicted_yield = dtr.predict(transformed_features)
        return predicted_yield[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

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
            
            # Debug: Print collected data
            print(f"Inputs received: Year={Year}, Rainfall={average_rain_fall_mm_per_year}, Pesticides={pesticides_tonnes}, Temp={avg_temp}, Area={Area}, Item={Item}")

            # Prediction
            result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)
            
            if result is not None:
                return render_template('index.html', prediction=result)
            else:
                return render_template('index.html', error="Prediction failed due to model error.")
        
        except ValueError as ve:
            print(f"Input error: {ve}")
            return render_template('index.html', error=f"Input error: {str(ve)}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return render_template('index.html', error=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
