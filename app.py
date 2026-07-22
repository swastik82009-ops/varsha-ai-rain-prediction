from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib, pandas as pd
import sklearn.compose._column_transformer
class _RemainderColsList(list): pass
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

app = Flask(__name__)
CORS(app)
model = joblib.load("weather_final_pipeline.pkl")

@app.route('/')
def home(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # IMPORTANT: Jo feature missing hai uska default de raha hu
    full_data = {
        'Location': data.get('Location','Sydney'), 'MinTemp': float(data.get('MinTemp',15)),
        'MaxTemp': float(data.get('MaxTemp',22)), 'Rainfall': float(data.get('Rainfall',0)),
        'Evaporation': 5.0, 'Sunshine': 8.0, 'WindGustDir': 'W', 'WindGustSpeed': 40,
        'WindDir9am': 'W', 'WindDir3pm': 'W', 'WindSpeed9am': 15, 'WindSpeed3pm': 20,
        'Humidity9am': float(data.get('Humidity9am',70)), 'Humidity3pm': float(data.get('Humidity3pm',70)),
        'Pressure9am': 1010, 'Pressure3pm': 1010, 'Cloud9am': 4, 'Cloud3pm': 5,
        'Temp9am': float(data.get('MinTemp',15))+2, 'Temp3pm': float(data.get('MaxTemp',22))-2,
        'RainToday': 'Yes' if float(data.get('Rainfall',0))>1 else 'No'
    }
    df = pd.DataFrame([full_data])
    prob = model.predict_proba(df)[0][1] # Yahi tera asli 86% wala result hai
    pred = model.predict(df)[0]
    return jsonify({"chance": int(prob*100), "result": int(pred)})

if __name__ == '__main__': app.run(host='0.0.0.0', port=10000)
