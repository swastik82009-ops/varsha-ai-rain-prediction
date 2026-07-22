from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sklearn.compose._column_transformer

class _RemainderColsList(list):
    pass

if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

app = Flask(__name__)
CORS(app)

# Model load
model = joblib.load('weather_final_pipeline.pkl')

@app.route('/')
def home():
    return jsonify({"status": "Varsha AI Backend is Live!", "model": "loaded"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # DataFrame bana ke predict - kyunki tera pipeline hai
        import pandas as pd
        df = pd.DataFrame([data])
        prob = model.predict_proba(df)[0][1] * 100

        return jsonify({
            'prob': round(float(prob), 1),
            'willRain': bool(prob > 50)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
