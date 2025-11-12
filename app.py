from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Train simple model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return "Iris Classifier API is live on GCP!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    pred = model.predict(features)[0]
    return jsonify({"prediction": iris.target_names[pred]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
