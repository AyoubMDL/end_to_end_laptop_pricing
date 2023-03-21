import pickle
from flask import Flask, render_template, request, app, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
reg_model = pickle.load(open("regmodel.pkl", 'rb'))
scaler = pickle.load(open("scaling.pkl", 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = reg_model.predict(scaled_data)
    print(prediction[0])
    return jsonify(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input_data = scaler.transform(np.array(data).reshape(1, -1))
    prediction = reg_model.predict(input_data)[0]

    return render_template('home.html', prediction_text=f"The house price prediction is {prediction}")


if __name__ == "__main__":
    app.run(debug=True)