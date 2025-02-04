import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
import pickle

app = Flask(__name__)

# Load the model
with open('voting_clf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('code.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    alcohol = float(request.form['alcohol'])
    malic_acid = float(request.form['malic_acid'])
    ash = float(request.form['ash'])
    alcalinity_of_ash = float(request.form['alcalinity_of_ash'])
    magnesium = float(request.form['magnesium'])
    total_phenols = float(request.form['total_phenols'])
    flavanoids = float(request.form['flavanoids'])
    nonflavanoid_phenols = float(request.form['nonflavanoid_phenols'])
    proanthocyanins = float(request.form['proanthocyanins'])
    color_intensity = float(request.form['color_intensity'])
    hue = float(request.form['hue'])
    od280_od315_of_diluted_wines = float(request.form['od280/od315_of_diluted_wines'])
    proline = float(request.form['proline'])

    input_data = np.array([alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline])

    df = pd.read_csv("x_model.csv")
    col = df.columns.to_list()
    input_data = pd.DataFrame([input_data], columns=col)
    input_data = input_data.reindex(columns=col, fill_value=0)

    prediction = model.predict(input_data)
    prediction_text = f"Predicted Wine Class: {prediction[0]}"

    return render_template('code.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=False)
