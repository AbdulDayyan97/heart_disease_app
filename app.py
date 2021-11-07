import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('svc_trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    features = [np.array(features)]
    prediction = model.predict(features)
    
    output = prediction[0]
    return render_template('index.html', prediction_text = 'Patient have a Heart Disease?: {}'.format('No' if output==0 else 'Yes'))

if __name__ == '__main__':
    app.run(debug=True)
