from flask import Flask, render_template, request
import json
from predict import train_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form['company']
    feature = request.form['feature']
    result = train_and_predict(company, feature)
    return render_template('result.html', result=json.dumps(result), company=company, feature=feature)

if __name__ == '__main__':
    app.run(debug=True)
