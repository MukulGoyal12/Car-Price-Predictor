from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import pickle

car = pd.read_csv('cleaned_car.csv')
app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel.pkl','rb'))

def get_unique_values(column, reverse=False):
    return sorted(car[column].unique(), reverse=reverse)

@app.route('/')
def index():
    return render_template('index.html',
                         companies=get_unique_values('company'),
                         car_models=get_unique_values('name'),
                         years=get_unique_values('year', reverse=True),
                         fule_types=get_unique_values('fuel_type'))

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))
    print(company, car_model, year, fuel_type, kms_driven)

    prediction = model.predict(pd.DataFrame([[car_model,company,year,fuel_type,kms_driven]],columns=['name','company','year','fuel_type','kms_driven']))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)