import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return np.log1p(X.values).reshape(-1, 1)  # Fix reshape issue
        return np.log1p(X) 


app = Flask(__name__)

regmodel = pickle.load(open('house_price_pipeline.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    data = pd.DataFrame(data, index=[0])
    prediction = regmodel.predict(data)
    print(np.expm1(prediction[0]))
    return jsonify(np.expm1(prediction[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data = request.form.to_dict()
    data['total_sqft'] = float(data['total_sqft'])
    data['bath'] = int(data['bath'])
    data = pd.DataFrame(data, index=[0])
    print(data) 
    prediction = regmodel.predict(data)
    print(np.expm1(prediction[0]))
    return render_template('home.html',prediction_text='House Price is {}'.format(np.expm1(prediction[0])))

if(__name__=='__main__'):
    app.run(debug=True)