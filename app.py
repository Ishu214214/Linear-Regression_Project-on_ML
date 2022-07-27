
import numpy as np
from flask import Flask, request, jsonify, render_template           # render_template it is used to redirect

import pickle


app = Flask(__name__)
model = pickle.load(open('linearregression.pkl','rb')) 



@app.route('/')     # it is a deacouterator
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted salary for given experinace is : {}'.format(prediction))


app.run()