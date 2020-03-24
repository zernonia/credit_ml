from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

# load the built-in model 
gbr = joblib.load('model.pkl')

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def get_pred():
    result=request.form
    gender = result['gender']
    age = result['age']
    debt = result['debt']
    married = result['married']
    bank = result['bank']
    edu = result['edu']
    ethnic = result['ethnic']
    yearsemp = result['yearsemp']
    priordef = result['priordef']
    employed = result['employed']
    creditscore = result['creditscore']
    citizen = result['citizen']

    # we create a json object that will hold data from user inputs
    user_input = {'Male':gender, 'Age':age, 'Debt':debt, 'Married':married, 'BankCustomer':bank,
                    'EducationLevel': edu, 'Ethnicity': ethnic, 'YearsEmployed': yearsemp , 
                    'PriorDefault': priordef, 'Employed': employed, 'CreditScore': creditscore,
                    'Citizen': citizen}

    # encode the json object to one hot encoding so that it could fit our model
    # a = input_to_one_hot(user_input)
    # get the price prediction
    list_input = [k for k in user_input.values()]
    data = np.array([np.asarray(list_input, dtype = float)])
    approved_pred = gbr.predict(data)[0]
    # return a json value
    return json.dumps({'Approved': approved_pred});

if __name__ == '__main__':
    app.run(debug=True)