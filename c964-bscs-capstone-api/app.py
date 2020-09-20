import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)
CORS(app)

# routes
@app.route('/', methods=['POST'])

def predict():
    # read request body
    data = request.get_json(force=True)

    # convert request body into a dataframe
    data.update((x, [y]) for x, y in data.items())
    df = pd.DataFrame.from_dict(data)

    # predictions
    prediction = model.predict(df)
    
    # return data
    return jsonify(cost=int(prediction[0]))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)