from flask import Flask, jsonify,render_template,request
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

app=Flask(__name__)

#load the model
# Load the model from file
with open('reliance.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    close_price = float(request.form['close'])

    # Create a DataFrame with the input values
    data = {'Open': [open_price], 'High': [high_price], 'Low': [low_price], 'Close': [close_price]}
    df = pd.DataFrame(data)

    # Predict the next day's closing price
    next_day_close = model.predict(df)[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': next_day_close})

if __name__ == '__main__':
    app.run(debug=True)