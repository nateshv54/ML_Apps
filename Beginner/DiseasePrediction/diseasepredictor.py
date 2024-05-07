from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the dataset
data = pd.read_csv('dataset.csv')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    cure = None
    alert = False
    if request.method == 'POST':
        # Get the input symptoms from the form
        symptoms = request.form['symptoms']
        
        # Only make prediction if symptoms are provided
        if symptoms:
            # Convert the input symptoms into bag-of-words representation
            symptoms_vectorized = vectorizer.transform([symptoms])
            
            # Make a prediction
            result = model.predict(symptoms_vectorized)[0]
            
            # Get the probability estimates for each class
            confidence = model.predict_proba(symptoms_vectorized).max() * 100
            
            # Get the cure for the predicted disease
            cure = data.loc[data['disease'] == result, 'cures'].values[0]
            
            # Check if the risk level is high
            risk_level = data.loc[data['disease'] == result, 'risk level'].values[0]
            if 'high' in risk_level:
                alert = True
    
    # Render the form with the result, confidence, cure, and alert
    return render_template('index.html', result=result, confidence=confidence, cure=cure, alert=alert)

if __name__ == '__main__':
    app.run(debug=True)
