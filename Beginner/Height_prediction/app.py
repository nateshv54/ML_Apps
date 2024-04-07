from flask import Flask,render_template,request
import joblib

app=Flask(__name__)

#load the trained model
model=joblib.load('rf_model.pkl')

@app.route("/",methods=['GET','POST'])
def predict_height():
    predicted_height=None
    if request.method=='POST':
        weight=float(request.form['Weight'])
        #converting to pounds according to dataset
        weight1=weight*2.20406
        predict_height=model.predict([[weight1]])[0]
    return render_template('Height_prediction.html',predicted_height=predict_height)

if __name__=='__main__':
    app.run(debug=True)