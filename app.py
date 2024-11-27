from flask import Flask,request,render_template
import joblib

model = joblib.load('final_model.pkl')

app=Flask(__name__)
                      
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/Prediction',methods=['POST'])
def detect():
   Satisfaction_Score=float(request.form['Satisfaction Score'])
   Discounted_Spend=float(request.form['Discounted Spend'])
   Spend_per_Item=float(request.form['Spend per Item'])
   Age=int(request.form['Age'])

   prediction=model.predict([[Satisfaction_Score,
                      Discounted_Spend,Spend_per_Item,Age]])
   return render_template('index.html',pred_res=prediction[0])

if __name__=='__main__':
  app.run(debug=True)