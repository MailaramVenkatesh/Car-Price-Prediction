from flask import Flask, render_template,jsonify,request
import pandas as pd
import pickle 

app = Flask(__name__)


@app.route("/")
def Home():
    return render_template('Home.html')

@app.route('/predict',methods= ['GET','POST'])
def predict():
        if request.method=='POST':
            make = request.form.get("make")
            model=request.form.get('model')
            year=request.form.get('year')
            Engine_HP=request.form.get('hp')
            Engine_Cylinders=request.form.get('engineCylinders')
            df=pd.read_json('new.json')
            print(make,model,year,Engine_HP,Engine_Cylinders)
            make_encode= df['Make_encode'][df['Make'] == make].values[0]
            model_encode= df['Model_encode'][df['Model'] == model].values[0]



            with open('model.pkl', 'rb') as mod:
                 mlmodel = pickle.load(mod)

            predit = mlmodel.predict([[make_encode,model_encode,float(year),float(Engine_HP),float(Engine_Cylinders)]])

            return render_template('Predicted.html',predicted_value=predit[0])
        

        else:
            return render_template("predict.html")
   



if __name__=='__main__':
    app.run(host = '0.0.0.0')
