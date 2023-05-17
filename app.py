from flask import Flask,request,render_template
import numpy as np
import pandas as pd



from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            Vehicle_condition=float(request.form.get('Vehicle_condition')),
            multiple_deliveries=float(request.form.get('multiple_deliveries')),
            Restaurant_Delivery_distance = float(request.form.get('Restaurant_Delivery_distance')),
            preparation_time=float(request.form.get('preparation_time')),
            Type_of_vehicle=(request.form.get('Type_of_vehicle')),
            Festival=(request.form.get('Festival')),
            Weather_conditions=(request.form.get('Weather_conditions')),
            City=(request.form.get('City'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True) 
