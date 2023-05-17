import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Delivery_person_Age: int,
        Delivery_person_Ratings: float,
        Road_traffic_density,
        Vehicle_condition,
        multiple_deliveries,
        Restaurant_Delivery_distance,
        preparation_time,
        Type_of_vehicle,
        Festival,
        Weather_conditions,
        City):

        self.Delivery_person_Age = Delivery_person_Age

        self.Delivery_person_Ratings = Delivery_person_Ratings

        self.Road_traffic_density = Road_traffic_density

        self.Vehicle_condition = Vehicle_condition

        self.multiple_deliveries = multiple_deliveries

        self.Restaurant_Delivery_distance = Restaurant_Delivery_distance

        self.preparation_time = preparation_time

        self.Type_of_vehicle = Type_of_vehicle

        self.Festival = Festival

        self.Weather_conditions = Weather_conditions

        self.City = City

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Road_traffic_density": [self.Road_traffic_density],
                "Vehicle_condition": [self.Vehicle_condition],
                "multiple_deliveries": [self.multiple_deliveries],
                "Restaurant_Delivery_distance": [self.Restaurant_Delivery_distance],
                "preparation_time": [self.preparation_time],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "Festival": [self.Festival],
                "Weather_conditions": [self.Weather_conditions],
                "City": [self.City]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
