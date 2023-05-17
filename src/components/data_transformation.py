import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from sklearn.pipeline import Pipeline #Pipeline Creation
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


from src.exception import CustomException
from src.logger import logging
import os
from geopy.distance import geodesic
from src.utils import save_object
from src.utils import filter_float_values

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,X):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            num_cols=X.select_dtypes(exclude='object').columns
            cat_cols=X.select_dtypes(include='object').columns

            weather_categories = ['Sunny','Stormy','Sandstorms','Windy','Cloudy','Fog']
            traffic_categories = ['Low','Medium','High','Jam']
            vehicleType_categories = ['electric_scooter','scooter','motorcycle']
            festival_categories=['No','Yes']
            city_categories= ['Urban','Metropolitian','Semi-Urban']

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[weather_categories,traffic_categories,vehicleType_categories,festival_categories,city_categories])),
                ('scaler',StandardScaler())
                ]

            )

            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {num_cols}")

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,cat_cols)
                ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_data_path):

        try:
            df = pd.read_csv(raw_data_path)

            logging.info("Read  data as dataframe")
            logging.info(f'Raw Dataframe Head : \n{df.head().to_string()}')
            logging.info("EDA and Feature Engineering Started")

            df = df.dropna()

            restaurant_coordinates = df[['Restaurant_latitude','Restaurant_longitude']].to_numpy()
            delivery_coordinates = df[['Delivery_location_latitude','Delivery_location_longitude']].to_numpy()

            df['distance(km)'] = ''
            for i in range(len(df)):
                df['distance(km)'].iloc[i]=geodesic(restaurant_coordinates[i],delivery_coordinates[i]).km
                
            df['Restaurant_Delivery_distance'] = df['distance(km)'].astype('float')
            logging.info('Added the distance column')
            df['Time_Orderd']=filter_float_values(df['Time_Orderd']).str.slice(0,5)
            df['Time_Order_picked']=filter_float_values(df['Time_Order_picked']).str.slice(0,5)
            ## Concatenating :00 atlast to make it date format and also we can calculate the preparation time using these values
            df['Time_Orderd']=df['Time_Orderd']+':00'
            df['Time_Order_picked']=df['Time_Order_picked']+':00'

            df=df.reset_index()
            df.drop(columns=['index'],axis=1,inplace=True)
            ## Let's calculate preparation time
            df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
            df['Time_Order_picked']=pd.to_timedelta(df['Time_Order_picked'])
            td = pd.Timedelta(1, "d") # to indicate 1 day

            df.loc[(df['Time_Order_picked'] < df['Time_Orderd']), 'preparation1'] = df['Time_Order_picked'] - df['Time_Orderd'] + td
            df.loc[(df['Time_Order_picked'] > df['Time_Orderd']), 'preparation2'] = df['Time_Order_picked'] - df['Time_Orderd']

            df['preparation1'].fillna(df['preparation2'],inplace=True)
            df['preparation_time(min)'] = pd.to_timedelta(df['preparation1'], "minute")

            for i in range(len(df['preparation_time(min)'])):
                df['preparation_time(min)'][i] = df['preparation_time(min)'][i].total_seconds()/60 # converting into minutes
            df['preparation_time'] = df['preparation_time(min)'].astype(float)

            df.drop(columns=['_id', 'ID', 'Delivery_person_ID','Restaurant_latitude',
            'Restaurant_longitude', 'Delivery_location_latitude',
            'Delivery_location_longitude', 'Order_Date', 'Time_Orderd',
            'Time_Order_picked','time','preparation1', 'preparation2','distance(km)','preparation_time(min)'],inplace=True)
            df.drop(columns='Type_of_order',inplace=True)

            logging.info("Splitting train and test split")
            logging.info(df.columns)
            X = df.drop(labels=['Time_taken (min)'],axis=1)
            Y = df[['Time_taken (min)']]



            preprocessing_obj=self.get_data_transformer_object(X)

            logging.info(f"Train Input features: \n {X.head().to_string()}")
            logging.info(f"Train Target features: \n {Y.head().to_string()}")

            input_feature_train_df,input_feature_test_df,target_feature_train_df,target_feature_test_df=train_test_split(X,Y,test_size=0.20,random_state=36)
            logging.info('Train and test split completed')
            
            # Saving train and test split data inside artifacts folder
            train_set = pd.concat([input_feature_train_df,target_feature_train_df],axis=1)
            test_set = pd.concat([input_feature_test_df,target_feature_test_df],axis=1)
            train_set.to_csv(self.data_transformation_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path,index=False,header=True)

            # Transformating using preprocessor obj
            input_feature_train_arr=pd.DataFrame(preprocessing_obj.fit_transform(input_feature_train_df),columns=preprocessing_obj.get_feature_names_out())
            input_feature_test_arr=pd.DataFrame(preprocessing_obj.transform(input_feature_test_df),columns=preprocessing_obj.get_feature_names_out())

            
            

            

            logging.info("Applying preprocessing object on training and testing datasets.")            

            train_arr = np.c_[input_feature_train_arr.to_numpy(), np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.to_numpy(), np.array(target_feature_test_df)] 

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)