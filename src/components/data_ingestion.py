import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from dataclasses import dataclass

import pymongo
from pymongo import MongoClient


@dataclass
class DataIngestionConfig:
    
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            connection=MongoClient('localhost',27017)
            db=connection.zomatodb
            data=db.zomato
            df=pd.DataFrame(list(data.find()))
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            
            

            

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.raw_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)


