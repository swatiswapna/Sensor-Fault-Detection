import sys 
import os 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd 
from sklearn.model_selection import train_test_split

logging.info("starting..")
@dataclass
class Dataclassconfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path  = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

class Datainjection:
    def __init__(self):
        self.injection_config = Dataclassconfig()

    def initiate_data_injection(self):
        try:
            logging.info("starting try block")
            df = pd.read_csv(os.path.join('notebook/data','wafer.csv'))
            os.makedirs(os.path.dirname(self.injection_config.raw_data_path),exist_ok=True)
            df.to_csv(self.injection_config.raw_data_path,index=False)

            logging.info("train test split starting .. ")

            train_set , test_set= train_test_split(df,test_size=0.2,random_state = 42)
            train_set.to_csv(self.injection_config.train_data_path,index= False,header=True)
            test_set.to_csv(self.injection_config.test_data_path,index= False,header=True)

            logging.info("train test split completed ")
            

            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )




        except Exception as e:
            logging.info("error occur in data injection")
            raise CustomException(e,sys)

