import os 
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# import yaml


class MainUtils:
    def __init__(self):
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys) from e
        

        
# @staticmethod
def save_object(file_path , obj):
    try:
        dir_path =os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e ,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    report = {}
    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_pred)
            report[model_name] = test_model_score

            

        return report
    except Exception as e:
        logging.info('Exception occurred in utils model training ')
        return None
    
def load_object(file_path:str):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in utils load model ')
        return None
    


